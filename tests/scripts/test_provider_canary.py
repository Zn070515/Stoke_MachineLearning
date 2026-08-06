"""Provider canary tests — pure offline (mocked fetch, NO network).

Validates ``fingerprint_frame`` / ``check_provider`` and the ``main()``
orchestration against a monkeypatched provider fetch.  Deliberately NOT marked
``network``/``slow`` — this is a normal fast offline test.
"""
import json

import numpy as np
import pandas as pd

import scripts.ops.provider_canary as canary
from stoke_ml.data.contract import get_contract

CONTRACT = get_contract("daily_equity")

TRADE_DATES = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]


def _fixture() -> pd.DataFrame:
    """A well-formed research_qfq_daily frame: qfq OHLC, volume in 股, amount in
    元, finite, correct pct_change (first day NaN), provenance in attrs."""
    closes = np.array([10.0, 10.5, 10.2, 10.8, 10.4], dtype="float64")
    df = pd.DataFrame({
        "date": pd.to_datetime(TRADE_DATES),
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": 1000.0,
        "amount": 1000.0 * closes,  # implied VWAP == close (well inside the band)
        "pct_change": pd.Series(closes).pct_change().mul(100.0).to_numpy(),
        "stock_code": ["600519"] * len(closes),
    })
    df.attrs["source"] = "test_provider"
    df.attrs["adjustment_mode"] = "qfq"
    return df


def _provider_with(monkeypatch, frame, available=True):
    """A REAL provider instance (constructs offline) with its online methods
    monkeypatched — no network, no online deps required."""
    from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource

    p = EfinanceSource()
    monkeypatch.setattr(p, "fetch_daily", lambda stock, start, end: frame)
    monkeypatch.setattr(p, "is_available", lambda: available)
    return p


class TestFingerprintFrame:
    def test_fingerprint_is_deterministic(self):
        df = _fixture()
        assert canary.fingerprint_frame(df) == canary.fingerprint_frame(df)

    def test_fingerprint_schema(self):
        df = _fixture()
        fp = canary.fingerprint_frame(df)
        assert fp["row_count"] == len(df)
        assert fp["source"] == "test_provider"
        assert fp["adjustment_mode"] == "qfq"
        # Column-name keys sorted; dtype tags canonicalized.
        assert list(fp["columns"]) == sorted(df.columns)
        assert fp["columns"]["date"] == "datetime64"
        assert fp["columns"]["stock_code"] == "string"
        assert fp["columns"]["volume"] == "float64"
        assert fp["columns"]["close"] == "float64"


class TestCheckProvider:
    def test_clean_frame_has_no_issues(self):
        fp, issues = canary.check_provider(_fixture(), CONTRACT)
        assert issues == []
        assert fp["row_count"] == 5

    def test_renamed_required_column_is_drift(self):
        df = _fixture().drop(columns=["volume"]).rename(columns={"volume": "vol"})
        _fp, issues = canary.check_provider(df, CONTRACT)
        assert issues
        assert any("missing_column:volume" in i for i in issues)

    def test_hand_scale_volume_is_drift(self):
        # volume 手-scale (~150x too small) pushes implied VWAP ~150x above the
        # qfq close — outside the contract's loose 100x diagnostic band.
        df = _fixture()
        df["volume"] = df["volume"] / 150.0
        _fp, issues = canary.check_provider(df, CONTRACT)
        assert issues
        assert any("amount_volume_unit_mismatch" in i for i in issues)


class TestMainOffline:
    def test_main_passes_and_writes_snapshot(self, tmp_path, monkeypatch):
        provider = _provider_with(monkeypatch, _fixture())
        monkeypatch.setattr(canary, "_build_providers", lambda: [provider])
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 0
        snap = json.loads(
            (tmp_path / "efinance__600519.json").read_text(encoding="utf-8")
        )
        assert snap["passed"] is True
        assert snap["issues"] == []
        assert snap["source"] == "efinance"  # provider name, not the frame attr
        assert snap["fingerprint"]["source"] == "test_provider"
        assert snap["fingerprint"]["row_count"] == 5

    def test_main_skips_unavailable_provider(self, tmp_path, monkeypatch):
        provider = _provider_with(monkeypatch, _fixture(), available=False)
        monkeypatch.setattr(canary, "_build_providers", lambda: [provider])
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 0  # unavailable providers are skipped, never a failure
        assert list(tmp_path.iterdir()) == []  # no snapshots written

    def test_main_fails_on_drift(self, tmp_path, monkeypatch):
        drifted = _fixture().drop(columns=["volume"])
        provider = _provider_with(monkeypatch, drifted)
        monkeypatch.setattr(canary, "_build_providers", lambda: [provider])
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 1
        snap = json.loads(
            (tmp_path / "efinance__600519.json").read_text(encoding="utf-8")
        )
        assert snap["passed"] is False
        assert snap["issues"]

    def test_main_fetch_exception_fails_cleanly(self, tmp_path, monkeypatch):
        """A fetch that RAISES fails the probe (rc 1) with no crash and no
        snapshot written for that probe."""
        from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource

        def _boom(stock, start, end):
            raise RuntimeError("provider blew up")

        p = EfinanceSource()
        monkeypatch.setattr(p, "fetch_daily", _boom)
        monkeypatch.setattr(p, "is_available", lambda: True)
        monkeypatch.setattr(canary, "_build_providers", lambda: [p])
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 1
        assert list(tmp_path.iterdir()) == []

    def test_main_is_available_exception_skips_provider(self, tmp_path, monkeypatch):
        """is_available() RAISING must SKIP the provider (never a failure),
        leaving rc == 0 with no snapshots written."""
        from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource

        def _boom():
            raise RuntimeError("availability probe failed")

        p = EfinanceSource()
        monkeypatch.setattr(p, "fetch_daily", lambda stock, start, end: _fixture())
        monkeypatch.setattr(p, "is_available", _boom)
        monkeypatch.setattr(canary, "_build_providers", lambda: [p])
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 0
        assert list(tmp_path.iterdir()) == []

    def test_main_cancels_watchdog_on_success(self, tmp_path, monkeypatch):
        """Regression: the watchdog Timer armed by main() must be CANCELLED on
        the normal return path.  If not, the default 300s daemon os._exit(2)
        timer fires LATER — long after main returned — killing the whole pytest
        process mid-suite (the full `-m ""` run exceeds 5 minutes)."""
        from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource

        p = EfinanceSource()
        monkeypatch.setattr(p, "fetch_daily", lambda stock, start, end: _fixture())
        monkeypatch.setattr(p, "is_available", lambda: True)
        monkeypatch.setattr(canary, "_build_providers", lambda: [p])

        started: list[float] = []
        cancelled: list[float] = []

        class _FakeTimer:
            def __init__(self, interval, fn):
                self.interval = interval
                self.fn = fn

            def start(self):
                started.append(self.interval)

            def cancel(self):
                cancelled.append(self.interval)

        monkeypatch.setattr(canary.threading, "Timer", _FakeTimer)
        rc = canary.main(["--state-dir", str(tmp_path)])
        assert rc == 0
        assert started == [canary.DEFAULT_TIMEOUT]
        assert cancelled == [canary.DEFAULT_TIMEOUT]


class TestCanonicalDtypeSync:
    """The canary replicates ``asset_contract._canonical_dtype``; keep them in
    sync so a silent divergence breaks CI here instead of corrupting the schema
    fingerprint the canary records on a live run."""

    def test_matches_asset_contract_across_representative_dtypes(self):
        from stoke_ml.data.asset_contract import _canonical_dtype as ref

        cases = {
            "datetime64[M]": np.dtype("datetime64[M]"),
            "datetime64[ns]": pd.Series(["2026-01-01"]).astype("datetime64[ns]").dtype,
            "object_string": pd.Series(["abc"], dtype=object).dtype,
            "pandas_string": pd.Series(["abc"], dtype="string").dtype,
            "float64": pd.Series([1.0], dtype="float64").dtype,
            "int64": pd.Series([1], dtype="int64").dtype,
            "bool": pd.Series([True], dtype=bool).dtype,
        }
        for name, dtype in cases.items():
            assert canary._canonical_dtype(dtype) == ref(dtype), name
