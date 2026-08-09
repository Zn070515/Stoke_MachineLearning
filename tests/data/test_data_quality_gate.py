"""Data quality gate tests.

The gate must FAIL when any check records a problem: read errors, missing
columns, or data inconsistencies must flip ``passed`` to False (a problem
recorded in the report must also flip the gate).  Sparsity uses NaN-excluded
coverage because ``(x != 0).mean()`` counts NaN as non-zero and inflates
coverage for missing-heavy features.

The required-dataset pre-gate must FAIL on empty/missing data by
default (0 file / 0 row = FAIL), only --allow-empty permitting it; and quick
sampling must be exchange-stratified, not biased toward low-code stocks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.production.data_quality_gate as gate_mod
from stoke_ml.data.calendar import (
    calendar_artifact_hash,
    get_research_calendar,
    load_calendar,
    most_recent_completed_trading_day,
    save_calendar,
)
from stoke_ml.data.storage import _provenance_from_attrs, _schema_hash
from scripts.production.data_quality_gate import (
    CHECKS,
    CheckResult,
    _manifest_contract_full_scan,
    _read_requested_file,
    _sample_files,
    check_aux_close_aligned,
    check_contract_schema,
    check_daily_internal,
    check_datasets,
    check_feature_pct,
    check_manifest,
    check_ohlc_sanity,
    check_sparsity,
    check_universe,
    contract_version,
    dataset_fingerprint,
    reconcile_requested_universe,
)


def _run_gate(argv, monkeypatch):
    """Invoke the gate's main() with a synthetic argv; returns its exit code."""
    monkeypatch.setattr(sys, "argv", argv)
    return gate_mod.main()


def _daily(dates, closes, code="000001", stock_code=None, **extra):
    """A well-formed daily frame: consistent OHLC, correct pct_change."""
    closes = pd.Series(closes, dtype="float64")
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": 1000.0,
        "amount": 10000.0,
        "turnover": 1.0,
        "pct_change": closes.pct_change() * 100.0,
        "stock_code": stock_code or code,
    })
    for k, v in extra.items():
        df[k] = v
    return df


def _write_daily(daily_dir, code, df, manifest=None):
    """Write a daily parquet + strongly-bound manifest sidecar, mirroring
    DataStorage: the gate must satisfy required_metadata from the sidecar."""
    df.to_parquet(daily_dir / f"{code}.parquet", index=False)
    mf = {"source": "unknown", "adjust": "unknown"} if manifest is None else manifest
    (daily_dir / f"{code}.manifest.json").write_text(
        json.dumps(mf), encoding="utf-8"
    )


def _write_daily_full(daily_dir, code, df, **overrides):
    """Parquet + a FULL contract manifest (rows/start/end/schema-hash/
    provenance) that ``validate_manifest`` accepts, mirroring the downloader's
    own write path.  ``overrides`` corrupt a single declared field on purpose.
    """
    df.to_parquet(daily_dir / f"{code}.parquet", index=False)
    prov = _provenance_from_attrs(df)
    segs = [{
        "source": prov["source"], "adjust": prov["adjust"],
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
    }]
    mf = {
        "stock": code,
        "start": segs[0]["start"],
        "end": segs[0]["end"],
        "rows": segs[0]["rows"],
        "source": prov["source"],
        "adjust": prov["adjust"],
        "units": prov["units"],
        "price_basis": prov["price_basis"],
        "calendar_version": prov["calendar_version"],
        "dataset_version": prov["dataset_version"],
        "schema_hash": _schema_hash(df),
        "source_segments": segs,
        "run_id": "test",
        "updated": "2026-08-04T00:00:00+00:00",
    }
    mf.update(overrides)
    (daily_dir / f"{code}.manifest.json").write_text(
        json.dumps(mf), encoding="utf-8"
    )


TRADE_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


class TestSparsityNaNCoverage:
    def test_nan_is_not_counted_as_nonzero(self, tmp_path, monkeypatch):
        """NaN != 0 is True, so it must be excluded from coverage."""
        feat_dir = tmp_path / "feat"
        feat_dir.mkdir()
        # 200 NaN + 1 non-zero + a few zeros: effective non-zero < 0.5%.
        n = 200
        df = pd.DataFrame({
            "date": pd.to_datetime(pd.date_range("2024-01-01", periods=n + 3, freq="D")),
            "sparse_feat": [np.nan] * n + [1.0, 0.0, 0.0],
            "dense_feat": np.arange(n + 3, dtype="float64"),
        })
        df.to_parquet(feat_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_sparsity(0)
        # Old (x != 0).mean() would report sparse_feat nz ~= 1.0 (not sparse);
        # NaN-excluded effective non-zero = 1/203 -> sparse.
        sparse_cols = {file for file, _detail in res.issues}
        assert "sparse_feat" in sparse_cols
        assert "dense_feat" not in sparse_cols
        assert "avg_finite_cov" in res.summary

    def test_sparsity_is_informational(self, tmp_path, monkeypatch):
        """Sparsity never fails the gate by itself."""
        feat_dir = tmp_path / "feat"
        feat_dir.mkdir()
        df = pd.DataFrame({
            "date": pd.to_datetime(pd.date_range("2024-01-01", periods=5, freq="D")),
            "x": np.arange(5, dtype="float64"),
        })
        df.to_parquet(feat_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_sparsity(0)
        assert res.passed is True

    def test_sparsity_counts_corrupt_files_unreadable(self, tmp_path, monkeypatch):
        """§T18: a corrupt feature parquet is counted via unreadable_files (the
        gate's existing §六-3 idiom), not silently dropped by a bare continue."""
        feat_dir = tmp_path / "feat"
        feat_dir.mkdir()
        (feat_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.production.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_sparsity(0)
        assert res.unreadable_files == 1
        assert res.passed is True  # sparsity itself stays informational


class TestLoadDailyNarrow:
    """§T18: _load_daily narrows its corrupt-parquet catch to (OSError, ValueError)."""

    def test_corrupt_daily_returns_none(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000003.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        # fail-closed semantics preserved: a corrupt file → None, not a crash.
        assert gate_mod._load_daily("000003", ["date", "close"]) is None

    def test_non_read_error_propagates(self, tmp_path, monkeypatch):
        """§T18: a failure outside (OSError, ValueError) must propagate instead
        of being silently turned into a fail-closed None."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "close": [1.0],
        }).to_parquet(daily_dir / "000004.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)

        def _raise(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr("scripts.production.data_quality_gate.pd.read_parquet", _raise)
        with pytest.raises(RuntimeError, match="unexpected"):
            gate_mod._load_daily("000004", ["date", "close"])


class TestFailOnReadError:
    def test_corrupt_daily_fails_daily_internal(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_daily_internal(0)
        assert res.passed is False
        assert any("000001" in f for f, _d in res.issues)

    def test_corrupt_daily_surfaces_category(self, tmp_path, monkeypatch, caplog):
        """A corrupt parquet read is logged with a taxonomy category, not swallowed."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        with caplog.at_level("WARNING", logger="scripts.production.data_quality_gate"):
            res = check_daily_internal(0)
        assert res.passed is False
        assert any(
            "000001" in rec.getMessage() and "category=" in rec.getMessage()
            for rec in caplog.records
        )

    def test_missing_col_fails_daily_internal(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # date + close only — no pct_change column.
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [10.0] * 5,
        }).to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_daily_internal(0)
        assert res.passed is False

    def test_missing_daily_fails_feature_pct(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        feat_dir = tmp_path / "feat"
        daily_dir.mkdir()
        feat_dir.mkdir()
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "pct_change": [np.nan, 1.0, 2.0, 3.0, 4.0],
        }).to_parquet(feat_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_feature_pct(0)
        assert res.passed is False


class TestOhlcSanity:
    def test_clean_file_passes(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0, 10.5, 10.2, 10.8, 10.4]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is True

    def test_weekend_row_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # 2024-01-06 is a Saturday — A-shares never trade weekends.
        _daily(["2024-01-05", "2024-01-06"], [10.0, 10.2]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("weekend" in d for _f, d in res.issues)

    def test_duplicate_date_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES[:3] + [TRADE_DATES[2]], [10.0] * 4).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("dup" in d for _f, d in res.issues)

    def test_low_gt_high_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5)
        df.loc[2, "low"] = 20.0  # low > high
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False

    def test_negative_volume_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5)
        df.loc[0, "volume"] = -100.0
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("volume" in d for _f, d in res.issues)

    def test_stock_code_mismatch_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # Filename 000001 but stock_code column says 600519.
        _daily(TRADE_DATES, [10.0] * 5, stock_code="600519").to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("stock_code" in d for _f, d in res.issues)

    def test_missing_stock_code_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5).drop(columns=["stock_code"])
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("stock_code" in d for _f, d in res.issues)


class TestContractSchema:
    def test_clean_file_passes(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _write_daily(daily_dir, "000001", _daily(TRADE_DATES, [10.0, 10.5, 10.2, 10.8, 10.4]))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is True
        assert res.issues == []
        assert res.files_scanned == 1

    def test_missing_required_column_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # "amount" is a DAILY_EQUITY required column — dropping it must fail the gate.
        _write_daily(
            daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5).drop(columns=["amount"])
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("missing_column:amount" in d for _f, d in res.issues)

    def test_duplicate_pk_date_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _write_daily(daily_dir, "000001", _daily(TRADE_DATES[:3] + [TRADE_DATES[2]], [10.0] * 4))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("pk_dup" in d for _f, d in res.issues)

    def test_missing_manifest_metadata_fails(self, tmp_path, monkeypatch):
        """A daily file whose sidecar declares no source/adjust fails — provenance
        is required (contract required_metadata), mirroring formal DataStorage reads."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _write_daily(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5), manifest={})
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("missing_metadata:source" in d for _f, d in res.issues)

    def test_official_holiday_fails(self, tmp_path, monkeypatch):
        """2024-01-01 is New Year's Day (exchange holiday): the gate must pass
        the official trading calendar so holidays are caught, not just weekends."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(["2024-01-01", "2024-01-02"], [10.0] * 2)  # holiday + trading day
        _write_daily(daily_dir, "000001", df)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("non_trading_day" in d for _f, d in res.issues)

    def test_corrupt_file_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("000001" in f for f, _d in res.issues)
        assert any("read_err" in d for _f, d in res.issues)


class TestManifest:
    """Manifest↔parquet consistency (§四-4): the manifest is the
    range/completeness authority; a file whose manifest is missing or whose
    start/end/rows/hash no longer match the parquet fails here."""

    def _daily_dir(self, tmp_path):
        """The gate derives the storage root from DAILY_DIR's parent: with
        DAILY_DIR = <root>/a_shares/daily, files live under <root>/a_shares/."""
        daily_dir = tmp_path / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir

    def test_clean_file_passes(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is True
        assert res.issues == []
        assert res.files_scanned == 1

    def test_missing_manifest_fails(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is False
        assert any("000001" in f for f, _d in res.issues)
        assert any("manifest missing" in d for _f, d in res.issues)

    def test_stale_row_count_fails(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5),
                          rows=999)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is False
        assert any("rows: manifest=999" in d for _f, d in res.issues)

    def test_stale_date_range_fails(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5),
                          start="2020-01-01")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is False
        assert any("start: manifest=" in d for _f, d in res.issues)

    def test_stale_schema_hash_fails(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5),
                          schema_hash="deadbeef")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is False
        assert any("schema_hash" in d for _f, d in res.issues)

    def test_re_adjusted_close_fails(self, tmp_path, monkeypatch):
        """An in-place re-adjustment edits close values but not the manifest:
        the content checksum inside the schema hash must catch it."""
        daily_dir = self._daily_dir(tmp_path)
        df = _daily(TRADE_DATES, [10.0] * 5)
        _write_daily_full(daily_dir, "000001", df)
        edited = _daily(TRADE_DATES, [10.0, 12.0, 9.0, 11.0, 10.5])
        edited.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_manifest(0)
        assert res.passed is False
        assert any("schema_hash" in d for _f, d in res.issues)


class TestDatasetsPreGate:
    """Empty/missing required data must FAIL; --allow-empty opt-out."""

    def test_empty_dir_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_datasets(0)
        assert res.passed is False
        assert any("files=0" in d for _f, d in res.issues)

    def test_missing_dir_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.DAILY_DIR", tmp_path / "does_not_exist"
        )
        res = check_datasets(0)
        assert res.passed is False
        assert any("missing_dir" in d for _f, d in res.issues)

    def test_min_stock_threshold_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_STOCKS", 2)
        res = check_datasets(0)
        assert res.passed is False
        assert any("valid_stocks" in d for _f, d in res.issues)

    def test_freshness_fails_on_stale_data(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        old = pd.Timestamp.now().normalize() - pd.Timedelta(days=90)
        dates = pd.bdate_range(old, old + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        _daily(dates, [10.0] * len(dates)).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 30)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        res = check_datasets(0)
        assert res.passed is False
        assert any("stale=" in d for _f, d in res.issues)

    def test_short_span_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 365)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.passed is False
        assert any("span=" in d for _f, d in res.issues)

    def test_healthy_dir_passes(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        dates = pd.bdate_range("2024-01-01", "2025-06-30").strftime("%Y-%m-%d")
        _daily(dates, np.arange(len(dates), dtype="float64") + 10.0).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.passed is True

    def test_allow_empty_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", tmp_path / "nope")
        monkeypatch.setattr("scripts.production.data_quality_gate.ALLOW_EMPTY", True)
        res = check_datasets(0)
        assert res.passed is True
        assert "allow-empty" in res.summary

    def test_required_features_dir(self, tmp_path, monkeypatch):
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        pd.DataFrame({
            "date": pd.to_datetime(pd.bdate_range("2024-01-01", "2025-06-30")),
            "x": np.arange(len(pd.bdate_range("2024-01-01", "2025-06-30")), dtype="float64"),
        }).to_parquet(feat_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.REQUIRED_DATASETS", ["features"])
        monkeypatch.setattr("scripts.production.data_quality_gate.FEAT_DIR", feat_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.passed is True

    def test_custom_dataset_resolves_under_data_root(self, tmp_path, monkeypatch):
        """§九.1: a non-canonical required dataset resolves under the REAL data
        root and is scanned there — no fixed-basename whitelist rejects it."""
        data_root = tmp_path / "data"
        custom = data_root / "features_panel_v2"
        custom.mkdir(parents=True)
        n = len(pd.bdate_range("2024-01-01", "2025-06-30"))
        pd.DataFrame({
            "date": pd.to_datetime(pd.bdate_range("2024-01-01", "2025-06-30")),
            "x": np.arange(n, dtype="float64"),
        }).to_parquet(custom / "000001.parquet", index=False)
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.A_SHARES", data_root / "a_shares"
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.DAILY_DIR",
            data_root / "a_shares" / "daily",
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.REQUIRED_DATASETS",
            ["features_panel_v2"],
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000
        )
        res = check_datasets(0)
        assert res.passed is True

    def test_custom_dataset_missing_dir_fails(self, tmp_path, monkeypatch):
        """A custom name whose dir is absent FAILS (missing_dir) — it must not
        silently scan DAILY_DIR the way the old fixed-basename fallback did."""
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.A_SHARES", data_root / "a_shares"
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.DAILY_DIR",
            data_root / "a_shares" / "daily",
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.REQUIRED_DATASETS",
            ["features_panel_v2"],
        )
        res = check_datasets(0)
        assert res.passed is False
        assert any("missing_dir" in d for _f, d in res.issues)

    def test_custom_dataset_outside_data_root_fails(self, tmp_path, monkeypatch):
        """§九.1: a custom name that escapes the data root is refused outright."""
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.A_SHARES", data_root / "a_shares"
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.DAILY_DIR",
            data_root / "a_shares" / "daily",
        )
        monkeypatch.setattr(
            "scripts.production.data_quality_gate.REQUIRED_DATASETS", ["../features"]
        )
        res = check_datasets(0)
        assert res.passed is False
        assert any("outside_data_root" in d for _f, d in res.issues)


class TestStratifiedSample:
    """Sampling must not be biased toward low-code stocks."""

    def test_sample_covers_multiple_exchanges(self):
        files = (
            [f"/data/0000{i:02d}.parquet" for i in range(1, 11)]  # SZ 000xxx
            + [f"/data/6000{i:02d}.parquet" for i in range(1, 11)]  # SH 600xxx
            + [f"/data/3000{i:02d}.parquet" for i in range(1, 6)]  # SZ 300xxx
            + [f"/data/8300{i:02d}.parquet" for i in range(1, 6)]  # BJ 830xxx
        )
        sample = _sample_files(files, 6)
        assert len(sample) == 6
        codes = [Path(f).stem for f in sample]
        # Not the plain sorted head (which would be all 000xxx).
        assert codes != [f"0000{i:02d}" for i in range(1, 7)]
        # At least one SH (600xxx) and one BJ (830xxx) stock present.
        assert any(c.startswith("6") for c in codes)
        assert any(c.startswith("83") for c in codes)

    def test_sample_is_deterministic(self):
        files = [f"/data/{c}.parquet" for c in
                 ["000001", "000002", "600001", "600002", "300001", "830001"]]
        a = _sample_files(files, 4)
        b = _sample_files(files, 4)
        assert a == b

    def test_sample_greater_than_size_returns_all(self):
        files = ["/data/000001.parquet", "/data/600001.parquet"]
        assert _sample_files(files, 5) == files


class TestUnreadableFilesAndFormalProfile:
    """§六-3/§六-4: unreadable files are always counted (and surface in the
    report), fail the dataset only when their share exceeds the max ratio, and
    the formal stock-ratio floor catches datasets with too few readable stocks."""

    def _daily_dir(self, tmp_path, *codes, corrupt=()):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        for c in codes:
            _daily(TRADE_DATES, [10.0] * 5).to_parquet(
                daily_dir / f"{c}.parquet", index=False
            )
        for c in corrupt:
            (daily_dir / f"{c}.parquet").write_bytes(b"not a parquet")
        return daily_dir

    def test_unreadable_exceeding_ratio_fails(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path, "000001", "000002", corrupt=("000003",))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.unreadable_files == 1
        assert res.passed is False
        assert any(d == "unreadable" and f == "000003" for f, d in res.issues)
        assert any("unreadable=1/3" in d for _f, d in res.issues)

    def test_unreadable_below_tolerance_is_counted_not_failing(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path, "000001", "000002", "000003",
                                    corrupt=("000004",))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_UNREADABLE_RATIO", 1.0)
        res = check_datasets(0)
        assert res.unreadable_files == 1
        assert res.passed is True
        assert not any("unreadable" in d for _f, d in res.issues)

    def test_formal_stock_ratio_fails_below_floor(self, tmp_path, monkeypatch):
        """A file whose dates are all unparseable reads fine but is not a
        valid stock: under a 100% floor that alone fails the dataset."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        pd.DataFrame({"date": ["not-a-date", "also-bad"]}).to_parquet(
            daily_dir / "000002.parquet", index=False
        )
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        monkeypatch.setattr("scripts.production.data_quality_gate.FORMAL_STOCK_RATIO", 1.0)
        res = check_datasets(0)
        assert res.passed is False
        assert any("valid_stocks=1/2" in d for _f, d in res.issues)


class TestVerifiedUntilBound:
    """§九-3: forward-estimate trading days past ``verified_until`` (2027+
    A-share closures are not verified exchange fact) must FAIL the required
    dataset when the bound is enforced (--profile formal in main()); off by
    default so exploratory bootstrap runs are unaffected."""

    _PAST_DATES = ["2026-12-30", "2026-12-31", "2027-01-04"]

    def _daily_dir(self, tmp_path):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(self._PAST_DATES, [10.0] * 3).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        return daily_dir

    def test_past_verified_until_fails_when_enforced(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        monkeypatch.setattr("scripts.production.data_quality_gate.ENFORCE_VERIFIED_UNTIL", True)
        res = check_datasets(0)
        assert res.passed is False
        assert any("extends_past_verified_until" in d for _f, d in res.issues)

    def test_past_verified_until_ok_when_not_enforced(self, tmp_path, monkeypatch):
        daily_dir = self._daily_dir(tmp_path)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 10000)
        # ENFORCE_VERIFIED_UNTIL stays at its default False.
        res = check_datasets(0)
        assert res.passed is True
        assert not any("extends_past_verified_until" in d for _f, d in res.issues)


class TestFingerprints:
    """§六-2 building blocks: the contract and dataset fingerprints the gate
    records so train_panel can verify a matching report."""

    def test_contract_version_is_deterministic_hex(self):
        a = contract_version()
        b = contract_version()
        assert a == b
        assert len(a) == 16
        assert all(ch in "0123456789abcdef" for ch in a)

    def test_dataset_fingerprint_deterministic_and_sensitive(self, tmp_path):
        root = tmp_path
        daily = root / "a_shares" / "daily"
        daily.mkdir(parents=True)
        self._write_daily_with_manifest(daily, "000001", TRADE_DATES, [10.0] * 5)
        h1 = dataset_fingerprint(root, ["daily"])
        assert dataset_fingerprint(root, ["daily"]) == h1
        # §v18-8: the fingerprint binds the per-stock MANIFEST (content hash),
        # not a byte re-scan.  Rebuilding the data MUST rewrite the manifest —
        # the canonical write path — and the digest flips.
        self._write_daily_with_manifest(
            daily, "000001", TRADE_DATES + ["2024-01-09"], [10.0] * 6)
        assert dataset_fingerprint(root, ["daily"]) != h1

    def test_dataset_fingerprint_missing_manifest_is_a_marker(self, tmp_path):
        """§v18-8: a parquet WITHOUT a manifest hashes a distinct marker — the
        fingerprint differs from the same parquet WITH a manifest (fail-closed:
        a manifest-less file is never silently treated as identical)."""
        root = tmp_path
        daily = root / "a_shares" / "daily"
        daily.mkdir(parents=True)
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(daily / "000001.parquet", index=False)
        h_no_manifest = dataset_fingerprint(root, ["daily"])
        self._write_daily_with_manifest(daily, "000001", TRADE_DATES, [10.0] * 5)
        h_with_manifest = dataset_fingerprint(root, ["daily"])
        assert h_no_manifest != h_with_manifest

    def test_dataset_fingerprint_noop_resave_is_stable(self, tmp_path):
        """§v18-8: a content-identical re-save (a no-op re-download) bumps only
        the per-write bookkeeping (run_id / updated) — the fingerprint must NOT
        flip.  The daily store writes run_id + updated on every save, so
        production stability requires the digest to ignore write-event noise."""
        root = tmp_path
        daily = root / "a_shares" / "daily"
        daily.mkdir(parents=True)
        self._write_daily_with_manifest(
            daily, "000001", TRADE_DATES, [10.0] * 5,
            run_id="r1", updated="2026-08-08T00:00:00+00:00")
        h1 = dataset_fingerprint(root, ["daily"])
        self._write_daily_with_manifest(
            daily, "000001", TRADE_DATES, [10.0] * 5,
            run_id="r2", updated="2026-08-09T00:00:00+00:00")
        assert dataset_fingerprint(root, ["daily"]) == h1

    @staticmethod
    def _write_daily_with_manifest(daily_dir, code, dates, closes,
                                   *, run_id="run-1",
                                   updated="2026-08-08T00:00:00+00:00"):
        """Production-shaped daily write: parquet + manifest sidecar carrying
        the per-write bookkeeping (run_id / updated) the real store emits — no
        written_at.  §v18-8: the fingerprint binds the manifest content with
        the per-write keys excluded."""
        from stoke_ml.data.asset_contract import schema_hash
        df = _daily(dates, closes)
        df.to_parquet(daily_dir / f"{code}.parquet", index=False)
        with open(daily_dir / f"{code}.manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "stock": code, "rows": len(df), "schema_hash": schema_hash(df),
                "source": "test", "start": dates[0], "end": dates[-1],
                "run_id": run_id, "updated": updated,
            }, f)

    def test_dataset_fingerprint_missing_dir_is_stable(self, tmp_path):
        h1 = dataset_fingerprint(tmp_path, ["daily", "features_panel"])
        h2 = dataset_fingerprint(tmp_path, ["daily", "features_panel"])
        assert h1 == h2
        assert len(h1) == 16


class TestUniverseReconciliation:
    """§P1-7: per-requested-stock reconciliation against a requested universe.

    The gate must account for every requested code individually: an absent
    parquet is MISSING, a present file with an invalid manifest is DEGRADED
    ("file exists" ≠ "data complete"), and the report must list them precisely.
    The whole check is OPT-IN — without a requested universe the gate's default
    behavior is unchanged (``universe`` not in ``CHECKS``, ``check_universe``
    no-ops to PASS).
    """

    def _daily_dir(self, tmp_path):
        daily_dir = tmp_path / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir

    def test_all_requested_present_ok(self, tmp_path):
        daily_dir = self._daily_dir(tmp_path)
        for c in ("000001", "600519"):
            _write_daily_full(daily_dir, c, _daily(TRADE_DATES, [10.0] * 5))
        rep = reconcile_requested_universe(["000001", "600519"], daily_dir=daily_dir)
        assert rep["ok"] is True
        assert rep["requested_count"] == 2
        assert rep["present_count"] == 2
        assert rep["missing_codes"] == []
        assert rep["degraded_codes"] == []

    def test_missing_codes_listed_precisely(self, tmp_path):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        rep = reconcile_requested_universe(
            ["000001", "600519", "300750"], daily_dir=daily_dir
        )
        assert rep["ok"] is False
        assert rep["requested_count"] == 3
        assert rep["present_count"] == 1
        assert rep["missing_codes"] == ["300750", "600519"]  # sorted
        assert rep["degraded_codes"] == []

    def test_present_but_invalid_manifest_is_degraded(self, tmp_path):
        """A parquet on disk with NO valid manifest is degraded, not "present"."""
        daily_dir = self._daily_dir(tmp_path)
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False  # no sidecar
        )
        _write_daily_full(daily_dir, "600519", _daily(TRADE_DATES, [10.0] * 5))
        rep = reconcile_requested_universe(["000001", "600519"], daily_dir=daily_dir)
        assert rep["ok"] is False
        assert rep["present_count"] == 2
        assert rep["missing_codes"] == []
        assert len(rep["degraded_codes"]) == 1
        assert rep["degraded_codes"][0]["code"] == "000001"
        assert "manifest_invalid" in rep["degraded_codes"][0]["reason"]

    def test_stale_manifest_is_degraded(self, tmp_path):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5), rows=999)
        rep = reconcile_requested_universe(["000001"], daily_dir=daily_dir)
        assert rep["ok"] is False
        assert rep["degraded_codes"][0]["code"] == "000001"
        assert "manifest_invalid" in rep["degraded_codes"][0]["reason"]

    def test_min_rows_flags_thin_history_degraded(self, tmp_path):
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        rep = reconcile_requested_universe(["000001"], daily_dir=daily_dir, min_rows=100)
        assert rep["ok"] is False
        assert rep["degraded_codes"][0]["code"] == "000001"
        assert "rows=5 < min=100" in rep["degraded_codes"][0]["reason"]
        # At/below the actual row count the same file is sound.
        rep2 = reconcile_requested_universe(["000001"], daily_dir=daily_dir, min_rows=5)
        assert rep2["ok"] is True
        assert rep2["degraded_codes"] == []

    def test_missing_within_tolerance_passes_but_is_listed(self, tmp_path):
        """A tolerated gap does not fail the gate, but is still reported."""
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        _write_daily_full(daily_dir, "600519", _daily(TRADE_DATES, [10.0] * 5))
        rep = reconcile_requested_universe(
            ["000001", "600519", "300750", "000002"], daily_dir=daily_dir,
            max_missing_ratio=0.5,  # 2 of 4 missing tolerated
        )
        assert rep["ok"] is True
        assert rep["missing_codes"] == ["000002", "300750"]

    def test_no_requested_universe_skips(self, monkeypatch):
        """Default behavior is unchanged: no universe -> check no-ops to PASS,
        and the universe check is not part of the default CHECKS set."""
        monkeypatch.setattr("scripts.production.data_quality_gate._UNIVERSE_REQUEST", None)
        res = check_universe(0)
        assert res.passed is True
        assert "skipped" in res.summary
        assert "universe" not in CHECKS

    def test_check_universe_wrapper_reports_and_fails(self, tmp_path, monkeypatch):
        """The gate wrapper surfaces the report + flips passed on a gap."""
        daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate._UNIVERSE_REQUEST", {
            "codes": ["000001", "600519"],
            "requested_start": None, "requested_end": None,
            "min_rows": 0, "min_coverage_ratio": 0.0,
            "max_missing_ratio": 0.0, "max_degraded_ratio": 0.0,
        })
        res = check_universe(0)
        assert res.passed is False
        assert res.details is not None
        assert res.details["missing_codes"] == ["600519"]
        assert ("600519", "missing") in res.issues

    # ── universe-source loaders ──────────────────────────────────────────
    def test_read_download_run_manifest(self, tmp_path):
        p = tmp_path / "download_manifest.json"
        p.write_text(json.dumps({
            "schema_version": "1.3", "market": "a_shares",
            "start_date": "2020-01-02", "requested_end": "2026-07-31",
            "requested": ["000001", "600519", "000001.0", "SH600519"],
        }), encoding="utf-8")
        info = _read_requested_file(str(p), require_manifest=True)
        assert info["codes"] == ["000001", "600519"]  # dedup + normalize
        assert info["requested_start"] == "2020-01-02"
        assert info["requested_end"] == "2026-07-31"

    def test_read_line_per_code_text_file(self, tmp_path):
        p = tmp_path / "universe.txt"
        p.write_text("# comment\n000001\n\n600519\n300750\n", encoding="utf-8")
        info = _read_requested_file(str(p))
        assert info["codes"] == ["000001", "600519", "300750"]

    def test_read_json_code_list(self, tmp_path):
        p = tmp_path / "codes.json"
        p.write_text('["000001", "600519"]', encoding="utf-8")
        info = _read_requested_file(str(p))
        assert info["codes"] == ["000001", "600519"]

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_requested_file(str(tmp_path / "nope.json"))

    def test_request_manifest_requires_requested_field(self, tmp_path):
        p = tmp_path / "not_a_manifest.json"
        p.write_text('{"foo": 1}', encoding="utf-8")
        with pytest.raises(ValueError):
            _read_requested_file(str(p), require_manifest=True)


class TestManifestContractFullScan:
    """v14 §八-1: ``manifest_contract_full_scan`` must be True only when BOTH
    the manifest and contract_schema checks actually ran, both passed, and both
    scanned every daily file.  A run scoped to ``--check manifest``
    (contract_schema never joined results) must NOT satisfy the floor — the old
    ``bool(full_audit)`` only needed one of the two to exist.

    ``unreadable_files`` is deliberately NOT part of this pair's decision: the
    count is only populated by the dataset check, not by manifest/contract_schema
    — a read error in either audit already flips its ``passed`` to False, so an
    explicit ``unreadable_files == 0`` clause would be dead."""

    @staticmethod
    def _audit(name, passed=True, scanned=None, unreadable=0, total=2):
        return CheckResult(
            name=name,
            passed=passed,
            summary="",
            files_scanned=total if scanned is None else scanned,
            unreadable_files=unreadable,
        )

    def test_manifest_only_is_false(self):
        """contract_schema absent -> the contract audit is unproven."""
        results = [self._audit("manifest", total=2)]
        assert _manifest_contract_full_scan(results, total_daily=2) is False

    def test_contract_only_is_false(self):
        """manifest absent -> the range/completeness audit is unproven."""
        results = [self._audit("contract_schema", total=2)]
        assert _manifest_contract_full_scan(results, total_daily=2) is False

    def test_both_covering_all_files_is_true(self):
        results = [
            self._audit("manifest", total=2),
            self._audit("contract_schema", total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is True

    def test_extra_checks_do_not_break_a_full_audit(self):
        """Unrelated checks may coexist; both audit checks still ran + covered."""
        results = [
            self._audit("manifest", total=2),
            self._audit("contract_schema", total=2),
            self._audit("datasets", total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is True

    def test_either_check_failed_is_false(self):
        results = [
            self._audit("manifest", passed=False, total=2),
            self._audit("contract_schema", total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is False
        # Other side failed too.
        results = [
            self._audit("manifest", total=2),
            self._audit("contract_schema", passed=False, total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is False

    def test_any_files_scanned_below_total_is_false(self):
        results = [
            self._audit("manifest", scanned=1, total=2),
            self._audit("contract_schema", total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is False

    def test_unreadable_count_does_not_drive_decision(self):
        """The unreadable_files count is only populated by the DATASET check,
        which is not part of this pair — so it must not gate the floor.  A read
        error inside manifest/contract_schema surfaces as passed=False instead,
        which this pair DOES enforce (see test_either_check_failed_is_false)."""
        results = [
            self._audit("manifest", total=2),
            self._audit("contract_schema", unreadable=1, total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is True
        # The same read error as a failed check still fails the floor.
        results = [
            self._audit("manifest", total=2),
            self._audit("contract_schema", passed=False, unreadable=1, total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is False

    def test_missing_daily_files_yields_false(self):
        """A check that scanned nothing cannot cover a non-empty pool."""
        results = [
            self._audit("manifest", scanned=0, total=2),
            self._audit("contract_schema", total=2),
        ]
        assert _manifest_contract_full_scan(results, total_daily=2) is False


class TestUniverseReconciliationFormalReport:
    """§八-2: the FORMAL gate (main) must reflect a requested universe's missing
    stocks as MISSING — never pretend "98% of what exists is valid" — and carry
    the download run manifest's own completion status next to the on-disk
    reconciliation.  A missing run manifest fails cleanly (not a traceback)."""

    def _data_root(self, tmp_path):
        root = tmp_path / "data"
        daily_dir = root / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        return root

    @staticmethod
    def _manifest(root, requested):
        path = root / "a_shares" / "download_manifest.json"
        path.write_text(json.dumps({
            "schema_version": "1.3", "market": "a_shares",
            "start_date": "2020-01-02", "requested_end": "2026-07-31",
            "status": "partial",
            "requested": list(requested),
            "failed": [], "missing": [], "complete": [],
            "requested_count": len(requested),
            "complete_count": 0, "failed_count": 0, "missing_count": 0,
            "all_complete": False,
        }), encoding="utf-8")
        return path

    def test_formal_report_reflects_missing_stocks(self, tmp_path, monkeypatch):
        """A requested universe with a stock missing from disk must be reported
        MISSING (exit 1) — the formal report never claims the on-disk pool is
        the universe."""
        root = self._data_root(tmp_path)
        manifest = self._manifest(root, ["000001", "600519"])
        rc = _run_gate([
            "data_quality_gate.py",
            "--data-dir", str(root),
            "--check", "universe",
            "--profile", "formal",
            "--request-manifest", str(manifest),
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is False
        recon = report["universe_reconciliation"]
        assert recon["ok"] is False
        assert recon["requested_count"] == 2
        assert recon["present_count"] == 1
        assert recon["missing_codes"] == ["600519"]
        # The run manifest's own status is recorded next to the on-disk truth.
        assert recon["run_manifest"]["status"] == "partial"
        assert recon["run_manifest"]["requested_count"] == 2

    def test_missing_manifest_fails_cleanly_in_formal_mode(self, tmp_path, monkeypatch):
        """§八-2: a missing --request-manifest FAILS the formal gate with a
        clean non-zero exit — it never silently resolves to whatever is on
        disk."""
        root = self._data_root(tmp_path)
        rc = _run_gate([
            "data_quality_gate.py",
            "--data-dir", str(root),
            "--check", "universe",
            "--profile", "formal",
            "--request-manifest", str(root / "a_shares" / "download_manifest.json"),
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        assert rc == 1


class TestChannelVintageFormalReport:
    """v15 §六/§十 / v16 §十二: the report surfaces the channel→3-dim vintage
    declaration under the run's vintage-admission policy — present in every run
    regardless of profile, each entry carrying exactly
    channel/source_vintage/transform/pit_alignment/rationale/allowed — and
    locks the documented revision-leakage sources (fundamental, macro) as
    latest_revised source.  Under the default revision-safe policy the
    latest_revised-sourced channels are marked allowed=False while
    immutable_snapshot-sourced channels (incl. the price channel) stay
    allowed=True."""

    def test_report_carries_channel_vintage_declaration(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        daily_dir = root / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        rc = _run_gate([
            "data_quality_gate.py",
            "--data-dir", str(root),
            "--check", "datasets",
            "--min-span-days", "0",
            "--max-stale-days", "10000",
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        # The report is written regardless of `passed`; span/freshness are
        # relaxed so the run is deterministic, but we do not depend on rc.
        assert rc in (0, 1)
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        section = report["channel_vintage"]
        assert report["vintage_policy"] == "revision-safe"
        assert isinstance(section, list) and len(section) > 0
        for entry in section:
            assert set(entry) == {"channel", "source_vintage", "transform",
                                  "pit_alignment", "rationale", "allowed"}
            assert entry["source_vintage"] in {"immutable_snapshot", "latest_revised"}
            assert isinstance(entry["allowed"], bool)
        by_name = {e["channel"]: e for e in section}
        assert by_name["fundamental"]["source_vintage"] == "latest_revised"
        assert by_name["macro"]["source_vintage"] == "latest_revised"
        # Default revision-safe policy: latest_revised-sourced denied,
        # immutable_snapshot-sourced admitted.
        assert by_name["fundamental"]["allowed"] is False
        assert by_name["macro"]["allowed"] is False
        assert by_name["sentiment"]["allowed"] is True
        assert by_name["daily_qfq"]["allowed"] is True

    # ── §T2 formal enforcement: reject branches ────────────────────────────
    # The crafted tests in test_vintage_policy.py prove vintage_report COMPUTES
    # the missing/denied flags; these prove main() turns them into a FAIL in
    # formal mode (and never does in bootstrap).  vintage_report is patched on
    # the RUN module object so main() picks up the fake; the fake returns a
    # COMPLETE report dict because the report builder reads channels + policy.
    #
    # The FORMAL_PROFILE relaxation below is REQUIRED: under --profile formal,
    # main() OVERRIDES the CLI span/stale flags with the frozen research floors
    # (5y span / 4 trading-day staleness), which the 5-day test dataset can
    # never satisfy.  Without relaxing those floors the datasets check alone
    # would force rc==1 and the reject tests would be tautological.  With the
    # floors relaxed (plus save_calendar for the calendar check), the vintage
    # enforcement is the SOLE remaining failure driver — proven by the clean
    # negative control below.

    @staticmethod
    def _vintage_gate_argv(root, output, *, profile):
        argv = [
            "data_quality_gate.py",
            "--data-dir", str(root),
            "--check", "datasets",
            "--min-files", "0",
            "--min-stocks", "0",
            "--min-rows", "0",
            "--min-span-days", "0",
            "--max-stale-days", "10000",
            "--output", str(output),
        ]
        if profile:
            argv.append("--profile")
            argv.append(profile)
        return argv

    def _vintage_gate_root(self, tmp_path):
        """A data root with one valid daily stock + the frozen calendar artifact.
        Formal runs must ALSO relax gate_mod.FORMAL_PROFILE (span/stale) so the
        vintage enforcement is the only possible reason for rc==1."""
        root = tmp_path / "data"
        daily_dir = root / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        save_calendar(root, "a_shares")
        return root

    def _relax_formal_profile(self, monkeypatch):
        """Lower the frozen formal span/stale floors so the 5-day test dataset's
        datasets check PASSES under --profile formal — isolating the vintage
        enforcement as the only remaining failure driver."""
        monkeypatch.setitem(gate_mod.FORMAL_PROFILE, "min_span_days", 0)
        monkeypatch.setitem(gate_mod.FORMAL_PROFILE, "max_stale_days", 10000)

    def _patch_vintage_report(self, monkeypatch, fake):
        monkeypatch.setattr(
            "scripts.production.data_quality_gate_run.vintage_report",
            lambda *a, **k: fake,
        )

    def test_formal_rejects_incomplete_vintage_declaration(self, tmp_path, monkeypatch):
        """§T2: formal mode FAILs when a documented use_* channel carries no
        vintage declaration (silently denied-by-default is a hard FAIL)."""
        root = self._vintage_gate_root(tmp_path)
        self._relax_formal_profile(monkeypatch)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": ["fundamental"],
            "daily_qfq_allowed": True,
            "declaration_complete": True,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile="formal"),
            monkeypatch,
        )
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is False
        assert report["vintage_policy"] == "revision-safe"
        assert report["channel_vintage"] == []  # the fake was consumed

    def test_formal_rejects_denied_price_channel(self, tmp_path, monkeypatch):
        """§T2: formal mode FAILs when the policy denies daily_qfq — a model
        cannot train without the price channel."""
        root = self._vintage_gate_root(tmp_path)
        self._relax_formal_profile(monkeypatch)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": [],
            "daily_qfq_allowed": False,
            "declaration_complete": True,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile="formal"),
            monkeypatch,
        )
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is False
        assert report["channel_vintage"] == []  # the fake was consumed

    def test_formal_passes_when_vintage_declaration_clean(self, tmp_path, monkeypatch):
        """§T2 negative control: the SAME relaxed formal run with a CLEAN fake
        (no missing channels, price channel allowed) passes — proving the two
        reject tests' rc==1 is driven by the vintage enforcement, not by the
        datasets/calendar checks."""
        root = self._vintage_gate_root(tmp_path)
        self._relax_formal_profile(monkeypatch)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": [],
            "daily_qfq_allowed": True,
            "declaration_complete": True,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile="formal"),
            monkeypatch,
        )
        assert rc == 0
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is True
        assert report["channel_vintage"] == []  # the fake was consumed

    def test_bootstrap_does_not_fail_on_incomplete_declaration(self, tmp_path, monkeypatch):
        """§T2 control: bootstrap REPORTS an incomplete declaration but never
        ENFORCES it — the same fake that fails formal leaves rc==0 here."""
        root = self._vintage_gate_root(tmp_path)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": ["fundamental"],
            "daily_qfq_allowed": True,
            "declaration_complete": True,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile=None),
            monkeypatch,
        )
        assert rc == 0
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is True
        assert report["channel_vintage"] == []  # the fake was consumed

    def test_formal_rejects_incomplete_three_dim_declaration(self, tmp_path, monkeypatch):
        """§T7: formal mode FAILs when the 3-dim declaration is incomplete —
        every declared channel must carry non-"unknown" source_vintage /
        transform / pit_alignment."""
        root = self._vintage_gate_root(tmp_path)
        self._relax_formal_profile(monkeypatch)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": [],
            "daily_qfq_allowed": True,
            "declaration_complete": False,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile="formal"),
            monkeypatch,
        )
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is False
        assert report["vintage_policy"] == "revision-safe"
        assert report["channel_vintage"] == []  # the fake was consumed

    def test_bootstrap_does_not_fail_on_incomplete_three_dim_declaration(self, tmp_path, monkeypatch):
        """§T7 control: bootstrap REPORTS an incomplete 3-dim declaration but
        never ENFORCES it — the same fake that fails formal leaves rc==0 here."""
        root = self._vintage_gate_root(tmp_path)
        self._patch_vintage_report(monkeypatch, {
            "vintage_policy": "revision-safe",
            "channels": [],
            "missing_channels": [],
            "daily_qfq_allowed": True,
            "declaration_complete": False,
        })
        rc = _run_gate(
            self._vintage_gate_argv(root, tmp_path / "report", profile=None),
            monkeypatch,
        )
        assert rc == 0
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8")
        )
        assert report["passed"] is True
        assert report["channel_vintage"] == []  # the fake was consumed


class TestCalendarArtifactAndFreshness:
    """v14 §九: the gate must (1) load the frozen calendar artifact for the
    data-dir being validated — never a module-import singleton, (2) record the
    artifact's CONTENT hash (not a version string), (3) fail in formal mode when
    the artifact is missing (no silent fallback to code holiday rules), and
    (4) judge freshness against the most recent COMPLETED trading day so a
    fully-current dataset never trips a natural-day ceiling over 春节/国庆."""

    @staticmethod
    def _daily_dir(tmp_path, root=None):
        root = root if root is not None else (tmp_path / "data")
        daily_dir = root / "a_shares" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        return root, daily_dir

    # ── artifact binding + report content hash ──────────────────────────

    def test_report_records_data_dir_artifact_content_hash(self, tmp_path, monkeypatch):
        """The gate resolves the calendar from the --data-dir being validated:
        two roots with DIFFERENT frozen artifacts yield DIFFERENT content hashes
        in their reports (never a built-in calendar or a bare version string)."""
        root_a, _ = self._daily_dir(tmp_path)
        root_b, _ = self._daily_dir(tmp_path, root=tmp_path / "data_b")
        for root in (root_a, root_b):
            save_calendar(root, "a_shares")
            _daily(TRADE_DATES, [10.0] * 5).to_parquet(
                root / "a_shares" / "daily" / "000001.parquet", index=False)
        # Tamper B's artifact: flip one real trading day to closed.
        frame = load_calendar(root_b, "a_shares")
        frame.loc[frame["date"] == pd.Timestamp("2010-02-24"), "is_open"] = False
        frame.to_parquet(root_b / "exchange_calendar" / "a_shares.parquet")
        hashes = {}
        for i, root in enumerate((root_a, root_b)):
            out = tmp_path / f"rep{i}"
            _run_gate([
                "data_quality_gate.py", "--data-dir", str(root),
                "--check", "datasets",
                "--min-span-days", "0", "--max-stale-days", "10000",
                "--output", str(out),
            ], monkeypatch)
            report = json.loads(
                (out / "data_quality_gate.json").read_text(encoding="utf-8"))
            assert report["calendar_artifact_present"] is True
            hashes[root] = report["calendar_artifact_hash"]
        # Different artifact content → different report hash, and it is THAT
        # root's canonical content hash — not a global version string.
        assert hashes[root_a] != hashes[root_b]
        assert hashes[root_a] == calendar_artifact_hash(root_a, "a_shares")
        assert hashes[root_b] == calendar_artifact_hash(root_b, "a_shares")

    # ── formal mode: missing artifact must fail, not fall back ──────────

    def test_formal_missing_artifact_fails(self, tmp_path, monkeypatch):
        """§九: formal mode with NO frozen calendar artifact FAILS the gate
        (rc 1, passed False, present False) instead of silently validating
        against the code holiday rules."""
        root, daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        rc = _run_gate([
            "data_quality_gate.py", "--data-dir", str(root),
            "--check", "datasets",
            "--profile", "formal",
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["calendar_artifact_present"] is False
        assert report["calendar_artifact_hash"] is None

    def test_formal_corrupt_artifact_fails_cleanly_with_report(self, tmp_path, monkeypatch):
        """§九: a present-but-corrupt frozen calendar artifact (wrong schema /
        empty / dup-date / gapped) must NOT crash the gate with a bare traceback
        before the report is written — it fails cleanly (rc 1), still writes the
        report, and reports the calendar as unusable / present=False."""
        root, daily_dir = self._daily_dir(tmp_path)
        _write_daily_full(daily_dir, "000001", _daily(TRADE_DATES, [10.0] * 5))
        cal_dir = root / "exchange_calendar"
        cal_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [pd.Timestamp("2026-01-01")]}).to_parquet(
            cal_dir / "a_shares.parquet")
        rc = _run_gate([
            "data_quality_gate.py", "--data-dir", str(root),
            "--check", "datasets",
            "--profile", "formal",
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        assert rc == 1
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["calendar_artifact_present"] is False
        assert report["calendar_artifact_hash"] is None

    def test_formal_with_artifact_not_refused_on_calendar(self, tmp_path, monkeypatch):
        """With the frozen artifact present, a fully-current 5-year dataset
        clears the formal profile — the calendar artifact is not the blocker."""
        root, daily_dir = self._daily_dir(tmp_path)
        save_calendar(root, "a_shares")
        cal = get_research_calendar(data_dir=root)
        last = most_recent_completed_trading_day(
            cal, pd.Timestamp.now().normalize().date())
        dates = pd.bdate_range("2019-01-02", last).strftime("%Y-%m-%d")
        _daily(dates, np.arange(len(dates), dtype="float64") + 10.0).to_parquet(
            daily_dir / "000001.parquet", index=False)
        rc = _run_gate([
            "data_quality_gate.py", "--data-dir", str(root),
            "--check", "datasets",
            "--profile", "formal",
            "--output", str(tmp_path / "report"),
        ], monkeypatch)
        assert rc == 0
        report = json.loads(
            (tmp_path / "report" / "data_quality_gate.json").read_text(encoding="utf-8"))
        assert report["calendar_artifact_present"] is True
        assert report["calendar_artifact_hash"] == calendar_artifact_hash(root, "a_shares")

    # ── freshness: positional against the most recent COMPLETED session ──

    def test_freshness_holiday_safe_over_spring_festival(self, tmp_path, monkeypatch):
        """A dataset current through the last COMPLETED session (2026-02-13) is
        FRESH when the gate runs 10 natural days later inside 春节 2026 — the
        natural-day ceiling must not trigger."""
        root, daily_dir = self._daily_dir(tmp_path)
        save_calendar(root, "a_shares")
        # 春节 2026 closures 2/16-2/23; last real session before is 2/13 (Fri).
        dates = pd.bdate_range("2026-01-05", "2026-02-13").strftime("%Y-%m-%d")
        _daily(dates, np.arange(len(dates), dtype="float64") + 10.0).to_parquet(
            daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.A_SHARES",
                            root / "a_shares")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 4)
        monkeypatch.setattr("scripts.production.data_quality_gate._now",
                            lambda: pd.Timestamp("2026-02-23"))
        res = check_datasets(0)
        assert res.passed is True
        assert not any("stale=" in d for _f, d in res.issues)

    def test_freshness_fails_when_behind_most_recent_completed(self, tmp_path, monkeypatch):
        """A dataset one session behind the last COMPLETED trading day IS stale
        under a zero-tolerance freshness floor (MAX_STALE_DAYS=0)."""
        root, daily_dir = self._daily_dir(tmp_path)
        save_calendar(root, "a_shares")
        # Ends 2026-02-12 — 2/13 was the last real session before 春节.
        dates = pd.bdate_range("2026-01-05", "2026-02-12").strftime("%Y-%m-%d")
        _daily(dates, np.arange(len(dates), dtype="float64") + 10.0).to_parquet(
            daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.production.data_quality_gate.A_SHARES",
                            root / "a_shares")
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.production.data_quality_gate.MIN_SPAN_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate.MAX_STALE_DAYS", 0)
        monkeypatch.setattr("scripts.production.data_quality_gate._now",
                            lambda: pd.Timestamp("2026-02-23"))
        res = check_datasets(0)
        assert res.passed is False
        assert any("stale=" in d for _f, d in res.issues)


class TestAuxCloseAligned:
    """§T19: aux processed close must equal canonical daily (basis-drift
    canary).  Per-date-price channels (board_processed) compare every row;
    forward-filled-close channels (dividend/lockup) compare only at genuine
    event rows (``dv_days_since == 0``), where close is a real daily price."""

    @staticmethod
    def _setup(tmp_path, monkeypatch):
        root = tmp_path / "a_shares"
        daily = root / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        # Aux fixture dirs the check globs (``A_SHARES / <dir> / *.parquet``)
        # must exist so the parquet writers have a target.
        for aux in ("board_processed", "dividend_processed"):
            (root / aux).mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("scripts.production.data_quality_gate.A_SHARES", root)
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", daily)
        # _load_daily caches by code; a fresh dir per test must not read a
        # prior test's cached frame.
        monkeypatch.setattr("scripts.production.data_quality_gate._DAILY_CACHE", {})
        return root, daily

    def _write_daily(self, daily_dir, code, closes):
        _daily(TRADE_DATES, closes, code=code).to_parquet(
            daily_dir / f"{code}.parquet", index=False
        )

    def test_per_date_channel_matching_close_passes(self, tmp_path, monkeypatch):
        root, daily = self._setup(tmp_path, monkeypatch)
        self._write_daily(daily, "000001", [10.0, 10.5, 10.2, 10.8, 10.4])
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [10.0, 10.5, 10.2, 10.8, 10.4],
        }).to_parquet(root / "board_processed" / "000001.parquet", index=False)
        res = check_aux_close_aligned(0)
        assert res.passed is True
        assert res.issues == []

    def test_per_date_channel_basis_drift_fails(self, tmp_path, monkeypatch):
        root, daily = self._setup(tmp_path, monkeypatch)
        self._write_daily(daily, "000001", [10.0, 10.5, 10.2, 10.8, 10.4])
        # Embedded close drifted +5 off canonical daily — must fail.
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [15.0, 15.5, 15.2, 15.8, 15.4],
        }).to_parquet(root / "board_processed" / "000001.parquet", index=False)
        res = check_aux_close_aligned(0)
        assert res.passed is False
        assert any("board_processed" in f for f, _d in res.issues)

    def test_forward_filled_close_compares_event_rows_only(self, tmp_path, monkeypatch):
        root, daily = self._setup(tmp_path, monkeypatch)
        self._write_daily(daily, "000001", [10.0, 10.5, 10.2, 10.8, 10.4])
        # Event on the first day (dv_days_since == 0): close = daily 10.0, then
        # forward-filled as a constant while daily drifts — genuine dividend
        # semantics, must PASS (only the event row is compared).
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "dv_days_since": [0, 1, 2, 3, 4],
        }).to_parquet(root / "dividend_processed" / "000001.parquet", index=False)
        res = check_aux_close_aligned(0)
        assert res.passed is True
        assert res.issues == []

    def test_forward_filled_channel_event_row_drift_still_fails(self, tmp_path, monkeypatch):
        root, daily = self._setup(tmp_path, monkeypatch)
        self._write_daily(daily, "000001", [10.0, 10.5, 10.2, 10.8, 10.4])
        # Event-row close drifted +0.3 off daily (10.3 vs 10.0) — the canary
        # must STILL fire at event rows even under the event-row restriction.
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [10.3, 10.3, 10.3, 10.3, 10.3],
            "dv_days_since": [0, 1, 2, 3, 4],
        }).to_parquet(root / "dividend_processed" / "000001.parquet", index=False)
        res = check_aux_close_aligned(0)
        assert res.passed is False
        assert any("dividend_processed" in f for f, _d in res.issues)
