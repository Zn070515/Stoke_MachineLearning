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
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.data.storage import _provenance_from_attrs, _schema_hash
from scripts.production.data_quality_gate import (
    _sample_files,
    check_contract_schema,
    check_daily_internal,
    check_datasets,
    check_feature_pct,
    check_manifest,
    check_ohlc_sanity,
    check_sparsity,
    contract_version,
    dataset_fingerprint,
)


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

    def test_unknown_dataset_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.production.data_quality_gate.REQUIRED_DATASETS", ["bogus"])
        monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR", tmp_path / "daily")
        res = check_datasets(0)
        assert res.passed is False
        assert any("unknown_dataset" in d for _f, d in res.issues)


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
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(daily / "000001.parquet", index=False)
        h1 = dataset_fingerprint(root, ["daily"])
        assert dataset_fingerprint(root, ["daily"]) == h1
        # Rebuild the same file with more rows: size + mtime change -> digest flips.
        _daily(TRADE_DATES + ["2024-01-09"], [10.0] * 6).to_parquet(
            daily / "000001.parquet", index=False
        )
        assert dataset_fingerprint(root, ["daily"]) != h1

    def test_dataset_fingerprint_missing_dir_is_stable(self, tmp_path):
        h1 = dataset_fingerprint(tmp_path, ["daily", "features_panel"])
        h2 = dataset_fingerprint(tmp_path, ["daily", "features_panel"])
        assert h1 == h2
        assert len(h1) == 16
