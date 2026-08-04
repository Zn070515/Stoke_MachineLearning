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
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_quality_gate import (
    _sample_files,
    check_contract_schema,
    check_daily_internal,
    check_datasets,
    check_feature_pct,
    check_ohlc_sanity,
    check_sparsity,
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
        monkeypatch.setattr("scripts.data_quality_gate.FEAT_DIR", feat_dir)
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
        monkeypatch.setattr("scripts.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_sparsity(0)
        assert res.passed is True


class TestFailOnReadError:
    def test_corrupt_daily_fails_daily_internal(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_daily_internal(0)
        assert res.passed is False
        assert any("000001" in f for f, _d in res.issues)

    def test_missing_col_fails_daily_internal(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # date + close only — no pct_change column.
        pd.DataFrame({
            "date": pd.to_datetime(TRADE_DATES),
            "close": [10.0] * 5,
        }).to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
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
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.data_quality_gate.FEAT_DIR", feat_dir)
        res = check_feature_pct(0)
        assert res.passed is False


class TestOhlcSanity:
    def test_clean_file_passes(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0, 10.5, 10.2, 10.8, 10.4]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is True

    def test_weekend_row_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # 2024-01-06 is a Saturday — A-shares never trade weekends.
        _daily(["2024-01-05", "2024-01-06"], [10.0, 10.2]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("weekend" in d for _f, d in res.issues)

    def test_duplicate_date_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES[:3] + [TRADE_DATES[2]], [10.0] * 4).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("dup" in d for _f, d in res.issues)

    def test_low_gt_high_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5)
        df.loc[2, "low"] = 20.0  # low > high
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False

    def test_negative_volume_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5)
        df.loc[0, "volume"] = -100.0
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
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
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("stock_code" in d for _f, d in res.issues)

    def test_missing_stock_code_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        df = _daily(TRADE_DATES, [10.0] * 5).drop(columns=["stock_code"])
        df.to_parquet(daily_dir / "000001.parquet", index=False)
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_ohlc_sanity(0)
        assert res.passed is False
        assert any("stock_code" in d for _f, d in res.issues)


class TestContractSchema:
    def test_clean_file_passes(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0, 10.5, 10.2, 10.8, 10.4]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is True
        assert res.issues == []
        assert res.files_scanned == 1

    def test_missing_required_column_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        # "amount" is a DAILY_EQUITY required column — dropping it must fail the gate.
        _daily(TRADE_DATES, [10.0] * 5).drop(columns=["amount"]).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("missing_column:amount" in d for _f, d in res.issues)

    def test_duplicate_pk_date_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES[:3] + [TRADE_DATES[2]], [10.0] * 4).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("pk_dup" in d for _f, d in res.issues)

    def test_corrupt_file_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "000001.parquet").write_bytes(b"not a parquet")
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_contract_schema(0)
        assert res.passed is False
        assert any("000001" in f for f, _d in res.issues)
        assert any("read_err" in d for _f, d in res.issues)


class TestDatasetsPreGate:
    """Empty/missing required data must FAIL; --allow-empty opt-out."""

    def test_empty_dir_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        res = check_datasets(0)
        assert res.passed is False
        assert any("files=0" in d for _f, d in res.issues)

    def test_missing_dir_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.data_quality_gate.DAILY_DIR", tmp_path / "does_not_exist"
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
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.data_quality_gate.MIN_STOCKS", 2)
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
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.data_quality_gate.MAX_STALE_DAYS", 30)
        monkeypatch.setattr("scripts.data_quality_gate.MIN_SPAN_DAYS", 0)
        res = check_datasets(0)
        assert res.passed is False
        assert any("stale=" in d for _f, d in res.issues)

    def test_short_span_fails(self, tmp_path, monkeypatch):
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        _daily(TRADE_DATES, [10.0] * 5).to_parquet(
            daily_dir / "000001.parquet", index=False
        )
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.data_quality_gate.MIN_SPAN_DAYS", 365)
        monkeypatch.setattr("scripts.data_quality_gate.MAX_STALE_DAYS", 10000)
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
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", daily_dir)
        monkeypatch.setattr("scripts.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.passed is True

    def test_allow_empty_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", tmp_path / "nope")
        monkeypatch.setattr("scripts.data_quality_gate.ALLOW_EMPTY", True)
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
        monkeypatch.setattr("scripts.data_quality_gate.REQUIRED_DATASETS", ["features"])
        monkeypatch.setattr("scripts.data_quality_gate.FEAT_DIR", feat_dir)
        monkeypatch.setattr("scripts.data_quality_gate.MAX_STALE_DAYS", 10000)
        res = check_datasets(0)
        assert res.passed is True

    def test_unknown_dataset_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.data_quality_gate.REQUIRED_DATASETS", ["bogus"])
        monkeypatch.setattr("scripts.data_quality_gate.DAILY_DIR", tmp_path / "daily")
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
