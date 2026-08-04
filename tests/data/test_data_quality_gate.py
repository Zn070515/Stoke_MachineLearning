"""Data quality gate tests (v6 §九).

The gate must FAIL when any check records a problem: read errors, missing
columns, or data inconsistencies must flip ``passed`` to False (a problem
recorded in the report must also flip the gate).  Sparsity uses NaN-excluded
coverage because ``(x != 0).mean()`` counts NaN as non-zero and inflates
coverage for missing-heavy features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_quality_gate import (
    check_contract_schema,
    check_daily_internal,
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
        """v6 §九: NaN != 0 is True, so it must be excluded from coverage."""
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
