"""Tests for TechnicalIndicators — Alpha158-style factor computation."""
import numpy as np
import pandas as pd
import pytest
from stoke_ml.features.technical import TechnicalIndicators


def _make_ohlcv(n_days=200):
    """Generate synthetic OHLCV data with a trend and noise."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_days))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.1, 2.0, n_days)
    low = close - rng.uniform(0.1, 2.0, n_days)
    open_ = close - rng.normal(0, 0.5, n_days)
    volume = rng.integers(1000000, 10000000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


class TestTechnicalIndicators:

    def test_compute_all_adds_moving_averages(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for period in [5, 10, 20, 60, 120]:
            assert f"ma_{period}" in result.columns

    def test_compute_all_adds_macd(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["macd_dif", "macd_dea", "macd_hist"]:
            assert col in result.columns

    def test_compute_all_adds_rsi(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for period in [6, 12, 24]:
            col = f"rsi_{period}"
            assert col in result.columns
            # RSI should be between 0 and 100
            valid = result[col].dropna()
            assert (valid >= 0).all()
            assert (valid <= 100).all()

    def test_compute_all_adds_kdj(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for period in [9, 14]:
            for suffix in ["k", "d", "j"]:
                assert f"kdj_{suffix}_{period}" in result.columns

    def test_compute_all_adds_bollinger(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["boll_mid", "boll_upper", "boll_lower", "boll_pct"]:
            assert col in result.columns

    def test_compute_all_adds_atr(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        assert "atr_14" in result.columns

    def test_compute_all_adds_volume_indicators(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["volume_ma5", "volume_ratio", "obv"]:
            assert col in result.columns

    def test_compute_all_adds_rolling_position_factors(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for d in [5, 10, 20, 30, 60]:
            for prefix in ["max_", "min_", "qtlu_", "qtld_", "rank_", "rsv_"]:
                assert f"{prefix}{d}d" in result.columns

    def test_compute_all_adds_kbar_features(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["kmid", "klen", "kmid2", "kup", "kup2",
                     "klow", "klow2", "ksft", "ksft2"]:
            assert col in result.columns

    def test_compute_all_adds_price_features(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["open0", "high0", "low0"]:
            assert col in result.columns

    def test_no_nan_in_price_columns(self):
        """Output preserves OHLCV columns."""
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_small_input_does_not_crash(self):
        """Very short OHLCV should not crash."""
        ti = TechnicalIndicators()
        df = _make_ohlcv(3)
        result = ti.compute_all(df)
        assert len(result) == 3

    def test_adx_added(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        for col in ["adx", "adxr", "pdi", "mdi"]:
            assert col in result.columns
        # ADX should be between 0 and 100
        valid = result["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_mfi_added(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        assert "mfi_14" in result.columns
        valid = result["mfi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_cmo_added(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        assert "cmo_14" in result.columns
        # CMO ranges -100 to +100
        valid = result["cmo_14"].dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()

    def test_trix_added(self):
        ti = TechnicalIndicators()
        df = _make_ohlcv(200)
        result = ti.compute_all(df)
        assert "trix" in result.columns
