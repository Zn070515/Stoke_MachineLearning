"""Tests for HigherOrderDeriver -- skew, kurtosis, realized vol, Amihud illiquidity."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.higher_order import HigherOrderDeriver


class TestHigherOrderDeriver:
    def test_computes_skew_and_kurtosis(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
        })
        result = hod.fit_transform(df)
        assert "skew_20d" in result.columns
        assert "kurt_20d" in result.columns

    def test_computes_realized_vol(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
        })
        result = hod.fit_transform(df)
        assert "realized_vol_5d" in result.columns
        assert "realized_vol_20d" in result.columns

    def test_computes_amihud_illiquidity(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
            "volume": np.abs(rng.randn(n) * 1e6) + 1e5,
        })
        result = hod.fit_transform(df)
        assert "amihud_illiq_20d" in result.columns

    def test_computes_max_drawdown(self):
        hod = HigherOrderDeriver()
        prices = np.array([100, 102, 105, 103, 98, 95, 97, 101, 99, 105,
                           108, 110, 107, 104, 100, 97, 94, 96, 100, 103])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(prices), freq="B"),
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1e6] * len(prices),
        })
        result = hod.fit_transform(df)
        assert "max_drawdown_20d" in result.columns
        assert "up_days_ratio_20d" in result.columns
        dd = result["max_drawdown_20d"].dropna()
        assert (dd >= 0).all()

    def test_empty_df(self):
        hod = HigherOrderDeriver()
        result = hod.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_disabled_is_noop(self):
        hod = HigherOrderDeriver(enabled=False)
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = hod.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)
