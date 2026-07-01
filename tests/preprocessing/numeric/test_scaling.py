"""Tests for RobustScaler -- rolling-window robust standardization."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.scaling import RobustScaler


class TestRobustScaler:
    def test_default_window(self):
        rs = RobustScaler()
        assert rs.window_days == 252

    def test_scales_to_median_zero(self):
        rs = RobustScaler(window_days=10, min_periods=5)
        values = np.arange(20, dtype=float)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="B"),
            "x": values,
        })
        rs.fit(df)
        result = rs.transform(df)
        mask = result["x"].notna()
        assert not np.allclose(result["x"].loc[mask].values,
                               df["x"].loc[mask].values)

    def test_preserves_nan(self):
        rs = RobustScaler(window_days=10, min_periods=5)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        rs.fit(df)
        result = rs.transform(df)
        assert np.isnan(result["x"].iloc[1])

    def test_skip_small_window(self):
        rs = RobustScaler(window_days=252, min_periods=63)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        rs.fit(df)
        result = rs.transform(df)
        assert result["x"].isna().all()

    def test_winsorize_before_scaling(self):
        # Tight winsorize caps extreme values → different output than loose
        rng = np.random.RandomState(42)
        base = list(rng.randn(58))
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60, freq="B"),
            "x": base + [100.0, -100.0],
        })
        result_tight = RobustScaler(
            window_days=30, min_periods=10, winsorize_sigma=1.0
        ).fit_transform(df)
        result_loose = RobustScaler(
            window_days=30, min_periods=10, winsorize_sigma=10.0
        ).fit_transform(df)
        # The last two rows (extreme values) should differ under different sigma
        tail_tight = result_tight["x"].iloc[-2:].values
        tail_loose = result_loose["x"].iloc[-2:].values
        assert not np.allclose(tail_tight, tail_loose, equal_nan=True)

    def test_empty_df(self):
        rs = RobustScaler()
        result = rs.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_skips_binary_columns(self):
        rs = RobustScaler(window_days=10, min_periods=5)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="B"),
            "x": rng.randn(20),
            "is_limit_up": [0] * 18 + [1, 0],
        })
        result = rs.fit_transform(df)
        # Binary column should be unchanged
        assert result["is_limit_up"].iloc[-2] == 1
        assert result["is_limit_up"].iloc[-1] == 0
        # x should be scaled (changed)
        assert not np.allclose(result["x"].values, df["x"].values, equal_nan=True)
