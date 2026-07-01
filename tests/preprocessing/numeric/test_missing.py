"""Tests for MissingImputer -- gap-classified interpolation (linear/Kalman/flag)."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.missing import MissingImputer


class TestMissingImputer:
    def test_short_gap_linear_interpolation(self):
        mi = MissingImputer(short_gap_max=2, medium_gap_max=10)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = mi.fit_transform(df)
        assert not np.isnan(result["x"].iloc[1])
        assert 1.5 < result["x"].iloc[1] < 2.5

    def test_medium_gap_attempts_kalman(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=5)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [10.0, np.nan, np.nan, np.nan, 50.0],
        })
        result = mi.fit_transform(df)
        # Medium gap: attempt Kalman, fallback to linear
        # At minimum, should not crash and should return same length
        assert len(result) == 5
        # At least one value should be filled (either Kalman or linear fallback)
        assert not np.isnan(result["x"].iloc[1]) or not np.isnan(result["x"].iloc[2])

    def test_long_gap_keeps_nan(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=2)
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 10 + [100.0]})
        result = mi.fit_transform(df)
        assert np.isnan(result["x"].iloc[5])

    def test_generates_gap_flags(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=1)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = mi.fit_transform(df)
        flag_cols = [c for c in result.columns if c.startswith("has_gap_")]
        assert len(flag_cols) >= 1

    def test_no_gaps_no_flags(self):
        mi = MissingImputer()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = mi.fit_transform(df)
        flag_cols = [c for c in result.columns if c.startswith("has_gap_")]
        assert len(flag_cols) == 0

    def test_empty_df(self):
        mi = MissingImputer()
        result = mi.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_respects_max_gap_settings(self):
        mi = MissingImputer(short_gap_max=0, medium_gap_max=0)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = mi.fit_transform(df)
        # With short_gap_max=0, medium_gap_max=0, this 1-step gap is >both
        # So it's a "long gap" -> NaN preserved
        assert len(result) == 3
        assert np.isnan(result["x"].iloc[1])

    def test_preserves_non_numeric_columns(self):
        mi = MissingImputer()
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "stock": ["A", "B", "C"],
        })
        result = mi.fit_transform(df)
        assert "stock" in result.columns
        assert result["stock"].tolist() == ["A", "B", "C"]
