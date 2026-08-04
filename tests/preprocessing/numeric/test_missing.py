"""Tests for MissingImputer -- gap-classified causal imputation.

No linear interpolation, no backfill, no bidirectional smoothing: every
fill must only use observations at or before the imputed row.
"""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.missing import MissingImputer


class TestMissingImputer:
    def test_short_gap_forward_fill(self):
        mi = MissingImputer(short_gap_max=2, medium_gap_max=10)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = mi.fit_transform(df)
        assert not np.isnan(result["x"].iloc[1])
        # Forward-fill holds the last observed value; it must NOT move toward
        # the future anchor 3.0 (old linear interp produced 2.0).
        assert result["x"].iloc[1] == 1.0

    def test_short_gap_no_future_leak(self):
        mi = MissingImputer(short_gap_max=2, medium_gap_max=10)
        df = pd.DataFrame({"x": [10.0, np.nan, np.nan, 1000.0]})
        result = mi.fit_transform(df)
        # Gap rows inherit the pre-gap value, entirely ignoring the post-gap
        # 1000.0 (the t-1=10, t=NaN, t+1=20 -> t=15 example).
        assert result["x"].iloc[1] == 10.0
        assert result["x"].iloc[2] == 10.0

    def test_truncation_invariance(self):
        """截断在 t: imputing the full series then truncating at t must
        equal imputing only the prefix up to t — no future rows involved."""
        full = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, 4.0, np.nan, np.nan, 7.0, np.nan],
        })
        imputed_full = MissingImputer().fit_transform(full)["x"]
        for t in (1, 2, 4, 5, 7):
            prefix = full.iloc[:t + 1].copy()
            imputed_prefix = MissingImputer().fit_transform(prefix)["x"]
            assert np.array_equal(
                imputed_prefix.to_numpy(),
                imputed_full.iloc[:t + 1].to_numpy(),
                equal_nan=True,
            )

    def test_medium_gap_attempts_kalman(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=5)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [10.0, np.nan, np.nan, np.nan, 50.0],
        })
        result = mi.fit_transform(df)
        assert len(result) == 5
        # All three gap positions should be filled (Kalman forecast or causal forward-fill fallback)
        assert not np.isnan(result["x"].iloc[1])
        assert not np.isnan(result["x"].iloc[2])
        assert not np.isnan(result["x"].iloc[3])
        assert not np.isnan(result["y"].iloc[1])
        assert not np.isnan(result["y"].iloc[2])
        assert not np.isnan(result["y"].iloc[3])

    def test_long_gap_keeps_nan(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=2)
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 10 + [100.0]})
        result = mi.fit_transform(df)
        assert np.isnan(result["x"].iloc[5])

    def test_generates_gap_flags(self):
        mi = MissingImputer()  # default medium_gap_max=10
        df = pd.DataFrame({
            "x": [1.0] + [np.nan] * 15 + [5.0] * 5,
            "y": [1.0] * 21,
        })
        result = mi.fit_transform(df)
        flag_cols = [c for c in result.columns if c.startswith("has_gap_")]
        assert len(flag_cols) >= 1
        # Gap positions should be flagged
        assert result["has_gap_x"].iloc[5] == 1

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
