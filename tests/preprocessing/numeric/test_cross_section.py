"""Tests for CrossSectionNormalizer -- sector/size/adaptive normalization."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer


class TestCrossSectionNormalizer:
    def test_default_stages(self):
        csn = CrossSectionNormalizer()
        assert csn.stages == ["sector", "size", "adaptive"]
        assert csn.enabled is True

    def test_disabled_is_noop(self):
        csn = CrossSectionNormalizer(enabled=False)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = csn.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_sector_stage_neutralizes(self):
        csn = CrossSectionNormalizer(stages=["sector"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "stock_code": ["A", "B", "A", "C", "B", "C"],
            "x": [100.0, 200.0, 110.0, 50.0, 190.0, 55.0],
            "sector": ["bank", "tech", "bank", "health", "tech", "health"],
        })
        result = csn.fit_transform(df)
        assert "x" in result.columns
        assert "x_raw" in result.columns
        # Within each (date, sector), median should be 0 after neutralization
        medians = result.groupby(["date", "sector"])["x"].median()
        assert (medians.abs() < 0.01).all()

    def test_empty_df(self):
        csn = CrossSectionNormalizer()
        result = csn.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_no_sector_column_skips_sector_stage(self):
        csn = CrossSectionNormalizer(stages=["sector"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "stock_code": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
        })
        result = csn.fit_transform(df)
        assert "x" in result.columns

    def test_columns_to_normalize_are_specified(self):
        csn = CrossSectionNormalizer(columns=["x", "y"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "z": [100.0, 200.0, 300.0],
        })
        result = csn.fit_transform(df)
        np.testing.assert_array_equal(result["z"].values, df["z"].values)

    def test_size_stage_computes_residuals(self):
        csn = CrossSectionNormalizer(stages=["size"])
        df = pd.DataFrame({
            "date": ["2024-01-02"] * 15,
            "stock_code": [f"S{i}" for i in range(15)],
            "x": [float(i * 10) for i in range(15)],
            "market_cap": [1e8 + i * 5e7 for i in range(15)],
        })
        result = csn.fit_transform(df)
        assert "x" in result.columns
        assert "x_pre_size" in result.columns

    def test_adaptive_stage_applies_alpha(self):
        csn = CrossSectionNormalizer(stages=["adaptive"])
        rng = np.random.RandomState(42)
        n = 40
        prices = 100 + rng.randn(n).cumsum()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "stock_code": ["S1"] * n,
            "close": prices,
            "x": rng.randn(n).astype(np.float64),
        })
        result = csn.fit_transform(df)
        assert "x" in result.columns
        # With adaptive scaling, values should differ from original
        not_nan = result["x"].notna() & df["x"].notna()
        assert not np.allclose(
            result["x"].loc[not_nan].values,
            df["x"].loc[not_nan].values,
        )
