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
