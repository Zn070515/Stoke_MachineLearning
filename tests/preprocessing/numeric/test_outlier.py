"""Tests for OutlierDetector -- MAD-based outlier clip with limit-up/down protection."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector


class TestOutlierDetector:
    def test_default_threshold(self):
        od = OutlierDetector()
        assert od.threshold == 5.0

    def test_clips_extreme_outliers(self):
        od = OutlierDetector(threshold=3.0)
        base_close = [10.0, 10.5, 10.2, 9.9, 10.1, 10.3, 9.8, 10.0, 10.4, 10.2]
        base_volume = [1e6, 1.1e6, 9.5e5, 1e6, 1.05e6, 9.8e5, 1.02e6, 9.9e5, 1e6, 9.7e5]
        df = pd.DataFrame({
            "close": base_close + [1000.0, 0.01, 10.1],
            "volume": base_volume + [1e6, 1e6, 1e6],
        })
        result = od.fit_transform(df)
        assert result["close"].max() < 1000.0
        assert result["close"].min() > 0.01

    def test_preserves_limit_moves(self):
        """Limit-up/down (+-9.5%+) should NOT be clipped."""
        od = OutlierDetector()
        df = pd.DataFrame({
            "close": [10.0, 10.95, 9.05, 10.0, 11.0],
            "pct_change": [0.0, 0.095, -0.095, 0.0, 0.10],
        })
        result = od.fit_transform(df)
        assert result["close"].iloc[1] > 10.9
        assert result["close"].iloc[2] < 9.1

    def test_empty_df(self):
        od = OutlierDetector()
        result = od.fit_transform(pd.DataFrame({"x": pd.Series([], dtype=float)}))
        assert len(result) == 0

    def test_no_clip_when_within_threshold(self):
        od = OutlierDetector(threshold=10.0)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = od.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_returns_outlier_stats(self):
        od = OutlierDetector()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.5, 2.0, 1.0, 3.0, 100.0, 5.0]})
        od.fit(df)
        result = od.transform(df)
        assert result["x"].max() < 100.0

    def test_fit_then_transform_is_idempotent(self):
        od = OutlierDetector(threshold=5.0)
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 5})
        test = pd.DataFrame({"x": [1.0, 2.0, 100.0, 4.0, 5.0]})
        od.fit(train)
        result1 = od.transform(test)
        result2 = od.transform(test)
        pd.testing.assert_frame_equal(result1, result2)

    def test_skips_limit_cols(self):
        od = OutlierDetector(threshold=3.0)
        df = pd.DataFrame({
            "close": [10.0, 10.5, 10.3, 9.9, 10.1, 10.2, 9.8, 10.0, 10.4, 10.1, 1000.0, 10.2],
            "pct_change": [0.01] * 11 + [500.0],
            "is_limit_up": [0] * 11 + [1],
        })
        result = od.fit_transform(df)
        # pct_change should NOT be clipped (in _LIMIT_COLS)
        assert result["pct_change"].iloc[-1] == 500.0
        # close SHOULD be clipped
        assert result["close"].max() < 1000.0
