"""Tests for BipolarClassifier — bull/bear/neutral from FinBERT scores."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier


class TestBipolarClassifier:
    def test_default_thresholds(self):
        bc = BipolarClassifier()
        assert bc.pos_threshold == 0.2
        assert bc.neg_threshold == -0.2

    def test_custom_thresholds(self):
        bc = BipolarClassifier(pos_threshold=0.3, neg_threshold=-0.3)
        assert bc.pos_threshold == 0.3

    def test_classifies_bull(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [0.5, 0.8, 0.25]})
        result = bc.fit_transform(df)
        assert result["is_bull_title"].tolist() == [1, 1, 1]
        assert result["is_bear_title"].tolist() == [0, 0, 0]
        assert result["is_neutral_title"].tolist() == [0, 0, 0]

    def test_classifies_bear(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [-0.5, -0.8, -0.25]})
        result = bc.fit_transform(df)
        assert result["is_bear_title"].tolist() == [1, 1, 1]
        assert result["is_bull_title"].tolist() == [0, 0, 0]

    def test_classifies_neutral(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [0.1, -0.1, 0.0, 0.19, -0.19]})
        result = bc.fit_transform(df)
        assert result["is_neutral_title"].tolist() == [1, 1, 1, 1, 1]

    def test_empty_df_passthrough(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": pd.Series([], dtype=float)})
        result = bc.fit_transform(df)
        assert len(result) == 0

    def test_missing_column_no_error(self):
        bc = BipolarClassifier(sentiment_cols=["nonexistent"])
        df = pd.DataFrame({"sentiment_title": [0.5]})
        result = bc.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_handles_multiple_sentiment_columns(self):
        bc = BipolarClassifier(sentiment_cols=["sentiment_title", "sentiment_body"])
        df = pd.DataFrame({
            "sentiment_title": [0.5, -0.5, 0.1],
            "sentiment_body": [0.8, -0.8, 0.0],
        })
        result = bc.fit_transform(df)
        assert "is_bull_title" in result.columns
        assert "is_bear_title" in result.columns
        assert "is_bull_body" in result.columns
        assert "is_bear_body" in result.columns
