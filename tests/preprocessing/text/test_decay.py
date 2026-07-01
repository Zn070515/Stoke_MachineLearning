"""Tests for TimeDecayWeighter — EMA-based time decay weighting."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter


class TestTimeDecayWeighter:
    def test_default_halflife(self):
        td = TimeDecayWeighter()
        assert td.halflife_days == 7

    def test_custom_halflife(self):
        td = TimeDecayWeighter(halflife_days=14)
        assert td.halflife_days == 14

    def test_adds_weight_column(self):
        td = TimeDecayWeighter()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"]),
            "sentiment_title": [0.5, -0.3, 0.8],
        })
        result = td.fit_transform(df)
        assert "decay_weight" in result.columns
        assert result["decay_weight"].iloc[-1] == 1.0

    def test_weights_decay_over_time(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "sentiment_title": [0.5, 0.3, 0.8],
        })
        result = td.fit_transform(df)
        assert result["decay_weight"].iloc[-1] == 1.0
        assert 0.45 < result["decay_weight"].iloc[1] < 0.55
        assert 0.2 < result["decay_weight"].iloc[0] < 0.3

    def test_respects_reference_date(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "sentiment_title": [0.5, 0.3],
        })
        result = td.fit_transform(df, reference_date="2024-01-12")
        assert 0.4 < result["decay_weight"].iloc[1] < 0.6

    def test_empty_df(self):
        td = TimeDecayWeighter()
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
        result = td.fit_transform(df)
        assert len(result) == 0

    def test_calculates_weighted_sentiment(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"]),
            "sentiment_title": [1.0, -1.0, 0.5],
        })
        result = td.fit_transform(df)
        assert "weighted_sent" in result.columns
        # weighted_sent = sentiment_title * decay_weight (per-row)
        for i in range(len(df)):
            expected = result["sentiment_title"].iloc[i] * result["decay_weight"].iloc[i]
            assert abs(result["weighted_sent"].iloc[i] - expected) < 0.001
