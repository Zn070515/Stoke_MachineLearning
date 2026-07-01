"""Tests for DailyAggregator — multi-dimensional daily sentiment aggregation."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.aggregation import DailyAggregator


class TestDailyAggregator:
    def test_aggregates_per_day(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "sentiment_title": [0.5, -0.3, 0.8],
            "decay_weight": [0.5, 1.0, 1.0],
        })
        result = agg.fit_transform(df)
        assert "date" in result.columns
        assert len(result) == 2

    def test_computes_bipolar_sent(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, 0.8, -0.5],
        })
        result = agg.fit_transform(df)
        # 2 bull (0.5, 0.8), 1 bear (-0.5), 0 neutral
        # bipolar = (2-1)/(2+1+1) = 1/4 = 0.25
        row = result.iloc[0]
        assert 0.2 < row["bipolar_sent"] < 0.3

    def test_computes_agreement_index(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"] * 5),
            "sentiment_title": [0.8, 0.7, 0.9, 0.6, 0.8],
        })
        result = agg.fit_transform(df)
        assert result["agreement"].iloc[0] > 0.8

    def test_computes_attention(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"] * 42),
            "sentiment_title": [0.5] * 42,
        })
        result = agg.fit_transform(df)
        assert 3.5 < result["attention"].iloc[0] < 4.0

    def test_computes_body_sentiment_separately(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, -0.3],
            "sentiment_body": [0.8, -0.5],
        })
        result = agg.fit_transform(df)
        assert "body_sent_mean" in result.columns
        assert 0.1 < result["body_sent_mean"].iloc[0] < 0.2

    def test_handles_missing_decay_column(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, -0.3],
        })
        result = agg.fit_transform(df)
        assert "bipolar_sent" in result.columns

    def test_single_post_per_day(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"]),
            "sentiment_title": [0.6],
        })
        result = agg.fit_transform(df)
        assert result["bipolar_sent"].iloc[0] > 0
        assert result["sent_divergence"].iloc[0] == 0.0

    def test_empty_df(self):
        agg = DailyAggregator()
        result = agg.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_adds_rolling_windows(self):
        agg = DailyAggregator(windows=[3, 5])
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime([
                "2024-01-02", "2024-01-03", "2024-01-04",
                "2024-01-05", "2024-01-08", "2024-01-09",
            ]),
            "sentiment_title": [0.5, -0.3, 0.2, 0.4, 0.1, -0.2],
        })
        result = agg.fit_transform(df)
        assert "bipolar_sent_3d_mean" in result.columns
        assert "bipolar_sent_5d_mean" in result.columns
        assert "attention_3d_mean" in result.columns
