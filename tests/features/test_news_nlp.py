"""Tests for NewsSentimentAnalyzer — lexicon and FinBERT sentiment scoring."""
from unittest.mock import patch

import numpy as np
import pytest
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer, compute_raw_sentiment,
    aggregate_daily_sentiment, build_sentiment_dataframe,
)


class TestLexiconSentiment:
    """Lexicon fallback path (no GPU required)."""

    def test_positive_text_scores_positive(self):
        analyzer = NewsSentimentAnalyzer()
        scores = analyzer.analyze(["利好大涨突破新高涨停"])
        assert len(scores) == 1
        assert scores[0] > 0.5

    def test_negative_text_scores_negative(self):
        analyzer = NewsSentimentAnalyzer()
        scores = analyzer.analyze(["利空大跌跳水跌停"])
        assert len(scores) == 1
        assert scores[0] < -0.5

    def test_empty_input_returns_empty(self):
        analyzer = NewsSentimentAnalyzer()
        scores = analyzer.analyze([])
        assert len(scores) == 0
        assert scores.dtype == np.float32

    def test_neutral_text_scores_near_zero(self):
        analyzer = NewsSentimentAnalyzer()
        with patch.object(analyzer, "_try_finbert", return_value=False):
            scores = analyzer.analyze(["今天天气不错"])
        assert len(scores) == 1
        assert abs(scores[0]) < 0.1

    def test_batch_scoring(self):
        analyzer = NewsSentimentAnalyzer()
        texts = ["利好大涨突破", "利空大跌跳水", "普通消息"]
        scores = analyzer.analyze(texts)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] < 0

    def test_use_finbert_flag_defaults_false(self):
        analyzer = NewsSentimentAnalyzer()
        assert analyzer._use_finbert is False
        assert analyzer._loaded is False
        assert analyzer._model_name == "lexicon"

    def test_ensure_loaded_sets_loaded_flag(self):
        analyzer = NewsSentimentAnalyzer()
        with patch.object(analyzer, "_try_finbert", return_value=False):
            analyzer._ensure_loaded()
        assert analyzer._loaded is True
        assert analyzer._model_name == "lexicon"
        assert analyzer._use_finbert is False


class TestComputeRawSentiment:
    """compute_raw_sentiment helper."""

    def test_adds_sentiment_title_column(self):
        import pandas as pd
        df = pd.DataFrame({"title": ["利好大涨", "利空大跌", "普通消息"]})
        analyzer = NewsSentimentAnalyzer()
        result = compute_raw_sentiment(df, analyzer)
        assert "sentiment_title" in result.columns
        assert len(result) == 3
        assert result["sentiment_title"].dtype == np.float32
        assert result["sentiment_title"].iloc[0] > 0
        assert result["sentiment_title"].iloc[1] < 0

    def test_adds_sentiment_body_when_body_present(self):
        import pandas as pd
        df = pd.DataFrame({
            "title": ["利好", "利空"],
            "body": ["好消息大涨", "坏消息大跌"],
        })
        analyzer = NewsSentimentAnalyzer()
        result = compute_raw_sentiment(df, analyzer)
        assert "sentiment_body" in result.columns
        assert result["sentiment_body"].dtype == np.float32

    def test_empty_df_returns_empty(self):
        import pandas as pd
        df = pd.DataFrame()
        result = compute_raw_sentiment(df)
        assert result.empty

    def test_works_without_passing_analyzer(self):
        import pandas as pd
        df = pd.DataFrame({"title": ["利好大涨"]})
        result = compute_raw_sentiment(df)
        assert "sentiment_title" in result.columns


class TestAggregateDailySentiment:
    """Daily aggregation helper."""

    def test_empty_titles_returns_zeros(self):
        result = aggregate_daily_sentiment([])
        assert result["news_count"] == 0
        assert result["sentiment_mean"] == 0.0
        assert result["positive_ratio"] == 0.0

    def test_single_title_aggregation(self):
        analyzer = NewsSentimentAnalyzer()
        result = aggregate_daily_sentiment(["利好大涨突破"], analyzer)
        assert result["news_count"] == 1
        assert result["positive_ratio"] == 1.0
        assert result["negative_ratio"] == 0.0

    def test_mixed_titles(self):
        analyzer = NewsSentimentAnalyzer()
        result = aggregate_daily_sentiment(
            ["利好大涨突破", "利空大跌跳水", "普通消息"], analyzer
        )
        assert result["news_count"] == 3
        assert 0 <= result["positive_ratio"] <= 1
        assert 0 <= result["negative_ratio"] <= 1


class TestBuildSentimentDataframe:
    """DataFrame builder for storage."""

    def test_builds_correct_columns(self):
        daily = {
            "date": ["2024-01-02", "2024-01-03"],
            "sentiment_mean": [0.5, -0.3],
            "sentiment_std": [0.1, 0.2],
            "news_count": [5, 3],
            "positive_ratio": [0.6, 0.0],
            "negative_ratio": [0.0, 0.33],
        }
        df = build_sentiment_dataframe(daily, "600519")
        assert "stock_code" in df.columns
        assert "has_news" in df.columns
        assert df["has_news"].iloc[0] == True  # noqa: E712
        assert df["has_news"].iloc[1] == True  # noqa: E712
        assert len(df) == 2
