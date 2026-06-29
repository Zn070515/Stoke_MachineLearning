"""Tests for FeaturePipeline — feature engineering and merge logic."""
import numpy as np
import pandas as pd
import pytest
from stoke_ml.features.pipeline import FeaturePipeline, SENTIMENT_COLS, GUBA_COLS


def _make_kl(n_days=200):
    """Synthetic K-line data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates, "open": close - rng.normal(0, 0.5, n_days),
        "high": close + rng.uniform(0.1, 2.0, n_days),
        "low": close - rng.uniform(0.1, 2.0, n_days),
        "close": close,
        "volume": rng.integers(1e6, 1e7, n_days).astype(float),
    })


def _make_sentiment(dates):
    """Daily sentiment DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "sentiment_mean": rng.uniform(-1, 1, len(dates)).astype(np.float32),
        "sentiment_std": rng.uniform(0, 0.5, len(dates)).astype(np.float32),
        "news_count": rng.integers(0, 10, len(dates)).astype("int16"),
        "positive_ratio": rng.uniform(0, 1, len(dates)).astype(np.float32),
        "negative_ratio": rng.uniform(0, 1, len(dates)).astype(np.float32),
        "has_news": [True] * len(dates),
    })


class TestFeaturePipelineBuild:

    def test_technical_only_returns_valid_shapes(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        X, y, aligned_close = pipe.build_features(df, target_col="close")
        assert X.ndim == 3  # (samples, seq_len, features)
        assert X.shape[0] > 0
        assert X.shape[1] == 20  # seq_len
        assert len(y) == X.shape[0]
        assert len(aligned_close) > len(y)

    def test_with_sentiment_merge(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=True, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        sentiment = _make_sentiment(df["date"])
        X, y, _ = pipe.build_features(df, sentiment_df=sentiment)
        assert X.shape[0] > 0

    def test_sentiment_disabled_skips_merge(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        sentiment = _make_sentiment(df["date"])
        X1, _, _ = pipe.build_features(df, sentiment_df=sentiment)
        X2, _, _ = pipe.build_features(df, sentiment_df=None)
        # Should produce identical shapes with or without data when disabled
        assert X1.shape == X2.shape

    def test_flat_mode_output(self):
        pipe = FeaturePipeline(
            seq_len=20, flat_mode=True, use_sentiment=False,
            use_announcements=False, use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        X, y, _ = pipe.build_features(df, target_col="close")
        assert X.ndim == 2  # (samples, seq_len * features)

    def test_build_features_insufficient_data(self):
        pipe = FeaturePipeline(seq_len=100)
        df = _make_kl(50)  # shorter than seq_len
        X, y, aligned_close = pipe.build_features(df)
        assert len(X) == 0
        assert len(y) == 0

    def test_target_is_binary(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        _, y, _ = pipe.build_features(df, target_col="close")
        assert set(np.unique(y)).issubset({0, 1})


class TestFeaturePipelineFlags:

    def test_new_flags_default_true(self):
        pipe = FeaturePipeline()
        assert pipe.use_margin is True
        assert pipe.use_northbound is True
        assert pipe.use_dragon_tiger is True
        assert pipe.use_fundamental is True
        assert pipe.use_etf_flow is True

    def test_new_flags_can_be_disabled(self):
        pipe = FeaturePipeline(
            use_margin=False, use_northbound=False,
            use_dragon_tiger=False, use_fundamental=False,
            use_etf_flow=False,
        )
        assert pipe.use_margin is False
        assert pipe.use_northbound is False

    def test_margin_merge_disabled_respects_flag(self):
        pipe = FeaturePipeline(seq_len=20, use_margin=False)
        df = _make_kl(300)
        margin_df = pd.DataFrame({
            "date": df["date"],
            "margin_balance": [1.0] * len(df),
            "margin_buy": [0.5] * len(df),
        })
        X1, _, _ = pipe.build_features(df, margin_df=margin_df)
        pipe2 = FeaturePipeline(seq_len=20, use_margin=True)
        X2, _, _ = pipe2.build_features(df, margin_df=margin_df)
        # With margin enabled, more columns → different shape
        assert X1.shape[2] != X2.shape[2]


class TestMergeHelpers:

    def test_merge_guba_adds_columns(self):
        pipe = FeaturePipeline(
            use_guba=True, use_sentiment=False, use_announcements=False,
            use_comment=False,
        )
        df = _make_kl(100)
        guba = pd.DataFrame({
            "date": df["date"],
            "guba_sentiment_mean": [0.3] * len(df),
            "guba_sentiment_std": [0.1] * len(df),
            "guba_post_count": [5] * len(df),
            "guba_positive_ratio": [0.4] * len(df),
            "guba_negative_ratio": [0.2] * len(df),
            "has_guba_post": [True] * len(df),
        })
        # Just test merge doesn't crash
        result = pipe._merge_guba(df.copy(), guba)
        assert "guba_sentiment_mean" in result.columns

    def test_merge_sentiment_lags_by_one_day(self):
        pipe = FeaturePipeline(use_sentiment=True)
        df = _make_kl(10)
        sentiment = pd.DataFrame({
            "date": df["date"],
            "sentiment_mean": np.arange(1, 11, dtype=np.float32),
            "sentiment_std": [0.1] * 10,
            "news_count": [1] * 10,
            "positive_ratio": [0.5] * 10,
            "negative_ratio": [0.1] * 10,
            "has_news": [True] * 10,
        })
        result = pipe._merge_sentiment(df.copy(), sentiment)
        # First row: shift + ZI fill → 0.0 (no prior-day sentiment available)
        assert result["sentiment_mean"].iloc[0] == 0.0
        # Second row should have first row's pre-shift sentiment (value 1.0)
        assert result["sentiment_mean"].iloc[1] == 1.0
