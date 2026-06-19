"""Feature pipeline orchestrating all feature engineering steps."""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.temporal import (
    add_lag_features, add_rolling_features, add_calendar_features,
)

# Sentiment columns merged from daily sentiment Gold layer
SENTIMENT_COLS = [
    "sentiment_mean", "sentiment_std", "news_count",
    "positive_ratio", "negative_ratio", "has_news",
]


class FeaturePipeline:
    """End-to-end feature engineering for stock prediction."""

    TARGET_COLS = ["open", "high", "low", "close", "volume"]
    LAGS = [1, 2, 3, 5, 10, 20]
    ROLLING_WINDOWS = [5, 10, 20, 60]

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        flat_mode: bool = False,
        use_technical: bool = True,
        use_scoring: bool = True,
        use_temporal: bool = True,
        use_sentiment: bool = True,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.flat_mode = flat_mode
        self.use_technical = use_technical
        self.use_scoring = use_scoring
        self.use_temporal = use_temporal
        self.use_sentiment = use_sentiment
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()

    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        sentiment_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self._engineer_features(df, sentiment_df)
        X, y, aligned_close = self._create_sequences(feats, target_col)
        return X, y, aligned_close

    def _engineer_features(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = df.copy()

        # Merge sentiment features before temporal transforms so lags
        # and rolling windows are computed on them too
        if self.use_sentiment and sentiment_df is not None and not sentiment_df.empty:
            s = sentiment_df.copy()
            s["date"] = pd.to_datetime(s["date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(
                s[["date"] + SENTIMENT_COLS],
                on="date", how="left",
            )
            # ZI method: missing-news days get zeros + has_news=False
            for col in SENTIMENT_COLS:
                if col == "has_news":
                    df[col] = df[col].fillna(False).astype(bool)
                elif col == "news_count":
                    df[col] = df[col].fillna(0).astype("int16")
                else:
                    df[col] = df[col].fillna(0.0).astype(np.float32)

        if self.use_technical:
            df = self._ti.compute_all(df)
        if self.use_scoring:
            df = self._scorer.score(df)
        if self.use_temporal:
            cols = self.TARGET_COLS + ["volume_ratio", "atr_14", "rsi_12"]
            if self.use_sentiment and sentiment_df is not None and not sentiment_df.empty:
                cols += [
                    "sentiment_mean", "sentiment_std", "positive_ratio",
                    "negative_ratio",
                ]
            df = add_lag_features(df, cols, self.LAGS)
            df = add_rolling_features(df, cols, self.ROLLING_WINDOWS)
            df = add_calendar_features(df)
        return df

    def _create_sequences(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        drop_cols = ["date", "stock_code"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        feat_df = feat_df.dropna()

        close = feat_df[target_col].values
        target = (close[self.horizon:] > close[: -self.horizon]).astype(int)

        price_cols = ["open", "high", "low", "close", "amount"]
        X_cols = [c for c in feat_df.columns if c not in price_cols]
        X_data = feat_df[X_cols].values.astype(np.float32)

        n_samples = len(X_data) - self.seq_len - self.horizon + 1
        if n_samples <= 0:
            empty = np.array([], dtype=np.float32)
            return empty, np.array([], dtype=np.int64), empty

        if self.flat_mode:
            X = np.array([
                X_data[i: i + self.seq_len].flatten()
                for i in range(n_samples)
            ], dtype=np.float32)
        else:
            X = np.array([
                X_data[i: i + self.seq_len]
                for i in range(n_samples)
            ], dtype=np.float32)

        y = target[self.seq_len - 1: self.seq_len - 1 + n_samples]
        # close prices aligned with predictions: n_samples + 1 points
        # giving n_samples price returns matching n_samples predictions
        aligned_close = close[self.seq_len - 1: self.seq_len + n_samples]
        return X, y, aligned_close.astype(np.float32)
