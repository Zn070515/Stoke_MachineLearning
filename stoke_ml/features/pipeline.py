"""Feature pipeline orchestrating all feature engineering steps."""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.temporal import (
    add_lag_features, add_rolling_features, add_calendar_features,
)


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
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.flat_mode = flat_mode
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()

    def build_features(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self._engineer_features(df)
        X, y, aligned_close = self._create_sequences(feats, target_col)
        return X, y, aligned_close

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ti.compute_all(df)
        df = self._scorer.score(df)
        cols = self.TARGET_COLS + ["volume_ratio", "atr_14", "rsi_12"]
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
