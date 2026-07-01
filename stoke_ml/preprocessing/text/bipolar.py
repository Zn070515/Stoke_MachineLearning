"""Bipolar sentiment classifier: bull / bear / neutral from FinBERT scores.

Produces per-row binary flags (is_bull, is_bear, is_neutral) from continuous
sentiment scores.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class BipolarClassifier(PreprocessingStep):
    """Classify sentiment scores into bull/bear/neutral.

    Default thresholds (+/-0.2) are tuned for FinBERT Chinese model which
    outputs P(positive) - P(negative) in [-1, 1].
    """

    def __init__(
        self,
        pos_threshold: float = 0.2,
        neg_threshold: float = -0.2,
        sentiment_cols: list[str] | None = None,
    ):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.sentiment_cols = sentiment_cols

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        cols = self.sentiment_cols
        if cols is None:
            cols = [c for c in df.columns
                    if c.startswith("sentiment_") and c not in (
                        "sentiment_mean", "sentiment_std",
                    )]
        if not cols:
            return df

        available = [c for c in cols if c in df.columns]
        if not available:
            return df

        for col in available:
            suffix = _col_suffix(col)
            values = df[col].values
            nan_mask = np.isnan(values)
            df[f"is_bull_{suffix}"] = np.where(
                nan_mask, np.nan, (values > self.pos_threshold).astype("int8"),
            )
            df[f"is_bear_{suffix}"] = np.where(
                nan_mask, np.nan, (values < self.neg_threshold).astype("int8"),
            )
            df[f"is_neutral_{suffix}"] = np.where(
                nan_mask, np.nan,
                ((values >= self.neg_threshold) & (values <= self.pos_threshold)).astype("int8"),
            )

        return df


def _col_suffix(col_name: str) -> str:
    """Extract short suffix from sentiment column name."""
    if col_name == "sentiment_title":
        return "title"
    if col_name == "sentiment_body":
        return "body"
    if col_name.startswith("sentiment_"):
        return col_name[len("sentiment_"):]
    return col_name
