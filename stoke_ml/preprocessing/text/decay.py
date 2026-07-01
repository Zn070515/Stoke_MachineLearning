"""Time-decay weighting for sentiment posts.

Each post gets a weight w = exp(-lambda * days_before_reference) where
lambda = ln(2) / halflife_days.  More recent posts carry more weight.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class TimeDecayWeighter(PreprocessingStep):
    """Apply exponential time decay to sentiment posts.

    w_i = exp(-lambda * days_since_post)
    where lambda = ln(2) / halflife_days

    Adds columns: decay_weight, weighted_sent
    """

    def __init__(self, halflife_days: float = 7.0):
        self.halflife_days = halflife_days
        self._lambda = np.log(2) / halflife_days

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, reference_date=None, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        if "date" not in df.columns:
            return df

        dates = pd.to_datetime(df["date"])
        if reference_date is None:
            ref = dates.max()
        else:
            ref = pd.Timestamp(reference_date)

        days_diff = (ref - dates).dt.days.values.astype(float)
        days_diff = np.maximum(days_diff, 0.0)
        weights = np.exp(-self._lambda * days_diff)
        df["decay_weight"] = weights.astype(np.float32)

        if "sentiment_title" in df.columns:
            df["weighted_sent"] = (
                df["sentiment_title"].fillna(0.0) * weights
            ).astype(np.float32)

        return df
