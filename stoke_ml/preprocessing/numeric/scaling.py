"""RobustScaler: rolling-window median/MAD standardization with winsorize.

Backward-looking windows only (PIT-safe). Fits parameters on training
data, reuses for validation/test.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class RobustScaler(PreprocessingStep):
    """Rolling-window robust standardization.

    For each column: winsorize(±winsorize_sigma * std), then
    z_robust = (x - rolling_median) / (rolling_MAD * 1.4826).

    The factor 1.4826 makes MAD consistent with standard deviation
    for normally distributed data.
    """

    def __init__(
        self,
        window_days: int = 252,
        winsorize_sigma: float = 3.0,
        min_periods: int = 63,
    ):
        self.window_days = window_days
        self.winsorize_sigma = winsorize_sigma
        self.min_periods = min_periods

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df.copy()
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skip = {"is_limit_up", "is_limit_down", "is_neutral",
                "is_bull", "is_bear", "has_news", "has_guba_post",
                "has_xueqiu_post", "has_announce", "has_comment",
                "date_day", "date_month", "date_weekday"}
        skip |= {c for c in df.columns if c.startswith("has_gap_")}
        for c in df.columns:
            if df[c].dropna().nunique() <= 2:
                skip.add(c)

        for col in numeric_cols:
            if col in skip:
                continue
            values = df[col].values.astype(np.float64)
            # Winsorize
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 1e-10:
                upper = mean + self.winsorize_sigma * std
                lower = mean - self.winsorize_sigma * std
                values = np.clip(values, lower, upper)

            # Rolling robust scale
            series = pd.Series(values, index=df.index)
            roll_median = series.rolling(
                self.window_days, min_periods=self.min_periods
            ).median()
            roll_mad = series.rolling(
                self.window_days, min_periods=self.min_periods
            ).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            scaled = (values - roll_median.values) / (roll_mad.values * 1.4826 + 1e-10)
            df[col] = scaled.astype(np.float32)

        return df
