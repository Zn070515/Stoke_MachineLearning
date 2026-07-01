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
            series = pd.Series(df[col].values, index=df.index)

            # Rolling winsorize (PIT-safe: backward-looking only)
            roll_mean = series.rolling(
                self.window_days, min_periods=self.min_periods
            ).mean()
            roll_std = series.rolling(
                self.window_days, min_periods=self.min_periods
            ).std()
            upper = roll_mean + self.winsorize_sigma * roll_std
            lower = roll_mean - self.winsorize_sigma * roll_std
            values = series.values.astype(np.float64)
            w_values = np.where(
                roll_std.values > 1e-10,
                np.clip(values, lower.values, upper.values),
                values,
            )

            # Rolling robust scale: z = (x - rolling_median) / (rolling_MAD * 1.4826)
            win_series = pd.Series(w_values, index=df.index)
            roll_median = win_series.rolling(
                self.window_days, min_periods=self.min_periods
            ).median()
            roll_mad = win_series.rolling(
                self.window_days, min_periods=self.min_periods
            ).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            # Adaptive epsilon: 0.1% of rolling median, floor at 1e-6
            med_abs = np.abs(roll_median.values)
            eps = np.maximum(med_abs * 1e-3, 1e-6)
            scaled = (w_values - roll_median.values) / (roll_mad.values * 1.4826 + eps)
            # Clip to prevent inf from float32 overflow
            scaled = np.clip(scaled, -1e4, 1e4)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            df[col] = scaled.astype(np.float32)

        return df
