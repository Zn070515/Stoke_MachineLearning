"""Gap-classified missing value imputation.

Short gaps (1-2 days): linear interpolation.
Medium gaps (3-10 days): Kalman smoother (statsmodels) with linear fallback.
Long gaps (>10 days): NaN preserved + has_gap_{col} flag generated.
"""

import numpy as np
from stoke_ml.preprocessing.base import PreprocessingStep


class MissingImputer(PreprocessingStep):
    """Impute missing values by gap length with interpolation strategy.

    Never uses ZI (zero-imputation) -- that's the core improvement
    over the legacy approach.
    """

    def __init__(
        self,
        short_gap_max: int = 2,
        short_gap_method: str = "linear",
        medium_gap_max: int = 10,
        medium_gap_method: str = "kalman",
    ):
        self.short_gap_max = short_gap_max
        self.short_gap_method = short_gap_method
        self.medium_gap_max = medium_gap_max
        self.medium_gap_method = medium_gap_method

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df.copy()
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        gap_flags = {}

        for col in numeric_cols:
            values = df[col].to_numpy(copy=True)
            n = len(values)

            is_nan = np.isnan(values)
            if not is_nan.any():
                continue

            # Find gap runs
            gap_starts = []
            i = 0
            while i < n:
                if is_nan[i]:
                    j = i
                    while j < n and is_nan[j]:
                        j += 1
                    gap_len = j - i
                    gap_starts.append((i, gap_len))
                    i = j
                else:
                    i += 1

            has_long_gap = False
            for start, length in gap_starts:
                end = start + length
                if length <= self.short_gap_max:
                    if start > 0 and end < n and not np.isnan(values[start - 1]) and not np.isnan(values[end]):
                        left = values[start - 1]
                        right = values[end]
                        step = (right - left) / (length + 1)
                        for k in range(length):
                            values[start + k] = left + step * (k + 1)
                elif length <= self.medium_gap_max:
                    filled = self._kalman_fill(values, start, end)
                    if filled is not None:
                        values[start:end] = filled
                    elif start > 0 and end < n and not np.isnan(values[start - 1]) and not np.isnan(values[end]):
                        left = values[start - 1]
                        right = values[end]
                        step = (right - left) / (length + 1)
                        for k in range(length):
                            values[start + k] = left + step * (k + 1)
                else:
                    has_long_gap = True

            df[col] = values

            if has_long_gap:
                gap_flags[col] = np.isnan(values)

        for col, nan_mask in gap_flags.items():
            df[f"has_gap_{col}"] = nan_mask.astype("int8")

        return df

    @staticmethod
    def _kalman_fill(values: np.ndarray, start: int, end: int) -> np.ndarray | None:
        """Attempt Kalman smoothing on a gap segment.

        Fits a local-level model on pre-gap observations, forecasts into
        the gap, and blends toward the post-gap anchor for continuity.
        """
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            return None

        pre = values[max(0, start - 5):start]
        post = values[end:min(len(values), end + 5)]
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]

        if len(pre) < 2:
            return None

        try:
            gap_len = end - start
            model = UnobservedComponents(
                pre, level='local level', irregular=True,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=gap_len)

            # Blend forecast toward post-gap anchor for smooth continuity
            if len(post) > 0:
                anchor = float(post[0])
                blend = np.linspace(0.0, 0.5, gap_len)
                return forecast * (1.0 - blend) + anchor * blend
            return forecast
        except Exception:
            return None
