"""Gap-classified missing value imputation (causal methods only).

Forecasting for time series may only use causal imputation — the value at
time ``t`` must never depend on observations after ``t``.  Every fill here
is causal:

* Short gaps (<= short_gap_max): forward-fill from the last observed value.
* Medium gaps (<= medium_gap_max): Kalman filter forecast (local-level model
  fit on pre-gap observations, then forecast into the gap) with a
  forward-fill fallback.
* Long gaps (> medium_gap_max): NaN preserved + ``has_gap_{col}`` flag.

Banned: linear interpolation, backfill, bidirectional smoothing,
Kalman smoother — they all pull future observations into imputed rows.
"""

import logging

import numpy as np

from stoke_ml.preprocessing.base import PreprocessingStep
from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)


class MissingImputer(PreprocessingStep):
    """Impute missing values by gap length using causal methods only.

    Never uses ZI (zero-imputation) -- that's the core improvement
    over the legacy approach.  Also never uses linear interpolation or
    any bidirectional fill -- those leak future observations into
    historical features.
    """

    def __init__(
        self,
        short_gap_max: int = 2,
        medium_gap_max: int = 10,
    ):
        self.short_gap_max = short_gap_max
        self.medium_gap_max = medium_gap_max

    def fit(self, df, **kwargs):
        return super().fit(df, **kwargs)

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
                # Only the last observed value before the gap may be used —
                # the post-gap value is in the future at imputation time.
                last_obs = values[start - 1] if start > 0 else np.nan
                if length <= self.short_gap_max:
                    # Forward-fill: causal, uses no future information.
                    if start > 0 and not np.isnan(last_obs):
                        values[start:end] = last_obs
                elif length <= self.medium_gap_max:
                    filled = self._kalman_fill(values, start, end)
                    if filled is not None:
                        values[start:end] = filled
                    elif start > 0 and not np.isnan(last_obs):
                        # Causal fallback (never the old linear blend).
                        values[start:end] = last_obs
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
        the gap causally — no post-gap data is used (no look-ahead leakage).
        """
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            return None

        pre = values[max(0, start - 5):start]
        pre = pre[~np.isnan(pre)]

        if len(pre) < 2:
            return None

        try:
            gap_len = end - start
            model = UnobservedComponents(
                pre, level='local level', irregular=True,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=gap_len)

            # Causal forecast only — do NOT blend toward the post-gap
            # anchor: that would leak future information into imputed rows.
            return forecast
        except Exception as exc:
            logger.warning(
                "Kalman gap fill failed (category=%s), falling back to "
                "causal ffill", classify_error(exc).value,
            )
            return None
