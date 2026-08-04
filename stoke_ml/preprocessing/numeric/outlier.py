"""MAD-based outlier detection and winsorization (PIT-safe).

Uses Median Absolute Deviation (robust to skewed financial data).
Limit-up/down moves (+-9.5% daily change) are real signals, not outliers.

Bounds are computed per row from a *trailing* window (expanding until
``window_days`` is reached), so each clip uses only data known up to that day.
The previous implementation estimated per-column median/MAD on the full sample
and clipped the entire history with those future-informed bounds — a look-ahead
leak whenever the step ran over full history.
"""

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep


class OutlierDetector(PreprocessingStep):
    """Detect and clip outliers via trailing-window MAD.

    |x - median| > threshold * MAD -> clip to [median +- threshold * MAD].
    Default threshold=5.0 is conservative (only extreme outliers).
    """

    _LIMIT_COLS = frozenset({"pct_change", "is_limit_up", "is_limit_down",
                              "gap_up_pct", "gap_down_pct"})

    def __init__(self, threshold: float = 5.0, clip: bool = True,
                 window_days: int = 252, min_periods: int = 10):
        self.threshold = threshold
        self.clip = clip
        self.window_days = window_days
        self.min_periods = min_periods
        self._bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df, **kwargs):
        """Stateless — bounds are computed point-in-time inside transform()."""
        return super().fit(df, **kwargs)

    def transform(self, df, **kwargs):
        if df.empty:
            return df.copy()
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self._LIMIT_COLS or not self.clip:
                continue
            values = df[col].to_numpy(dtype=np.float64)
            if len(values) < self.min_periods:
                continue
            df[col] = df[col].astype(np.float64)
            lower, upper = self._trailing_bounds(values)
            clipped = np.clip(values, lower, upper)
            # Keep rows with insufficient history or zero MAD (constant window):
            # clipping those to a point bound would destroy them.
            keep = np.isnan(lower) | np.isnan(upper) | ((upper - lower) < 1e-10)
            if keep.any():
                df.loc[~keep, col] = clipped[~keep]
            else:
                df[col] = clipped
        return df

    def _trailing_bounds(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row trailing-window median±threshold*MAD bounds (look-back only)."""
        n = len(values)
        w = self.window_days
        lower = np.full(n, np.nan, dtype=np.float64)
        upper = np.full(n, np.nan, dtype=np.float64)

        # Expanding fallback for early positions (before the first full window).
        for i in range(self.min_periods - 1, min(w - 1, n)):
            past = values[:i + 1]
            med = float(np.median(past))
            mad = float(np.median(np.abs(past - med)))
            if mad < 1e-10:
                continue
            lower[i] = med - self.threshold * mad
            upper[i] = med + self.threshold * mad

        if n >= w:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(values, w)
            med = np.median(win, axis=1)
            mad = np.median(np.abs(win - med[:, None]), axis=1)
            lower[w - 1:] = med - self.threshold * mad
            upper[w - 1:] = med + self.threshold * mad
        return lower, upper
