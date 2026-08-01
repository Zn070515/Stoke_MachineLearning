"""MAD-based outlier detection and winsorization.

Uses Median Absolute Deviation (robust to skewed financial data).
Limit-up/down moves (+-9.5% daily change) are real signals, not outliers.
"""

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep


class OutlierDetector(PreprocessingStep):
    """Detect and clip outliers via MAD method.

    |x - median| > threshold * MAD -> clip to [median +- threshold * MAD].
    Default threshold=5.0 is conservative (only extreme outliers).
    """

    _LIMIT_COLS = frozenset({"pct_change", "is_limit_up", "is_limit_down",
                              "gap_up_pct", "gap_down_pct"})

    def __init__(self, threshold: float = 5.0, clip: bool = True):
        self.threshold = threshold
        self.clip = clip
        self._bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df, **kwargs):
        """Compute the per-column clip set (which columns get winsorized).

        NOTE: fit() uses FULL-SAMPLE statistics to decide the clip column-set,
        and skips columns whose full-sample MAD < 1e-10.  transform() is causal
        (trailing window), but the *include-set* decided here can leak the
        future: a column constant for most of history then volatile late would
        only be included because of that later data.  Call fit() on TRAINING
        windows only — never on a window that includes the evaluation period.
        """
        self._bounds = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self._LIMIT_COLS:
                continue
            values = df[col].dropna().values
            if len(values) < 10:
                continue
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad < 1e-10:
                continue
            lower = median - self.threshold * mad
            upper = median + self.threshold * mad
            self._bounds[col] = (lower, upper)
        return self

    def transform(self, df, **kwargs):
        if df.empty or not self._bounds:
            return df.copy()
        df = df.copy()
        for col in self._bounds:
            if col not in df.columns:
                continue
            if not self.clip:
                continue
            if df[col].dtype.kind == "i":
                df[col] = df[col].astype(np.float64)
            s = df[col].astype(np.float64)
            # Causal (trailing, up-to-current-point) MAD winsorization: the
            # clip bounds for row i use ONLY rows up to i, so no future data
            # leaks into the clip.  For series <= 252 rows this is a true
            # expanding window; for longer series it is a bounded 252-day
            # trailing window (efficient, still causal).  Rows with fewer than
            # 10 observations up to and including themselves are left as-is.
            w = min(len(s), 252)
            cnt = s.rolling(w, min_periods=1).count()
            med = s.rolling(w, min_periods=10).median()
            abs_dev = (s - med).abs()
            # min_periods=1 so MAD is available as soon as the median is
            # (the <10-observation guard is enforced via *valid* below).
            mad = abs_dev.rolling(w, min_periods=1).median()
            lower = med - self.threshold * mad
            upper = med + self.threshold * mad
            valid = (cnt >= 10) & med.notna() & mad.notna()
            df[col] = s.mask(valid, s.clip(lower=lower, upper=upper))
        return df
