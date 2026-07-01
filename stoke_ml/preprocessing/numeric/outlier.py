"""MAD-based outlier detection and winsorization.

Uses Median Absolute Deviation (robust to skewed financial data).
Limit-up/down moves (+-9.5% daily change) are real signals, not outliers.
"""

import numpy as np
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
        for col, (lower, upper) in self._bounds.items():
            if col not in df.columns:
                continue
            if self.clip:
                if df[col].dtype.kind == "i":
                    df[col] = df[col].astype(np.float64)
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].clip(lower, upper)
        return df
