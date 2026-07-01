"""Feature drift monitor: compare current statistics against baseline."""

import numpy as np
from stoke_ml.preprocessing.base import PreprocessingStep


class DriftMonitor(PreprocessingStep):
    """Track feature distribution drift vs a stored baseline.

    On fit(): records baseline statistics (mean, std, median) for each
    numeric column. On transform(): computes current statistics and
    flags columns whose sigma_distance exceeds sigma_threshold.

    Sigma distance = |current_mean - baseline_mean| / baseline_std
    """

    def __init__(self, sigma_threshold: float = 3.0):
        self.sigma_threshold = sigma_threshold
        self.baseline_: dict[str, dict] = {}
        self._drift_report: list[dict] = []

    def fit(self, df, **kwargs):
        self.baseline_ = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna().values
            if len(values) < 10:
                continue
            self.baseline_[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }
        return self

    def transform(self, df, **kwargs):
        if not self.baseline_:
            return df.copy()
        df = df.copy()
        self._drift_report = []

        for col, bl in self.baseline_.items():
            if col not in df.columns:
                self._drift_report.append({
                    "feature": col,
                    "sigma_distance": float("nan"),
                    "status": "missing",
                })
                continue

            values = df[col].dropna().values
            if len(values) < 10:
                continue

            current_mean = float(np.mean(values))
            bl_std = bl["std"]
            if bl_std < 1e-10:
                continue

            sigma_dist = abs(current_mean - bl["mean"]) / bl_std
            if sigma_dist >= self.sigma_threshold:
                self._drift_report.append({
                    "feature": col,
                    "sigma_distance": round(sigma_dist, 3),
                    "baseline_mean": round(bl["mean"], 4),
                    "current_mean": round(current_mean, 4),
                    "status": "drifted",
                })

        return df

    @property
    def drift_report(self) -> list[dict]:
        return self._drift_report

    @property
    def n_drifted(self) -> int:
        return len(self._drift_report)
