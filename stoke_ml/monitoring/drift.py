"""Data drift detection via KS test + Population Stability Index.

Compares two time windows of feature distributions and flags
columns with significant drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class DriftMonitor:
    """Compare feature distributions between reference and current windows.

    KS test flags distributional shift; PSI quantifies magnitude.
    Thresholds follow industry convention: PSI < 0.1 OK, < 0.25 moderate, ≥ 0.25 high.
    """

    def __init__(self, psi_bins: int = 10):
        self.psi_bins = psi_bins

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run KS + PSI on each numeric column.

        Returns DataFrame: column, ks_stat, ks_pvalue, psi, drift_flag.
        """
        if columns is None:
            columns = reference.select_dtypes(include=[np.number]).columns
        columns = [c for c in columns if c in current.columns]

        rows = []
        for col in columns:
            ref = reference[col].dropna().values
            cur = current[col].dropna().values

            if len(ref) < 30 or len(cur) < 30:
                continue

            ks_stat, ks_p = stats.ks_2samp(ref, cur)
            psi = self._compute_psi(ref, cur)
            flag = "high" if psi >= 0.25 else "moderate" if psi >= 0.1 else "ok"

            rows.append({
                "column": col,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 4),
                "psi": round(float(psi), 4),
                "drift_flag": flag,
            })

        return pd.DataFrame(rows).sort_values("psi", ascending=False)

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index with automatic bin edges."""
        combined = np.concatenate([reference, current])
        bins = np.percentile(combined, np.linspace(0, 100, self.psi_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        eps = 1e-10

        psi = 0.0
        for r, c in zip(ref_hist, cur_hist):
            r_c = max(r, eps)
            c_c = max(c, eps)
            psi += (c_c - r_c) * np.log(c_c / r_c)
        return float(psi)
