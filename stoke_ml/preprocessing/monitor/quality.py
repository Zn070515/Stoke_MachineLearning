"""Data quality monitor: missing, constant, infinite, duplicate checks."""

import numpy as np
from stoke_ml.preprocessing.base import PreprocessingStep


class QualityMonitor(PreprocessingStep):
    """Check data quality and emit alerts.

    Runs on transform() and returns a report. Does NOT modify the DataFrame.
    """

    def __init__(
        self,
        missing_warn_threshold: float = 0.2,
        missing_error_threshold: float = 0.5,
        constant_warn_threshold: float = 0.99,
    ):
        self.missing_warn_threshold = missing_warn_threshold
        self.missing_error_threshold = missing_error_threshold
        self.constant_warn_threshold = constant_warn_threshold
        self._report: list[dict] = []

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        self._report = []
        df = df.copy()

        n_rows = len(df)
        if n_rows == 0:
            self._report.append({
                "level": "WARN", "message": "Empty DataFrame passed to QualityMonitor",
            })
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = df[col].values
            missing_rate = np.isnan(values).mean()

            if missing_rate >= self.missing_error_threshold:
                self._report.append({
                    "level": "ERROR",
                    "column": col,
                    "message": f"Missing rate {missing_rate:.2%} >= {self.missing_error_threshold:.0%}",
                    "missing_rate": float(missing_rate),
                })
            elif missing_rate >= self.missing_warn_threshold:
                self._report.append({
                    "level": "WARN",
                    "column": col,
                    "message": f"Missing rate {missing_rate:.2%} >= {self.missing_warn_threshold:.0%}",
                    "missing_rate": float(missing_rate),
                })

            # Check for infinite values
            finites = np.isfinite(values)
            if not finites.all():
                n_inf = (~finites).sum()
                self._report.append({
                    "level": "ERROR",
                    "column": col,
                    "message": f"Found {n_inf} infinite values",
                    "infinite_count": int(n_inf),
                })

            # Check for near-constant columns
            valid = values[~np.isnan(values)]
            if len(valid) > 1:
                unique_ratio = len(np.unique(valid)) / len(valid)
                if unique_ratio < (1.0 - self.constant_warn_threshold):
                    self._report.append({
                        "level": "WARN",
                        "column": col,
                        "message": f"Near-constant: {unique_ratio:.1%} unique",
                        "unique_ratio": float(unique_ratio),
                    })

        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            self._report.append({
                "level": "WARN",
                "message": f"Found {dup_count} duplicate rows ({dup_count/n_rows:.2%})",
                "duplicate_count": int(dup_count),
            })

        return df

    @property
    def report(self) -> list[dict]:
        return self._report

    @property
    def has_errors(self) -> bool:
        return any(r["level"] == "ERROR" for r in self._report)

    @property
    def has_warnings(self) -> bool:
        return any(r["level"] == "WARN" for r in self._report)
