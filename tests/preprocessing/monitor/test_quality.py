"""Tests for QualityMonitor -- missing/duplicate/constant/inf checks."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.monitor.quality import QualityMonitor


class TestQualityMonitor:
    def test_detects_high_missing_rate(self):
        qm = QualityMonitor(missing_warn_threshold=0.2)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        qm.fit_transform(df)
        alerts = [r for r in qm.report if r["level"] in ("WARN", "ERROR")]
        has_missing = any("missing" in r.get("message", "").lower() for r in alerts)
        assert has_missing

    def test_detects_inf_values(self):
        qm = QualityMonitor()
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
        qm.fit_transform(df)
        assert qm.has_errors

    def test_detects_near_constant_columns(self):
        qm = QualityMonitor(constant_warn_threshold=0.95)
        # 49 ones and one 2 -> 2/50 = 4% unique, below 5% threshold (1-0.95)
        df = pd.DataFrame({
            "x": [1.0] * 49 + [2.0],
        })
        qm.fit_transform(df)
        const_alerts = [r for r in qm.report if "constant" in r.get("message", "")]
        assert len(const_alerts) >= 1

    def test_detects_duplicate_rows(self):
        qm = QualityMonitor()
        df = pd.DataFrame({"x": [1.0, 2.0, 1.0], "y": [10.0, 20.0, 10.0]})
        qm.fit_transform(df)
        assert any("duplicate" in r.get("message", "") for r in qm.report)

    def test_empty_df_warns(self):
        qm = QualityMonitor()
        qm.fit_transform(pd.DataFrame({"x": pd.Series([], dtype=float)}))
        assert any("Empty" in r["message"] for r in qm.report)

    def test_clean_data_no_alerts(self):
        qm = QualityMonitor()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        qm.fit_transform(df)
        assert len(qm.report) == 0

    def test_error_threshold_triggers_error_level(self):
        qm = QualityMonitor(missing_error_threshold=0.5)
        df = pd.DataFrame({
            "x": [1.0] + [np.nan] * 9,
        })
        qm.fit_transform(df)
        assert qm.has_errors
