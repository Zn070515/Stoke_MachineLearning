"""Tests for DriftMonitor -- feature distribution drift detection."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.monitor.drift import DriftMonitor


class TestDriftMonitor:
    def test_records_baseline_on_fit(self):
        dm = DriftMonitor()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 3})
        dm.fit(df)
        assert "x" in dm.baseline_
        assert "mean" in dm.baseline_["x"]

    def test_detects_drifted_feature(self):
        dm = DriftMonitor(sigma_threshold=1.0)
        train = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        test = pd.DataFrame({"x": np.random.normal(5, 1, 100)})
        dm.fit(train)
        dm.transform(test)
        assert dm.n_drifted >= 1
        assert dm.drift_report[0]["feature"] == "x"

    def test_no_drift_when_similar(self):
        dm = DriftMonitor(sigma_threshold=5.0)
        rng = np.random.RandomState(42)
        train = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        test = pd.DataFrame({"x": rng.normal(0.1, 1, 100)})
        dm.fit(train)
        dm.transform(test)
        assert dm.n_drifted == 0

    def test_missing_baseline_skips(self):
        dm = DriftMonitor()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        dm.transform(df)  # no fit → no baseline
        assert dm.n_drifted == 0

    def test_column_missing_in_new_data(self):
        dm = DriftMonitor()
        train = pd.DataFrame({"x": np.random.normal(0, 1, 100), "y": np.random.normal(0, 1, 100)})
        test = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        dm.fit(train)
        dm.transform(test)
        missing = [r for r in dm.drift_report if r["status"] == "missing"]
        assert len(missing) == 1
        assert missing[0]["feature"] == "y"

    def test_empty_df(self):
        dm = DriftMonitor()
        result = dm.fit_transform(pd.DataFrame())
        assert len(result) == 0
