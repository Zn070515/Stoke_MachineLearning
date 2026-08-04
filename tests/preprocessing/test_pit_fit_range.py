"""Scaler/normalizer fit-range PIT audit.

Every PreprocessingStep records ``fit_start`` / ``fit_end`` — the date range
of the data it was last fit on — so a reviewer can audit that no step was fit
over full history instead of per-fold.  The stronger guarantee tested here is
truncation-invariance: the output for any row depends only on data known up to
that row, so appending future rows (a "2010-2026 fit" used for a 2015
training window) cannot change past outputs.  CrossSectionNormalizer is
per-date, so its per-date rows are identical whether fit on a full panel or on
a panel truncated at an earlier date.
"""

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep, PreprocessingChain
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.scaling import RobustScaler


class _Identity(PreprocessingStep):
    def transform(self, df, **kwargs):
        return df.copy()


class TestFitRangeRecording:
    def test_records_range_from_date_column(self):
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        step = _Identity()
        step.fit_transform(pd.DataFrame({"date": dates, "x": range(10)}))
        assert step.fit_start == dates[0]
        assert step.fit_end == dates[-1]

    def test_records_range_from_datetime_index(self):
        dates = pd.date_range("2020-06-01", periods=8, freq="B")
        df = pd.DataFrame({"x": range(8)}, index=dates)
        step = _Identity()
        step.fit(df)
        assert step.fit_start == dates[0]
        assert step.fit_end == dates[-1]

    def test_no_date_axis_leaves_fit_range_none(self):
        step = _Identity()
        step.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        assert step.fit_start is None
        assert step.fit_end is None

    def test_chain_records_overall_fit_range(self):
        chain = PreprocessingChain([_Identity()])
        dates = pd.date_range("2022-01-03", periods=5, freq="B")
        chain.fit(pd.DataFrame({"date": dates, "x": range(5)}))
        assert chain.fit_start == dates[0]
        assert chain.fit_end == dates[-1]

    def test_chain_fit_transform_records_overall_fit_range(self):
        """PreprocessingPipeline.run() uses fit_transform(), so the
        chain itself must record provenance on that path, not just fit()."""
        chain = PreprocessingChain([_Identity()])
        dates = pd.date_range("2022-01-03", periods=5, freq="B")
        chain.fit_transform(pd.DataFrame({"date": dates, "x": range(5)}))
        assert chain.fit_start == dates[0]
        assert chain.fit_end == dates[-1]

    def test_missing_imputer_records_fit_range(self):
        mi = MissingImputer()
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-04", periods=40, freq="B"),
            "x": list(range(40)),
        })
        df.loc[10, "x"] = np.nan
        mi.fit_transform(df)
        assert mi.fit_start == pd.Timestamp("2021-01-04")
        assert mi.fit_end == df["date"].max()


class TestScalerPitNoFutureLeak:
    """Appending future rows must not change the outputs for existing rows."""

    @staticmethod
    def _frame(n=200, seed=7, outlier_idx=None, outlier_val=30.0):
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        x = rng.randn(n).cumsum().astype(np.float32)
        if outlier_idx is not None:
            x[outlier_idx] += outlier_val
        return pd.DataFrame({"date": dates, "x": x})

    def test_robust_scaler_future_rows_do_not_change_past_output(self):
        df = self._frame(outlier_idx=50)
        rs = RobustScaler()
        out_full = rs.fit_transform(df)
        assert rs.fit_start == df["date"].min()
        assert rs.fit_end == df["date"].max()

        cut = 120
        out_trunc = RobustScaler().fit_transform(df.iloc[:cut])
        np.testing.assert_allclose(
            out_full["x"].values[:cut], out_trunc["x"].values,
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )

    def test_outlier_detector_future_rows_do_not_change_past_output(self):
        df = self._frame(seed=11, outlier_idx=60, outlier_val=40.0)
        od = OutlierDetector(threshold=3.0)
        out_full = od.fit_transform(df)
        assert od.fit_start == df["date"].min()
        assert od.fit_end == df["date"].max()

        cut = 120
        out_trunc = OutlierDetector(threshold=3.0).fit_transform(df.iloc[:cut])
        np.testing.assert_allclose(
            out_full["x"].values[:cut], out_trunc["x"].values,
            rtol=1e-6, atol=1e-6,
        )


class TestCrossSectionNormalizerPit:
    def test_records_full_panel_fit_range(self):
        panel = self._panel()
        csn = CrossSectionNormalizer(stages=["sector", "rank"])
        csn.fit_transform(panel)
        assert str(csn.fit_start) == "2024-01-02"
        assert str(csn.fit_end) == "2024-01-04"

    def test_per_date_rows_identical_when_fit_on_truncated_panel(self):
        panel = self._panel()
        out_full = CrossSectionNormalizer(stages=["sector", "rank"]).fit_transform(panel)

        trunc = panel[panel["date"] <= "2024-01-03"]
        out_trunc = CrossSectionNormalizer(stages=["sector", "rank"]).fit_transform(trunc)

        mask = out_full["date"] <= "2024-01-03"
        full_head = out_full[mask].sort_values(["date", "stock_code"]).reset_index(drop=True)
        trunc_sorted = out_trunc.sort_values(["date", "stock_code"]).reset_index(drop=True)
        np.testing.assert_allclose(
            full_head["x"].values, trunc_sorted["x"].values,
            rtol=1e-6, atol=1e-6,
        )

    @staticmethod
    def _panel():
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        return pd.DataFrame({
            "date": [dates[0]] * 6 + [dates[1]] * 6 + [dates[2]] * 6,
            "stock_code": ["A", "B", "C", "D", "E", "F"] * 3,
            "sector": ["bank", "bank", "tech", "tech", "health", "health"] * 3,
            "x": [float(i) for i in range(18)],
        })
