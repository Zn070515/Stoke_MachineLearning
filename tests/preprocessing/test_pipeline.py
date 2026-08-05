"""Integration tests for PreprocessingPipeline — end-to-end text+numeric chains."""
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import pytest
from stoke_ml.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingQualityError,
    PreprocessingScopeError,
    DriftMonitorError,
)
from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.monitor.quality import QualityMonitor


class TestPreprocessingPipeline:
    def test_registers_chains(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), TimeDecayWeighter()], name="text"
        ))
        assert "text" in pp.list_chains()

    def test_runs_text_chain(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), DailyAggregator()], name="text"
        ))
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "sentiment_title": [0.5, -0.3, 0.8],
        })
        result = pp.run("text", df.copy())
        assert "bipolar_sent" in result.columns
        assert "agreement" in result.columns

    def test_runs_numeric_chain(self):
        pp = PreprocessingPipeline()
        pp.register_chain("numeric", PreprocessingChain(
            [OutlierDetector(threshold=5.0), MissingImputer()], name="numeric"
        ))
        df = pd.DataFrame({
            "close": [10.0, 10.5, np.nan, 10.2, 10.8],
            "volume": [1e6, 1.1e6, 1e6, 1e6, 1e6],
        })
        result = pp.run("numeric", df.copy())
        # Missing value in close should be filled
        assert not result["close"].isna().all()

    def test_run_nonexistent_chain_raises(self):
        pp = PreprocessingPipeline()
        with pytest.raises(KeyError):
            pp.run("nonexistent", pd.DataFrame())

    def test_full_text_pipeline(self):
        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            BipolarClassifier(),
            TimeDecayWeighter(halflife_days=7),
            DailyAggregator(windows=(3, 5)),
        ], name="text_full")
        pp.register_chain("text", chain)

        n_posts = 100
        rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        df = pd.DataFrame({
            "aligned_date": rng.choice(dates, n_posts),
            "sentiment_title": rng.uniform(-1, 1, n_posts).astype(np.float32),
        })

        result = pp.run("text", df)
        # Should have daily aggregation with rolling windows
        assert "bipolar_sent" in result.columns
        assert "bipolar_sent_3d_mean" in result.columns
        assert "attention" in result.columns

    def test_run_with_stock_code_context(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), DailyAggregator()], name="text"
        ))
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, 0.8],
        })
        result = pp.run("text", df.copy(), stock_code="000001")
        assert "bipolar_sent" in result.columns

    def test_from_config_builds_pipeline(self):
        """Verify from_config builds pipeline with text+numeric chains."""
        config = {
            "text": {
                "bipolar": {"threshold_positive": 0.25, "threshold_negative": -0.25},
                "time_decay": {"halflife_days": 14},
                "aggregation": {"windows": [5, 10]},
            },
            "numeric": {
                "outlier": {"threshold": 4.0},
                "missing": {"short_gap_max": 3},
                "cross_section": {"enabled": True},
                "scaling": {"window_days": 120},
                "higher_order": {"enabled": True},
            },
        }
        pp = PreprocessingPipeline.from_config(config)
        assert "text" in pp.list_chains()
        assert "numeric" in pp.list_chains()


class TestFromConfigErrors:
    """A malformed preprocessing config must BLOCK, not silently
    degrade to an empty pipeline.  None is the legitimate 'no section' case."""

    def test_none_builds_empty_pipeline(self):
        pp = PreprocessingPipeline.from_config(None)
        assert pp.list_chains() == []

    def test_malformed_non_dict_config_raises(self):
        with pytest.raises(ValueError):
            PreprocessingPipeline.from_config("not a config")

    def test_list_chains_empty(self):
        pp = PreprocessingPipeline()
        assert pp.list_chains() == []

    def test_multiple_chains(self):
        pp = PreprocessingPipeline()
        pp.register_chain("news", PreprocessingChain(
            [BipolarClassifier()], name="news"
        ))
        pp.register_chain("guba", PreprocessingChain(
            [BipolarClassifier()], name="guba"
        ))
        assert set(pp.list_chains()) == {"guba", "news"}


class TestStrictMode:
    """Strict mode blocks error-level quality issues instead of
    degrading silently.  The raised exception keeps the staging output and
    full quality report for the caller to persist / audit."""

    def _pipeline_with_monitor(self, **qm_kwargs):
        pp = PreprocessingPipeline()
        pp.register_chain("numeric", PreprocessingChain([], name="numeric"))
        pp._quality_monitor = QualityMonitor(**qm_kwargs)
        return pp

    def test_strict_raises_on_error_level(self):
        pp = self._pipeline_with_monitor(missing_error_threshold=0.5)
        df = pd.DataFrame({"x": [1.0] + [np.inf] * 9})
        with pytest.raises(PreprocessingQualityError):
            pp.run("numeric", df, strict=True)

    def test_error_carries_report_and_staging_df(self):
        pp = self._pipeline_with_monitor(missing_error_threshold=0.5)
        df = pd.DataFrame({"x": [1.0] + [np.inf] * 9})
        try:
            pp.run("numeric", df, strict=True)
            raise AssertionError("expected PreprocessingQualityError")
        except PreprocessingQualityError as exc:
            assert exc.chain_name == "numeric"
            assert any(r["level"] == "ERROR" for r in exc.report)
            assert isinstance(exc.df, pd.DataFrame)
            assert len(exc.df) == len(df)

    def test_strict_clean_data_no_raise(self):
        pp = self._pipeline_with_monitor()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = pp.run("numeric", df, strict=True)
        assert len(result) == len(df)

    def test_non_strict_returns_despite_errors(self):
        """Legacy behavior preserved: without strict, errors log but don't block."""
        pp = self._pipeline_with_monitor(missing_error_threshold=0.5)
        df = pd.DataFrame({"x": [1.0] + [np.inf] * 9})
        result = pp.run("numeric", df, strict=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_strict_defaults_to_false(self):
        pp = self._pipeline_with_monitor(missing_error_threshold=0.5)
        df = pd.DataFrame({"x": [1.0] + [np.inf] * 9})
        # No strict kwarg → behaves like strict=False (returns, does not raise).
        result = pp.run("numeric", df)
        assert isinstance(result, pd.DataFrame)

    def test_no_monitor_strict_is_noop(self):
        """Strict without a monitor attached cannot detect issues — returns."""
        pp = PreprocessingPipeline()
        pp.register_chain("numeric", PreprocessingChain([], name="numeric"))
        df = pd.DataFrame({"x": [1.0] + [np.inf] * 9})
        result = pp.run("numeric", df, strict=True)
        assert isinstance(result, pd.DataFrame)


class TestFormalScope:
    """§十-1: formal full-history preprocessing refuses fold_train_only steps.

    The offline pass must never fit a fold_train_only step over the full
    window; the split chains let a caller re-route those steps per fold.
    """

    @pytest.fixture()
    def concept_pipeline(self):
        from stoke_ml.preprocessing.categorical.encoder import ConceptBlockEncoder
        pp = PreprocessingPipeline()
        pp.register_chain("concept", PreprocessingChain(
            [ConceptBlockEncoder(top_n=5, min_stocks_per_board=1)],
            name="concept",
        ))
        return pp

    @pytest.fixture()
    def pit_pipeline(self):
        pp = PreprocessingPipeline()
        pp.register_chain("flow", PreprocessingChain(
            [OutlierDetector(threshold=5.0), MissingImputer()], name="flow"
        ))
        return pp

    def test_formal_refuses_fold_train_only_chain(self, concept_pipeline):
        df = pd.DataFrame({
            "stock_code": ["000001", "000001"],
            "board_name": ["白酒", "人工智能"],
        })
        with pytest.raises(PreprocessingScopeError, match="ConceptBlockEncoder"):
            concept_pipeline.run("concept", df, formal=True)

    def test_formal_allows_stateless_pit_chain(self, pit_pipeline):
        df = pd.DataFrame({"close": [10.0, 10.5, np.nan]})
        result = pit_pipeline.run("flow", df, formal=True)
        assert isinstance(result, pd.DataFrame)

    def test_formal_default_false_preserves_legacy(self, concept_pipeline):
        """Without formal=True the legacy path still runs the chain — existing
        offline consumers keep working until they opt into the guard."""
        df = pd.DataFrame({
            "stock_code": ["000001"],
            "board_name": ["白酒"],
        })
        result = concept_pipeline.run("concept", df)
        assert "cb_0" in result.columns

    def test_scope_error_carries_split_chains(self, concept_pipeline):
        df = pd.DataFrame({"stock_code": ["000001"], "board_name": ["白酒"]})
        try:
            concept_pipeline.run("concept", df, formal=True)
            raise AssertionError("expected PreprocessingScopeError")
        except PreprocessingScopeError as exc:
            assert exc.chain_name == "concept"
            assert exc.step_types == ["ConceptBlockEncoder"]
            assert exc.offline_chain.steps == []
            assert [type(s).__name__ for s in exc.fold_chain.steps] == [
                "ConceptBlockEncoder"
            ]

    def test_from_config_concept_chain_refused_formal(self):
        """The config-assembled concept chain carries the fold_train_only scope
        and is refused by a formal full-history pass (§十-1)."""
        config = {"concept": {"enabled": True, "top_n": 5, "min_stocks_per_board": 1}}
        pp = PreprocessingPipeline.from_config(config)
        df = pd.DataFrame({
            "stock_code": ["000001"],
            "board_name": ["白酒"],
        })
        with pytest.raises(PreprocessingScopeError):
            pp.run("concept", df, formal=True)
        # Non-formal still runs (existing offline consumers unaffected).
        assert "cb_0" in pp.run("concept", df).columns

    def test_offline_pit_chains_map_excludes_fold_steps(self, concept_pipeline, pit_pipeline):
        offline = concept_pipeline.offline_pit_chains()
        assert offline["concept"].steps == []
        assert pit_pipeline.offline_pit_chains()["flow"].steps == \
            pit_pipeline.get_chain("flow").steps


class TestDriftLifecycle:
    """§十-3: DriftMonitor is executable as fit-on-train / compare-on-eval,
    not merely mounted.  A check with no baseline raises instead of silently
    pretending monitoring happened."""

    @pytest.fixture()
    def drift_pipeline(self):
        from stoke_ml.preprocessing.monitor.drift import DriftMonitor
        pp = PreprocessingPipeline()
        pp.register_chain("numeric", PreprocessingChain([], name="numeric"))
        pp._drift_monitor = DriftMonitor(sigma_threshold=1.0)
        return pp

    def test_no_monitor_raises_on_fit(self):
        pp = PreprocessingPipeline()
        with pytest.raises(RuntimeError, match="DriftMonitor"):
            pp.drift_fit(pd.DataFrame({"x": [1.0, 2.0]}))

    def test_check_without_fit_raises(self, drift_pipeline):
        with pytest.raises(RuntimeError, match="drift_fit"):
            drift_pipeline.drift_check(pd.DataFrame({"x": [1.0, 2.0]}))

    def test_fit_then_check_detects_drift(self, drift_pipeline):
        rng = np.random.RandomState(0)
        drift_pipeline.drift_fit(pd.DataFrame({"x": rng.normal(0, 1, 200)}))
        report = drift_pipeline.drift_check(
            pd.DataFrame({"x": rng.normal(10, 1, 200)})
        )
        assert any(r["status"] == "drifted" for r in report)

    def test_clean_eval_no_drift(self, drift_pipeline):
        rng = np.random.RandomState(1)
        drift_pipeline.drift_fit(pd.DataFrame({"x": rng.normal(0, 1, 200)}))
        report = drift_pipeline.drift_check(
            pd.DataFrame({"x": rng.normal(0.05, 1, 200)})
        )
        assert report == []

    def test_on_drift_raise(self, drift_pipeline):
        rng = np.random.RandomState(2)
        drift_pipeline.drift_fit(pd.DataFrame({"x": rng.normal(0, 1, 200)}))
        with pytest.raises(DriftMonitorError, match="drifted"):
            drift_pipeline.drift_check(
                pd.DataFrame({"x": rng.normal(10, 1, 200)}), on_drift="raise"
            )

    def test_bad_on_drift_value(self, drift_pipeline):
        drift_pipeline.drift_fit(pd.DataFrame({"x": [1.0, 2.0]}))
        with pytest.raises(ValueError, match="on_drift"):
            drift_pipeline.drift_check(
                pd.DataFrame({"x": [1.0, 2.0]}), on_drift="explode"
            )
