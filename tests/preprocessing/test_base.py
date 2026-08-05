"""Tests for PreprocessingStep base class and PreprocessingChain."""
import numpy as np
import pandas as pd
import pytest
from stoke_ml.preprocessing.base import (
    PreprocessingStep,
    PreprocessingChain,
    FIT_SCOPES,
)


class _AddOne(PreprocessingStep):
    def fit(self, df, **kwargs):
        return self
    def transform(self, df, **kwargs):
        df = df.copy()
        df["x"] = df["x"] + 1
        return df


class _ScaleByFit(PreprocessingStep):
    def fit(self, df, **kwargs):
        self.mean_ = df["x"].mean()
        return self
    def transform(self, df, **kwargs):
        df = df.copy()
        df["x"] = df["x"] - self.mean_
        return df


class _DropColumn(PreprocessingStep):
    def transform(self, df, **kwargs):
        return df.drop(columns=["y"], errors="ignore")


class _FilterMinLength(PreprocessingStep):
    def __init__(self, min_length=5):
        self.min_length = min_length
    def transform(self, df, **kwargs):
        return df[df["text"].str.len() >= self.min_length].copy()


class TestPreprocessingStep:
    def test_fit_returns_self_by_default(self):
        step = _AddOne()
        result = step.fit(pd.DataFrame({"x": [1]}))
        assert result is step

    def test_default_fit_scope_is_stateless_pit(self):
        """§十-1: steps default to the offline-safe scope; stateful steps must
        opt into fold_train_only / global_frozen explicitly."""
        assert _AddOne().fit_scope == "stateless_pit"
        assert FIT_SCOPES == ("stateless_pit", "fold_train_only", "global_frozen")

    def test_scope_partition_of_stateful_steps(self):
        """§十-1: the components named in the review are classified correctly."""
        from stoke_ml.preprocessing.numeric.scaling import RobustScaler
        from stoke_ml.preprocessing.categorical.encoder import ConceptBlockEncoder
        from stoke_ml.preprocessing.monitor.drift import DriftMonitor
        from stoke_ml.preprocessing.text.topics import TopicModeler
        assert RobustScaler().fit_scope == "fold_train_only"
        assert ConceptBlockEncoder().fit_scope == "fold_train_only"
        assert DriftMonitor().fit_scope == "fold_train_only"
        assert TopicModeler(enabled=False).fit_scope == "global_frozen"

    def test_transform_not_implemented(self):
        class _BadStep(PreprocessingStep):
            pass
        with pytest.raises(TypeError, match="abstract"):
            _BadStep()

    def test_fit_transform_calls_both(self):
        step = _ScaleByFit()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = step.fit_transform(df)
        np.testing.assert_array_almost_equal(result["x"].values, [-1.0, 0.0, 1.0])

    def test_repr_shows_init_params(self):
        step = _FilterMinLength(min_length=10)
        r = repr(step)
        assert "min_length=10" in r


class TestPreprocessingChain:
    def test_empty_chain_passthrough(self):
        chain = PreprocessingChain()
        df = pd.DataFrame({"x": [1, 2]})
        result = chain.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_chain_order(self):
        chain = PreprocessingChain([_AddOne(), _AddOne()])
        df = pd.DataFrame({"x": [0, 0]})
        result = chain.fit_transform(df)
        assert result["x"].tolist() == [2, 2]

    def test_fit_calls_all_steps(self):
        step1 = _ScaleByFit()
        chain = PreprocessingChain([step1])
        chain.fit(pd.DataFrame({"x": [1.0, 5.0]}))
        assert step1.mean_ == 3.0

    def test_transform_after_fit(self):
        chain = PreprocessingChain([_ScaleByFit()])
        train = pd.DataFrame({"x": [1.0, 5.0]})
        test = pd.DataFrame({"x": [3.0, 7.0]})
        chain.fit(train)
        result = chain.transform(test)
        np.testing.assert_array_almost_equal(result["x"].values, [0.0, 4.0])

    def test_step_drops_columns(self):
        chain = PreprocessingChain([_DropColumn()])
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        result = chain.fit_transform(df)
        assert "y" not in result.columns
        assert "x" in result.columns

    def test_fit_preserves_state_across_steps(self):
        class _Counter(PreprocessingStep):
            def fit(self, df, **kwargs):
                self.n_fit_rows_ = len(df)
                return self
            def transform(self, df, **kwargs):
                return df
        chain = PreprocessingChain([_Counter(), _AddOne(), _Counter()])
        chain.fit(pd.DataFrame({"x": [1, 2, 3]}))
        assert chain.steps[0].n_fit_rows_ == 3
        assert chain.steps[2].n_fit_rows_ == 3

    def test_pipeline_has_name(self):
        chain = PreprocessingChain([_AddOne()], name="test_pipe")
        assert chain.name == "test_pipe"

    def test_to_config_roundtrip(self):
        chain = PreprocessingChain([_FilterMinLength(min_length=10)], name="filter")
        cfg = chain.to_config()
        assert cfg["name"] == "filter"
        assert cfg["steps"][0]["type"] == "_FilterMinLength"
        assert cfg["steps"][0]["params"]["min_length"] == 10

    def test_to_config_records_fit_scope(self):
        """§十-1: the serialized step record carries fit_scope for auditability."""
        from stoke_ml.preprocessing.numeric.scaling import RobustScaler
        chain = PreprocessingChain([RobustScaler()], name="num")
        cfg = chain.to_config()
        assert cfg["steps"][0]["fit_scope"] == "fold_train_only"

    def test_offline_pit_chain_drops_fold_train_only(self):
        """§十-1: offline_pit_chain keeps only steps safe for full-history
        preprocessing; fold_fitted_chain keeps the fold_train_only complement."""
        from stoke_ml.preprocessing.numeric.scaling import RobustScaler
        from stoke_ml.preprocessing.numeric.missing import MissingImputer
        chain = PreprocessingChain([MissingImputer(), RobustScaler()], name="num_full")
        offline = chain.offline_pit_chain()
        assert [type(s).__name__ for s in offline.steps] == ["MissingImputer"]
        fitted = chain.fold_fitted_chain()
        assert [type(s).__name__ for s in fitted.steps] == ["RobustScaler"]
        # The two sub-chains partition the original (no step lost or doubled).
        assert len(offline.steps) + len(fitted.steps) == len(chain.steps)
        assert chain.fold_train_only_steps() == fitted.steps

    def test_global_frozen_kept_in_offline_pit_chain(self):
        """§十-1: a global_frozen step (fit once, pinned artifact) is allowed in
        the offline pass — only fold_train_only steps are excluded."""
        from stoke_ml.preprocessing.text.topics import TopicModeler
        from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
        chain = PreprocessingChain(
            [TopicModeler(enabled=False), BipolarClassifier()], name="topic"
        )
        offline = chain.offline_pit_chain()
        assert [type(s).__name__ for s in offline.steps] == [
            "TopicModeler", "BipolarClassifier",
        ]
        assert chain.fold_fitted_chain().steps == []

    def test_add_rejects_unknown_fit_scope(self):
        """§十-1: a typo in a future step's scope fails fast at assembly time."""
        class _BadScope(_AddOne):
            fit_scope = "leak_everything"
        chain = PreprocessingChain()
        with pytest.raises(ValueError, match="unknown fit_scope"):
            chain.add(_BadScope())

    def test_chain_with_kwargs_passthrough(self):
        class _ContextStep(PreprocessingStep):
            def transform(self, df, stock_code=None):
                df = df.copy()
                df["stock"] = stock_code or "unknown"
                return df
        chain = PreprocessingChain([_ContextStep()])
        result = chain.fit_transform(pd.DataFrame({"x": [1]}), stock_code="000001")
        assert result["stock"].iloc[0] == "000001"
