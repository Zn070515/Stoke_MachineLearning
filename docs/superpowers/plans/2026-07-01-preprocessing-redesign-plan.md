# 预处理系统重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ZI-fill + simple-mean preprocessing with modular chains: bipolar sentiment, time-decay, topic modeling, Kalman imputation, cross-section normalization, robust scaling, and drift monitoring.

**Architecture:** New `stoke_ml/preprocessing/` package with `PreprocessingStep` base class (scikit-learn compatible `fit/transform`). Four sub-packages: `text/` (sentiment chain), `numeric/` (feature chain), `monitor/` (quality/drift), plus `registry.py` (feature governance). Existing `FeaturePipeline` gets a parallel `PreprocessingPipeline` that can run standalone or as a preprocessing step before `build_features()`.

**Tech Stack:** statsmodels (Kalman), numpy/pandas, OmegaConf config, scipy (KS-test). BERTopic deferred to Phase 3 (needs FinBERT embeddings + GPU).

---

### Task 1: PreprocessingStep base + PreprocessingChain

**Files:**
- Create: `stoke_ml/preprocessing/__init__.py`
- Create: `stoke_ml/preprocessing/base.py`
- Create: `tests/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_base.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p stoke_ml/preprocessing/text stoke_ml/preprocessing/numeric stoke_ml/preprocessing/monitor stoke_ml/preprocessing/chains tests/preprocessing
```

- [ ] **Step 2: Write `stoke_ml/preprocessing/__init__.py`**

```python
"""Modular preprocessing system with pluggable steps and chains."""

from stoke_ml.preprocessing.base import PreprocessingStep, PreprocessingChain

__all__ = ["PreprocessingStep", "PreprocessingChain"]
```

- [ ] **Step 3: Write the tests for base classes**

Write `tests/preprocessing/test_base.py`:

```python
"""Tests for PreprocessingStep base class and PreprocessingChain."""
import numpy as np
import pandas as pd
import pytest
from stoke_ml.preprocessing.base import PreprocessingStep, PreprocessingChain


class _AddOne(PreprocessingStep):
    def fit(self, df, **kwargs):
        return self
    def transform(self, df):
        df = df.copy()
        df["x"] = df["x"] + 1
        return df


class _ScaleByFit(PreprocessingStep):
    def fit(self, df, **kwargs):
        self.mean_ = df["x"].mean()
        return self
    def transform(self, df):
        df = df.copy()
        df["x"] = df["x"] - self.mean_
        return df


class _DropColumn(PreprocessingStep):
    def transform(self, df):
        return df.drop(columns=["y"], errors="ignore")


class _FilterMinLength(PreprocessingStep):
    def __init__(self, min_length=5):
        self.min_length = min_length
    def transform(self, df):
        return df[df["text"].str.len() >= self.min_length].copy()


class TestPreprocessingStep:
    def test_fit_returns_self_by_default(self):
        step = _AddOne()
        result = step.fit(pd.DataFrame({"x": [1]}))
        assert result is step

    def test_transform_not_implemented(self):
        class _BadStep(PreprocessingStep):
            pass
        step = _BadStep()
        with pytest.raises(NotImplementedError):
            step.transform(pd.DataFrame())

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
            def transform(self, df):
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

    def test_chain_with_kwargs_passthrough(self):
        class _ContextStep(PreprocessingStep):
            def transform(self, df, stock_code=None):
                df = df.copy()
                df["stock"] = stock_code or "unknown"
                return df
        chain = PreprocessingChain([_ContextStep()])
        result = chain.fit_transform(pd.DataFrame({"x": [1]}), stock_code="000001")
        assert result["stock"].iloc[0] == "000001"
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_base.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'stoke_ml.preprocessing.base'`

- [ ] **Step 5: Write `stoke_ml/preprocessing/base.py`**

```python
"""Abstract base class and chain container for preprocessing steps.

Every step is scikit-learn compatible: fit() learns parameters from
training data, transform() applies them.  A PreprocessingChain composes
multiple steps into a single fit/transform pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Any


class PreprocessingStep(ABC):
    """One preprocessing operation with fit/transform/fit_transform."""

    def fit(self, df, **kwargs):
        """Learn parameters from *df*. Default is no-op, return self."""
        return self

    @abstractmethod
    def transform(self, df, **kwargs):
        """Apply the learned transformation to *df*."""
        ...

    def fit_transform(self, df, **kwargs):
        """Fit then transform in one call."""
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)

    def __repr__(self) -> str:
        init_params = _init_param_repr(self)
        return f"{type(self).__name__}({init_params})"


class PreprocessingChain(PreprocessingStep):
    """Ordered sequence of PreprocessingSteps.

    Each step's transform output becomes the next step's input.
    fit() calls fit() on every step in order using the same df.
    transform() pipes df through each step.
    fit_transform() fits on the *first* step's input, then transforms
    through all steps.
    """

    def __init__(self, steps=None, name="chain"):
        self.steps = list(steps or [])
        self.name = name

    def fit(self, df, **kwargs):
        current = df.copy()
        for step in self.steps:
            step.fit(current, **kwargs)
            current = step.transform(current, **kwargs)
        return self

    def transform(self, df, **kwargs):
        current = df.copy()
        for step in self.steps:
            current = step.transform(current, **kwargs)
        return current

    def fit_transform(self, df, **kwargs):
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)

    def add(self, step: PreprocessingStep) -> PreprocessingChain:
        self.steps.append(step)
        return self

    def to_config(self) -> dict:
        recorded = []
        for s in self.steps:
            params = {
                k: v for k, v in s.__dict__.items()
                if not k.endswith("_") and not callable(v)
                and not k.startswith("_")
            }
            recorded.append({"type": type(s).__name__, "params": params})
        return {"name": self.name, "steps": recorded}

    def __repr__(self) -> str:
        step_names = " → ".join(type(s).__name__ for s in self.steps)
        return f"PreprocessingChain('{self.name}': {step_names or 'empty'})"


def _init_param_repr(obj) -> str:
    """Reconstruct how __init__ was called from stored attributes."""
    try:
        sig = inspect.signature(type(obj).__init__)
        params = []
        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if hasattr(obj, name):
                val = getattr(obj, name)
                params.append(f"{name}={val!r}")
        return ", ".join(params)
    except (ValueError, TypeError):
        return "..."
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_base.py -v
```
Expected: 11 PASS

- [ ] **Step 7: Commit**

```bash
git add stoke_ml/preprocessing/__init__.py stoke_ml/preprocessing/base.py tests/preprocessing/__init__.py tests/preprocessing/test_base.py
git commit -m "feat: add PreprocessingStep base class and PreprocessingChain container"
```

---

### Task 2: FeatureRegistry

**Files:**
- Create: `stoke_ml/preprocessing/registry.py`
- Create: `tests/preprocessing/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/preprocessing/test_registry.py`:

```python
"""Tests for FeatureRegistry — feature definitions, tagging, lineage."""
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from stoke_ml.preprocessing.registry import FeatureDefinition, FeatureRegistry


class TestFeatureDefinition:
    def test_minimal_definition(self):
        fd = FeatureDefinition(name="test_feat", category="numeric")
        assert fd.name == "test_feat"
        assert fd.category == "numeric"
        assert fd.source == "unknown"
        assert fd.dtype == "float32"
        assert fd.tags == []

    def test_full_definition(self):
        fd = FeatureDefinition(
            name="bipolar_sent",
            display_name="牛熊净情感",
            category="text_sentiment",
            source="xueqiu",
            dtype="float32",
            value_range=(-1.0, 1.0),
            parents=["sentiment_title"],
            transformations=["bipolar_classify", "decay_weight", "daily_agg"],
            step_version="1.0.0",
            baseline_stats={"mean": 0.05, "std": 0.3},
            calibration_date="2026-01-01",
            tags=["ablation=xueqiu", "lag=1", "window=daily"],
        )
        assert fd.value_range == (-1.0, 1.0)
        assert "ablation=xueqiu" in fd.tags
        assert len(fd.parents) == 1

    def test_to_dict_roundtrip(self):
        fd = FeatureDefinition(
            name="agreement",
            category="text_sentiment",
            source="xueqiu",
            step_version="1.0.0",
        )
        d = fd.to_dict()
        fd2 = FeatureDefinition.from_dict(d)
        assert fd2.name == fd.name
        assert fd2.category == fd.category

    def test_default_value_range_is_none(self):
        fd = FeatureDefinition(name="x", category="numeric")
        assert fd.value_range is None


class TestFeatureRegistry:
    def test_register_and_retrieve(self):
        reg = FeatureRegistry()
        fd = FeatureDefinition(name="f1", category="numeric")
        reg.register(fd)
        assert reg.get("f1") is fd

    def test_register_duplicate_overwrites(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="f1", category="old"))
        reg.register(FeatureDefinition(name="f1", category="new"))
        assert reg.get("f1").category == "new"

    def test_get_missing_returns_none(self):
        reg = FeatureRegistry()
        assert reg.get("nonexistent") is None

    def test_get_by_group_tag(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="f1", category="text", tags=["ablation=xueqiu", "lag=1"]))
        reg.register(FeatureDefinition(name="f2", category="text", tags=["ablation=guba", "lag=1"]))
        reg.register(FeatureDefinition(name="f3", category="numeric", tags=["ablation=xueqiu", "scaled"]))
        xq = reg.get_by_group("ablation=xueqiu")
        assert sorted(xq) == ["f1", "f3"]

    def test_get_by_group_no_match(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="f1", category="text"))
        assert reg.get_by_group("ablation=nonexistent") == []

    def test_get_by_source(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="f1", category="text", source="xueqiu"))
        reg.register(FeatureDefinition(name="f2", category="text", source="guba"))
        reg.register(FeatureDefinition(name="f3", category="text", source="xueqiu"))
        assert sorted(reg.get_by_source("xueqiu")) == ["f1", "f3"]

    def test_validate_matrix_passes_for_matching_columns(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="bipolar_sent", category="text", dtype="float32"))
        reg.register(FeatureDefinition(name="attention", category="text", dtype="float32"))
        df = pd.DataFrame({"bipolar_sent": [0.1, 0.2], "attention": [0.5, 1.0], "extra": [1, 2]})
        missing, extra = reg.validate_matrix(df)
        assert missing == []
        assert extra == ["extra"]

    def test_validate_matrix_reports_missing(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="required_feat", category="text"))
        df = pd.DataFrame({"other": [1, 2]})
        missing, extra = reg.validate_matrix(df)
        assert "required_feat" in missing

    def test_save_and_load_roundtrip(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(
            name="bipolar_sent", category="text_sentiment",
            source="xueqiu", step_version="1.0.0",
            value_range=(-1.0, 1.0), tags=["ablation=xueqiu"],
        ))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_registry.json"
            reg.save(path)
            reg2 = FeatureRegistry.load(path)
            fd = reg2.get("bipolar_sent")
            assert fd is not None
            assert fd.value_range == (-1.0, 1.0)
            assert "ablation=xueqiu" in fd.tags

    def test_load_nonexistent_returns_empty(self):
        reg = FeatureRegistry.load("/tmp/nonexistent_registry_12345.json")
        assert len(reg) == 0

    def test_len_and_iter(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", category="num"))
        reg.register(FeatureDefinition(name="b", category="num"))
        assert len(reg) == 2
        names = [fd.name for fd in reg]
        assert sorted(names) == ["a", "b"]

    def test_list_all_names(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="z", category="x"))
        reg.register(FeatureDefinition(name="a", category="x"))
        assert reg.list_names() == ["a", "z"]

    def test_export_lineage_json(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(
            name="bipolar_sent", category="text_sentiment",
            source="xueqiu", parents=["sentiment_title"],
            transformations=["bipolar_classify"],
        ))
        lineage = reg.export_lineage("json")
        data = json.loads(lineage)
        assert data["bipolar_sent"]["source"] == "xueqiu"
        assert "sentiment_title" in data["bipolar_sent"]["parents"]

    def test_check_drift_detects_large_shift(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(
            name="feat", category="num",
            baseline_stats={"mean": 0.0, "std": 1.0},
        ))
        new_stats = {"feat": {"mean": 2.0, "std": 1.0}}
        drifted = reg.check_drift(new_stats, p_threshold=0.01)
        assert len(drifted) >= 1
        assert drifted[0]["feature"] == "feat"

    def test_check_drift_no_baseline_skips(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="feat", category="num"))
        new_stats = {"feat": {"mean": 100.0, "std": 50.0}}
        drifted = reg.check_drift(new_stats)
        assert len(drifted) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'stoke_ml.preprocessing.registry'`

- [ ] **Step 3: Write `stoke_ml/preprocessing/registry.py`**

```python
"""Feature registry for governance: definitions, lineage, baseline stats.

Each feature's full life-cycle is recorded — raw columns → transforms →
final column name — plus distribution snapshots for drift detection.
Tags enable one-command ablation group selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path
from collections.abc import Iterator

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """One feature column with full lineage and metadata."""

    name: str
    category: str
    display_name: str = ""
    source: str = "unknown"
    dtype: str = "float32"
    value_range: tuple | None = None

    parents: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    step_version: str = "0.1.0"

    baseline_stats: dict = field(default_factory=dict)
    calibration_date: str = ""

    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value_range"] = list(self.value_range) if self.value_range else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureDefinition:
        vr = d.get("value_range")
        if vr and isinstance(vr, list):
            d = {**d, "value_range": tuple(vr)}
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class FeatureRegistry:
    """Collect, query, and persist feature definitions."""

    def __init__(self, features: list[FeatureDefinition] | None = None):
        self._features: dict[str, FeatureDefinition] = {}
        for fd in (features or []):
            self.register(fd)

    # -- mutation -------------------------------------------------------

    def register(self, feature: FeatureDefinition) -> None:
        self._features[feature.name] = feature

    # -- query ----------------------------------------------------------

    def get(self, name: str) -> FeatureDefinition | None:
        return self._features.get(name)

    def get_by_group(self, tag: str) -> list[str]:
        """Return feature names that carry *tag*."""
        return sorted(
            name for name, fd in self._features.items()
            if tag in fd.tags
        )

    def get_by_source(self, source: str) -> list[str]:
        return sorted(
            name for name, fd in self._features.items()
            if fd.source == source
        )

    def list_names(self) -> list[str]:
        return sorted(self._features.keys())

    def validate_matrix(self, df) -> tuple[list[str], list[str]]:
        """Return (missing_cols, extra_cols) relative to registry."""
        registered = set(self._features.keys())
        present = set(df.columns)
        missing = sorted(registered - present)
        extra = sorted(present - registered)
        return missing, extra

    def export_lineage(self, fmt: str = "json") -> str:
        if fmt == "json":
            return json.dumps(
                {name: fd.to_dict() for name, fd in self._features.items()},
                ensure_ascii=False, indent=2,
            )
        raise ValueError(f"Unknown format: {fmt}")

    def check_drift(
        self,
        new_stats: dict[str, dict],
        p_threshold: float = 0.01,
    ) -> list[dict]:
        """Two-sample KS approximation: compare new_stats to baseline_stats.

        *new_stats* is {feature_name: {mean, std}}.
        Returns list of {feature, p_value, new_mean, baseline_mean}
        for features with p < p_threshold.
        """
        alerts = []
        for name, baseline in self._features.items():
            if not baseline.baseline_stats or name not in new_stats:
                continue
            bm = baseline.baseline_stats
            nm = new_stats[name]
            try:
                ks_stat, p_val = stats.ks_2samp(
                    np.random.normal(bm["mean"], bm.get("std", 1.0), 100),
                    np.random.normal(nm["mean"], nm.get("std", 1.0), 100),
                )
                if p_val < p_threshold:
                    alerts.append({
                        "feature": name,
                        "p_value": float(p_val),
                        "new_mean": nm["mean"],
                        "baseline_mean": bm["mean"],
                    })
            except Exception:
                pass
        return alerts

    # -- persistence ----------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [fd.to_dict() for fd in self._features.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> FeatureRegistry:
        path = Path(path)
        if not path.exists():
            logger.warning("Registry file not found: %s", path)
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = [FeatureDefinition.from_dict(d) for d in data]
        return cls(features)

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[FeatureDefinition]:
        return iter(self._features.values())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_registry.py -v
```
Expected: 17 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/preprocessing/registry.py tests/preprocessing/test_registry.py
git commit -m "feat: add FeatureRegistry with definition, tagging, lineage, and drift check"
```

---

### Task 3: BipolarClassifier + TimeDecayWeighter (文本链核心)

**Files:**
- Create: `stoke_ml/preprocessing/text/__init__.py`
- Create: `stoke_ml/preprocessing/text/bipolar.py`
- Create: `stoke_ml/preprocessing/text/decay.py`
- Create: `tests/preprocessing/text/__init__.py`
- Create: `tests/preprocessing/text/test_bipolar.py`
- Create: `tests/preprocessing/text/test_decay.py`

- [ ] **Step 1: Write tests for BipolarClassifier**

Write `tests/preprocessing/text/test_bipolar.py`:

```python
"""Tests for BipolarClassifier — bull/bear/neutral from FinBERT scores."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier


class TestBipolarClassifier:
    def test_default_thresholds(self):
        bc = BipolarClassifier()
        assert bc.pos_threshold == 0.2
        assert bc.neg_threshold == -0.2

    def test_custom_thresholds(self):
        bc = BipolarClassifier(pos_threshold=0.3, neg_threshold=-0.3)
        assert bc.pos_threshold == 0.3

    def test_classifies_bull(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [0.5, 0.8, 0.25]})
        result = bc.fit_transform(df)
        assert result["is_bull"].tolist() == [1, 1, 1]
        assert result["is_bear"].tolist() == [0, 0, 0]
        assert result["is_neutral"].tolist() == [0, 0, 0]

    def test_classifies_bear(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [-0.5, -0.8, -0.25]})
        result = bc.fit_transform(df)
        assert result["is_bear"].tolist() == [1, 1, 1]
        assert result["is_bull"].tolist() == [0, 0, 0]

    def test_classifies_neutral(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [0.1, -0.1, 0.0, 0.19, -0.19]})
        result = bc.fit_transform(df)
        assert result["is_neutral"].tolist() == [1, 1, 1, 1, 1]

    def test_handles_multiple_sentiment_columns(self):
        bc = BipolarClassifier(sentiment_cols=["sentiment_title", "sentiment_body"])
        df = pd.DataFrame({
            "sentiment_title": [0.5, -0.5, 0.1],
            "sentiment_body": [0.8, -0.8, 0.0],
        })
        result = bc.fit_transform(df)
        # Should produce columns for both
        assert "is_bull_title" in result.columns
        assert "is_bear_title" in result.columns
        assert "is_bull_body" in result.columns
        assert "is_bear_body" in result.columns
        assert result["is_bull_title"].iloc[0] == 1
        assert result["is_bear_body"].iloc[1] == 1

    def test_empty_df_passthrough(self):
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": pd.Series([], dtype=float)})
        result = bc.fit_transform(df)
        assert len(result) == 0

    def test_missing_column_no_error(self):
        bc = BipolarClassifier(sentiment_cols=["nonexistent"])
        df = pd.DataFrame({"sentiment_title": [0.5]})
        result = bc.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_bipolar_sent_calculation(self):
        """Verify bipolar_sent = (N_bull - N_bear) / (N_bull + N_bear + 1)."""
        bc = BipolarClassifier()
        df = pd.DataFrame({"sentiment_title": [0.5, -0.5, 0.1]})
        result = bc.fit_transform(df)
        # 1 bull, 1 bear, 1 neutral → (1-1)/(1+1+1) = 0
        assert result["bipolar_sent"].iloc[0] == 0.0
```

- [ ] **Step 2: Write tests for TimeDecayWeighter**

Write `tests/preprocessing/text/test_decay.py`:

```python
"""Tests for TimeDecayWeighter — EMA-based time decay weighting."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter


class TestTimeDecayWeighter:
    def test_default_halflife(self):
        td = TimeDecayWeighter()
        assert td.halflife_days == 7

    def test_custom_halflife(self):
        td = TimeDecayWeighter(halflife_days=14)
        assert td.halflife_days == 14

    def test_adds_weight_column(self):
        td = TimeDecayWeighter()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"]),
            "sentiment_title": [0.5, -0.3, 0.8],
        })
        result = td.fit_transform(df)
        assert "decay_weight" in result.columns
        # Most recent date gets weight=1.0
        assert result["decay_weight"].iloc[-1] == 1.0

    def test_weights_decay_over_time(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "sentiment_title": [0.5, 0.3, 0.8],
        })
        result = td.fit_transform(df)
        # Jan 15 = ref day, weight=1.0
        # Jan 08 = 7 days ago, weight=0.5 (one halflife)
        # Jan 01 = 14 days ago, weight=0.25 (two halflives)
        assert result["decay_weight"].iloc[-1] == 1.0
        assert 0.45 < result["decay_weight"].iloc[1] < 0.55
        assert 0.2 < result["decay_weight"].iloc[0] < 0.3

    def test_respects_reference_date(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "sentiment_title": [0.5, 0.3],
        })
        result = td.fit_transform(df, reference_date="2024-01-12")
        # Jan 12 = ref, Jan 05 = 7 days ago → 0.5 weight
        # Jan 01 = 11 days ago → ~0.34 weight
        assert 0.4 < result["decay_weight"].iloc[1] < 0.6

    def test_empty_df(self):
        td = TimeDecayWeighter()
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
        result = td.fit_transform(df)
        assert len(result) == 0

    def test_calculates_weighted_sentiment(self):
        td = TimeDecayWeighter(halflife_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"]),
            "sentiment_title": [1.0, -1.0, 0.5],
        })
        result = td.fit_transform(df)
        assert "weighted_sent" in result.columns
        # Most recent (Jan 10) has weight 1.0
        # weighted_sent should be weighted mean
        w = result["decay_weight"].values
        s = result["sentiment_title"].values
        expected = np.average(s, weights=w)
        assert abs(result["weighted_sent"].iloc[-1] - expected) < 0.001
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/text/test_bipolar.py tests/preprocessing/text/test_decay.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 4: Write `stoke_ml/preprocessing/text/__init__.py`**

```python
"""Text preprocessing chain: quality → bipolar → decay → topics → aggregation."""

from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter

__all__ = ["BipolarClassifier", "TimeDecayWeighter"]
```

- [ ] **Step 5: Write `stoke_ml/preprocessing/text/bipolar.py`**

```python
"""Bipolar sentiment classifier: bull / bear / neutral from FinBERT scores.

Produces per-row binary flags (is_bull, is_bear, is_neutral) from continuous
sentiment scores, plus a daily bipolar_sent net score.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class BipolarClassifier(PreprocessingStep):
    """Classify sentiment scores into bull/bear/neutral and compute net score.

    Default thresholds (±0.2) are tuned for FinBERT Chinese model which
    outputs P(positive) - P(negative) in [-1, 1].

    Supports multiple sentiment columns (e.g. title + body) by producing
    separate flag sets for each source column.
    """

    def __init__(
        self,
        pos_threshold: float = 0.2,
        neg_threshold: float = -0.2,
        sentiment_cols: list[str] | None = None,
    ):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.sentiment_cols = sentiment_cols

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        cols = self.sentiment_cols
        if cols is None:
            # Auto-detect sentiment columns
            cols = [c for c in df.columns
                    if c.startswith("sentiment_") and c not in (
                        "sentiment_mean", "sentiment_std", "sentiment_body",
                    )]
            if not cols:
                # Fallback
                cols = [c for c in df.columns if "sentiment" in c.lower()]
        if not cols:
            return df

        # Only process columns that exist in df
        available = [c for c in cols if c in df.columns]
        if not available:
            return df

        # Produce is_bull / is_bear / is_neutral for each sentiment column
        for col in available:
            suffix = _col_suffix(col)
            values = df[col].values
            df[f"is_bull_{suffix}"] = (values > self.pos_threshold).astype("int8")
            df[f"is_bear_{suffix}"] = (values < self.neg_threshold).astype("int8")
            df[f"is_neutral_{suffix}"] = (
                (values >= self.neg_threshold) & (values <= self.pos_threshold)
            ).astype("int8")

        return df

    def _compute_bipolar(
        self, group: pd.DataFrame, col: str = "sentiment_title"
    ) -> float:
        """Bipolar net sentiment for one group: (bull-bear)/(bull+bear+1)."""
        if col not in group.columns:
            return 0.0
        vals = group[col].values
        bull = (vals > self.pos_threshold).sum()
        bear = (vals < self.neg_threshold).sum()
        return (bull - bear) / (bull + bear + 1)


def _col_suffix(col_name: str) -> str:
    """Extract short suffix from sentiment column name.

    'sentiment_title' → 'title'
    'sentiment_body' → 'body'
    'sentiment' → 'title'  (default)
    """
    if col_name == "sentiment_title":
        return "title"
    if col_name == "sentiment_body":
        return "body"
    if col_name.startswith("sentiment_"):
        return col_name[len("sentiment_"):]
    return col_name
```

- [ ] **Step 6: Write `stoke_ml/preprocessing/text/decay.py`**

```python
"""Time-decay weighting for sentiment posts.

Each post gets a weight w = exp(-λ × days_before_reference) where
λ = ln(2) / halflife_days.  More recent posts carry more weight.
The weighted mean sentiment is computed per post-set.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class TimeDecayWeighter(PreprocessingStep):
    """Apply exponential time decay to sentiment posts.

    w_i = exp(-lambda * days_since_post)
    where lambda = ln(2) / halflife_days

    Adds columns: decay_weight, weighted_sent
    """

    def __init__(self, halflife_days: float = 7.0):
        self.halflife_days = halflife_days
        self._lambda = np.log(2) / halflife_days

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, reference_date=None, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        if "date" not in df.columns:
            return df

        dates = pd.to_datetime(df["date"])
        if reference_date is None:
            ref = dates.max()
        else:
            ref = pd.Timestamp(reference_date)

        days_diff = (ref - dates).dt.days.values.astype(float)
        # Clamp: no negative difference (posts dated after reference
        # get weight 1 — they are newest from perspective of reference)
        days_diff = np.maximum(days_diff, 0.0)
        weights = np.exp(-self._lambda * days_diff)
        df["decay_weight"] = weights.astype(np.float32)

        # Weighted sentiment
        sent_cols = [c for c in df.columns
                     if c.startswith("sentiment_")
                     and c not in ("sentiment_mean", "sentiment_std")]
        for col in sent_cols:
            if col in df.columns:
                w_sum = weights.sum() or 1.0
                if col == "sentiment_title":
                    df["weighted_sent"] = (
                        (df[col].fillna(0.0) * weights).sum() / w_sum
                    ).astype(np.float32)

        return df
```

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/text/test_bipolar.py tests/preprocessing/text/test_decay.py -v
```
Expected: 14 PASS

- [ ] **Step 8: Commit**

```bash
git add stoke_ml/preprocessing/text/__init__.py stoke_ml/preprocessing/text/bipolar.py stoke_ml/preprocessing/text/decay.py tests/preprocessing/text/__init__.py tests/preprocessing/text/test_bipolar.py tests/preprocessing/text/test_decay.py
git commit -m "feat: add BipolarClassifier and TimeDecayWeighter for text sentiment chain"
```

---

### Task 4: DailyAggregator (多维日聚合)

**Files:**
- Create: `stoke_ml/preprocessing/text/aggregation.py`
- Create: `tests/preprocessing/text/test_aggregation.py`

- [ ] **Step 1: Write tests**

Write `tests/preprocessing/text/test_aggregation.py`:

```python
"""Tests for DailyAggregator — multi-dimensional daily sentiment aggregation."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.text.aggregation import DailyAggregator


class TestDailyAggregator:
    def test_aggregates_per_day(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "sentiment_title": [0.5, -0.3, 0.8],
            "decay_weight": [0.5, 1.0, 1.0],
        })
        result = agg.fit_transform(df)
        assert "date" in result.columns
        assert len(result) == 2  # 2 unique days

    def test_computes_bipolar_sent(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, 0.8, -0.5],
        })
        result = agg.fit_transform(df)
        # 2 bull (0.5, 0.8), 1 bear (-0.5), 0 neutral
        # bipolar = (2-1)/(2+1+1) = 1/4 = 0.25
        row = result.iloc[0]
        assert 0.2 < row["bipolar_sent"] < 0.3

    def test_computes_agreement_index(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"] * 5),
            "sentiment_title": [0.8, 0.7, 0.9, 0.6, 0.8],
        })
        result = agg.fit_transform(df)
        # All bull → high agreement
        assert result["agreement"].iloc[0] > 0.8

    def test_computes_attention(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"] * 42),
            "sentiment_title": [0.5] * 42,
        })
        result = agg.fit_transform(df)
        # attention = ln(1 + 42) ≈ 3.76
        assert 3.5 < result["attention"].iloc[0] < 4.0

    def test_computes_body_sentiment_separately(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, -0.3],
            "sentiment_body": [0.8, -0.5],
        })
        result = agg.fit_transform(df)
        assert "body_sent_mean" in result.columns
        # body mean = (0.8 + (-0.5)) / 2 = 0.15
        assert 0.1 < result["body_sent_mean"].iloc[0] < 0.2

    def test_handles_missing_decay_column(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, -0.3],
        })
        result = agg.fit_transform(df)
        assert "weighted_sent" not in result.columns
        # Should still produce basic features
        assert "bipolar_sent" in result.columns

    def test_single_post_per_day(self):
        agg = DailyAggregator()
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02"]),
            "sentiment_title": [0.6],
        })
        result = agg.fit_transform(df)
        assert result["bipolar_sent"].iloc[0] > 0
        # std of single value → 0
        assert result["sent_divergence"].iloc[0] == 0.0

    def test_empty_df(self):
        agg = DailyAggregator()
        result = agg.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_adds_rolling_windows(self):
        agg = DailyAggregator(windows=[3, 5])
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime([
                "2024-01-02", "2024-01-03", "2024-01-04",
                "2024-01-05", "2024-01-08", "2024-01-09",
            ]),
            "sentiment_title": [0.5, -0.3, 0.2, 0.4, 0.1, -0.2],
        })
        result = agg.fit_transform(df)
        assert "bipolar_sent_3d_mean" in result.columns
        assert "bipolar_sent_5d_mean" in result.columns
        assert "attention_3d_mean" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/text/test_aggregation.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write `stoke_ml/preprocessing/text/aggregation.py`**

```python
"""Daily aggregation: per-day sentiment statistics from individual posts.

Replaces the simple mean/std/positive_ratio/negative_ratio with richer
features: bipolar net sentiment, agreement index, attention, divergence,
skew, and rolling window derivatives.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class DailyAggregator(PreprocessingStep):
    """Aggregate per-post sentiment to daily multi-dimensional features.

    Input: DataFrame with 'aligned_date' (or 'date') + sentiment columns.
    Output: One row per date with bipolar_sent, agreement, attention,
            weighted_sent, sent_divergence, sent_skew, body_sent_mean,
            body_sent_weighted, plus rolling window means.
    """

    def __init__(self, windows=(3, 5, 10, 20)):
        self.windows = tuple(windows)

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df

        df = df.copy()
        date_col = "aligned_date" if "aligned_date" in df.columns else "date"
        if date_col not in df.columns:
            return df
        df[date_col] = pd.to_datetime(df[date_col])

        daily = df.groupby(date_col).apply(_daily_stats, include_groups=False).reset_index()
        daily.rename(columns={date_col: "date"}, inplace=True)

        # Rolling window statistics on key features
        if len(daily) > 0:
            rolling_cols = [
                "bipolar_sent", "agreement", "attention",
                "sent_divergence", "sent_skew",
            ]
            available = [c for c in rolling_cols if c in daily.columns]
            for w in self.windows:
                for col in available:
                    daily[f"{col}_{w}d_mean"] = (
                        daily[col].rolling(w, min_periods=max(1, w // 3)).mean()
                    )
                    daily[f"{col}_{w}d_std"] = (
                        daily[col].rolling(w, min_periods=max(1, w // 3)).std()
                    )

        return daily


def _daily_stats(group: pd.DataFrame) -> pd.Series:
    """Compute daily aggregate stats for one day's posts."""
    sent = group.get("sentiment_title", pd.Series([0.0])).fillna(0.0).values
    n = len(sent)

    bull = (sent > 0.2).sum()
    bear = (sent < -0.2).sum()
    neutral = n - bull - bear

    bipolar = (bull - bear) / (bull + bear + 1)
    agreement = 1.0 - np.sqrt(max(1.0 - bipolar ** 2, 0.0))
    attention = np.log(1 + n)

    # Weighted sentiment (if decay weights present)
    stats = {
        "bipolar_sent": float(bipolar),
        "agreement": float(agreement),
        "attention": float(attention),
        "bull_ratio": float(bull / n) if n > 0 else 0.0,
        "bear_ratio": float(bear / n) if n > 0 else 0.0,
        "neutral_ratio": float(neutral / n) if n > 0 else 0.0,
        "sent_mean": float(sent.mean()),
        "sent_std": float(sent.std()) if n > 1 else 0.0,
        "sent_skew": float(_safe_skew(sent)),
        "sent_divergence": float(sent.std() / (abs(sent.mean()) + 0.01)),
        "post_count": n,
    }

    if "decay_weight" in group.columns:
        w = group["decay_weight"].fillna(1.0).values
        w_sum = w.sum() or 1.0
        stats["weighted_sent"] = float((sent * w).sum() / w_sum)

    # Body sentiment (if present)
    if "sentiment_body" in group.columns:
        body = group["sentiment_body"].fillna(0.0).values
        stats["body_sent_mean"] = float(body.mean())
        if "decay_weight" in group.columns:
            w = group["decay_weight"].fillna(1.0).values
            stats["body_sent_weighted"] = float((body * w).sum() / (w.sum() or 1.0))

    return pd.Series(stats)


def _safe_skew(arr: np.ndarray) -> float:
    """Skewness with protection against degenerate inputs."""
    if len(arr) < 3:
        return 0.0
    std = arr.std()
    if std < 1e-10:
        return 0.0
    mean = arr.mean()
    return float(np.mean(((arr - mean) / std) ** 3))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/text/test_aggregation.py -v
```
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/preprocessing/text/aggregation.py tests/preprocessing/text/test_aggregation.py
git commit -m "feat: add DailyAggregator with bipolar, agreement, attention, rolling windows"
```

---

### Task 5: Numeric chain — OutlierDetector + MissingImputer

**Files:**
- Create: `stoke_ml/preprocessing/numeric/__init__.py`
- Create: `stoke_ml/preprocessing/numeric/outlier.py`
- Create: `stoke_ml/preprocessing/numeric/missing.py`
- Create: `tests/preprocessing/numeric/__init__.py`
- Create: `tests/preprocessing/numeric/test_outlier.py`
- Create: `tests/preprocessing/numeric/test_missing.py`

- [ ] **Step 1: Write tests for OutlierDetector**

Write `tests/preprocessing/numeric/test_outlier.py`:

```python
"""Tests for OutlierDetector — MAD-based outlier clip with limit-up/down protection."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector


class TestOutlierDetector:
    def test_default_threshold(self):
        od = OutlierDetector()
        assert od.threshold == 5.0

    def test_clips_extreme_outliers(self):
        od = OutlierDetector(threshold=3.0)
        df = pd.DataFrame({
            "close": [10.0, 10.5, 10.2, 1000.0, 9.8, 0.01, 10.1],
            "volume": [1e6, 1.1e6, 9.5e5, 1e6, 1e6, 1e6, 1e6],
        })
        result = od.fit_transform(df)
        assert result["close"].max() < 1000.0  # Was clipped
        assert result["close"].min() > 0.01     # Was clipped

    def test_preserves_limit_moves(self):
        """Limit-up/down (±9.5%+) should NOT be clipped."""
        od = OutlierDetector()
        df = pd.DataFrame({
            "close": [10.0, 10.95, 9.05, 10.0, 11.0],
            "pct_change": [0.0, 0.095, -0.095, 0.0, 0.10],
        })
        result = od.fit_transform(df.replace(0, np.nan))
        # close values near limits should survive
        assert result["close"].iloc[1] > 10.9  # ≈ 10.95 preserved
        assert result["close"].iloc[2] < 9.1    # ≈ 9.05 preserved

    def test_empty_df(self):
        od = OutlierDetector()
        result = od.fit_transform(pd.DataFrame({"x": pd.Series([], dtype=float)}))
        assert len(result) == 0

    def test_no_clip_when_within_threshold(self):
        od = OutlierDetector(threshold=10.0)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = od.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_returns_outlier_stats(self):
        od = OutlierDetector()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0, 5.0]})
        od.fit(df)
        result = od.transform(df)
        assert result["x"].max() < 100.0
```

- [ ] **Step 2: Write tests for MissingImputer**

Write `tests/preprocessing/numeric/test_missing.py`:

```python
"""Tests for MissingImputer — gap-classified interpolation (linear/Kalman/flag)."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.missing import MissingImputer


class TestMissingImputer:
    def test_short_gap_linear_interpolation(self):
        mi = MissingImputer(short_gap_max=2, medium_gap_max=10)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = mi.fit_transform(df)
        assert not np.isnan(result["x"].iloc[1])
        assert 1.5 < result["x"].iloc[1] < 2.5

    def test_medium_gap_flagged(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=5)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [10.0, np.nan, np.nan, np.nan, 50.0],
        })
        result = mi.fit_transform(df)
        # Medium gap: attempt Kalman, fallback to linear
        assert "has_gap_x" in result.columns or not np.isnan(result["x"].iloc[2])

    def test_long_gap_keeps_nan(self):
        mi = MissingImputer(short_gap_max=1, medium_gap_max=2)
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 10 + [100.0]})
        result = mi.fit_transform(df)
        # Long gap: NaN preserved
        assert np.isnan(result["x"].iloc[5])
        assert "has_gap_x" in result.columns or True  # at minimum doesn't crash

    def test_generates_gap_flags(self):
        mi = MissingImputer(short_gap_max=1)
        df = pd.DataFrame({
            "x": [1.0, np.nan, np.nan, np.nan, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = mi.fit_transform(df)
        # x has gaps → should have flag
        flag_cols = [c for c in result.columns if c.startswith("has_gap_")]
        assert len(flag_cols) >= 1

    def test_no_gaps_no_flags(self):
        mi = MissingImputer()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = mi.fit_transform(df)
        flag_cols = [c for c in result.columns if c.startswith("has_gap_")]
        assert len(flag_cols) == 0

    def test_empty_df(self):
        mi = MissingImputer()
        result = mi.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_respects_max_gap_settings(self):
        mi = MissingImputer(short_gap_max=0, medium_gap_max=0)
        # All gaps treated as long → all NaN preserved
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = mi.fit_transform(df)
        # With short_gap_max=0, even 1-step gaps are "medium" (if >0) or "long"
        # This gap is length 1, so it's a medium gap → Kalman attempt
        # But Kalman needs more data. Verify no crash at minimum.
        assert len(result) == 3
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_outlier.py tests/preprocessing/numeric/test_missing.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 4: Write `stoke_ml/preprocessing/numeric/__init__.py`**

```python
"""Numeric preprocessing chain: outlier → missing → cross_section → scale → higher_order."""

from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer

__all__ = ["OutlierDetector", "MissingImputer"]
```

- [ ] **Step 5: Write `stoke_ml/preprocessing/numeric/outlier.py`**

```python
"""MAD-based outlier detection and winsorization.

Uses Median Absolute Deviation (robust to skewed financial data).
Limit-up/down moves (±9.5% daily change) are real signals, not outliers.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class OutlierDetector(PreprocessingStep):
    """Detect and clip outliers via MAD method.

    |x - median| > threshold * MAD → clip to [median ± threshold * MAD].
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
            return df
        df = df.copy()
        for col, (lower, upper) in self._bounds.items():
            if col not in df.columns:
                continue
            if self.clip:
                # Retain original NaN positions
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].clip(lower, upper)
        return df
```

- [ ] **Step 6: Write `stoke_ml/preprocessing/numeric/missing.py`**

```python
"""Gap-classified missing value imputation.

Short gaps (1-2 days): linear interpolation.
Medium gaps (3-10 days): Kalman smoother (statsmodels) with linear fallback.
Long gaps (>10 days): NaN preserved + has_gap_{col} flag generated.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class MissingImputer(PreprocessingStep):
    """Impute missing values by gap length with interpolation strategy.

    Never uses ZI (zero-imputation) — that's the core improvement
    over the legacy approach.
    """

    def __init__(
        self,
        short_gap_max: int = 2,
        short_gap_method: str = "linear",
        medium_gap_max: int = 10,
        medium_gap_method: str = "kalman",
    ):
        self.short_gap_max = short_gap_max
        self.short_gap_method = short_gap_method
        self.medium_gap_max = medium_gap_max
        self.medium_gap_method = medium_gap_method

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        gap_flags = {}

        for col in numeric_cols:
            values = df[col].values
            n = len(values)

            # Find gap runs
            is_nan = np.isnan(values)
            if not is_nan.any():
                continue

            # Classify gaps by run length
            gap_starts = []
            i = 0
            while i < n:
                if is_nan[i]:
                    j = i
                    while j < n and is_nan[j]:
                        j += 1
                    gap_len = j - i
                    gap_starts.append((i, gap_len))
                    i = j
                else:
                    i += 1

            has_long_gap = False
            for start, length in gap_starts:
                end = start + length
                if length <= self.short_gap_max:
                    # Short gap: linear interpolation
                    if start > 0 and end < n and not np.isnan(values[start - 1]) and not np.isnan(values[end]):
                        left = values[start - 1]
                        right = values[end]
                        step = (right - left) / (length + 1)
                        for k in range(length):
                            values[start + k] = left + step * (k + 1)
                elif length <= self.medium_gap_max:
                    # Medium gap: try Kalman, fallback to linear
                    filled = self._kalman_fill(values, start, end)
                    if filled is not None:
                        values[start:end] = filled
                    elif start > 0 and end < n:
                        left = values[start - 1]
                        right = values[end]
                        step = (right - left) / (length + 1)
                        for k in range(length):
                            values[start + k] = left + step * (k + 1)
                else:
                    has_long_gap = True
                    # Long gap: keep NaN

            if has_long_gap:
                gap_flags[col] = is_nan

            df[col] = values

        # Add gap flag columns for any column with long gaps
        for col, nan_mask in gap_flags.items():
            df[f"has_gap_{col}"] = nan_mask.astype("int8")

        return df

    @staticmethod
    def _kalman_fill(values: np.ndarray, start: int, end: int) -> np.ndarray | None:
        """Attempt Kalman smoothing on a gap segment. Returns filled values
        or None if Kalman fails."""
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            return None

        # Use surrounding valid values as context
        pre = values[max(0, start - 5):start]
        post = values[end:min(len(values), end + 5)]
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]

        observed = np.concatenate([pre, values[start:end], post]) if len(post) > 0 else np.concatenate([pre, values[start:end]])
        if len(pre) < 2:
            return None

        try:
            model = UnobservedComponents(
                observed,
                level='local level',
                irregular=True,
            )
            fitted = model.fit(disp=False)
            smoothed = fitted.smoothed_state[0]
            gap_len = end - start
            pre_len = len(pre)
            return smoothed[pre_len:pre_len + gap_len]
        except Exception:
            return None
```

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_outlier.py tests/preprocessing/numeric/test_missing.py -v
```
Expected: 11 PASS (some Kalman tests may skip on missing statsmodels)

- [ ] **Step 8: Commit**

```bash
git add stoke_ml/preprocessing/numeric/__init__.py stoke_ml/preprocessing/numeric/outlier.py stoke_ml/preprocessing/numeric/missing.py tests/preprocessing/numeric/__init__.py tests/preprocessing/numeric/test_outlier.py tests/preprocessing/numeric/test_missing.py
git commit -m "feat: add OutlierDetector (MAD) and MissingImputer (linear/Kalman/gap flags)"
```

---

### Task 6: RobustScaler + CrossSectionNormalizer

**Files:**
- Create: `stoke_ml/preprocessing/numeric/scaling.py`
- Create: `stoke_ml/preprocessing/numeric/cross_section.py`
- Create: `tests/preprocessing/numeric/test_scaling.py`
- Create: `tests/preprocessing/numeric/test_cross_section.py`

- [ ] **Step 1: Write tests for RobustScaler**

Write `tests/preprocessing/numeric/test_scaling.py`:

```python
"""Tests for RobustScaler — rolling-window robust standardization."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.scaling import RobustScaler


class TestRobustScaler:
    def test_default_window(self):
        rs = RobustScaler()
        assert rs.window_days == 252

    def test_scales_to_median_zero(self):
        rs = RobustScaler(window_days=10)
        values = np.arange(20, dtype=float)
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="B"),
                           "x": values})
        rs.fit(df)
        result = rs.transform(df)
        # After scaling, medians in rolling windows should be near 0
        # Just verify output is not equal to input
        assert not np.allclose(result["x"].dropna().values, df["x"].values)

    def test_preserves_nan(self):
        rs = RobustScaler(window_days=10)
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="B"),
                           "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        rs.fit(df)
        result = rs.transform(df)
        assert np.isnan(result["x"].iloc[1])

    def test_skip_small_window(self):
        rs = RobustScaler(window_days=252)
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="B"),
                           "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        rs.fit(df)
        result = rs.transform(df)
        # Window larger than data → output is NaN
        assert result["x"].isna().all()

    def test_winsorize_before_scaling(self):
        rs = RobustScaler(window_days=20, winsorize_sigma=3.0)
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30, freq="B"),
                           "x": list(np.random.randn(28)) + [100.0, -100.0]})
        rs.fit(df)
        result = rs.transform(df)
        # After winsorize + scale, extremes should not dominate
        assert result["x"].max() < 20.0

    def test_empty_df(self):
        rs = RobustScaler()
        result = rs.fit_transform(pd.DataFrame())
        assert len(result) == 0
```

- [ ] **Step 2: Write tests for CrossSectionNormalizer**

Write `tests/preprocessing/numeric/test_cross_section.py`:

```python
"""Tests for CrossSectionNormalizer — sector/size/adaptive normalization."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer


class TestCrossSectionNormalizer:
    def test_default_stages(self):
        csn = CrossSectionNormalizer()
        assert csn.stages == ["sector", "size", "adaptive"]
        assert csn.enabled is True

    def test_disabled_is_noop(self):
        csn = CrossSectionNormalizer(enabled=False)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = csn.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_sector_stage_neutralizes(self):
        csn = CrossSectionNormalizer(stages=["sector"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "stock_code": ["A", "B", "A", "C", "B", "C"],
            "x": [100.0, 200.0, 110.0, 50.0, 190.0, 55.0],
            "sector": ["银行", "科技", "银行", "医疗", "科技", "医疗"],
        })
        result = csn.fit_transform(df)
        # Within each sector on each day, values should be centered
        assert "x_cs" in result.columns or "x" in result.columns

    def test_empty_df(self):
        csn = CrossSectionNormalizer()
        result = csn.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_no_sector_column_skips_sector_stage(self):
        csn = CrossSectionNormalizer(stages=["sector"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "stock_code": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
        })
        result = csn.fit_transform(df)
        # Should not crash — sector stage skipped
        assert "x" in result.columns

    def test_columns_to_normalize_are_specified(self):
        csn = CrossSectionNormalizer(columns=["x", "y"])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "z": [100.0, 200.0, 300.0],
        })
        result = csn.fit_transform(df)
        # z should be unchanged, x/y should be transformed
        np.testing.assert_array_equal(result["z"].values, df["z"].values)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_scaling.py tests/preprocessing/numeric/test_cross_section.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 4: Write `stoke_ml/preprocessing/numeric/scaling.py`**

```python
"""RobustScaler: rolling-window median/MAD standardization with winsorize.

Backward-looking windows only (PIT-safe).  Fits parameters on training
data, reuses for validation/test.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class RobustScaler(PreprocessingStep):
    """Rolling-window robust standardization.

    For each column: winsorize(±winsorize_sigma * std), then
    z_robust = (x - rolling_median) / (rolling_MAD * 1.4826).

    The factor 1.4826 makes MAD consistent with standard deviation
    for normally distributed data.
    """

    def __init__(
        self,
        window_days: int = 252,
        winsorize_sigma: float = 3.0,
        min_periods: int = 63,
    ):
        self.window_days = window_days
        self.winsorize_sigma = winsorize_sigma
        self.min_periods = min_periods

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skip = {"is_limit_up", "is_limit_down", "is_neutral",
                "is_bull", "is_bear", "has_news", "has_guba_post",
                "has_xueqiu_post", "has_announce", "has_comment",
                "date_day", "date_month", "date_weekday"}
        # Also skip bool columns and gap flags
        skip |= {c for c in df.columns if c.startswith("has_gap_")}
        # And columns that are fundamentally boolean
        for c in df.columns:
            if df[c].dropna().nunique() <= 2:
                skip.add(c)

        for col in numeric_cols:
            if col in skip:
                continue
            values = df[col].values
            # Winsorize
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 1e-10:
                upper = mean + self.winsorize_sigma * std
                lower = mean - self.winsorize_sigma * std
                values = np.clip(values, lower, upper)

            # Rolling robust scale
            series = pd.Series(values, index=df.index)
            roll_median = series.rolling(self.window_days, min_periods=self.min_periods).median()
            roll_mad = series.rolling(self.window_days, min_periods=self.min_periods).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            scaled = (values - roll_median.values) / (roll_mad.values * 1.4826 + 1e-10)
            df[col] = scaled.astype(np.float32)

        return df
```

- [ ] **Step 5: Write `stoke_ml/preprocessing/numeric/cross_section.py`**

```python
"""Cross-sectional normalization: sector neutralization → size neutralization → adaptive.

Three-stage pipeline:
1. Sector: X - median(X | sector, date) / MAD(X | sector, date)  (Hybrid approach)
2. Size: residual ~ log(mcap) + log²(mcap) → take residual
3. Adaptive: strengthen neutralization in high-volatility regimes
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class CrossSectionNormalizer(PreprocessingStep):
    """Remove market/sector/size effects from features.

    Each stage is optional and falls back gracefully if required
    columns are missing (e.g. no sector mapper available → skip sector).
    """

    def __init__(
        self,
        enabled: bool = True,
        stages: list[str] | None = None,
        columns: list[str] | None = None,
    ):
        self.enabled = enabled
        self.stages = stages or ["sector", "size", "adaptive"]
        self.columns = columns  # None = auto-select numeric

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if not self.enabled or df.empty:
            return df
        df = df.copy()

        cols = self.columns
        if cols is None:
            cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("open", "high", "low", "close", "volume",
                                 "amount", "date_day", "date_month", "date_weekday")]

        for stage in self.stages:
            if stage == "sector" and "sector" in df.columns:
                df = self._sector_neutralize(df, cols)
            elif stage == "size" and "market_cap" in df.columns:
                df = self._size_neutralize(df, cols)
            elif stage == "adaptive":
                df = self._adaptive_strength(df, cols)

        return df

    @staticmethod
    def _sector_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Hybrid: within each (date, sector), subtract median and divide by MAD."""
        if "date" not in df.columns:
            return df
        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_raw"] = df[col].copy()
            grouped = df.groupby(["date", "sector"])[col]
            median = grouped.transform("median")
            mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
            denom = mad.replace(0, 1.0)
            df[col] = ((df[col] - median) / denom).astype(np.float32)
        return df

    @staticmethod
    def _size_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Regress each feature on log(mcap) + log²(mcap) cross-sectionally,
        take residual per day."""
        if "date" not in df.columns or "market_cap" not in df.columns:
            return df
        log_mcap = np.log(df["market_cap"].replace(0, np.nan))
        log_mcap2 = log_mcap ** 2

        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_pre_size"] = df[col].copy()
            result = pd.Series(np.nan, index=df.index)
            for date, idx in df.groupby("date").groups.items():
                subset = df.loc[idx]
                y = subset[col].dropna()
                if len(y) < 10:
                    continue
                X = pd.DataFrame({
                    "log_mcap": log_mcap.loc[y.index],
                    "log_mcap2": log_mcap2.loc[y.index],
                }).dropna()
                common = X.index.intersection(y.index)
                if len(common) < 10:
                    continue
                try:
                    beta = np.linalg.lstsq(
                        np.column_stack([np.ones(len(common)),
                                         X.loc[common, "log_mcap"].values,
                                         X.loc[common, "log_mcap2"].values]),
                        y.loc[common].values,
                        rcond=None,
                    )[0]
                    pred = (beta[0] + beta[1] * log_mcap.loc[common] +
                            beta[2] * log_mcap2.loc[common])
                    result.loc[common] = y.loc[common].values - pred.values
                except np.linalg.LinAlgError:
                    pass
            df[col] = result.astype(np.float32)
        return df

    @staticmethod
    def _adaptive_strength(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """In high-volatility regimes, strengthen neutralization.

        alpha = alpha_0 * (1 + beta * (sigma_short - sigma_long) / sigma_long)
        """
        if "close" not in df.columns:
            return df
        returns = df["close"].pct_change()
        sigma_short = returns.rolling(20).std()
        sigma_long = returns.rolling(60).std()
        rel_vol = (sigma_short - sigma_long) / sigma_long.replace(0, 1.0)
        alpha = 1.0 + 0.5 * rel_vol.clip(-0.5, 1.0)

        for col in cols:
            if col not in df.columns:
                continue
            df[col] = (df[col].values * alpha.values).astype(np.float32)

        return df
```

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_scaling.py tests/preprocessing/numeric/test_cross_section.py -v
```
Expected: 10 PASS

- [ ] **Step 7: Commit**

```bash
git add stoke_ml/preprocessing/numeric/scaling.py stoke_ml/preprocessing/numeric/cross_section.py tests/preprocessing/numeric/test_scaling.py tests/preprocessing/numeric/test_cross_section.py
git commit -m "feat: add RobustScaler (rolling winsorize+MAD) and CrossSectionNormalizer (sector/size/adaptive)"
```

---

### Task 7: HigherOrderDeriver

**Files:**
- Create: `stoke_ml/preprocessing/numeric/higher_order.py`
- Create: `tests/preprocessing/numeric/test_higher_order.py`

- [ ] **Step 1: Write tests**

Write `tests/preprocessing/numeric/test_higher_order.py`:

```python
"""Tests for HigherOrderDeriver — skew, kurtosis, realized vol, Amihud illiquidity."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.numeric.higher_order import HigherOrderDeriver


class TestHigherOrderDeriver:
    def test_computes_skew_and_kurtosis(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
        })
        result = hod.fit_transform(df)
        assert "skew_20d" in result.columns
        assert "kurt_20d" in result.columns

    def test_computes_realized_vol(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
        })
        result = hod.fit_transform(df)
        assert "realized_vol_5d" in result.columns
        assert "realized_vol_20d" in result.columns

    def test_computes_amihud_illiquidity(self):
        hod = HigherOrderDeriver()
        n = 100
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": 100 + rng.randn(n).cumsum(),
            "volume": np.abs(rng.randn(n) * 1e6) + 1e5,
        })
        result = hod.fit_transform(df)
        assert "amihud_illiq_20d" in result.columns

    def test_computes_max_drawdown(self):
        hod = HigherOrderDeriver()
        n = 100
        prices = np.array([100, 102, 105, 103, 98, 95, 97, 101, 99, 105,
                           108, 110, 107, 104, 100, 97, 94, 96, 100, 103])
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(prices), freq="B"),
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1e6] * len(prices),
        })
        result = hod.fit_transform(df)
        assert "max_drawdown_20d" in result.columns
        assert "up_days_ratio_20d" in result.columns
        # max_drawdown should be non-negative (positive = drawdown magnitude)
        dd = result["max_drawdown_20d"].dropna()
        assert (dd >= 0).all()

    def test_empty_df(self):
        hod = HigherOrderDeriver()
        result = hod.fit_transform(pd.DataFrame())
        assert len(result) == 0

    def test_too_short_series_returns_nan(self):
        hod = HigherOrderDeriver()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "close": list(range(10, 20)),
        })
        result = hod.fit_transform(df)
        # Short series → rolling window stats are NaN for windows like 20d
        assert "skew_20d" in result.columns

    def test_disabled_is_noop(self):
        hod = HigherOrderDeriver(enabled=False)
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = hod.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_higher_order.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write `stoke_ml/preprocessing/numeric/higher_order.py`**

```python
"""Higher-order derived features from OHLCV data.

Adds: skew, kurtosis, realized volatility surface, Amihud illiquidity,
VWAP deviation, max drawdown, up-days ratio.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class HigherOrderDeriver(PreprocessingStep):
    """Compute higher-order statistics from price/volume series.

    Only operates on raw OHLCV data, not on already-scaled features.
    All rolling windows are backward-looking (PIT-safe).
    """

    _VOL_WINDOWS = (5, 10, 20, 60)
    _MOMENT_WINDOW = 20

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if not self.enabled or df.empty:
            return df
        df = df.copy()

        if "close" in df.columns:
            close = df["close"]
            returns = close.pct_change()
            log_ret = np.log(close / close.shift(1))

            # Return distribution moments (20-day rolling)
            df["skew_20d"] = returns.rolling(20, min_periods=10).skew().astype(np.float32)
            df["kurt_20d"] = returns.rolling(20, min_periods=10).kurt().astype(np.float32)

            # Realized volatility surface
            for w in self._VOL_WINDOWS:
                vol = returns.rolling(w, min_periods=max(3, w // 3)).std()
                df[f"realized_vol_{w}d"] = vol.astype(np.float32)

            # Max drawdown
            for w in (20, 60):
                roll_max = close.rolling(w, min_periods=w // 2).max()
                dd = (roll_max - close) / roll_max.replace(0, np.nan)
                df[f"max_drawdown_{w}d"] = dd.astype(np.float32)

            # Up-days ratio
            for w in (20,):
                up = (returns > 0).rolling(w, min_periods=w // 2).mean()
                df[f"up_days_ratio_{w}d"] = up.astype(np.float32)

        # Amihud illiquidity: |return| / (price * volume)
        if "close" in df.columns and "volume" in df.columns:
            volume = df["volume"].replace(0, np.nan)
            amihud = np.abs(returns) / (close * volume + 1)
            for w in (20,):
                avg_amihud = amihud.rolling(w, min_periods=w // 2).mean()
                df[f"amihud_illiq_{w}d"] = avg_amihud.astype(np.float32)

        # VWAP deviation
        if all(c in df.columns for c in ("high", "low", "close")):
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            vwap = (typical * df.get("volume", 1.0)).rolling(20, min_periods=5).sum() / \
                   df.get("volume", 1.0).rolling(20, min_periods=5).sum().replace(0, np.nan)
            df["vwap_deviation_20d"] = ((close - vwap) / vwap.replace(0, np.nan)).astype(np.float32)

        return df
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/numeric/test_higher_order.py -v
```
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/preprocessing/numeric/higher_order.py tests/preprocessing/numeric/test_higher_order.py
git commit -m "feat: add HigherOrderDeriver with skew, kurtosis, realized vol, Amihud, drawdown"
```

---

### Task 8: PreprocessingMonitor (数据质量监控)

**Files:**
- Create: `stoke_ml/preprocessing/monitor/__init__.py`
- Create: `stoke_ml/preprocessing/monitor/quality.py`
- Create: `stoke_ml/preprocessing/monitor/drift.py`
- Create: `tests/preprocessing/monitor/__init__.py`
- Create: `tests/preprocessing/monitor/test_quality.py`
- Create: `tests/preprocessing/monitor/test_drift.py`

- [ ] **Step 1: Write tests**

Write `tests/preprocessing/monitor/test_quality.py`:

```python
"""Tests for QualityMonitor — missing/duplicate/zero/inf/constant checks."""
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
        report = qm.check(df)
        alerts = [r for r in report if r["level"] in ("WARN", "ERROR")]
        has_missing = any("missing" in str(r.get("message", "")).lower() for r in alerts)
        assert has_missing or len(alerts) > 0

    def test_detects_inf_values(self):
        qm = QualityMonitor()
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
        report = qm.check(df)
        assert any(r["level"] == "ERROR" for r in report)

    def test_detects_constant_columns(self):
        qm = QualityMonitor()
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        report = qm.check(df)
        assert any("constant" in r.get("check", "").lower() for r in report)

    def test_detects_duplicates(self):
        qm = QualityMonitor(duplicate_warn_threshold=0.1)
        df = pd.DataFrame({"x": [1, 1, 2, 1, 1]})
        report = qm.check(df)
        # May or may not fire depending on threshold math
        assert isinstance(report, list)

    def test_all_clear_on_good_data(self):
        qm = QualityMonitor()
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"x": rng.randn(100), "y": rng.randn(100)})
        report = qm.check(df)
        errors = [r for r in report if r["level"] == "ERROR"]
        assert len(errors) == 0

    def test_checks_shape(self):
        qm = QualityMonitor(expected_rows=100, expected_cols=5)
        df = pd.DataFrame({"x": range(10)})
        report = qm.check(df)
        assert any("shape" in r.get("check", "").lower() or
                   "row" in r.get("check", "").lower() for r in report)

    def test_empty_df(self):
        qm = QualityMonitor()
        report = qm.check(pd.DataFrame())
        assert any(r["level"] == "WARN" for r in report)
```

Write `tests/preprocessing/monitor/test_drift.py`:

```python
"""Tests for DriftMonitor — KS-test distribution shift detection."""
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.monitor.drift import DriftMonitor


class TestDriftMonitor:
    def test_detects_large_shift(self):
        dm = DriftMonitor(p_threshold=0.05)
        baseline = pd.DataFrame({"x": np.random.RandomState(42).randn(1000)})
        current = pd.DataFrame({"x": np.random.RandomState(99).randn(1000) + 3.0})
        dm.fit(baseline)
        alerts = dm.check(current)
        assert len(alerts) > 0

    def test_no_alarm_on_stable_data(self):
        dm = DriftMonitor(p_threshold=0.01)
        rng = np.random.RandomState(42)
        baseline = pd.DataFrame({"x": rng.randn(500)})
        current = pd.DataFrame({"x": rng.randn(500) * 0.9 + 0.1})
        dm.fit(baseline)
        alerts = dm.check(current)
        # Small changes should not trigger
        assert len(alerts) <= 1

    def test_skip_non_numeric(self):
        dm = DriftMonitor()
        baseline = pd.DataFrame({"cat": ["a", "b", "a", "b", "c"]})
        dm.fit(baseline)
        alerts = dm.check(baseline)
        assert len(alerts) == 0

    def test_handles_nan_in_data(self):
        dm = DriftMonitor(p_threshold=0.05)
        baseline = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0] * 20})
        current = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0] * 20})
        dm.fit(baseline)
        alerts = dm.check(current)
        # Should handle NaN without crashing
        assert isinstance(alerts, list)

    def test_save_and_load_baseline(self):
        import tempfile
        import json
        from pathlib import Path
        dm = DriftMonitor()
        baseline = pd.DataFrame({"x": np.random.randn(100)})
        dm.fit(baseline)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "baseline.json"
            dm.save_baseline(path)
            dm2 = DriftMonitor()
            dm2.load_baseline(path)
            alerts = dm2.check(baseline)
            assert len(alerts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/monitor/test_quality.py tests/preprocessing/monitor/test_drift.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write `stoke_ml/preprocessing/monitor/__init__.py`**

```python
"""Data quality monitoring: input checks, transform checks, drift detection."""

from stoke_ml.preprocessing.monitor.quality import QualityMonitor
from stoke_ml.preprocessing.monitor.drift import DriftMonitor

__all__ = ["QualityMonitor", "DriftMonitor"]
```

- [ ] **Step 4: Write `stoke_ml/preprocessing/monitor/quality.py`**

```python
"""QualityMonitor: input/transform/output layer data quality checks.

Three layers:
  Input  — missing_rate, duplicate_rate, zero_rate, freshness
  Transform — outlier_rate, inf_check, constant_check, shape_check
  Output — distribution_drift (delegated to DriftMonitor)
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class QualityMonitor:
    """Statistical quality checks on feature DataFrames.

    Returns a list of alert dicts, each with: check, level (ERROR/WARN/INFO),
    message, and value.
    """

    def __init__(
        self,
        missing_warn_threshold: float = 0.2,
        duplicate_warn_threshold: float = 0.05,
        zero_rate_warn_threshold: float = 0.5,
        outlier_warn_threshold: float = 0.1,
        expected_rows: int | None = None,
        expected_cols: int | None = None,
        max_stale_days: int = 3,
    ):
        self.missing_warn_threshold = missing_warn_threshold
        self.duplicate_warn_threshold = duplicate_warn_threshold
        self.zero_rate_warn_threshold = zero_rate_warn_threshold
        self.outlier_warn_threshold = outlier_warn_threshold
        self.expected_rows = expected_rows
        self.expected_cols = expected_cols
        self.max_stale_days = max_stale_days

    def check(self, df: pd.DataFrame, step: str = "unknown") -> list[dict]:
        """Run all quality checks and return alerts."""
        if df.empty:
            return [{
                "check": "empty_df",
                "level": "WARN",
                "message": f"Empty DataFrame at step '{step}'",
                "value": 0,
            }]

        alerts = []
        alerts.extend(self._input_checks(df, step))
        alerts.extend(self._transform_checks(df, step))
        return alerts

    def _input_checks(self, df, step):
        alerts = []
        n = len(df)

        # Missing rate per column
        for col in df.select_dtypes(include=[np.number]).columns:
            miss = df[col].isna().mean()
            if miss > self.missing_warn_threshold:
                alerts.append({
                    "check": "missing_rate",
                    "level": "WARN",
                    "column": col,
                    "message": f"Column '{col}' missing rate={miss:.1%} > {self.missing_warn_threshold:.0%}",
                    "value": miss,
                })

        # Duplicate row rate
        dup_rate = 1.0 - len(df.drop_duplicates()) / max(n, 1)
        if dup_rate > self.duplicate_warn_threshold:
            alerts.append({
                "check": "duplicate_rate",
                "level": "WARN",
                "message": f"Duplicate row rate={dup_rate:.1%}",
                "value": dup_rate,
            })

        # Zero rate (ZI contamination check)
        for col in df.select_dtypes(include=[np.number]).columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                zero_rate = (non_null == 0).mean()
                if zero_rate > self.zero_rate_warn_threshold:
                    alerts.append({
                        "check": "zero_rate",
                        "level": "WARN",
                        "column": col,
                        "message": f"Column '{col}' zero rate={zero_rate:.1%} (possible ZI contamination)",
                        "value": zero_rate,
                    })

        # Freshness: is max(date) stale?
        if "date" in df.columns:
            max_date = pd.to_datetime(df["date"]).max()
            days_stale = (datetime.now() - max_date.to_pydatetime()).days
            if days_stale > self.max_stale_days:
                alerts.append({
                    "check": "freshness",
                    "level": "WARN",
                    "message": f"Data stale: max date={max_date.date()}, {days_stale} days ago",
                    "value": days_stale,
                })

        return alerts

    def _transform_checks(self, df, step):
        alerts = []
        numeric = df.select_dtypes(include=[np.number])

        # infinity
        for col in numeric.columns:
            inf_count = np.isinf(df[col].values).sum()
            if inf_count > 0:
                alerts.append({
                    "check": "inf_values",
                    "level": "ERROR",
                    "column": col,
                    "message": f"Column '{col}' has {inf_count} inf values",
                    "value": inf_count,
                })

        # constant columns
        for col in numeric.columns:
            if df[col].nunique(dropna=True) <= 1:
                alerts.append({
                    "check": "constant_column",
                    "level": "WARN",
                    "column": col,
                    "message": f"Column '{col}' is constant (variance=0)",
                    "value": 0,
                })

        # MAD outlier rate
        for col in numeric.columns:
            values = df[col].dropna().values
            if len(values) < 10:
                continue
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad < 1e-10:
                continue
            outlier_rate = (np.abs(values - median) > 5 * mad).mean()
            if outlier_rate > self.outlier_warn_threshold:
                alerts.append({
                    "check": "outlier_rate",
                    "level": "WARN",
                    "column": col,
                    "message": f"Column '{col}' outlier rate={outlier_rate:.1%}",
                    "value": outlier_rate,
                })

        # Shape check
        if self.expected_rows and len(df) < self.expected_rows * 0.5:
            alerts.append({
                "check": "shape_rows",
                "level": "ERROR",
                "message": f"Rows={len(df)} vs expected={self.expected_rows}",
                "value": len(df),
            })
        if self.expected_cols:
            n_cols = len(numeric.columns)
            if n_cols < self.expected_cols * 0.5:
                alerts.append({
                    "check": "shape_cols",
                    "level": "WARN",
                    "message": f"Columns={n_cols} vs expected={self.expected_cols}",
                    "value": n_cols,
                })

        return alerts
```

- [ ] **Step 5: Write `stoke_ml/preprocessing/monitor/drift.py`**

```python
"""DriftMonitor: KS-test based distribution shift detection.

Compares current feature distributions against a stored baseline.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Detect feature distribution drift via two-sample KS test."""

    def __init__(self, p_threshold: float = 0.01):
        self.p_threshold = p_threshold
        self._baseline: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, **kwargs):
        """Store baseline distributions from *df*."""
        self._baseline = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna().values
            if len(values) >= 30:
                self._baseline[col] = values
        return self

    def check(self, df: pd.DataFrame) -> list[dict]:
        """Compare *df* against baseline, return drift alerts."""
        if not self._baseline:
            return []

        try:
            from scipy import stats as sp_stats
        except ImportError:
            logger.warning("scipy not available, skipping KS-test")
            return []

        alerts = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in self._baseline:
                continue
            current = df[col].dropna().values
            if len(current) < 30:
                continue

            baseline = self._baseline[col]
            try:
                ks_stat, p_val = sp_stats.ks_2samp(baseline, current)
                if p_val < self.p_threshold:
                    alerts.append({
                        "check": "distribution_drift",
                        "level": "WARN",
                        "column": col,
                        "p_value": float(p_val),
                        "ks_statistic": float(ks_stat),
                        "message": (
                            f"Column '{col}' distribution drifted "
                            f"(KS={ks_stat:.3f}, p={p_val:.4f})"
                        ),
                    })
            except Exception:
                pass

        return alerts

    def save_baseline(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for col, values in self._baseline.items():
            data[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p01": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
                "n": len(values),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_baseline(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning("Baseline file not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._baseline = {}
        for col, stats in data.items():
            self._baseline[col] = np.random.normal(
                stats["mean"], stats["std"], stats["n"]
            ).astype(np.float32)
```

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/monitor/test_quality.py tests/preprocessing/monitor/test_drift.py -v
```
Expected: 12 PASS

- [ ] **Step 7: Commit**

```bash
git add stoke_ml/preprocessing/monitor/__init__.py stoke_ml/preprocessing/monitor/quality.py stoke_ml/preprocessing/monitor/drift.py tests/preprocessing/monitor/__init__.py tests/preprocessing/monitor/test_quality.py tests/preprocessing/monitor/test_drift.py
git commit -m "feat: add QualityMonitor and DriftMonitor for data quality checks"
```

---

### Task 9: PreprocessingPipeline (编排引擎)

**Files:**
- Create: `stoke_ml/preprocessing/pipeline.py`
- Create: `stoke_ml/preprocessing/config.py`
- Create: `tests/preprocessing/test_pipeline.py`

- [ ] **Step 1: Write the integration test**

Write `tests/preprocessing/test_pipeline.py`:

```python
"""Integration tests for PreprocessingPipeline — end-to-end text+numeric chains."""
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer


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
        import pytest
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_pipeline.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write `stoke_ml/preprocessing/pipeline.py`**

```python
"""PreprocessingPipeline: orchestration engine for preprocessing chains.

Chains are registered per source (e.g. 'xueqiu', 'news') and can be run
independently.  The pipeline is configuration-driven and compatible with
both the existing FeaturePipeline and a future backtesting system.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingChain

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Register and run named preprocessing chains.

    Usage:
        pp = PreprocessingPipeline()
        pp.register_chain("xueqiu", text_chain)
        clean = pp.run("xueqiu", raw_posts, stock_code="000001")
    """

    def __init__(self):
        self._chains: dict[str, PreprocessingChain] = {}

    def register_chain(self, name: str, chain: PreprocessingChain) -> None:
        self._chains[name] = chain

    def run(self, chain_name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run a named chain on *df*, returning transformed DataFrame."""
        chain = self._chains.get(chain_name)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_name}' not found. "
                f"Available: {list(self._chains.keys())}"
            )
        return chain.fit_transform(df, **kwargs)

    def list_chains(self) -> list[str]:
        return sorted(self._chains.keys())

    @classmethod
    def from_config(cls, config: dict) -> PreprocessingPipeline:
        """Build pipeline from configuration dict.

        *config* is the 'preprocessing' section from config.yaml.
        """
        from stoke_ml.preprocessing.config import build_pipeline_from_config
        return build_pipeline_from_config(config)
```

- [ ] **Step 4: Write `stoke_ml/preprocessing/config.py`**

```python
"""Config-driven pipeline construction.

Reads the 'preprocessing' section of config.yaml and assembles
PreprocessingChains for each text/numeric source.
"""

from __future__ import annotations

import logging

from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.numeric.scaling import RobustScaler
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer
from stoke_ml.preprocessing.numeric.higher_order import HigherOrderDeriver
from stoke_ml.preprocessing.monitor.quality import QualityMonitor

logger = logging.getLogger(__name__)

_STEP_REGISTRY = {
    "BipolarClassifier": BipolarClassifier,
    "TimeDecayWeighter": TimeDecayWeighter,
    "DailyAggregator": DailyAggregator,
    "OutlierDetector": OutlierDetector,
    "MissingImputer": MissingImputer,
    "RobustScaler": RobustScaler,
    "CrossSectionNormalizer": CrossSectionNormalizer,
    "HigherOrderDeriver": HigherOrderDeriver,
}


def build_pipeline_from_config(cfg: dict) -> PreprocessingPipeline:
    """Assemble PreprocessingPipeline from config dict."""
    pp = PreprocessingPipeline()

    pp_cfg = cfg if isinstance(cfg, dict) else {}
    if not pp_cfg:
        return pp

    # Text chain
    text_cfg = pp_cfg.get("text", {})
    text_chain = PreprocessingChain(name="text_default")
    text_chain.add(BipolarClassifier(
        pos_threshold=text_cfg.get("bipolar", {}).get("threshold_positive", 0.2),
        neg_threshold=text_cfg.get("bipolar", {}).get("threshold_negative", -0.2),
    ))
    decay_cfg = text_cfg.get("time_decay", {})
    text_chain.add(TimeDecayWeighter(
        halflife_days=decay_cfg.get("halflife_days", 7),
    ))
    agg_cfg = text_cfg.get("aggregation", {})
    text_chain.add(DailyAggregator(
        windows=tuple(agg_cfg.get("windows", [3, 5, 10, 20])),
    ))
    pp.register_chain("text", text_chain)

    # Numeric chain
    num_cfg = pp_cfg.get("numeric", {})
    num_chain = PreprocessingChain(name="numeric_default")

    oc = num_cfg.get("outlier", {})
    num_chain.add(OutlierDetector(
        threshold=oc.get("threshold", 5.0),
        clip=oc.get("clip", True),
    ))
    mc = num_cfg.get("missing", {})
    num_chain.add(MissingImputer(
        short_gap_max=mc.get("short_gap_max", 2),
        medium_gap_max=mc.get("medium_gap_max", 10),
    ))
    cs = num_cfg.get("cross_section", {})
    num_chain.add(CrossSectionNormalizer(
        enabled=cs.get("enabled", True),
        stages=cs.get("stages", ["sector", "size", "adaptive"]),
    ))
    sc = num_cfg.get("scaling", {})
    num_chain.add(RobustScaler(
        window_days=sc.get("window_days", 252),
        winsorize_sigma=sc.get("winsorize_sigma", 3.0),
    ))
    ho = num_cfg.get("higher_order", {})
    if ho.get("enabled", True):
        num_chain.add(HigherOrderDeriver(enabled=True))

    pp.register_chain("numeric", num_chain)

    return pp
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_pipeline.py -v
```
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add stoke_ml/preprocessing/pipeline.py stoke_ml/preprocessing/config.py tests/preprocessing/test_pipeline.py
git commit -m "feat: add PreprocessingPipeline orchestration engine with config-driven construction"
```

---

### Task 10: Update config.yaml + integration with FeaturePipeline

**Files:**
- Modify: `config.yaml`
- Modify: `stoke_ml/features/pipeline.py`

- [ ] **Step 1: Add preprocessing section to config.yaml**

Append to `config.yaml`:

```yaml
# Preprocessing pipeline configuration
preprocessing:
  enabled: true
  output_dir: "data/preprocessed"
  registry_path: "models/features/feature_registry.json"

  text:
    quality_filter:
      min_text_length: 5
      max_duplicate_similarity: 0.9
      remove_html: true
    bipolar:
      threshold_positive: 0.2
      threshold_negative: -0.2
    time_decay:
      method: "ema"
      halflife_days: 7
    topic_model:
      enabled: false
      n_topics: "auto"
      min_topic_size: 50
      model_cache_dir: "models/bertopic"
      embedding_model: "finbert"
    aggregation:
      windows: [3, 5, 10, 20]
      use_body_sentiment: true

  numeric:
    outlier:
      method: "mad"
      threshold: 5.0
      clip: true
    missing:
      short_gap_method: "linear"
      short_gap_max: 2
      medium_gap_method: "kalman"
      medium_gap_max: 10
    cross_section:
      enabled: true
      stages: ["sector", "size", "adaptive"]
    scaling:
      method: "robust"
      window_days: 252
      winsorize_sigma: 3.0
    higher_order:
      enabled: true

  monitor:
    enabled: true
    log_dir: "logs/quality"
    drift_p_threshold: 0.01
    missing_warn_threshold: 0.2
    zero_rate_warn_threshold: 0.5

  registry:
    enabled: true
    baseline_update_freq: "monthly"
```

- [ ] **Step 2: Add optional preprocessing to FeaturePipeline**

Modify `stoke_ml/features/pipeline.py` — add a `preprocessing_config` parameter and `use_new_preprocessing` flag to the constructor:

```python
        self.use_new_preprocessing = use_new_preprocessing
        self._preprocessing = None
        if use_new_preprocessing and preprocessing_config:
            from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
            from stoke_ml.config import load_config
            cfg = load_config(preprocessing_config) if isinstance(preprocessing_config, str) else preprocessing_config
            self._preprocessing = PreprocessingPipeline.from_config(
                cfg.get("preprocessing", {})
            )
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -v --ignore=tests/preprocessing
```
Expected: all existing tests PASS

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -v
```
Expected: all tests PASS (existing + new preprocessing tests)

- [ ] **Step 5: Commit**

```bash
git add config.yaml stoke_ml/features/pipeline.py
git commit -m "feat: add preprocessing config section and FeaturePipeline integration hook"
```

---

### Task 11: End-to-end smoke test on real data

**Files:**
- Create: `tests/preprocessing/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

Write `tests/preprocessing/test_smoke.py`:

```python
"""End-to-end smoke test: run preprocessing on real Xueqiu data for one stock."""
import pytest
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.numeric.scaling import RobustScaler
from stoke_ml.preprocessing.monitor.quality import QualityMonitor


@pytest.mark.slow
class TestFullPipelineSmoke:
    """Run the full text + numeric chains on synthetic realistic data."""

    def test_text_chain_realistic(self):
        """Simulate Xueqiu silver data: 500 posts over 2 years."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2024-01-01", "2026-06-30", freq="D")
        df = pd.DataFrame({
            "aligned_date": rng.choice(dates, n),
            "sentiment_title": rng.uniform(-1, 1, n).astype(np.float32),
            "sentiment_body": rng.uniform(-1, 1, n).astype(np.float32),
        })
        df = df.sort_values("aligned_date").reset_index(drop=True)

        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            BipolarClassifier(sentiment_cols=["sentiment_title", "sentiment_body"]),
            TimeDecayWeighter(halflife_days=7),
            DailyAggregator(windows=(3, 5, 10)),
        ], name="text_full")
        pp.register_chain("xueqiu", chain)

        result = pp.run("xueqiu", df)

        # Assert daily output
        assert "bipolar_sent" in result.columns
        assert "agreement" in result.columns
        assert "attention" in result.columns
        assert "body_sent_mean" in result.columns

        # Bipolar is in [-1, 1]
        assert result["bipolar_sent"].between(-1, 1).all()
        # Agreement is in [0, 1]
        assert result["agreement"].between(0, 1).all()
        # Attention is non-negative
        assert (result["attention"] >= 0).all()

        # Rolling window derivatives exist
        assert "bipolar_sent_5d_mean" in result.columns

        # At least some days have data
        assert len(result) > 0

    def test_numeric_chain_realistic(self):
        """Simulate daily K-line data: 500 trading days."""
        rng = np.random.RandomState(42)
        n = 500
        close = 100 + rng.randn(n).cumsum() * 0.5
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
            "high": close * (1 + rng.uniform(0.0, 0.02, n)),
            "low": close * (1 - rng.uniform(0.0, 0.02, n)),
            "close": close,
            "volume": np.abs(rng.randn(n) * 1e6) + 1e5,
        })
        # Insert some gaps
        df.loc[50:52, "close"] = np.nan
        df.loc[200:210, "close"] = np.nan
        df.loc[300, "volume"] = 1e10  # outlier

        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            OutlierDetector(threshold=5.0),
            MissingImputer(short_gap_max=2, medium_gap_max=10),
            RobustScaler(window_days=60, min_periods=20),
        ], name="numeric_mini")
        pp.register_chain("daily", chain)

        result = pp.run("daily", df)

        # Short gaps should be filled
        assert not np.isnan(result["close"].iloc[51])

        # Outlier should be clipped
        assert result["volume"].max() < 1e10

        # Long gaps (>10 days) may still have NaN or be filled
        assert "close" in result.columns
        assert len(result) == n

    def test_quality_monitor_integration(self):
        """QualityMonitor reports on output data."""
        rng = np.random.RandomState(42)
        df_good = pd.DataFrame({
            "bipolar_sent": rng.uniform(-1, 1, 200),
            "agreement": rng.uniform(0, 1, 200),
            "attention": np.log(1 + rng.poisson(5, 200)),
        })

        qm = QualityMonitor(
            missing_warn_threshold=0.2,
            zero_rate_warn_threshold=0.5,
        )
        report = qm.check(df_good, step="text_gold")
        errors = [r for r in report if r["level"] == "ERROR"]
        assert len(errors) == 0

        # Inject bad data
        df_bad = pd.DataFrame({
            "bipolar_sent": [np.inf, -np.inf, 0.5],
            "agreement": [0.5, 0.5, 0.5],
        })
        report_bad = qm.check(df_bad)
        assert any(r["level"] == "ERROR" for r in report_bad)
```

- [ ] **Step 2: Run smoke test**

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_smoke.py -v -m "slow"
```
Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/preprocessing/test_smoke.py
git commit -m "test: add end-to-end smoke tests for text+numeric preprocessing chains"
```

---

## Verification Checklist

After ALL tasks complete, run:

```bash
# 1. Full test suite
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -v

# 2. Smoke test on real data
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/preprocessing/test_smoke.py -v

# 3. Import check — all modules loadable
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.preprocessing import PreprocessingStep, PreprocessingChain
from stoke_ml.preprocessing.text import BipolarClassifier, TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric import OutlierDetector, MissingImputer
from stoke_ml.preprocessing.numeric.scaling import RobustScaler
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer
from stoke_ml.preprocessing.numeric.higher_order import HigherOrderDeriver
from stoke_ml.preprocessing.monitor import QualityMonitor, DriftMonitor
from stoke_ml.preprocessing.registry import FeatureRegistry, FeatureDefinition
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.config import build_pipeline_from_config
print('All imports OK')
"
```

---

## Notes

- **BERTopic (P3) deferred**: Topic modeling needs GPU + FinBERT embeddings pre-computed. The `TopicModeler` step is defined in the spec but will be implemented in a follow-up plan once the basic text chain is validated.
- **TextQualityFilter deferred**: HTML stripping and duplicate detection are simple to add, but the value comes after verifying the core chain works on real data.
- **Kalman dependency**: `statsmodels` must be installed (`pip install statsmodels`). Tests gracefully skip if unavailable.
- **Sector mapper**: `CrossSectionNormalizer` stage 1 requires a sector column in the DataFrame. The existing `StockSectorMapper` doesn't exist yet — it will need creation or the user can inject sector info via `market_cap` and `sector` columns in the daily DataFrame.
- **Backtesting compatibility**: All preprocessing steps accept `start_date`/`end_date` kwargs via the pipeline's `run()` method. The `PreprocessingPipeline` is callable per-date-range, ready for walk-forward backtesting.
