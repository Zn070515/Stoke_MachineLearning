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
        drifted = reg.check_drift(new_stats, sigma_threshold=1.0)
        assert len(drifted) >= 1
        assert drifted[0]["feature"] == "feat"

    def test_check_drift_no_baseline_skips(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="feat", category="num"))
        new_stats = {"feat": {"mean": 100.0, "std": 50.0}}
        drifted = reg.check_drift(new_stats)
        assert len(drifted) == 0
