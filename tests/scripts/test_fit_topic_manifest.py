"""§十三: fit_topic_model.py run manifest carries full reproducibility info.

Every field that determines what the topic model represents — pickle SHA-256,
corpus file hashes, embedding model, min_topic_size, seed, dependency/Python
versions, sampling method + selected codes, config hash, calendar hash — must
be pinned in the manifest.  The fit itself is mocked (no network, no BERTopic
training); only the manifest-building wiring is exercised.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "production" / "fit_topic_model.py"
_spec = importlib.util.spec_from_file_location("fit_topic_model_mod", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_ALL_CODES = ["000001", "000002", "000003", "000004"]


class _FakeModel:
    def get_topic_info(self):
        return [{"Topic": -1}, {"Topic": 0}, {"Topic": 1}]


class _FakeTM:
    embedding_model = "finbert"
    min_topic_size = 50
    corpus_hash_ = "abcd1234abcd1234"
    _model = _FakeModel()
    last_fit_kwargs = None

    def __init__(self, model_cache_dir):
        self.model_cache_dir = str(model_cache_dir)
        self.fit_kwargs = {}

    def _corpus_key(self, cutoff):
        return pd.Timestamp(cutoff).strftime("%Y-%m-%d")

    def _read_meta(self, source, key):
        return {}

    def fit(self, clean, **kwargs):
        self.fit_kwargs = kwargs
        _FakeTM.last_fit_kwargs = kwargs
        return self


class _FakePP:
    @classmethod
    def from_config(cls, cfg):
        pp = cls()
        pp.topic_modeler = _FakeTM(_TMP / "bertopic")
        return pp

    def run(self, chain, df):
        return df


_TMP = None  # populated in the fixture


@pytest.fixture
def run_fit(monkeypatch, tmp_path):
    global _TMP
    _TMP = tmp_path
    data_dir = tmp_path / "data"
    silver = data_dir / "a_shares" / "news_silver"
    silver.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "title": ["x"], "body": ["y"],
        "aligned_date": pd.to_datetime(["2024-01-01"]),
    }).to_parquet(silver / "000001.parquet", index=False)

    def _fake_load_config(config_path=None):
        return OmegaConf.create({
            "project": {
                "data_dir": str(data_dir),
                "model_dir": str(tmp_path / "models"),
            },
            "preprocessing": {
                "text": {"topic_model": {
                    "enabled": True,
                    "model_cache_dir": str(tmp_path / "bertopic"),
                }},
            },
        })

    def _fake_discover(data_dir, source):
        return list(_ALL_CODES)

    def _fake_collect(data_dir, source, codes, cutoff):
        return pd.DataFrame({
            "title": ["a", "b", "c"],
            "body": ["x", "y", "z"],
            "aligned_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "stock_code": ["000001", "000002", "000003"],
        })

    monkeypatch.setattr(mod, "load_config", _fake_load_config)
    monkeypatch.setattr(mod, "_discover_stocks", _fake_discover)
    monkeypatch.setattr(mod, "_collect_silver", _fake_collect)
    monkeypatch.setattr(mod, "PreprocessingPipeline", _FakePP)

    sys.argv = [
        "fit_topic_model.py", "--source", "news", "--cutoff", "2024-12-31",
        "--stocks", "2", "--seed", "7",
    ]
    mod.main()
    manifest_path = tmp_path / "models" / "topic" / "topic_fit_news.json"
    assert manifest_path.exists(), "run manifest not written"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class TestFitManifestReproducibility:
    def test_all_reproducibility_fields_present(self, run_fit):
        m = run_fit
        for key in (
            "source", "corpus_cutoff", "corpus_hash", "n_topics", "n_posts",
            "fit_date",
            "model_pickle_sha256", "corpus_files", "corpus_file_hash",
            "embedding_model", "embedding_model_used", "min_topic_size",
            "seed", "bertopic_version", "sentence_transformers_version",
            "python_version", "sampling_method", "stock_codes",
            "config_hash", "calendar_hash",
        ):
            assert key in m, f"manifest missing {key}"

    def test_legacy_fields_still_present(self, run_fit):
        m = run_fit
        assert m["source"] == "news"
        assert m["corpus_cutoff"] == "2024-12-31"
        assert m["corpus_hash"] == "abcd1234abcd1234"
        assert m["n_topics"] == 3
        assert m["n_posts"] == 3

    def test_sampling_and_seed_recorded(self, run_fit):
        m = run_fit
        assert m["seed"] == 7
        assert m["sampling_method"] == "deterministic_equidistant_sorted"
        assert m["stock_codes"] == ["000001", "000003"]
        assert m["min_topic_size"] == 50
        assert m["embedding_model"] == "finbert"
        assert m["embedding_model_used"] == "finbert"

    def test_pickle_sha256_and_corpus_file_hash(self, run_fit):
        m = run_fit
        # fake tm wrote no pickle → sha recorded as unknown.
        assert m["model_pickle_sha256"] == "unknown"
        # exactly the sampled codes' silver parquet files are hashed.
        assert len(m["corpus_files"]) == 1
        assert m["corpus_files"][0]["file"].endswith("000001.parquet")
        assert len(m["corpus_files"][0]["sha256"]) == 64
        assert m["corpus_file_hash"]

    def test_version_and_hash_fields(self, run_fit):
        m = run_fit
        assert m["python_version"]
        assert m["config_hash"] and m["calendar_hash"]
        assert isinstance(m["bertopic_version"], str)
        assert isinstance(m["sentence_transformers_version"], str)

    def test_seed_and_stock_codes_passed_to_fit(self, run_fit):
        assert _FakeTM.last_fit_kwargs["seed"] == 7
        assert _FakeTM.last_fit_kwargs["stock_codes"] == ["000001", "000003"]
