"""Tests for the TopicModeler corpus-scope discipline.

The topic representation space must never see documents after a fold's
training cutoff.  We verify:
  * _truncate_to_cutoff keeps only rows at/before the cutoff;
  * the cache filename encodes the corpus version, so a model fit on a
    truncated corpus can never be confused with a full-history model;
  * fitting with two different cutoffs writes two distinct cache files.
"""

import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest

from stoke_ml.preprocessing.text.topics import (
    TopicModeler,
    _truncate_to_cutoff,
)


class _FakeHDBSCAN:
    def __init__(self, **kwargs):
        pass


class _FakeUMAP:
    def __init__(self, **kwargs):
        pass


class _FakeBERTopic:
    def __init__(self, **kwargs):
        pass

    def fit(self, texts, embeddings=None):
        pass

    def get_topic_info(self):
        return pd.DataFrame({"Topic": [-1, 0, 1]})


@pytest.fixture
def fake_topic_deps(monkeypatch):
    """Stub bertopic/hdbscan/umap so fit() needs no real training or model."""
    for name, stub in (
        ("bertopic", types.SimpleNamespace(BERTopic=_FakeBERTopic)),
        ("hdbscan", types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN)),
        ("umap", types.SimpleNamespace(UMAP=_FakeUMAP)),
    ):
        monkeypatch.setitem(sys.modules, name, stub)


def _make_topic_modeler(tmp_path, monkeypatch):
    tm = TopicModeler(
        enabled=True,
        min_topic_size=2,
        model_cache_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        tm, "_get_embeddings", lambda texts: np.zeros((len(texts), 8))
    )
    # _restore_embedder would load FinBERT over the network when a cached
    # model is reloaded — not needed for these unit tests.
    monkeypatch.setattr(tm, "_restore_embedder", lambda source, key: None)
    return tm


@pytest.fixture
def posts():
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-02", "2024-01-15", "2024-02-03", "2024-03-04"]
        ),
        "title": ["a", "b", "c", "d"],
        "body": ["x", "y", "z", "w"],
    })


# ---------------------------------------------------------------------------
# _truncate_to_cutoff
# ---------------------------------------------------------------------------

def test_truncate_to_cutoff_date_column(posts):
    out = _truncate_to_cutoff(posts, "2024-01-31")
    assert list(out["date"]) == list(posts["date"][:2])
    assert len(out) == 2


def test_truncate_to_cutoff_datetime_index(posts):
    df = posts.set_index("date")
    out = _truncate_to_cutoff(df, "2024-01-31")
    assert len(out) == 2
    assert out.index.max() <= pd.Timestamp("2024-01-31")


def test_truncate_to_cutoff_none_is_noop(posts):
    out = _truncate_to_cutoff(posts, None)
    assert out is posts


# ---------------------------------------------------------------------------
# Cache identity encodes the corpus version
# ---------------------------------------------------------------------------

def test_corpus_key_uses_explicit_cutoff(tmp_path, monkeypatch):
    tm = _make_topic_modeler(tmp_path, monkeypatch)
    assert tm._corpus_key("2024-01-31") == "2024-01-31"


def test_corpus_key_falls_back_to_fit_end(tmp_path, monkeypatch):
    tm = _make_topic_modeler(tmp_path, monkeypatch)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-02-03"]),
        "title": ["a", "b"],
        "body": ["x", "y"],
    })
    tm._record_fit_range(df)
    assert tm._corpus_key(None) == "2024-02-03"


def test_fit_with_cutoff_writes_cutoff_named_cache(
    fake_topic_deps, tmp_path, monkeypatch, posts
):
    dumped = {}

    def fake_dump(model, path):
        dumped[path] = model

    monkeypatch.setattr(joblib, "dump", fake_dump)
    tm = _make_topic_modeler(tmp_path, monkeypatch)

    tm.fit(posts.copy(), source="news", corpus_cutoff="2024-01-31")

    assert len(dumped) == 1
    path = next(iter(dumped))
    assert path.endswith("bertopic_news_cutoff_2024-01-31.pkl")
    # The model was fit on the truncated corpus, not full history.
    assert tm.fit_end == pd.Timestamp("2024-01-15")


def test_different_cutoffs_never_share_cache(
    fake_topic_deps, tmp_path, monkeypatch, posts
):
    """Two fold models with different cutoffs must use distinct cache files,
    so a fold fit can never silently reuse another fold's (or the full
    corpus's) model."""
    dumped = {}

    def fake_dump(model, path):
        dumped[path] = model

    monkeypatch.setattr(joblib, "dump", fake_dump)

    tm1 = _make_topic_modeler(tmp_path, monkeypatch)
    tm1.fit(posts.copy(), source="news", corpus_cutoff="2024-01-31")

    tm2 = _make_topic_modeler(tmp_path, monkeypatch)
    tm2.fit(posts.copy(), source="news", corpus_cutoff="2024-02-29")

    tm3 = _make_topic_modeler(tmp_path, monkeypatch)
    tm3.fit(posts.copy(), source="news")  # full corpus

    assert len(dumped) == 3
    names = sorted(p.split("\\")[-1] for p in dumped)
    assert names == [
        "bertopic_news_cutoff_2024-01-31.pkl",
        "bertopic_news_cutoff_2024-02-29.pkl",
        "bertopic_news_cutoff_2024-03-04.pkl",
    ]


def test_cached_fold_model_is_reused_for_same_cutoff(
    fake_topic_deps, tmp_path, monkeypatch, posts
):
    """Same cutoff → same cache path, so the second fit loads from disk
    instead of retraining (and never silently reuses a different corpus)."""
    dumped = {}
    loaded = {}

    def fake_dump(model, path):
        dumped[path] = model
        with open(path, "wb") as f:   # make the cache exist on disk
            f.write(b"dummy")

    def fake_load(path):
        loaded[path] = True
        return _FakeBERTopic()

    monkeypatch.setattr(joblib, "dump", fake_dump)
    monkeypatch.setattr(joblib, "load", fake_load)

    tm1 = _make_topic_modeler(tmp_path, monkeypatch)
    tm1.fit(posts.copy(), source="news", corpus_cutoff="2024-01-31")

    tm2 = _make_topic_modeler(tmp_path, monkeypatch)
    tm2.fit(posts.copy(), source="news", corpus_cutoff="2024-01-31")

    assert len(dumped) == 1          # trained once
    path = next(iter(dumped))
    assert path in loaded            # second fit reloaded from disk
    assert tm2._model is not None
