"""BERTopic topic modeler: per-post topic assignment via FinBERT embeddings.

Fits once on cross-stock data (fit), then assigns topic_id and
topic_probability per post (transform).  Model is cached to disk for reuse.

Gracefully degrades to no-op when bertopic/umap/hdbscan are not installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep
from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)


def dependency_versions() -> dict[str, str]:
    """Runtime + dependency versions for the topic stack (§十三).

    Best-effort ``importlib.metadata`` lookups; a distribution that is not
    installed (or an import that fails) is reported as ``"unknown"`` so a fit
    manifest is always complete.  The ``"python"`` entry is the interpreter
    version.
    """
    from importlib import metadata

    out = {}
    for dist in ("bertopic", "sentence-transformers", "umap", "hdbscan"):
        try:
            out[dist] = metadata.version(dist)
        except Exception:
            out[dist] = "unknown"
    out["python"] = platform.python_version()
    return out


def _sha256_file(path: str) -> str:
    """Full SHA-256 hex digest of a file, or ``"unknown"`` when unreadable.

    Used for the persisted model pickle (``model_pickle_sha256``) so a cached
    artifact can be verified byte-for-byte and a corrupt pickle surfaces
    (§十三).  Returns ``"unknown"`` for a missing/unreadable file rather than
    raising, so metadata writing never crashes the fit.
    """
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"


def _truncate_to_cutoff(df, corpus_cutoff=None):
    """Keep only rows at or before *corpus_cutoff* (PIT fold discipline).

    The topic representation space must never see documents after a fold's
    training cutoff.  Accepts either a ``date`` column or
    a DatetimeIndex.
    """
    if corpus_cutoff is None:
        return df
    cutoff = pd.Timestamp(corpus_cutoff)
    if "date" in getattr(df, "columns", ()):
        return df[pd.to_datetime(df["date"]) <= cutoff].copy()
    if isinstance(getattr(df, "index", None), pd.DatetimeIndex):
        return df[df.index <= cutoff].copy()
    return df


class TopicModeler(PreprocessingStep):
    """BERTopic topic modeler with FinBERT embeddings.

    Trained on all available posts (cross-stock) to discover a global
    topic space, then applied per-stock to assign topic IDs.

    Parameters
    ----------
    enabled:
        If False, the step is a no-op pass-through.
    n_topics:
        Number of topics for BERTopic.  ``"auto"`` lets HDBSCAN decide.
    min_topic_size:
        Minimum cluster size for HDBSCAN (controls topic granularity).
    model_cache_dir:
        Directory for cached BERTopic models and metadata JSON.
    embedding_model:
        ``"finbert"`` for sentence-transformers, ``"tfidf"`` for jieba+CountVectorizer.
    seed:
        Random seed for UMAP (deterministic reproducibility).  Also recorded in
        the cache metadata / fit manifest (§十三).
    """

    # Fit ONCE on a pinned corpus (see corpus_cutoff) and frozen to disk; a
    # formal offline pass must consume the pinned artifact, never re-fit on the
    # full window (§十-1, §十-2).
    fit_scope = "global_frozen"

    def __init__(
        self,
        enabled: bool = True,
        n_topics: str | int = "auto",
        min_topic_size: int = 50,
        model_cache_dir: str = "models/bertopic",
        embedding_model: str = "finbert",
        corpus_cutoff: str | None = None,
        seed: int = 42,
    ):
        self.enabled = enabled
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model_cache_dir = model_cache_dir
        self.embedding_model = embedding_model
        self.seed = int(seed)
        # Pinned production corpus cutoff (TRAIN_END).  When set, transform()
        # auto-loads the cached model fit on this corpus instead of running
        # unfitted, and never silently drops the topic features (§十-2).
        self.corpus_cutoff = corpus_cutoff
        # SHA-1 (first 16 hex) of the training corpus text, recorded in cache
        # metadata so a model is only reused when the corpus CONTENT matches.
        self.corpus_hash_ = None
        # Stock codes that contributed to the training corpus, recorded in cache
        # metadata + the fit manifest for reproducibility (§十三).
        self._stock_codes: list[str] | None = None

        self._model = None
        self._finbert_model = None
        self._tfidf_vectorizer = None
        self._enabled = enabled and self._check_deps()
        if self._enabled:
            os.makedirs(self.model_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df, **kwargs):
        """Train BERTopic on *df* and cache to disk.

        The training corpus is truncated to ``corpus_cutoff`` (inclusive) when
        given, so a fold fits ONLY on corpus up to its training cutoff — the
        topic representation never sees later vocabulary/documents
        The effective corpus end date is encoded into the
        cache filename, so two runs fit on different corpora can never silently
        reuse each other's model.

        Keyword Args:
            source: Name used in cache filename (e.g. ``"news"``, ``"guba"``).
            force_retrain: If True, ignore cached model.
            corpus_cutoff: str/date — fit only on rows with date <= cutoff.
        """
        if df.empty or not self._enabled:
            self._record_fit_range(df)
            return self

        source = kwargs.get("source", "default")
        force_retrain = kwargs.get("force_retrain", False)
        corpus_cutoff = kwargs.get("corpus_cutoff", self.corpus_cutoff)
        seed = int(kwargs.get("seed", self.seed))
        self.seed = seed
        self._stock_codes = kwargs.get("stock_codes")

        fit_df = _truncate_to_cutoff(df, corpus_cutoff)
        self._record_fit_range(fit_df)
        if len(fit_df) < self.min_topic_size:
            logger.warning(
                "Only %d rows (min_topic_size=%d) after cutoff %s, "
                "disabling topic modeler",
                len(fit_df), self.min_topic_size, corpus_cutoff or "none",
            )
            self._enabled = False
            return self

        # Corpus version key → cache identity.
        corpus_key = self._corpus_key(corpus_cutoff)
        self.corpus_hash_ = self._corpus_hash(fit_df)
        cache_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_cutoff_{corpus_key}.pkl"
        )

        # Try loading from cache — reuse ONLY when the cached model was fit on
        # identical corpus CONTENT.  A same-cutoff corpus that changed (edited
        # / re-downloaded posts) must force a retrain, never silently reuse the
        # stale representation (§十-2).
        if not force_retrain and os.path.exists(cache_path):
            cached_hash = self._read_meta(source, corpus_key).get("corpus_hash")
            if cached_hash != self.corpus_hash_:
                logger.info(
                    "Cached model at %s was fit on different corpus content "
                    "(cached=%s, current=%s), retraining",
                    cache_path, cached_hash, self.corpus_hash_,
                )
            else:
                try:
                    import joblib
                    self._model = joblib.load(cache_path)
                    self._restore_embedder(source, corpus_key)
                    logger.info("Loaded cached BERTopic model from %s", cache_path)
                    return self
                except Exception as exc:
                    logger.warning(
                        "Corrupted BERTopic cache at %s (category=%s), will retrain",
                        cache_path, classify_error(exc).value,
                    )

        texts = self._build_texts(fit_df)
        if len(texts) < self.min_topic_size:
            logger.warning(
                "Only %d texts (min_topic_size=%d), disabling topic modeler",
                len(texts), self.min_topic_size,
            )
            self._enabled = False
            return self

        embeddings = self._get_embeddings(texts)
        if embeddings is None:
            self._enabled = False
            return self

        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from umap import UMAP

            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=self.seed,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric="euclidean",
                prediction_data=True,
            )
            nr_topics = None if self.n_topics == "auto" else int(self.n_topics)

            self._model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=None,
                nr_topics=nr_topics,
                calculate_probabilities=True,
                verbose=True,
            )
            self._model.fit(texts, embeddings=embeddings)

            n_found = len(self._model.get_topic_info())
            logger.info("BERTopic trained: %d topics from %d texts", n_found, len(texts))

            # Persist
            import joblib
            joblib.dump(self._model, cache_path)
            self._save_metadata(
                source, corpus_key, n_found, len(texts), self.corpus_hash_
            )

        except Exception as e:
            logger.warning(
                "BERTopic training failed (category=%s): %s",
                classify_error(e).value, e,
            )
            self._enabled = False

        return self

    def transform(self, df, *, source="default", formal: bool = False, **kwargs):
        """Assign topic_id and topic_probability columns.

        Production transform needs a fitted model.  When none is loaded yet
        and a pinned ``corpus_cutoff`` is configured, the matching cached
        model is auto-loaded from disk.  When neither exists, the call RAISES
        instead of silently dropping the topic features — the old behavior
        meant the topic columns could be absent from production data with no
        error at all (§十-2).

        ``formal`` selects failure semantics for the per-post assignment step:

          * ``formal=False`` (default, offline/research) — a transform failure
            is logged and the topic columns degrade to ``topic_id=-1`` /
            ``topic_probability=0`` so downstream processing keeps running.
          * ``formal=True`` — a transform failure RAISES instead of silently
            degrading the whole channel to a constant ``-1``/``0``, which the
            old behavior hid as if the model had worked (§二十一 A4).
        """
        if df.empty or not self._enabled:
            return df
        if self._model is None:
            self._auto_load(source)
        if self._model is None:
            raise RuntimeError(
                "TopicModeler.transform() has no fitted model to assign topics "
                "(pinned corpus_cutoff=%s). Run "
                "`python scripts/production/fit_topic_model.py --source %s --cutoff TRAIN_END` "
                "and set preprocessing.text.topic_model.corpus_cutoff, or the "
                "topic_* features are silently absent."
                % (self.corpus_cutoff or "unset", source)
            )

        df = df.copy()
        texts = self._build_texts(df)

        try:
            from hdbscan.prediction import approximate_predict

            embeddings = self._get_embeddings(texts)
            if embeddings is None:
                raise RuntimeError("Failed to produce embeddings")
            # Manually run UMAP → HDBSCAN (bypasses BERTopic 0.17.4
            # transform() bug with pre-computed embeddings).
            reduced = self._model.umap_model.transform(embeddings)
            topics, probs = approximate_predict(
                self._model.hdbscan_model, reduced,
            )
            df["topic_id"] = np.asarray(topics, dtype="int16")
            df["topic_probability"] = np.asarray(probs, dtype=np.float32)
        except Exception as e:
            if formal:
                raise
            logger.warning(
                "Topic transform failed (category=%s): %s",
                classify_error(e).value, e,
            )
            df["topic_id"] = -1
            df["topic_probability"] = np.float32(0.0)

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_deps(self) -> bool:
        try:
            import bertopic  # noqa: F401
            import umap  # noqa: F401
            import hdbscan  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "BERTopic/UMAP/HDBSCAN not installed — TopicModeler disabled"
            )
            return False

    def _build_texts(self, df: pd.DataFrame) -> list[str]:
        """Concatenate title + body into a single text per row."""
        texts = []
        has_title = "title" in df.columns
        has_body = "body" in df.columns

        for i in range(len(df)):
            parts = []
            if has_title:
                t = df.iloc[i]["title"]
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            if has_body:
                b = df.iloc[i]["body"]
                if isinstance(b, str) and b.strip():
                    parts.append(b.strip())
            texts.append(" ".join(parts) if parts else "")

        return texts

    def _corpus_hash(self, df: pd.DataFrame) -> str:
        """Deterministic SHA-1 digest (first 16 hex) of the corpus text.

        Recorded in cache metadata so a cached model is reused only when the
        training corpus CONTENT is unchanged — an edited corpus with the same
        cutoff date can no longer silently reuse a stale model (§十-2).
        """
        texts = self._build_texts(df)
        digest = "\n".join(texts).encode("utf-8")
        return hashlib.sha1(digest).hexdigest()[:16]

    def _read_meta(self, source: str, corpus_key: str) -> dict:
        """Read cached-model metadata, or an empty dict when unavailable."""
        meta_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_cutoff_{corpus_key}_meta.json"
        )
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Cannot read BERTopic meta %s (category=%s)",
                meta_path, classify_error(exc).value,
            )
            return {}

    def _auto_load(self, source: str) -> None:
        """Load the cached model pinned by ``corpus_cutoff``, if configured.

        Called from transform() when no model is fitted.  With a pinned cutoff
        this loads the production artifact fit on TRAIN_END; without one the
        model stays unfitted and transform() raises so the missing topic
        features are surfaced instead of silently dropped (§十-2).
        """
        if self.corpus_cutoff is None:
            logger.error(
                "TopicModeler has no fitted model and no pinned corpus_cutoff "
                "(preprocessing.text.topic_model.corpus_cutoff); "
                "topic features will not be generated."
            )
            return
        corpus_key = self._corpus_key(self.corpus_cutoff)
        cache_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_cutoff_{corpus_key}.pkl"
        )
        if not os.path.exists(cache_path):
            logger.error(
                "Pinned topic cache %s not found. Run "
                "`python scripts/production/fit_topic_model.py --source %s --cutoff %s`.",
                cache_path, source, self.corpus_cutoff,
            )
            return
        try:
            import joblib
            self._model = joblib.load(cache_path)
            self._restore_embedder(source, corpus_key)
            logger.info("Auto-loaded pinned BERTopic model from %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to auto-load pinned BERTopic cache %s (category=%s): %s",
                cache_path, classify_error(exc).value, exc,
            )

    def _get_embeddings(self, texts: list[str]):
        """Produce document embeddings for BERTopic.

        Caches the embedding model on first call (during fit) and reuses it
        on subsequent calls (during per-stock transform) to guarantee
        consistent embedding dimensionality.
        """
        import os
        import torch

        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)

        # Already cached: reuse to guarantee dimension consistency
        if self._finbert_model is not None:
            try:
                logger.info(
                    "Computing FinBERT embeddings for %d texts "
                    "(batch_size=128, threads=%d)...",
                    len(texts), cpu_count,
                )
                return self._finbert_model.encode(
                    texts, show_progress_bar=False, batch_size=128,
                )
            except Exception as e:
                logger.warning("FinBERT embeddings failed during transform: %s", e)
                return None

        if self._tfidf_vectorizer is not None:
            try:
                import jieba
                tokenized = [" ".join(jieba.cut(t)) for t in texts]
                return self._tfidf_vectorizer.transform(tokenized)
            except Exception as e:
                logger.warning("TF-IDF transform failed: %s", e)
                return None

        # First call (during fit): determine and cache the embedding method
        if self.embedding_model == "finbert":
            try:
                from sentence_transformers import SentenceTransformer
                # Try local-only first to avoid HF timeouts when model is cached
                try:
                    self._finbert_model = SentenceTransformer(
                        "yiyanghkust/finbert-tone-chinese",
                        cache_folder=self.model_cache_dir,
                        local_files_only=True,
                    )
                except Exception:
                    self._finbert_model = SentenceTransformer(
                        "yiyanghkust/finbert-tone-chinese",
                        cache_folder=self.model_cache_dir,
                    )
                logger.info(
                    "Computing FinBERT embeddings for %d texts "
                    "(batch_size=128, threads=%d)...",
                    len(texts), cpu_count,
                )
                return self._finbert_model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=128,
                )
            except Exception as e:
                logger.warning(
                    "FinBERT embeddings unavailable (%s), falling back to TF-IDF", e
                )

        # TF-IDF fallback (pre-tokenize to avoid deprecated sklearn tokenizer param)
        try:
            import jieba
            from sklearn.feature_extraction.text import CountVectorizer

            tokenized = [" ".join(jieba.cut(t)) for t in texts]
            self._tfidf_vectorizer = CountVectorizer(max_features=5000)
            return self._tfidf_vectorizer.fit_transform(tokenized)
        except Exception as e:
            logger.warning("TF-IDF fallback also failed: %s", e)
            return None

    def _save_metadata(
        self, source: str, corpus_key: str, n_topics: int, n_docs: int,
        corpus_hash: str,
    ) -> None:
        meta_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_cutoff_{corpus_key}_meta.json"
        )
        cache_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_cutoff_{corpus_key}.pkl"
        )
        used_embedding = "finbert" if self._finbert_model is not None else "tfidf"
        versions = dependency_versions()
        meta = {
            "source": source,
            "corpus_cutoff": corpus_key,
            "corpus_hash": corpus_hash,
            "n_topics_found": n_topics,
            "n_docs_trained": n_docs,
            "corpus_date_min": (self.fit_start.strftime("%Y-%m-%d")
                                if self.fit_start is not None else None),
            "corpus_date_max": (self.fit_end.strftime("%Y-%m-%d")
                                if self.fit_end is not None else None),
            "min_topic_size": self.min_topic_size,
            "seed": self.seed,
            "embedding_model": used_embedding,
            "embedding_model_config": self.embedding_model,
            "bertopic_version": versions["bertopic"],
            "sentence_transformers_version": versions["sentence-transformers"],
            "python_version": versions["python"],
            "stock_codes": list(self._stock_codes) if self._stock_codes else None,
            "model_pickle_sha256": _sha256_file(cache_path),
            "training_date": pd.Timestamp.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _corpus_key(self, corpus_cutoff=None) -> str:
        """Stable cache-key fragment for the training corpus version.

        Uses the explicit cutoff when given, else the corpus end date recorded
        by ``_record_fit_range``.  Either way the key changes whenever the
        training corpus's end changes, so a model fit on full history can never
        be silently reused for a fold that must stop earlier — the fix for the
        representation-space leak.
        """
        if corpus_cutoff is not None:
            return pd.Timestamp(corpus_cutoff).strftime("%Y-%m-%d")
        if self.fit_end is not None:
            return pd.Timestamp(self.fit_end).strftime("%Y-%m-%d")
        return "unknown"

    def _restore_embedder(self, source: str, corpus_key: str) -> None:
        """Pre-load the correct embedding model to match cached BERTopic.

        Called after loading a cached BERTopic model from disk to ensure
        ``_get_embeddings()`` uses the same embedding type the model was
        trained with.
        """
        meta_path = os.path.join(
            self.model_cache_dir,
            f"bertopic_{source}_cutoff_{corpus_key}_meta.json",
        )
        used_embedding = self.embedding_model  # default
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                used_embedding = meta.get("embedding_model", self.embedding_model)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Cannot read BERTopic meta %s (category=%s)",
                    meta_path, classify_error(exc).value,
                )

        if used_embedding == "finbert":
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._finbert_model = SentenceTransformer(
                        "yiyanghkust/finbert-tone-chinese",
                        cache_folder=self.model_cache_dir,
                        local_files_only=True,
                    )
                except Exception:
                    self._finbert_model = SentenceTransformer(
                        "yiyanghkust/finbert-tone-chinese",
                        cache_folder=self.model_cache_dir,
                    )
                logger.info("Restored FinBERT embedder to match cached model")
            except Exception as e:
                logger.warning("Cannot restore FinBERT embedder: %s", e)
        else:
            logger.warning(
                "Cached model used TF-IDF embeddings, but restore requires "
                "re-fitting the vectorizer. Use force_retrain=True."
            )
