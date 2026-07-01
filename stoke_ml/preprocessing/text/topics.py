"""BERTopic topic modeler: per-post topic assignment via FinBERT embeddings.

Fits once on cross-stock data (fit), then assigns topic_id and
topic_probability per post (transform).  Model is cached to disk for reuse.

Gracefully degrades to no-op when bertopic/umap/hdbscan are not installed.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        enabled: bool = True,
        n_topics: str | int = "auto",
        min_topic_size: int = 50,
        model_cache_dir: str = "models/bertopic",
        embedding_model: str = "finbert",
    ):
        self.enabled = enabled
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model_cache_dir = model_cache_dir
        self.embedding_model = embedding_model

        self._model = None
        self._enabled = enabled and self._check_deps()
        if self._enabled:
            os.makedirs(self.model_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df, **kwargs):
        """Train BERTopic on *df* and cache to disk.

        Keyword Args:
            source: Name used in cache filename (e.g. ``"news"``, ``"guba"``).
            force_retrain: If True, ignore cached model.
        """
        if df.empty or not self._enabled:
            return self

        source = kwargs.get("source", "default")
        force_retrain = kwargs.get("force_retrain", False)
        cache_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}.pkl"
        )

        # Try loading from cache
        if not force_retrain and os.path.exists(cache_path):
            try:
                import joblib
                self._model = joblib.load(cache_path)
                logger.info("Loaded cached BERTopic model from %s", cache_path)
                return self
            except Exception:
                logger.warning(
                    "Corrupted BERTopic cache at %s, will retrain", cache_path
                )

        texts = self._build_texts(df)
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
                random_state=42,
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
            self._save_metadata(source, n_found, len(texts))

        except Exception as e:
            logger.warning("BERTopic training failed: %s", e)
            self._enabled = False

        return self

    def transform(self, df, **kwargs):
        """Assign topic_id and topic_probability columns."""
        if df.empty or not self._enabled or self._model is None:
            return df

        df = df.copy()
        texts = self._build_texts(df)

        try:
            topics, probs = self._model.transform(texts)
            df["topic_id"] = topics.astype("int16")
            df["topic_probability"] = probs.astype(np.float32)
        except Exception as e:
            logger.warning("Topic transform failed: %s", e)
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

    def _get_embeddings(self, texts: list[str]):
        """Produce document embeddings for BERTopic.

        Returns None if no embedding method is available.
        """
        if self.embedding_model == "finbert":
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(
                    "yiyanghkust/finbert-tone-chinese",
                    cache_folder=self.model_cache_dir,
                )
                logger.info(
                    "Computing FinBERT embeddings for %d texts...", len(texts)
                )
                return model.encode(
                    texts,
                    show_progress_bar=True,
                    batch_size=32,
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
            vectorizer = CountVectorizer(max_features=5000)
            return vectorizer.fit_transform(tokenized)
        except Exception as e:
            logger.warning("TF-IDF fallback also failed: %s", e)
            return None

    def _save_metadata(
        self, source: str, n_topics: int, n_docs: int
    ) -> None:
        meta_path = os.path.join(
            self.model_cache_dir, f"bertopic_{source}_meta.json"
        )
        meta = {
            "source": source,
            "n_topics_found": n_topics,
            "n_docs_trained": n_docs,
            "min_topic_size": self.min_topic_size,
            "embedding_model": self.embedding_model,
            "training_date": pd.Timestamp.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
