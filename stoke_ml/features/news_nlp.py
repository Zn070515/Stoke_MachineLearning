"""News sentiment analysis for Chinese financial text.

L1: SnowNLP (offline, fast, good Chinese support).
L2: FinBERT Chinese (planned — needs HuggingFace access or local mirror).

Pipeline:
  raw news → compute_sentiment() → raw news with scores
  → PIT-align via NewsStorage.bronze_to_silver()
  → aggregate_daily_sentiment() → daily features
  → NewsStorage.silver_to_gold() → Gold layer with ZI filling
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A-share market close time (CST)
MARKET_CLOSE_HOUR = 15


class NewsSentimentAnalyzer:
    """Chinese financial news sentiment using SnowNLP (L1)."""

    def __init__(self):
        self._loaded = False
        self._model = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if self._try_snownlp():
            logger.info("Sentiment: using SnowNLP (offline Chinese NLP)")
            self._model = "snownlp"
            return

        logger.warning("No sentiment model available; scores will be 0")
        self._model = None

    @staticmethod
    def _try_snownlp() -> bool:
        try:
            from snownlp import SnowNLP  # noqa: F401
            return True
        except ImportError:
            return False

    def analyze(self, texts: list[str]) -> np.ndarray:
        """Return sentiment scores [-1, 1] for each text."""
        if not texts:
            return np.array([], dtype=np.float32)
        self._ensure_loaded()
        if self._model == "snownlp":
            return self._snownlp_sentiment(texts)
        return np.zeros(len(texts), dtype=np.float32)

    @staticmethod
    def _snownlp_sentiment(texts: list[str]) -> np.ndarray:
        try:
            from snownlp import SnowNLP
            scores = []
            for t in texts:
                s = SnowNLP(t)
                scores.append(s.sentiments * 2 - 1)
            return np.array(scores, dtype=np.float32)
        except ImportError:
            logger.warning("SnowNLP not installed, returning zeros")
            return np.zeros(len(texts), dtype=np.float32)


def compute_raw_sentiment(
    df: pd.DataFrame,
    analyzer: NewsSentimentAnalyzer | None = None,
) -> pd.DataFrame:
    """Score titles (and bodies if present) in a raw news DataFrame.

    Adds columns: sentiment_title, sentiment_body (if body present).
    """
    if df.empty:
        return df

    if analyzer is None:
        analyzer = NewsSentimentAnalyzer()

    df = df.copy()
    df["sentiment_title"] = analyzer.analyze(df["title"].tolist())

    if "body" in df.columns:
        non_null = df["body"].notna() & (df["body"] != "")
        bodies = df.loc[non_null, "body"].tolist()
        if bodies:
            df.loc[non_null, "sentiment_body"] = analyzer.analyze(bodies)
        else:
            df["sentiment_body"] = np.nan
        df["sentiment_body"] = df["sentiment_body"].astype(np.float32)

    df["sentiment_title"] = df["sentiment_title"].astype(np.float32)
    return df


def aggregate_daily_sentiment(
    titles: list[str],
    analyzer: NewsSentimentAnalyzer | None = None,
) -> dict:
    """Compute daily aggregate sentiment features from news titles.

    Returns dict with: sentiment_mean, sentiment_std, news_count,
    positive_ratio, negative_ratio.
    """
    if not titles:
        return {
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "news_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
        }

    if analyzer is not None:
        scores = analyzer.analyze(titles)
    else:
        scores = np.zeros(len(titles), dtype=np.float32)

    n = len(titles)
    pos = float((scores > 0.2).sum())
    neg = float((scores < -0.2).sum())

    return {
        "sentiment_mean": float(scores.mean()),
        "sentiment_std": float(scores.std()),
        "news_count": n,
        "positive_ratio": pos / n if n > 0 else 0.0,
        "negative_ratio": neg / n if n > 0 else 0.0,
    }


def build_sentiment_dataframe(
    daily_scores: dict[str, list],
    stock_code: str,
) -> pd.DataFrame:
    """Convert dict-of-lists from batch aggregation to a DataFrame
    suitable for NewsStorage.save_daily_sentiment().
    """
    n = len(next(iter(daily_scores.values()), []))
    if n == 0:
        return pd.DataFrame()

    df = pd.DataFrame(daily_scores)
    df["stock_code"] = stock_code
    df["has_news"] = df["news_count"] > 0
    df["sentiment_mean"] = df["sentiment_mean"].astype(np.float32)
    df["sentiment_std"] = df["sentiment_std"].astype(np.float32)
    df["news_count"] = df["news_count"].astype("int16")
    df["positive_ratio"] = df["positive_ratio"].astype(np.float32)
    df["negative_ratio"] = df["negative_ratio"].astype(np.float32)
    return df
