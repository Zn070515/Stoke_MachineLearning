"""News sentiment analysis for Chinese financial text.

L1: SnowNLP (general Chinese NLP, offline).
L2: Financial lexicon + SnowNLP hybrid (domain-aware, offline).
L3: FinBERT Chinese (planned — needs accessible model mirror).

Pipeline:
  raw news → compute_sentiment() → raw news with scores
  → PIT-align via NewsStorage.bronze_to_silver()
  → aggregate_daily_sentiment() → daily features
  → NewsStorage.silver_to_gold() → Gold layer with ZI filling
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A-share market close time (CST)
MARKET_CLOSE_HOUR = 15

# Chinese financial sentiment lexicon — domain-specific positive/negative
# terms that carry strong signal in A-share news context.
_FIN_POSITIVE = frozenset([
    "利好", "大涨", "暴涨", "飙升", "涨停", "突破", "新高",
    "增长", "增速", "提速", "回暖", "复苏", "反弹", "走强",
    "买入", "增持", "看好", "超预期", "优于预期", "业绩预增",
    "分红", "回购", "高送转", "派息", "股息",
    "盈利", "扭亏", "净利润增长", "营收增长", "毛利率提升",
    "中标", "签约", "订单", "扩产", "投产",
    "创新高", "历史新高", "领涨", "龙头", "标杆",
    "政策利好", "扶持", "补贴", "减税", "降准",
    "资金流入", "北向增持", "机构调研", "主力加仓",
])

_FIN_NEGATIVE = frozenset([
    "利空", "大跌", "暴跌", "跳水", "跌停", "破位", "新低",
    "下滑", "萎缩", "衰退", "低迷", "疲软", "走弱", "承压",
    "卖出", "减持", "清仓", "看空", "低于预期", "业绩预亏",
    "亏损", "净亏损", "净利润下滑", "营收下降", "毛利率下降",
    "违约", "处罚", "调查", "诉讼", "退市", "ST",
    "质押", "爆仓", "平仓", "强制平仓", "追加保证金",
    "资金流出", "北向减持", "主力出逃", "机构减仓",
    "政策利空", "监管", "整顿", "去杠杆", "收紧",
])


class NewsSentimentAnalyzer:
    """Chinese financial news sentiment (SnowNLP + financial lexicon hybrid).

    Uses SnowNLP as base signal, blended with financial lexicon ratio
    for domain-aware scoring.
    """

    def __init__(self):
        self._loaded = False
        self._model = None
        self._has_snownlp = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        self._has_snownlp = self._try_snownlp()

        if self._has_snownlp:
            self._model = "hybrid"
            logger.info("Sentiment: hybrid (SnowNLP + financial lexicon)")
        else:
            self._model = "lexicon"
            logger.info("Sentiment: financial lexicon only (no SnowNLP)")

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

        lex_scores = np.array(
            [self._lexicon_score(t) for t in texts], dtype=np.float32
        )

        if self._has_snownlp:
            snow_scores = self._snownlp_sentiment(texts)
            # Blend: lexicon is more domain-precise, SnowNLP provides nuance
            return (0.55 * lex_scores + 0.45 * snow_scores).astype(np.float32)

        return lex_scores

    @staticmethod
    def _lexicon_score(text: str) -> float:
        """Score a single text using financial lexicon ratio [-1, 1]."""
        if not text or not isinstance(text, str):
            return 0.0
        pos = sum(1 for w in _FIN_POSITIVE if w in text)
        neg = sum(1 for w in _FIN_NEGATIVE if w in text)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

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
