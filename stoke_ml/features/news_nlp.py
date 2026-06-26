"""News sentiment analysis for Chinese financial text.

L3: FinBERT Chinese (yiyanghkust/finbert-tone-chinese) with GPU batching.
    Fallback: financial lexicon for CPU-only environments.

Pipeline:
  raw news → compute_sentiment() → raw news with scores
  → PIT-align via NewsStorage.bronze_to_silver()
  → aggregate_daily_sentiment() → daily features
  → NewsStorage.silver_to_gold() → Gold layer with ZI filling
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FINBERT_MODEL = "yiyanghkust/finbert-tone-chinese"
_BATCH_SIZE = 64  # GPU batch size for sentiment inference

# Chinese financial sentiment lexicon — fallback when FinBERT unavailable
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

# Label mapping for yiyanghkust/finbert-tone-chinese
_LABEL_TO_SCORE = {
    "Positive": 1.0,
    "Negative": -1.0,
    "Neutral": 0.0,
}


class NewsSentimentAnalyzer:
    """Chinese financial news sentiment with FinBERT (GPU) + lexicon fallback.

    Uses yiyanghkust/finbert-tone-chinese on GPU for high-quality
    financial sentiment. Falls back to financial lexicon on CPU.
    """

    def __init__(self):
        self._loaded = False
        self._model_name = "lexicon"
        self._pipe = None
        self._device = "cpu"

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if self._try_finbert():
            self._model_name = "finbert"
            logger.info("Sentiment: FinBERT Chinese (GPU batch)")
        else:
            self._model_name = "lexicon"
            logger.info("Sentiment: financial lexicon (CPU fallback)")

    def _try_finbert(self) -> bool:
        """Load FinBERT model to GPU. Returns True on success."""
        try:
            import torch
            from transformers import pipeline

            if not torch.cuda.is_available():
                logger.info("FinBERT: no GPU available, using lexicon fallback")
                return False

            self._device = "cuda"
            self._pipe = pipeline(
                "sentiment-analysis",
                model=_FINBERT_MODEL,
                device=0,  # first GPU
                batch_size=_BATCH_SIZE,
            )
            # Warm-up: run a single text to prime CUDA context
            _ = self._pipe("测试")
            logger.info("FinBERT loaded on %s", torch.cuda.get_device_name(0))
            return True
        except ImportError:
            logger.info("FinBERT: transformers/torch not installed")
            return False
        except Exception as e:
            logger.info("FinBERT: failed to load (%s), using lexicon", e)
            return False

    def analyze(self, texts: list[str]) -> np.ndarray:
        """Return sentiment scores [-1, 1] for each text."""
        if not texts:
            return np.array([], dtype=np.float32)
        self._ensure_loaded()

        if self._model_name == "finbert":
            return self._finbert_sentiment(texts)

        return self._lexicon_batch(texts)

    def _finbert_sentiment(self, texts: list[str]) -> np.ndarray:
        """Run FinBERT inference with GPU batching.

        Uses P(positive) - P(negative) for nuanced scoring instead of
        argmax, since financial text is often neutral-dominant but
        carries weak directional signal in the probability margins.

        Text is truncated to ~300 chars (safe for 512 BERT tokens even
        with multi-byte Chinese characters).
        """
        truncated = [t[:300] for t in texts]
        results = self._pipe(truncated, truncation=True, max_length=512,
                             top_k=None)
        scores = np.zeros(len(texts), dtype=np.float32)
        for i, r in enumerate(results):
            probs = {item["label"]: item["score"] for item in r}
            pos = probs.get("Positive", 0.0)
            neg = probs.get("Negative", 0.0)
            scores[i] = pos - neg
        return scores

    @staticmethod
    def _lexicon_batch(texts: list[str]) -> np.ndarray:
        """Score texts with financial lexicon ratio."""
        scores = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                continue
            pos = sum(1 for w in _FIN_POSITIVE if w in text)
            neg = sum(1 for w in _FIN_NEGATIVE if w in text)
            total = pos + neg
            if total > 0:
                scores[i] = (pos - neg) / total
        return scores


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
