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
_FINBERT_MODELS = [
    "yiyanghkust/finbert-tone-chinese",     # 0.88 accuracy, analyst reports
    "bardsai/finance-sentiment-zh-base",    # 0.973 accuracy, financial phrase bank
]
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

    Tries models in order:
    1. yiyanghkust/finbert-tone-chinese (0.88 acc, analyst reports)
    2. bardsai/finance-sentiment-zh-base (0.973 acc, financial phrase bank)
    3. Financial lexicon fallback (CPU)
    """

    def __init__(self):
        self._loaded = False
        self._model_name = "lexicon"
        self._pipe = None
        self._device = "cpu"
        self._use_finbert = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self._try_finbert():
            self._loaded = True
            logger.info("Sentiment: %s (%s)", self._model_name, self._device)
        else:
            self._loaded = True
            self._model_name = "lexicon"
            logger.info("Sentiment: financial lexicon (CPU fallback)")

    def _try_finbert(self) -> bool:
        """Load best available FinBERT model.

        Tries models in _FINBERT_MODELS order, each with mirror → direct
        → offline cache fallback. Returns True on first success.
        """
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            logger.info("FinBERT: transformers/torch not installed")
            return False

        # Resolve device
        if torch.cuda.is_available():
            self._device = "cuda"
            device = 0
        else:
            self._device = "cpu"
            device = -1

        for model_name in _FINBERT_MODELS:
            loaded = self._try_load_model(model_name, device, pipeline)
            if loaded:
                return True
        return False

    def _try_load_model(self, model_name: str, device, pipeline) -> bool:
        """Try loading a specific model with mirror/direct/offline fallback."""
        # Try with network access (mirror then direct)
        for attempt in range(2):
            try:
                if attempt == 0:
                    import os as _os
                    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=device,
                )
                _ = self._pipe("测试")  # warm-up
                self._model_name = model_name
                self._use_finbert = True
                logger.info("%s loaded on %s", model_name, self._device)
                return True
            except Exception:
                pass

        # Final attempt: load from local cache only
        try:
            self._pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
                local_files_only=True,
            )
            _ = self._pipe("测试")
            self._model_name = model_name
            self._use_finbert = True
            logger.info("%s loaded from cache on %s", model_name, self._device)
            return True
        except Exception:
            pass

        return False

    def analyze(self, texts: list[str]) -> np.ndarray:
        """Return sentiment scores [-1, 1] for each text."""
        if not texts:
            return np.array([], dtype=np.float32)
        self._ensure_loaded()

        if self._use_finbert:
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
