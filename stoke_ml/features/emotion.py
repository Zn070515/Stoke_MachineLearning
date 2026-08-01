"""Emotion refinement features from sentiment data.

Computes momentum, reversal, disagreement, attention, and cross-source
features from news + Guba sentiment columns.  Stateless — pure functions
operating on a single stock's daily DataFrame.
"""
import numpy as np
import pandas as pd

from stoke_ml.features._rolling import (
    rolling_mean, rolling_std, rolling_min, rolling_quantile,
    sign_streak, skew_proxy,
)

# Column name patterns for each source.  Columns matching these prefixes
# are treated as belonging to that source's sentiment family.
_NEWS_SENT_COLS = [
    "sentiment_mean", "sentiment_std", "news_count",
    "positive_ratio", "negative_ratio",
]
_GUBA_SENT_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio",
]


class EmotionRefiner:
    """Compute emotion features from sentiment data.

    Generates ~15 news features, ~15 guba features, and ~5 cross-source
    features.  Gracefully handles missing columns — if a source's columns
    are absent, its features are simply not computed.
    """

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        # Detect which sources are present
        has_news = any(c in result.columns for c in _NEWS_SENT_COLS)
        has_guba = any(c in result.columns for c in _GUBA_SENT_COLS)

        if has_news:
            result = self._compute_source_features(
                result,
                sent_col="sentiment_mean",
                std_col="sentiment_std",
                count_col="news_count",
                pos_col="positive_ratio",
                neg_col="negative_ratio",
                prefix="news",
            )

        if has_guba:
            result = self._compute_source_features(
                result,
                sent_col="guba_sentiment_mean",
                std_col="guba_sentiment_std",
                count_col="guba_post_count",
                pos_col="guba_positive_ratio",
                neg_col="guba_negative_ratio",
                prefix="guba",
            )

        if has_news and has_guba:
            result = self._compute_cross_features(result)

        return result

    # ------------------------------------------------------------------
    # Per-source features
    # ------------------------------------------------------------------

    def _compute_source_features(
        self,
        df: pd.DataFrame,
        sent_col: str,
        std_col: str,
        count_col: str,
        pos_col: str,
        neg_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        sent = df.get(sent_col)
        std = df.get(std_col)
        count = df.get(count_col)
        pos = df.get(pos_col)
        neg = df.get(neg_col)

        if sent is None:
            return df

        sent_v = sent.values.astype(np.float64)
        eps = 1e-8

        # 1. Momentum: ma5 of sentiment
        df[f"{prefix}_sent_momentum_5d"] = rolling_mean(sent_v, 5).astype(np.float32)

        # 2. Acceleration: ma5 - ma20
        ma5 = rolling_mean(sent_v, 5)
        ma20 = rolling_mean(sent_v, 20)
        df[f"{prefix}_sent_accel"] = (ma5 - ma20).astype(np.float32)

        # 3. Reversal: sentiment - min(sentiment, 5d)
        rev = sent_v - rolling_min(sent_v, 5)
        df[f"{prefix}_sent_reversal_5d"] = rev.astype(np.float32)

        # 4. Disagreement: std / (|mean| + eps)
        if std is not None:
            std_v = std.values.astype(np.float64)
            df[f"{prefix}_disagreement"] = (
                std_v / (np.abs(sent_v) + eps)
            ).astype(np.float32)

        # 5. Attention z-score: (count - ma20) / std20
        if count is not None:
            cnt_v = count.values.astype(np.float64)
            cnt_ma20 = rolling_mean(cnt_v, 20)
            cnt_std20 = rolling_std(cnt_v, 20)
            z = np.full(len(cnt_v), np.nan, dtype=np.float64)
            valid = cnt_std20 > eps
            z[valid] = (cnt_v[valid] - cnt_ma20[valid]) / cnt_std20[valid]
            df[f"{prefix}_attention_z"] = np.nan_to_num(z, 0.0).astype(np.float32)

            # 6. Sentiment-volume interaction: sentiment × log(count + 1)
            df[f"{prefix}_sent_volume"] = (
                sent_v * np.log(np.maximum(cnt_v, 0) + 1)
            ).astype(np.float32)

            # 13. Count momentum: count / ma5(count)
            cnt_ma5 = rolling_mean(cnt_v, 5)
            safe_ratio = np.full_like(cnt_ma5, 1.0, dtype=np.float64)
            valid = cnt_ma5 > eps
            safe_ratio[valid] = cnt_v[valid] / cnt_ma5[valid]
            df[f"{prefix}_count_momentum"] = safe_ratio.astype(np.float32)

        # 7. Net bullish: positive_ratio - negative_ratio
        if pos is not None and neg is not None:
            df[f"{prefix}_net_bullish"] = (
                pos.values.astype(np.float64) - neg.values.astype(np.float64)
            ).astype(np.float32)

        # 8. Sentiment streak: consecutive days of same sign
        df[f"{prefix}_sent_streak"] = sign_streak(sent_v).astype(np.float32)

        # 9. Sentiment volatility ratio: std5 / std20
        std5 = rolling_std(sent_v, 5)
        std20 = rolling_std(sent_v, 20)
        with np.errstate(invalid="ignore"):
            df[f"{prefix}_sent_vol_ratio"] = np.where(
                std20 > eps, std5 / std20, 1.0
            ).astype(np.float32)

        # 10. Sentiment extreme: >80th or <20th percentile (20d window)
        p80 = rolling_quantile(sent_v, 20, 0.8)
        p20 = rolling_quantile(sent_v, 20, 0.2)
        df[f"{prefix}_sent_extreme"] = (
            (sent_v > p80) | (sent_v < p20)
        ).astype(np.float32)

        # 11. Positive momentum: positive_ratio - ma5(positive_ratio)
        if pos is not None:
            pos_v = pos.values.astype(np.float64)
            df[f"{prefix}_pos_momentum"] = (pos_v - rolling_mean(pos_v, 5)).astype(np.float32)

        # 12. Negative momentum: negative_ratio - ma5(negative_ratio)
        if neg is not None:
            neg_v = neg.values.astype(np.float64)
            df[f"{prefix}_neg_momentum"] = (neg_v - rolling_mean(neg_v, 5)).astype(np.float32)

        # 14. Sentiment skew proxy: (mean - median) / std (20d window)
        df[f"{prefix}_sent_skew"] = skew_proxy(sent_v, 20).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Cross-source features
    # ------------------------------------------------------------------

    def _compute_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        news_sent = df.get("sentiment_mean")
        guba_sent = df.get("guba_sentiment_mean")
        news_count = df.get("news_count")
        guba_count = df.get("guba_post_count")
        guba_neg = df.get("guba_negative_ratio")

        if news_sent is None or guba_sent is None:
            return df

        ns = news_sent.values.astype(np.float64)
        gs = guba_sent.values.astype(np.float64)
        eps = 1e-8

        # 1. Divergence: news - guba
        df["news_guba_divergence"] = (ns - gs).astype(np.float32)

        # 2. Source mix: news_count / (guba_count + 1)
        if news_count is not None and guba_count is not None:
            nc = news_count.values.astype(np.float64)
            gc = guba_count.values.astype(np.float64)
            df["news_guba_ratio"] = (nc / (gc + 1)).astype(np.float32)

            # 3. Total attention
            df["total_attention"] = (nc + gc).astype(np.float32)

        # 4. Cross-source agreement: sign match
        df["cross_source_agreement"] = (
            (np.sign(ns) == np.sign(gs)).astype(np.float32)
        )

        # 5. Retail panic: guba_neg > 0.7 AND news_sent near neutral
        if guba_neg is not None:
            gn = guba_neg.values.astype(np.float64)
            df["retail_panic"] = (
                (gn > 0.7) & (np.abs(ns) < 0.05)
            ).astype(np.float32)

        return df


# Rolling helpers imported from stoke_ml.features._rolling
