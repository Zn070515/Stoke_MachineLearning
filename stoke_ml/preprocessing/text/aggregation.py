"""Daily aggregation: per-day sentiment statistics from individual posts.

Replaces the simple mean/std/positive_ratio/negative_ratio with richer
features: bipolar net sentiment, agreement index, attention, divergence,
skew, and rolling window derivatives.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class DailyAggregator(PreprocessingStep):
    """Aggregate per-post sentiment to daily multi-dimensional features.

    Input: DataFrame with 'aligned_date' (or 'date') + sentiment columns.
    Output: One row per date with bipolar_sent, agreement, attention,
            weighted_sent, sent_divergence, sent_skew, body_sent_mean,
            body_sent_weighted, plus rolling window means.
    """

    def __init__(self, windows=(3, 5, 10, 20)):
        self.windows = tuple(windows)

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df.copy()

        df = df.copy()
        date_col = "aligned_date" if "aligned_date" in df.columns else "date"
        if date_col not in df.columns:
            return df
        df[date_col] = pd.to_datetime(df[date_col])

        daily = df.groupby(date_col).apply(
            _daily_stats, include_groups=False
        ).reset_index()
        daily.rename(columns={date_col: "date"}, inplace=True)

        if len(daily) > 0:
            rolling_cols = [
                "bipolar_sent", "agreement", "attention",
                "sent_divergence", "sent_skew",
            ]
            available = [c for c in rolling_cols if c in daily.columns]
            for w in self.windows:
                for col in available:
                    daily[f"{col}_{w}d_mean"] = (
                        daily[col].rolling(w, min_periods=max(1, w // 3)).mean()
                    )
                    daily[f"{col}_{w}d_std"] = (
                        daily[col].rolling(w, min_periods=max(1, w // 3)).std()
                    )

        return daily


def _daily_stats(group: pd.DataFrame) -> pd.Series:
    """Compute daily aggregate stats for one day's posts."""
    if "sentiment_title" not in group.columns:
        return pd.Series({
            "bipolar_sent": np.nan, "agreement": np.nan, "attention": np.nan,
            "bull_ratio": np.nan, "bear_ratio": np.nan, "neutral_ratio": np.nan,
            "sent_mean": np.nan, "sent_std": np.nan, "sent_skew": np.nan,
            "sent_divergence": np.nan, "post_count": 0,
        })

    sent = group["sentiment_title"].fillna(0.0).values
    n = len(sent)

    bull = (sent > 0.2).sum()
    bear = (sent < -0.2).sum()
    neutral = n - bull - bear

    bipolar = (bull - bear) / (bull + bear + 1)
    agreement = max(0.0, 1.0 - float(np.std(sent)))
    attention = np.log(1 + n)

    stats = {
        "bipolar_sent": float(bipolar),
        "agreement": float(agreement),
        "attention": float(attention),
        "bull_ratio": float(bull / n) if n > 0 else 0.0,
        "bear_ratio": float(bear / n) if n > 0 else 0.0,
        "neutral_ratio": float(neutral / n) if n > 0 else 0.0,
        "sent_mean": float(sent.mean()),
        "sent_std": float(sent.std()) if n > 1 else 0.0,
        "sent_skew": float(_safe_skew(sent)),
        "sent_divergence": float(sent.std() / (abs(sent.mean()) + 0.05)),
        "post_count": n,
    }

    if "decay_weight" in group.columns:
        w = group["decay_weight"].fillna(1.0).values
        w_sum = w.sum() or 1.0
        if "weighted_sent" in group.columns:
            stats["weighted_sent"] = float(
                group["weighted_sent"].fillna(0.0).sum() / w_sum
            )
        else:
            stats["weighted_sent"] = float((sent * w).sum() / w_sum)

    if "sentiment_body" in group.columns:
        body = group["sentiment_body"].fillna(0.0).values
        stats["body_sent_mean"] = float(body.mean())
        if "decay_weight" in group.columns:
            w = group["decay_weight"].fillna(1.0).values
            stats["body_sent_weighted"] = float(
                (body * w).sum() / (w.sum() or 1.0)
            )

    # --- Topic features (from TopicModeler) -----------------------------
    if "topic_id" in group.columns and "topic_probability" in group.columns:
        topics = group["topic_id"].values
        probs = group["topic_probability"].values

        valid_mask = topics >= 0
        valid_topics = topics[valid_mask]

        if len(valid_topics) > 0:
            unique_topics = np.unique(valid_topics)

            for tid in unique_topics:
                mask = topics == tid
                tid_int = int(tid)
                if "sentiment_title" in group.columns:
                    topic_sent = group.loc[mask, "sentiment_title"].fillna(0.0).mean()
                    stats[f"topic_{tid_int}_sent"] = float(topic_sent)
                stats[f"topic_{tid_int}_ratio"] = (
                    float(mask.sum() / n) if n > 0 else 0.0
                )

            if len(unique_topics) > 1:
                topic_counts = np.array([
                    (topics == tid).sum() for tid in unique_topics
                ])
                topic_props = topic_counts / topic_counts.sum()
                topic_ent = -float(np.sum(
                    topic_props * np.log(topic_props + 1e-10)
                ))
            else:
                topic_ent = 0.0
            stats["topic_entropy"] = topic_ent

            topic_counts = [(int(tid), int((topics == tid).sum())) for tid in unique_topics]
            dominant = max(topic_counts, key=lambda x: x[1]) if topic_counts else (-1, 0)
            stats["topic_dominant"] = dominant[0]

            topic_sents = []
            for tid in unique_topics:
                mask = topics == tid
                if "sentiment_title" in group.columns and mask.any():
                    topic_sents.append(
                        group.loc[mask, "sentiment_title"].fillna(0.0).mean()
                    )
            stats["topic_sent_dispersion"] = (
                float(np.std(topic_sents)) if len(topic_sents) > 1 else 0.0
            )
        else:
            stats["topic_entropy"] = 0.0
            stats["topic_dominant"] = -1
            stats["topic_sent_dispersion"] = 0.0

    return pd.Series(stats)


def _safe_skew(arr: np.ndarray) -> float:
    """Skewness with protection against degenerate inputs."""
    if len(arr) < 3:
        return 0.0
    std = arr.std()
    if std < 1e-10:
        return 0.0
    mean = arr.mean()
    return float(np.mean(((arr - mean) / std) ** 3))
