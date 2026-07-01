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
            return df

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
    sent = group.get("sentiment_title", pd.Series([0.0])).fillna(0.0).values
    n = len(sent)

    bull = (sent > 0.2).sum()
    bear = (sent < -0.2).sum()

    bipolar = (bull - bear) / (bull + bear + 1)
    agreement = max(0.0, 1.0 - float(np.std(sent)))
    attention = np.log(1 + n)

    stats = {
        "bipolar_sent": float(bipolar),
        "agreement": float(agreement),
        "attention": float(attention),
        "bull_ratio": float(bull / n) if n > 0 else 0.0,
        "bear_ratio": float(bear / n) if n > 0 else 0.0,
        "sent_mean": float(sent.mean()),
        "sent_std": float(sent.std()) if n > 1 else 0.0,
        "sent_skew": float(_safe_skew(sent)),
        "sent_divergence": float(sent.std() / (abs(sent.mean()) + 0.01)),
        "post_count": n,
    }

    if "decay_weight" in group.columns and "weighted_sent" in group.columns:
        ws = group["weighted_sent"].fillna(0.0)
        w = group["decay_weight"].fillna(1.0)
        w_sum = w.sum() or 1.0
        stats["weighted_sent"] = float(ws.sum() / w_sum)

    if "sentiment_body" in group.columns:
        body = group["sentiment_body"].fillna(0.0).values
        stats["body_sent_mean"] = float(body.mean())
        if "decay_weight" in group.columns:
            w = group["decay_weight"].fillna(1.0).values
            stats["body_sent_weighted"] = float(
                (body * w).sum() / (w.sum() or 1.0)
            )

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
