"""Sentiment-price and cross-source interaction features.

Combines independent feature dimensions to capture nonlinear interactions:
- Sentiment × momentum: conviction-weighted trend signals
- Text source agreement: consensus/dispersion across multiple sentiment sources
- Volume × sentiment: conviction-weighted volume signals
"""

import numpy as np
import pandas as pd

# Columns from each text source expected in the merged DataFrame
_SENTIMENT_PAIRS = [
    ("sentiment_mean", "news"),
    ("guba_sentiment_mean", "guba"),
    ("xueqiu_sentiment_mean", "xueqiu"),
    ("ann_sentiment_mean", "ann"),
]


class InteractionFeatures:
    """Compute cross-features between text sentiment and price indicators."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        new = {}

        new.update(_sentiment_momentum(result))
        new.update(_source_agreement(result))
        new.update(_sentiment_volume(result))

        if new:
            new_df = pd.DataFrame(new, index=result.index)
            result = pd.concat([result, new_df], axis=1)
        return result


def _sentiment_momentum(df: pd.DataFrame) -> dict:
    """Sentiment × momentum cross-features.

    High sentiment + positive recent return = bullish conviction.
    High sentiment + negative recent return = potential reversal.
    """
    out = {}
    for sent_col, prefix in _SENTIMENT_PAIRS:
        if sent_col not in df.columns:
            continue

        sent = df[sent_col].values

        for period in [5, 20]:
            roc_col = f"roc_{period}"
            rsi_col = f"rsi_{period}" if period in [6, 12] else None

            if roc_col in df.columns:
                roc_val = df[roc_col].values
                # signed strength: sentiment × ROC direction and magnitude
                out[f"{prefix}_sent_x_roc{period}"] = sent * np.sign(roc_val)

            if rsi_col and rsi_col in df.columns:
                rsi_val = df[rsi_col].values
                out[f"{prefix}_sent_x_rsi{period}"] = sent * (rsi_val - 50) / 50

            # MACD × sentiment
            if "macd_hist" in df.columns:
                macd = df["macd_hist"].values
                out[f"{prefix}_sent_x_macd"] = sent * np.sign(macd)

    return out


def _source_agreement(df: pd.DataFrame) -> dict:
    """Cross-source sentiment agreement/dispersion.

    When multiple text sources agree on sentiment direction, the signal
    should be stronger than a single source alone.
    """
    out = {}
    sent_cols = [c for c, _ in _SENTIMENT_PAIRS if c in df.columns]
    if len(sent_cols) < 2:
        return out

    sent_data = df[sent_cols].to_numpy(dtype=np.float64, copy=True)  # (n, k)
    signs = np.sign(sent_data)  # -1, 0, +1

    # Agreement: fraction of available sources that agree on direction
    pos_agree = (signs > 0).mean(axis=1)
    neg_agree = (signs < 0).mean(axis=1)
    out["sent_agree_pos"] = pos_agree.astype(np.float32)
    out["sent_agree_neg"] = neg_agree.astype(np.float32)

    # Net consensus: mean signed direction (-1 to +1)
    out["sent_consensus"] = signs.mean(axis=1).astype(np.float32)

    # Dispersion: std of sentiment values (normalized)
    sent_std = np.nan_to_num(sent_data, 0).std(axis=1)
    sent_abs_mean = np.abs(np.nan_to_num(sent_data, 0)).mean(axis=1) + 1e-10
    out["sent_dispersion"] = (sent_std / sent_abs_mean).astype(np.float32)

    return out


def _sentiment_volume(df: pd.DataFrame) -> dict:
    """Sentiment × volume interaction.

    High volume + positive sentiment = institutional conviction.
    Low volume + positive sentiment = weak signal.
    """
    out = {}
    vol_cols = [c for c in ["volume_ratio", "volume_ratio_20"] if c in df.columns]

    for sent_col, prefix in _SENTIMENT_PAIRS:
        if sent_col not in df.columns:
            continue
        sent = df[sent_col].values

        for vcol in vol_cols:
            vol = np.nan_to_num(df[vcol].values, 1.0)
            # Positive sentiment + heavy volume = strong bullish
            pos_sent = np.maximum(sent, 0)
            neg_sent = np.abs(np.minimum(sent, 0))
            out[f"{prefix}_sent_bull_vol"] = (pos_sent * np.log1p(np.maximum(vol, 0))).astype(np.float32)
            out[f"{prefix}_sent_bear_vol"] = (neg_sent * np.log1p(np.maximum(vol, 0))).astype(np.float32)

    return out
