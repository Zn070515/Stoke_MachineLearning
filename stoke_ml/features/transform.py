"""Temporal statistics on past-observed columns.

Replaces the old add_rolling_features for PO columns with focused
operators: rolling_mean (5/10/20), rolling_std (20), accel (ma5-ma20),
zscore ((v-ma20)/std20).  Column type determines which operators apply.

Structurally sparse columns (zero fraction > 80 % after ZI fill) skip
z-score — rolling std on ZI-filled data is near-zero, making _z20 NaN.
Event-time features (days_since, count_20d, etc.) from EventToDaily
serve as the replacement.
"""
import numpy as np
import pandas as pd

from stoke_ml.features._rolling import rolling_mean, rolling_std, accel, zscore

# Zero fraction above which a column is considered structurally sparse
# and _z20 is skipped (ZI fill → long zero runs → std≈0 → z≈NaN).
_SPARSE_ZERO_FRACTION = 0.8

# Prefixes that identify known PK columns (excluded from PO transform).
# These are computed from OHLCV, not merged from auxiliary data.
_PK_PREFIXES = frozenset({
    "open", "high", "low", "close", "volume", "amount",
    "ma_", "ema_", "macd_", "rsi_", "kdj_", "boll_", "atr_",
    "roc_", "wr_", "cci_", "vol_", "volume_", "amount_",
    "kmid", "klen", "kup", "klow", "ksft",
    "open0", "high0", "low0",
    "adx", "adxr", "pdi", "mdi",
    "mfi_", "cmo_", "trix",
    "obv", "turnover",
    "is_limit_", "gap_", "volume_anomaly", "limit_up_",
    "is_one_word", "seal_quality",
    "day_of_", "month", "quarter",
    "minutes_", "is_am_", "is_pm_", "session_", "bar_of_", "opening_",
    "trend_level", "bias_", "buy_signal",
    "pct_change",
    # Alpha158 rolling-window factors
    "max_", "min_", "qtlu_", "qtld_", "rank_", "rsv_",
    "corr_", "cord_", "beta_", "rsqr_", "resi_",
    "vma_", "vstd_",
    "cntp_", "cntn_", "cntd_",
    "sump_", "sumn_", "sumd_",
    "imax_", "imin_", "imxd_",
    "wvma_", "vsump_", "vsumn_", "vsumd_",
    # Interaction features (derived from PK technical + sentiment)
    "interaction_",
    # Fundamental (quarterly reports + derived from PK)
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
    "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm",
    "f_score", "quality_composite", "earnings_quality",
    "profitability_stability", "margin_stability", "growth_quality",
    "pe_percentile_", "pb_percentile_", "pe_pb_divergence", "deep_value",
    "roe_accel", "revenue_trend_", "margin_trend_", "earnings_surprise",
    # Emotion refinement (derived features, not raw PO)
    "news_", "guba_",
    "news_guba_divergence", "news_guba_ratio",
    "total_attention", "cross_source_agreement", "retail_panic",
})

# Columns that are metadata, not features.
_SKIP_COLS = frozenset({
    "date", "stock_code", "sector", "sector_code", "size_proxy",
})


def _is_po_column(col: str) -> bool:
    """Return True if *col* is a past-observed (aux-merged) feature column."""
    if col in _SKIP_COLS:
        return False
    if col.startswith("has_"):
        return True  # ZI-fill flags are PO
    for prefix in _PK_PREFIXES:
        if col == prefix or col.startswith(prefix):
            return False
    return True


class TemporalTransformer:
    """Apply temporal statistics to past-observed columns.

    Operators (column-type-dependent):

    | Type       | Pattern                      | Operators                    |
    |------------|------------------------------|------------------------------|
    | boolean    | is_*, has_*                  | rolling_mean only            |
    | ratio      | *_ratio, *_pct, *_proportion | all four                     |
    | continuous | everything else              | all four                     |
    """

    WINDOWS_MEAN = [5, 10, 20]
    WINDOW_STD = 20
    WINDOW_ACCEL_FAST = 5
    WINDOW_ACCEL_SLOW = 20
    WINDOW_ZSCORE = 20

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal statistic columns to *df*, return modified copy.

        Structurally sparse columns (zero fraction > 80 %) skip z-score
        because ZI-filled rolling std ≈ 0, making _z20 pervasively NaN.
        """
        result = df.copy()
        po_cols = [c for c in df.columns if _is_po_column(c)]
        if not po_cols:
            return result

        bool_cols = [c for c in po_cols if c.startswith(("is_", "has_"))]
        ratio_cols = [
            c for c in po_cols
            if c not in bool_cols and c.endswith(("_ratio", "_pct", "_proportion"))
        ]
        cont_cols = [c for c in po_cols if c not in bool_cols and c not in ratio_cols]

        # Identify structurally sparse columns — zero fraction from ZI fill
        # makes rolling z-score meaningless.  Still compute mean/std/accel
        # (rolling averages over mostly-zeros are informative).
        sparse_cols: set[str] = set()
        n = len(result)
        if n > 0:
            for col in ratio_cols + cont_cols:
                vals = result[col].values
                valid = ~np.isnan(vals)
                if valid.sum() == 0:
                    sparse_cols.add(col)
                else:
                    zero_frac = (np.abs(vals[valid]) < 1e-12).mean()
                    if zero_frac >= _SPARSE_ZERO_FRACTION:
                        sparse_cols.add(col)

        new_cols = {}

        for col in bool_cols:
            arr = result[col].values.astype(np.float64)
            for w in self.WINDOWS_MEAN:
                new_cols[f"{col}_ma{w}"] = rolling_mean(arr, w)

        for col in ratio_cols + cont_cols:
            arr = result[col].values.astype(np.float64)
            for w in self.WINDOWS_MEAN:
                new_cols[f"{col}_ma{w}"] = rolling_mean(arr, w)
            new_cols[f"{col}_std{self.WINDOW_STD}"] = rolling_std(arr, self.WINDOW_STD)
            new_cols[f"{col}_accel"] = accel(arr, self.WINDOW_ACCEL_FAST, self.WINDOW_ACCEL_SLOW)
            if col not in sparse_cols:
                new_cols[f"{col}_z{self.WINDOW_ZSCORE}"] = zscore(arr, self.WINDOW_ZSCORE)

        if new_cols:
            result = pd.concat(
                [result, pd.DataFrame(new_cols, index=result.index).astype(np.float32)],
                axis=1,
            )
        return result


# Rolling helpers imported from stoke_ml.features._rolling
