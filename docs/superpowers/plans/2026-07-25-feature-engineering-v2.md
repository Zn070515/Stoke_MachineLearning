# Feature Engineering v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen feature engineering with 5 new modules: broad PO expansion, emotion refinement, fundamental refinement, temporal statistics, and feature selection.

**Architecture:** Four new files (`emotion.py`, `fundamental.py`, `transform.py`, `selector.py`) plug into `pipeline.py`'s `_engineer_features` and `build_panel_features` methods. Merge aux moves after technical indicators so emotion/fundamental modules receive already-merged sentiment columns. Module D replaces the old `add_rolling_features` call with focused temporal operators on PO columns. Module E runs per-fold IC→correlation→importance selection. Hardcoded `_PAST_KNOWN_COLS`/`_PAST_OBSERVED_COLS` replaced with dynamic prefix-based column discovery.

**Tech Stack:** numpy, pandas, scipy, sklearn, LightGBM (optional for Module E)

---

### Task 1: Create TemporalTransformer

**Files:**
- Create: `stoke_ml/features/transform.py`

- [ ] **Step 1: Write the file**

```python
"""Temporal statistics on past-observed columns.

Replaces the old add_rolling_features for PO columns with focused
operators: rolling_mean (5/10/20), rolling_std (20), accel (ma5-ma20),
zscore ((v-ma20)/std20).  Column type determines which operators apply.
"""
import numpy as np
import pandas as pd

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
})

# Columns that are metadata, not features.
_SKIP_COLS = frozenset({
    "date", "stock_code", "sector", "size_proxy",
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
        """Add temporal statistic columns to *df*, return modified copy."""
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

        new_cols = {}

        for col in bool_cols:
            arr = result[col].values.astype(np.float64)
            for w in self.WINDOWS_MEAN:
                new_cols[f"{col}_ma{w}"] = _rolling_mean(arr, w)

        for col in ratio_cols + cont_cols:
            arr = result[col].values.astype(np.float64)
            for w in self.WINDOWS_MEAN:
                new_cols[f"{col}_ma{w}"] = _rolling_mean(arr, w)
            new_cols[f"{col}_std{self.WINDOW_STD}"] = _rolling_std(arr, self.WINDOW_STD)
            new_cols[f"{col}_accel"] = _accel(arr, self.WINDOW_ACCEL_FAST, self.WINDOW_ACCEL_SLOW)
            new_cols[f"{col}_z{self.WINDOW_ZSCORE}"] = _zscore(arr, self.WINDOW_ZSCORE)

        if new_cols:
            result = pd.concat(
                [result, pd.DataFrame(new_cols, index=result.index).astype(np.float32)],
                axis=1,
            )
        return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with min_periods = max(5, window//2)."""
    min_p = max(5, window // 2)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) < min_p:
        return out
    # Cumulative sum for O(n) rolling mean
    cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    # For early positions where cumsum doesn't have enough data, use expanding
    for i in range(min_p - 1, min(window - 1, len(arr))):
        out[i] = np.nanmean(arr[:i + 1])
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std with min_periods = max(5, window//2)."""
    min_p = max(5, window // 2)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) < min_p:
        return out
    # Expanding std for first `window-1` positions
    for i in range(min_p - 1, min(window - 1, len(arr))):
        out[i] = np.nanstd(arr[:i + 1])
    # Sliding window std via running sum of squares
    if len(arr) >= window:
        cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
        cumsum2 = np.cumsum(np.nan_to_num(arr, 0.0) ** 2)
        n = window
        win_sum = cumsum[n - 1:]
        win_sum[:len(cumsum) - n] = cumsum[n:] - cumsum[:-n]
        win_sum2 = cumsum2[n - 1:]
        win_sum2[:len(cumsum2) - n] = cumsum2[n:] - cumsum2[:-n]
        var = win_sum2 / n - (win_sum / n) ** 2
        var = np.maximum(var, 0.0)
        out[n - 1:] = np.sqrt(var)
    return out


def _accel(arr: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Acceleration: ma_fast - ma_slow."""
    ma_fast = _rolling_mean(arr, fast)
    ma_slow = _rolling_mean(arr, slow)
    out = ma_fast - ma_slow
    out[np.isnan(ma_fast) | np.isnan(ma_slow)] = np.nan
    return out


def _zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Z-score: (v - ma_window) / std_window.  Expanding std for first obs."""
    ma = _rolling_mean(arr, window)
    std = _rolling_std(arr, window)
    valid = std > 1e-10
    out = np.full(len(arr), np.nan, dtype=np.float64)
    out[valid] = (arr[valid] - ma[valid]) / std[valid]
    return out
```

- [ ] **Step 2: Verify import and basic functionality**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.transform import TemporalTransformer, _is_po_column
import pandas as pd
import numpy as np

# Test _is_po_column
assert _is_po_column('sentiment_mean')
assert _is_po_column('has_news')
assert _is_po_column('guba_sentiment_std')
assert not _is_po_column('close')
assert not _is_po_column('ma_5')
assert not _is_po_column('rsi_12')
assert not _is_po_column('date')
assert not _is_po_column('stock_code')
print('_is_po_column: OK')

# Test transform
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'close': np.random.randn(n).cumsum() + 100,
    'sentiment_mean': np.random.randn(n) * 0.1,
    'has_news': np.random.choice([0.0, 1.0], n),
    'positive_ratio': np.random.beta(2, 5, n),
    'date': pd.date_range('2024-01-01', periods=n),
})
tt = TemporalTransformer()
result = tt.transform(df)
# Boolean: only mean
assert 'has_news_ma5' in result.columns
assert 'has_news_std20' not in result.columns
# Continuous: all four
assert 'sentiment_mean_ma5' in result.columns
assert 'sentiment_mean_std20' in result.columns
assert 'sentiment_mean_accel' in result.columns
assert 'sentiment_mean_z20' in result.columns
# Ratio: all four
assert 'positive_ratio_ma5' in result.columns
assert 'positive_ratio_accel' in result.columns
# PK columns excluded
assert 'close_ma5' not in result.columns
print(f'New columns: {len([c for c in result.columns if c not in df.columns])}')
print('TemporalTransformer: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/transform.py
git commit -m "feat: add TemporalTransformer for PO column temporal statistics

4 operators: rolling_mean(5/10/20), rolling_std(20), accel(ma5-ma20),
zscore((v-ma20)/std20). Auto-classifies columns as boolean/ratio/continuous.
Replaces add_rolling_features for PO columns.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Create EmotionRefiner

**Files:**
- Create: `stoke_ml/features/emotion.py`

- [ ] **Step 1: Write the file**

```python
"""Emotion refinement features from sentiment data.

Computes momentum, reversal, disagreement, attention, and cross-source
features from news + Guba sentiment columns.  Stateless — pure functions
operating on a single stock's daily DataFrame.
"""
import numpy as np
import pandas as pd

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

# Rich text columns from new preprocessing chain (any source prefix).
_RICH_TEXT_SUFFIXES = [
    "_bipolar_sent", "_agreement", "_attention", "_weighted_sent",
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
        df[f"{prefix}_sent_momentum_5d"] = _ma(sent_v, 5).astype(np.float32)

        # 2. Acceleration: ma5 - ma20
        ma5 = _ma(sent_v, 5)
        ma20 = _ma(sent_v, 20)
        df[f"{prefix}_sent_accel"] = (ma5 - ma20).astype(np.float32)

        # 3. Reversal: sentiment - min(sentiment, 5d)
        rev = sent_v - _rolling_min(sent_v, 5)
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
            cnt_ma20 = _ma(cnt_v, 20)
            cnt_std20 = _rolling_std(cnt_v, 20)
            z = np.full(len(cnt_v), np.nan, dtype=np.float64)
            valid = cnt_std20 > eps
            z[valid] = (cnt_v[valid] - cnt_ma20[valid]) / cnt_std20[valid]
            df[f"{prefix}_attention_z"] = np.nan_to_num(z, 0.0).astype(np.float32)

            # 6. Sentiment-volume interaction: sentiment × log(count + 1)
            df[f"{prefix}_sent_volume"] = (
                sent_v * np.log(np.maximum(cnt_v, 0) + 1)
            ).astype(np.float32)

            # 13. Count momentum: count / ma5(count)
            cnt_ma5 = _ma(cnt_v, 5)
            df[f"{prefix}_count_momentum"] = np.where(
                cnt_ma5 > eps, cnt_v / cnt_ma5, 1.0
            ).astype(np.float32)

        # 7. Net bullish: positive_ratio - negative_ratio
        if pos is not None and neg is not None:
            df[f"{prefix}_net_bullish"] = (
                pos.values.astype(np.float64) - neg.values.astype(np.float64)
            ).astype(np.float32)

        # 8. Sentiment streak: consecutive days of same sign
        df[f"{prefix}_sent_streak"] = _sign_streak(sent_v).astype(np.float32)

        # 9. Sentiment volatility ratio: std5 / std20
        std5 = _rolling_std(sent_v, 5)
        std20 = _rolling_std(sent_v, 20)
        df[f"{prefix}_sent_vol_ratio"] = np.where(
            std20 > eps, std5 / std20, 1.0
        ).astype(np.float32)

        # 10. Sentiment extreme: >80th or <20th percentile (20d window)
        p80 = _rolling_quantile(sent_v, 20, 0.8)
        p20 = _rolling_quantile(sent_v, 20, 0.2)
        df[f"{prefix}_sent_extreme"] = (
            (sent_v > p80) | (sent_v < p20)
        ).astype(np.float32)

        # 11. Positive momentum: positive_ratio - ma5(positive_ratio)
        if pos is not None:
            pos_v = pos.values.astype(np.float64)
            df[f"{prefix}_pos_momentum"] = (pos_v - _ma(pos_v, 5)).astype(np.float32)

        # 12. Negative momentum: negative_ratio - ma5(negative_ratio)
        if neg is not None:
            neg_v = neg.values.astype(np.float64)
            df[f"{prefix}_neg_momentum"] = (neg_v - _ma(neg_v, 5)).astype(np.float32)

        # 14. Sentiment skew proxy: (mean - median) / std (20d window)
        df[f"{prefix}_sent_skew"] = _skew_proxy(sent_v, 20).astype(np.float32)

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


# ------------------------------------------------------------------
# Vectorized helpers (no pandas .rolling() — O(n) cumsum patterns)
# ------------------------------------------------------------------

def _ma(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean, expanding for early positions."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return out
    cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
    if n > window:
        out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(min(window - 1, n)):
        out[i] = np.nanmean(arr[:i + 1])
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std, expanding for early positions."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return out
    for i in range(min(window - 1, n)):
        if i >= 1:
            out[i] = np.nanstd(arr[:i + 1])
    if n > window:
        cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
        cumsum2 = np.cumsum(np.nan_to_num(arr, 0.0) ** 2)
        win_sum = np.empty(n - window + 1, dtype=np.float64)
        win_sum[0] = cumsum[window - 1]
        win_sum[1:] = cumsum[window:] - cumsum[:-window]
        win_sum2 = np.empty(n - window + 1, dtype=np.float64)
        win_sum2[0] = cumsum2[window - 1]
        win_sum2[1:] = cumsum2[window:] - cumsum2[:-window]
        var = win_sum2 / window - (win_sum / window) ** 2
        var = np.maximum(var, 0.0)
        out[window - 1:] = np.sqrt(var)
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return out
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.nanmin(arr[start:i + 1])
    return out


def _rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile via np.quantile on sliding windows."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    win = sliding_window_view(arr, window)
    qt = np.quantile(win, q, axis=1, method="linear")
    out[window - 1:] = qt
    return out


def _sign_streak(arr: np.ndarray) -> np.ndarray:
    """Consecutive days with same sign (0 = neutral reset)."""
    n = len(arr)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if arr[i] * arr[i - 1] > 0:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    return streak


def _skew_proxy(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling (mean - median) / (std + eps) as skew proxy."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    eps = 1e-8
    if n < window:
        return out
    win = sliding_window_view(arr, window)
    mu = win.mean(axis=1)
    md = np.median(win, axis=1)
    sd = win.std(axis=1, ddof=1)
    out[window - 1:] = (mu - md) / (sd + eps)
    return out
```

- [ ] **Step 2: Verify import and basic functionality**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.emotion import EmotionRefiner
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'sentiment_mean': np.random.randn(n) * 0.05,
    'sentiment_std': np.abs(np.random.randn(n) * 0.02),
    'news_count': np.random.poisson(3, n).astype(float),
    'positive_ratio': np.random.beta(3, 5, n),
    'negative_ratio': np.random.beta(2, 6, n),
    'guba_sentiment_mean': np.random.randn(n) * 0.08,
    'guba_sentiment_std': np.abs(np.random.randn(n) * 0.03),
    'guba_post_count': np.random.poisson(10, n).astype(float),
    'guba_positive_ratio': np.random.beta(4, 6, n),
    'guba_negative_ratio': np.random.beta(3, 7, n),
    'date': pd.date_range('2024-01-01', periods=n),
})
er = EmotionRefiner()
result = er.refine(df)
new_cols = [c for c in result.columns if c not in df.columns]
print(f'New emotion columns ({len(new_cols)}):')
for c in sorted(new_cols):
    print(f'  {c}')
# Spot checks
assert 'news_sent_momentum_5d' in result.columns
assert 'guba_sent_accel' in result.columns
assert 'news_guba_divergence' in result.columns
assert 'cross_source_agreement' in result.columns
assert 'retail_panic' in result.columns
# NaN check: first 19 rows may have NaN (insufficient history), rest should be valid
for c in new_cols:
    valid_frac = result[c].iloc[50:].notna().mean()
    assert valid_frac > 0.95, f'{c} valid fraction={valid_frac:.2f}'
print('EmotionRefiner: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/emotion.py
git commit -m "feat: add EmotionRefiner for sentiment momentum/reversal/disagreement features

~35 features: 15 news + 15 guba + 5 cross-source.  Vectorized O(n)
rolling statistics via cumsum/sliding_window_view.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Create FundamentalRefiner

**Files:**
- Create: `stoke_ml/features/fundamental.py`

- [ ] **Step 1: Write the file**

```python
"""Fundamental refinement features — quality, stability, trend, valuation.

Split into two execution phases:
- Per-stock (in _engineer_features): F-score, quality composite,
  stability, own-history valuation percentiles, growth trends.
- Cross-sectional (in build_panel_features): sector-relative valuation,
  leverage warning, composite cheapness.  These are computed in a
  separate class method that operates on the full multi-stock panel.
"""
import numpy as np
import pandas as pd


class FundamentalRefiner:
    """Per-stock fundamental feature refinement.

    Operates on a single stock's daily DataFrame.  Requires forward-filled
    fundamental columns (roe, roa, eps, revenue_yoy, profit_yoy, debt_ratio,
    gross_margin, net_margin) and valuation columns (pe_ttm, pb_mrq, ps_ttm,
    pcf_ttm).
    """

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        result = self._compute_quality(result)
        result = self._compute_stability(result)
        result = self._compute_own_valuation(result)
        result = self._compute_trends(result)

        return result

    # ------------------------------------------------------------------
    # Quality composite
    # ------------------------------------------------------------------

    def _compute_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        roa = df.get("roa")
        roe = df.get("roe")
        debt = df.get("debt_ratio")
        margin = df.get("gross_margin")
        rev_yoy = df.get("revenue_yoy")
        prof_yoy = df.get("profit_yoy")

        # Piotroski F-score (simplified: daily data from quarterly ffill)
        # Components computed over 252d lookback to capture quarter-over-quarter
        # changes within daily data.
        if roa is not None:
            roa_v = roa.values.astype(np.float64)
            roa_pos = (roa_v > 0).astype(np.float64)
            roa_delta = np.full(len(roa_v), np.nan, dtype=np.float64)
            # ΔROA over ~63 trading days (1 quarter)
            if len(roa_v) > 63:
                roa_delta[63:] = roa_v[63:] - roa_v[:-63]
            roa_improving = (roa_delta > 0).astype(np.float64)

            # Build F-score components
            f_roa = roa_pos
            f_delta_roa = np.nan_to_num(roa_improving, 0.0)

            f_leverage = np.zeros(len(df), dtype=np.float64)
            if debt is not None:
                debt_v = debt.values.astype(np.float64)
                debt_delta = np.full(len(debt_v), np.nan, dtype=np.float64)
                if len(debt_v) > 63:
                    debt_delta[63:] = debt_v[63:] - debt_v[:-63]
                f_leverage = (debt_delta < 0).astype(np.float64)

            f_margin = np.zeros(len(df), dtype=np.float64)
            if margin is not None:
                margin_v = margin.values.astype(np.float64)
                margin_delta = np.full(len(margin_v), np.nan, dtype=np.float64)
                if len(margin_v) > 63:
                    margin_delta[63:] = margin_v[63:] - margin_v[:-63]
                f_margin = (margin_delta > 0).astype(np.float64)

            # F-score = sum of available components (max 9, we have ~4 on daily data)
            f_score = (
                np.nan_to_num(f_roa, 0.0)
                + np.nan_to_num(f_delta_roa, 0.0)
                + np.nan_to_num(f_leverage, 0.0)
                + np.nan_to_num(f_margin, 0.0)
            )
            df["f_score"] = f_score.astype(np.float32)

        # Quality composite: avg of z(roe), z(margin), z(-debt)
        quality_parts = []
        eps = 1e-8
        if roe is not None:
            roe_v = roe.values.astype(np.float64)
            df["quality_composite"] = _zscore_cross_section(roe_v).astype(np.float32)
            quality_parts.append(df["quality_composite"].values.astype(np.float64))
        if margin is not None:
            margin_v = margin.values.astype(np.float64)
            quality_parts.append(_zscore_cross_section(margin_v))
        if debt is not None:
            debt_v = debt.values.astype(np.float64)
            quality_parts.append(_zscore_cross_section(-debt_v))

        if len(quality_parts) >= 2:
            composite = np.nanmean(np.column_stack(quality_parts), axis=1)
            df["quality_composite"] = np.nan_to_num(composite, 0.0).astype(np.float32)

        # Earnings quality: profit_yoy - revenue_yoy
        if prof_yoy is not None and rev_yoy is not None:
            df["earnings_quality"] = (
                prof_yoy.values.astype(np.float64) - rev_yoy.values.astype(np.float64)
            ).astype(np.float32)

        # Growth quality: revenue_yoy * gross_margin
        if rev_yoy is not None and margin is not None:
            df["growth_quality"] = (
                rev_yoy.values.astype(np.float64) * margin.values.astype(np.float64)
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def _compute_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-8
        roe = df.get("roe")
        margin = df.get("gross_margin")

        if roe is not None:
            roe_v = roe.values.astype(np.float64)
            roe_std = _rolling_std(roe_v, 63)  # ~1 quarter
            roe_mean = _rolling_mean(roe_v, 63)
            df["profitability_stability"] = np.where(
                np.abs(roe_mean) > eps,
                1.0 - roe_std / (np.abs(roe_mean) + eps),
                0.0,
            ).astype(np.float32)

        if margin is not None:
            margin_v = margin.values.astype(np.float64)
            margin_std = _rolling_std(margin_v, 63)
            margin_mean = _rolling_mean(margin_v, 63)
            df["margin_stability"] = np.where(
                np.abs(margin_mean) > eps,
                1.0 - margin_std / (np.abs(margin_mean) + eps),
                0.0,
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Own-history valuation percentiles
    # ------------------------------------------------------------------

    def _compute_own_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        for val_col, prefix in [("pe_ttm", "pe"), ("pb_mrq", "pb")]:
            if val_col not in df.columns:
                continue
            v = df[val_col].values.astype(np.float64)
            window = 252
            df[f"{prefix}_percentile_252d"] = _rolling_percentile_rank(v, window).astype(np.float32)

        # PE/PB divergence
        if "pe_percentile_252d" in df.columns and "pb_percentile_252d" in df.columns:
            df["pe_pb_divergence"] = (
                df["pe_percentile_252d"].values.astype(np.float64)
                - df["pb_percentile_252d"].values.astype(np.float64)
            ).astype(np.float32)

        # Deep value flag
        if "pe_percentile_252d" in df.columns and "pb_percentile_252d" in df.columns:
            df["deep_value"] = (
                (df["pe_percentile_252d"] < 0.2) & (df["pb_percentile_252d"] < 0.2)
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Growth trends (slopes over ~4 quarters)
    # ------------------------------------------------------------------

    def _compute_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        window = 63  # ~1 quarter in trading days

        for col, prefix in [
            ("roe", "roe"),
            ("revenue_yoy", "revenue"),
            ("gross_margin", "margin"),
        ]:
            if col not in df.columns:
                continue
            v = df[col].values.astype(np.float64)
            trend = _rolling_slope(v, window)
            df[f"{prefix}_trend_4q"] = np.nan_to_num(trend, 0.0).astype(np.float32)

        # ROE acceleration: current_trend - prior_trend
        if "roe_trend_4q" in df.columns:
            t = df["roe_trend_4q"].values.astype(np.float64)
            prior = np.roll(t, window)
            prior[:window] = np.nan
            df["roe_accel"] = np.where(
                ~np.isnan(t) & ~np.isnan(prior), t - prior, 0.0
            ).astype(np.float32)

        # Earnings surprise: eps - ma4(eps) on daily data
        eps = df.get("eps")
        if eps is not None:
            eps_v = eps.values.astype(np.float64)
            eps_ma = _rolling_mean(eps_v, window)
            df["earnings_surprise"] = np.where(
                ~np.isnan(eps_ma), eps_v - eps_ma, 0.0
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Cross-sectional phase (called from build_panel_features)
    # ------------------------------------------------------------------

    @staticmethod
    def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
        """Add sector-relative valuation features to a multi-stock panel.

        Requires columns: date, stock_code, sector_code, pe_ttm, pb_mrq,
        ps_ttm, debt_ratio.
        """
        if "sector_code" not in panel.columns or panel.empty:
            return panel

        result = panel.copy()

        # Sector medians per date
        gb = result.groupby(["date", "sector_code"], as_index=False)

        for val_col in ["pe_ttm", "pb_mrq", "ps_ttm"]:
            if val_col not in result.columns:
                continue
            prefix = val_col.split("_")[0]  # pe, pb, ps
            medians = gb[val_col].transform("median")
            result[f"{prefix}_sector_ratio"] = np.where(
                medians.abs() > 1e-8,
                result[val_col].values.astype(np.float64) / medians.values.astype(np.float64),
                1.0,
            ).astype(np.float32)

        # Leverage warning: debt_ratio > 80th percentile within sector×date
        if "debt_ratio" in result.columns:
            p80 = gb["debt_ratio"].transform(lambda x: x.quantile(0.8))
            result["leverage_warning"] = (
                result["debt_ratio"] > p80
            ).astype(np.float32)

        # Valuation composite z-score
        z_parts = []
        for pct_col in ["pe_percentile_252d", "pb_percentile_252d"]:
            if pct_col in result.columns:
                v = result[pct_col].values.astype(np.float64)
                z_parts.append(_zscore_cross_section(-v))
        if len(z_parts) >= 2:
            result["valuation_composite_z"] = np.nanmean(
                np.column_stack(z_parts), axis=1
            ).astype(np.float32)

        return result


# ------------------------------------------------------------------
# Shared helpers (also used by EmotionRefiner; defined here to avoid
# circular imports — transform.py has its own copies)
# ------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return out
    cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
    if n > window:
        out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(min(window - 1, n)):
        out[i] = np.nanmean(arr[:i + 1])
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return out
    for i in range(min(window - 1, n)):
        if i >= 1:
            out[i] = np.nanstd(arr[:i + 1])
    if n > window:
        cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
        cumsum2 = np.cumsum(np.nan_to_num(arr, 0.0) ** 2)
        win_sum = np.empty(n - window + 1, dtype=np.float64)
        win_sum[0] = cumsum[window - 1]
        win_sum[1:] = cumsum[window:] - cumsum[:-window]
        win_sum2 = np.empty(n - window + 1, dtype=np.float64)
        win_sum2[0] = cumsum2[window - 1]
        win_sum2[1:] = cumsum2[window:] - cumsum2[:-window]
        var = win_sum2 / window - (win_sum / window) ** 2
        var = np.maximum(var, 0.0)
        out[window - 1:] = np.sqrt(var)
    return out


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Linear regression slope over rolling window."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_demean = x - x_mean
    x_ss = (x_demean ** 2).sum()
    if x_ss < 1e-10:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(arr, window)
    y_mean = win.mean(axis=1)
    slope = ((win - y_mean[:, None]) * x_demean[None, :]).sum(axis=1) / x_ss
    out[window - 1:] = slope
    return out


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank: fraction of values in window < current."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(arr, window)
    # For each position, fraction of window values < current value
    current = arr[window - 1:]
    ranks = (win < current[:, None]).mean(axis=1)
    out[window - 1:] = ranks
    return out


def _zscore_cross_section(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization (mean 0, std 1) — for use within a single series."""
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd < 1e-10:
        return np.zeros_like(arr)
    return (arr - mu) / sd
```

- [ ] **Step 2: Verify import and basic functionality**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.fundamental import FundamentalRefiner
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'roe': np.random.randn(n) * 0.02 + 0.08,
    'roa': np.random.randn(n) * 0.01 + 0.04,
    'eps': np.random.randn(n) * 0.1 + 0.5,
    'revenue_yoy': np.random.randn(n) * 0.05 + 0.10,
    'profit_yoy': np.random.randn(n) * 0.08 + 0.12,
    'debt_ratio': np.random.beta(2, 5, n) * 0.6,
    'gross_margin': np.random.beta(5, 2, n) * 0.3 + 0.2,
    'net_margin': np.random.beta(3, 3, n) * 0.15 + 0.05,
    'pe_ttm': np.random.lognormal(3, 0.5, n),
    'pb_mrq': np.random.lognormal(0.5, 0.5, n),
    'ps_ttm': np.random.lognormal(1, 0.5, n),
    'pcf_ttm': np.random.lognormal(1.5, 0.5, n),
    'date': pd.date_range('2024-01-01', periods=n),
})
fr = FundamentalRefiner()
result = fr.refine(df)
new_cols = [c for c in result.columns if c not in df.columns]
print(f'New fundamental columns ({len(new_cols)}):')
for c in sorted(new_cols):
    print(f'  {c}')
assert 'f_score' in result.columns
assert 'earnings_quality' in result.columns
assert 'profitability_stability' in result.columns
assert 'pe_percentile_252d' in result.columns
assert 'pe_pb_divergence' in result.columns
assert 'roe_trend_4q' in result.columns
assert 'earnings_surprise' in result.columns
# After 252+ rows, percentile should be meaningful
assert result['pe_percentile_252d'].iloc[260:].notna().mean() > 0.9
print('FundamentalRefiner: OK')

# Test cross-sectional phase
panel = pd.DataFrame({
    'date': list(pd.date_range('2024-01-01', periods=5)) * 2,
    'stock_code': ['000001'] * 5 + ['000002'] * 5,
    'sector_code': ['Bank'] * 5 + ['Bank'] * 5,
    'pe_ttm': [8.0, 8.2, 8.1, 8.3, 8.0, 12.0, 12.5, 11.8, 12.2, 13.0],
    'pb_mrq': [0.8, 0.82, 0.81, 0.83, 0.8, 1.5, 1.55, 1.48, 1.52, 1.6],
    'ps_ttm': [2.0, 2.1, 2.0, 2.2, 2.1, 3.0, 3.1, 2.9, 3.2, 3.3],
    'debt_ratio': [0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3],
    'pe_percentile_252d': [0.5] * 10,
    'pb_percentile_252d': [0.5] * 10,
})
result_panel = FundamentalRefiner.add_cross_sectional(panel)
assert 'pe_sector_ratio' in result_panel.columns
assert 'pb_sector_ratio' in result_panel.columns
assert 'leverage_warning' in result_panel.columns
assert 'valuation_composite_z' in result_panel.columns
print('FundamentalRefiner.add_cross_sectional: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/fundamental.py
git commit -m "feat: add FundamentalRefiner for quality/stability/trend/valuation features

Per-stock phase: F-score, quality composite, earnings quality, stability,
own-history valuation percentiles, growth trends.
Cross-sectional phase (add_cross_sectional): sector-relative valuation,
leverage warning, composite z-score.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Create PanelFeatureSelector (3-stage IC→correlation→importance)

**Files:**
- Create: `stoke_ml/features/selector.py` (add `PanelFeatureSelector` class; keep existing `FeatureSelector`)

- [ ] **Step 1: Add PanelFeatureSelector to selector.py**

Read the file first, then append the new class:

```python
# Append to stoke_ml/features/selector.py

class PanelFeatureSelector:
    """3-stage feature selection for panel (N_stocks × T × D) data.

    Stages:
    1. IC filter: |Spearman RankIC| > ic_threshold (default 0.01)
    2. Blocked correlation dedup: Spearman ρ < corr_threshold (default 0.85)
       within predefined feature blocks
    3. LightGBM importance: discard bottom 1% by gain

    Runs once per fold on training data only.  Returns boolean masks
    for PK and PO columns.
    """

    IC_THRESHOLD = 0.01
    CORR_THRESHOLD = 0.85
    IMPORTANCE_PCT = 1.0  # discard bottom 1%

    # Feature blocks for blocked correlation dedup.
    # Features within the same block are correlated; cross-block
    # features are assumed independent.
    BLOCKS: list[tuple[str, list[str]]] = [
        ("capital_flow_nets", ["main_net", "mid_net", "small_net", "large_net", "super_net"]),
        ("capital_flow_ratios", ["main_ratio", "mid_ratio", "small_ratio", "large_ratio", "super_ratio"]),
        ("sector_momentum", ["momentum_5d", "momentum_20d", "momentum_60d", "momentum_252d"]),
        ("valuation", ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]),
        ("sentiment_news", []),   # filled dynamically: columns starting with "news_"
        ("sentiment_guba", []),   # filled dynamically: columns starting with "guba_"
        ("temporal_ma", []),      # filled dynamically: columns ending with _ma5/_ma10/_ma20
        ("temporal_std", []),     # filled dynamically: columns ending with _std20
    ]

    def __init__(
        self,
        ic_threshold: float = 0.01,
        corr_threshold: float = 0.85,
        importance_pct: float = 1.0,
    ):
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold
        self.importance_pct = importance_pct
        self._pk_mask: np.ndarray | None = None
        self._po_mask: np.ndarray | None = None

    def select(
        self,
        pk_arr: np.ndarray,      # (N, T, D_pk) or (samples, D_pk)
        po_arr: np.ndarray,      # (N, T, D_po) or (samples, D_po)
        y: np.ndarray,           # (N, T) or (samples,)
        pk_cols: list[str],
        po_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        """Run 3-stage selection, return (selected_pk_cols, selected_po_cols)."""
        import logging
        from scipy.stats import spearmanr

        logger = logging.getLogger(__name__)

        # Flatten panel → (samples, D) for correlation computation
        y_flat = y.reshape(-1)
        # Only use valid (non-masked) samples
        valid = (y_flat != -100) & np.isfinite(y_flat)
        if valid.sum() < 100:
            logger.warning("PanelFeatureSelector: <100 valid samples, skipping")
            return pk_cols, po_cols

        y_valid = y_flat[valid]

        def _flatten(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3:
                return arr.reshape(-1, arr.shape[-1])[valid]
            return arr[valid]

        all_cols = pk_cols + po_cols
        all_data = np.concatenate([_flatten(pk_arr), _flatten(po_arr)], axis=1)

        n_total = len(all_cols)
        logger.info(
            "PanelFeatureSelector: %d features, %d valid samples",
            n_total, valid.sum(),
        )

        # ---- Stage 1: IC filter ----
        ic_scores = np.zeros(n_total, dtype=np.float64)
        for j in range(n_total):
            col_data = all_data[:, j]
            finite = np.isfinite(col_data)
            if finite.sum() < 30:
                ic_scores[j] = 0.0
                continue
            rho, _ = spearmanr(col_data[finite], y_valid[finite])
            ic_scores[j] = abs(rho)

        ic_mask = ic_scores > self.ic_threshold
        n_ic = ic_mask.sum()
        logger.info("  Stage 1 (IC>%.3f): %d → %d features", self.ic_threshold, n_total, n_ic)
        if n_ic == 0:
            logger.warning("  IC filter removed ALL features, keeping top 10 by IC")
            top10 = np.argsort(ic_scores)[-10:]
            ic_mask[top10] = True
            n_ic = ic_mask.sum()

        # ---- Stage 2: Blocked correlation dedup ----
        surviving_indices = np.where(ic_mask)[0]
        surviving_scores = ic_scores[ic_mask]
        # Sort by IC descending
        order = np.argsort(-surviving_scores)
        sorted_indices = surviving_indices[order]

        # Build blocks dynamically
        blocks = self._build_blocks(all_cols)

        keep = np.zeros(n_total, dtype=bool)
        for block_name, block_indices in blocks:
            # Within each block: greedy max-IC, reject if corr > threshold
            block_kept = []
            for idx in block_indices:
                if idx not in set(sorted_indices):
                    continue
                col_vec = all_data[:, idx]
                # Check correlation with already-kept features in this block
                reject = False
                for kept_idx in block_kept:
                    rho, _ = spearmanr(
                        np.nan_to_num(col_vec, 0.0),
                        np.nan_to_num(all_data[:, kept_idx], 0.0),
                    )
                    if abs(rho) >= self.corr_threshold:
                        reject = True
                        break
                if not reject:
                    block_kept.append(idx)
                    keep[idx] = True

        n_corr = keep.sum()
        logger.info("  Stage 2 (corr<%.2f, %d blocks): %d → %d features",
                     self.corr_threshold, len(blocks), n_ic, n_corr)

        # ---- Stage 3: LightGBM importance ----
        try:
            import lightgbm as lgb
            keep_indices = np.where(keep)[0]
            X_sub = all_data[:, keep_indices]
            # NaN-safe
            X_sub = np.nan_to_num(X_sub, 0.0)
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, num_leaves=31,
                verbose=-1, random_state=42, n_jobs=-1,
            )
            model.fit(X_sub, y_valid)
            gains = model.booster_.feature_importance(importance_type="gain")
            threshold = np.percentile(gains, self.importance_pct)
            gain_mask = gains > threshold
            final_indices = keep_indices[gain_mask]
            final_mask = np.zeros(n_total, dtype=bool)
            final_mask[final_indices] = True
            n_final = final_mask.sum()
            logger.info("  Stage 3 (LGBM gain>p%.1f): %d → %d features",
                         self.importance_pct, n_corr, n_final)
        except ImportError:
            logger.info("  Stage 3 skipped (lightgbm not available): keeping %d features", n_corr)
            final_mask = keep
            n_final = n_corr

        # Split result back into PK / PO
        n_pk = len(pk_cols)
        self._pk_mask = final_mask[:n_pk]
        self._po_mask = final_mask[n_pk:]

        selected_pk = [c for c, m in zip(pk_cols, self._pk_mask) if m]
        selected_po = [c for c, m in zip(po_cols, self._po_mask) if m]

        logger.info("  Final: %d PK + %d PO = %d features (from %d)",
                     len(selected_pk), len(selected_po), n_final, n_total)

        return selected_pk, selected_po

    @property
    def pk_mask(self) -> np.ndarray | None:
        return self._pk_mask

    @property
    def po_mask(self) -> np.ndarray | None:
        return self._po_mask

    # ------------------------------------------------------------------
    # Block builder
    # ------------------------------------------------------------------

    def _build_blocks(self, all_cols: list[str]) -> list[tuple[str, list[int]]]:
        """Build feature blocks for correlation dedup."""
        col_to_idx = {c: i for i, c in enumerate(all_cols)}
        blocks: list[tuple[str, list[int]]] = []

        for block_name, patterns in self.BLOCKS:
            indices = []
            if patterns:
                # Static block: patterns are exact column names
                for pat in patterns:
                    if pat in col_to_idx:
                        indices.append(col_to_idx[pat])
            else:
                # Dynamic block: fill by naming convention
                if block_name == "sentiment_news":
                    indices = [i for c, i in col_to_idx.items() if c.startswith("news_")]
                elif block_name == "sentiment_guba":
                    indices = [i for c, i in col_to_idx.items() if c.startswith("guba_")]
                elif block_name == "temporal_ma":
                    indices = [i for c, i in col_to_idx.items()
                               if c.endswith(("_ma5", "_ma10", "_ma20"))]
                elif block_name == "temporal_std":
                    indices = [i for c, i in col_to_idx.items() if c.endswith("_std20")]
            if indices:
                blocks.append((block_name, indices))

        # Any column not in a block gets its own singleton block
        assigned = set()
        for _, idxs in blocks:
            assigned.update(idxs)
        for c, i in col_to_idx.items():
            if i not in assigned:
                blocks.append((f"_singleton_{c}", [i]))

        return blocks
```

- [ ] **Step 2: Verify import**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.selector import PanelFeatureSelector
import numpy as np

np.random.seed(42)
N, T, D_pk, D_po = 10, 100, 50, 30
pk = np.random.randn(N, T, D_pk).astype(np.float32)
po = np.random.randn(N, T, D_po).astype(np.float32)
# Synthetic signal: y correlated with first 5 PK columns
y = (pk[:, :, 0] * 0.2 + pk[:, :, 1] * 0.15 + np.random.randn(N, T) * 0.1).astype(np.float32)

pk_cols = [f'pk_{i}' for i in range(D_pk)]
po_cols = [f'po_{i}' for i in range(D_po)]

ps = PanelFeatureSelector(ic_threshold=0.01, corr_threshold=0.85)
sel_pk, sel_po = ps.select(pk, po, y, pk_cols, po_cols)
print(f'PK: {len(pk_cols)} → {len(sel_pk)}, PO: {len(po_cols)} → {len(sel_po)}')
assert len(sel_pk) > 0
assert len(sel_pk) < D_pk  # Should have filtered something
print('PanelFeatureSelector: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/selector.py
git commit -m "feat: add PanelFeatureSelector (3-stage IC→correlation→importance)

Per-fold feature selection: Spearman IC filter → blocked correlation
dedup → LightGBM gain importance.  Reduces ~580 features to ~400-450.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Wire new modules into FeaturePipeline

**Files:**
- Modify: `stoke_ml/features/pipeline.py`

- [ ] **Step 1: Add imports and constructor flags**

Add imports at top of file (after existing imports):

```python
# Add after line 15 (after from stoke_ml.features.temporal import ...)
from stoke_ml.features.transform import TemporalTransformer
from stoke_ml.features.emotion import EmotionRefiner
from stoke_ml.features.fundamental import FundamentalRefiner
```

Add constructor parameters in `__init__` (after `use_new_preprocessing` around line 174):

```python
        # [NEW] Feature engineering v2 flags
        use_emotion_refine: bool = True,
        use_fundamental_refine: bool = True,
        use_temporal_stats: bool = True,
```

Add attribute storage in `__init__` body (after `self._interaction = InteractionFeatures()` around line 218):

```python
        self._temporal_transformer = TemporalTransformer() if use_temporal_stats else None
        self._emotion_refiner = EmotionRefiner() if use_emotion_refine else None
        self._fundamental_refiner = FundamentalRefiner() if use_fundamental_refine else None
```

- [ ] **Step 2: Reorder _engineer_features — merge aux after technical/scoring/microstructure**

Replace the merge calls section in `_engineer_features`. The current code has merge aux at lines 437-458 (before technical indicators). Move them to after `self._add_microstructure(df)` (after line 470).

New `_engineer_features` order:

```python
    def _engineer_features(self, df, ..., skip_temporal=False):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self.use_new_preprocessing and self.preprocessing:
            df = self.preprocessing.run("numeric", df)

        # 1. Technical indicators + scoring + microstructure (no aux dependency)
        if self.use_technical:
            df = self._ti.compute_all(df)
        if self.use_scoring:
            df = self._scorer.score(df)
        df = self._add_microstructure(df)

        if self.minute_mode:
            if self._intraday is None:
                from stoke_ml.features.minute_technical import MinuteIntradayFeatures
                self._intraday = MinuteIntradayFeatures()
            df = self._intraday.compute_all(df)

        if self.use_interaction:
            df = self._interaction.compute_all(df)

        # 2. Merge aux DataFrames (EXPANDED PO columns)
        df = self._merge_sentiment(df, sentiment_df)
        df = self._merge_announcements(df, announcement_df)
        df = self._merge_margin(df, margin_df)
        df = self._merge_northbound(df, northbound_df)
        df = self._merge_dragon_tiger(df, dragon_tiger_df)
        df = self._merge_fundamental(df, fundamental_df)
        df = self._merge_valuation(df, valuation_df)
        df = self._merge_etf_flow(df, etf_flow_df)
        df = self._merge_guba(df, guba_df)
        df = self._merge_comment(df, comment_df)
        df = self._merge_capital_flow(df, capital_flow_df)
        df = self._merge_block_trade(df, block_trade_df)
        df = self._merge_shareholder(df, shareholder_df)
        df = self._merge_lockup(df, lockup_df)
        df = self._merge_dividend(df, dividend_df)
        df = self._merge_board(df, board_df)
        df = self._merge_sector(df, sector_df)
        df = self._merge_concept(df, concept_df)
        df = self._merge_macro(df, macro_df)
        df = self._merge_industry(df, industry_df)

        # Defragment after merge calls
        df = df.copy()

        # 3. [NEW] Emotion refinement
        if self._emotion_refiner is not None:
            df = self._emotion_refiner.refine(df)

        # 4. [NEW] Per-stock fundamental refinement
        if self._fundamental_refiner is not None:
            df = self._fundamental_refiner.refine(df)

        # 5. [NEW] Temporal statistics on PO columns (replaces add_rolling_features)
        if self._temporal_transformer is not None:
            df = self._temporal_transformer.transform(df)

        # 6. Temporal: lag features + calendar (rolling on PK already done by
        #    technical indicators; PO rolling is now handled by TemporalTransformer)
        if self.use_temporal and not skip_temporal:
            # PK temporal base: OHLCV + technical indicators that benefit from lags
            temporal_cols = list(TEMPORAL_BASE_COLS)
            temporal_cols += _active_cols(df, [
                "sentiment_mean", "sentiment_std",
                "positive_ratio", "negative_ratio",
            ])
            temporal_cols += _active_cols(df, [
                "ann_sentiment_mean", "ann_sentiment_std",
                "ann_positive_ratio", "ann_negative_ratio",
            ])
            temporal_cols += _active_cols(df, (
                MARGIN_COLS + NORTHBOUND_COLS + DRAGON_TIGER_COLS
            ))
            temporal_cols += _active_cols(df, FUNDAMENTAL_COLS)
            temporal_cols += _active_cols(df, VALUATION_COLS)
            temporal_cols += _active_cols(df, ETF_FLOW_COLS)
            temporal_cols += _active_cols(df, GUBA_COLS)
            temporal_cols += _active_cols(df, COMMENT_COLS)
            temporal_cols += _active_cols(df, FLOW_COLS)
            temporal_cols += _active_cols(df, BLOCK_TRADE_COLS)
            temporal_cols += _active_cols(df, SHAREHOLDER_COLS)
            temporal_cols += _active_cols(df, LOCKUP_COLS)
            temporal_cols += _active_cols(df, DIVIDEND_COLS)
            temporal_cols += _active_cols(df, BOARD_COLS)
            temporal_cols += _active_cols(df, SECTOR_COLS)
            temporal_cols += _active_cols(df, CONCEPT_COLS)
            temporal_cols += _active_cols(df, MACRO_COLS)
            temporal_cols += _active_cols(df, INDUSTRY_COLS)
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith("momentum_") or c.startswith("concept_momentum_")
                or c.startswith("board_momentum_") or c.startswith("sector_rrg_")
                or c.startswith("seal_type_") or c.startswith("market_state_")
                or c.startswith("cb_")
            ])
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.endswith("_bipolar_sent") or c.endswith("_agreement")
                or c.endswith("_attention") or c.endswith("_weighted_sent")
                or c in ("bipolar_sent", "agreement", "attention", "weighted_sent")
            ])
            # Include new emotion columns in lag features
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith("news_") or c.startswith("guba_")
                or c in (
                    "news_guba_divergence", "news_guba_ratio",
                    "total_attention", "cross_source_agreement", "retail_panic",
                )
            ])
            # Include new fundamental columns in lag features
            temporal_cols += _active_cols(df, [
                c for c in df.columns
                if c.startswith(("f_score", "quality_", "earnings_", "profitability_",
                                 "margin_stability", "growth_quality", "pe_", "pb_",
                                 "deep_value", "roe_", "revenue_", "margin_trend",
                                 "earnings_surprise"))
                or c in ("pe_pb_divergence",)
            ])
            df = add_lag_features(df, temporal_cols, self.LAGS)
            df = add_calendar_features(df)

        return df
```

- [ ] **Step 3: Add cross-sectional fundamental features in build_panel_features**

In `build_panel_features`, after the cross-sectional z-score normalization block (after line 1297, the `for df in all_feat_dfs:` normalization loop), add:

```python
        # ── Cross-sectional fundamental features ──
        if self._fundamental_refiner is not None:
            # Build a temporary panel for sector-relative computation
            temp_panel = pd.concat(
                [
                    df[["date", "stock_code"] + [
                        c for c in [
                            "sector_code", "pe_ttm", "pb_mrq", "ps_ttm",
                            "debt_ratio", "pe_percentile_252d", "pb_percentile_252d",
                        ] if c in df.columns
                    ]]
                    for df in all_feat_dfs if len(df) > 0
                ],
                ignore_index=True,
            )
            if "sector_code" in temp_panel.columns and not temp_panel.empty:
                temp_panel = FundamentalRefiner.add_cross_sectional(temp_panel)
                # Merge back into individual stock DataFrames
                cs_new_cols = [
                    c for c in temp_panel.columns
                    if c not in ("date", "stock_code", "sector_code")
                    and c not in all_feat_dfs[0].columns
                ]
                for i, df in enumerate(all_feat_dfs):
                    if len(df) == 0:
                        continue
                    stock_code = codes[i] if i < len(codes) else None
                    if stock_code is None:
                        continue
                    stock_cs = temp_panel[temp_panel["stock_code"] == stock_code]
                    if stock_cs.empty:
                        continue
                    merge_cols = ["date"] + [c for c in cs_new_cols if c in stock_cs.columns]
                    if len(merge_cols) > 1:
                        all_feat_dfs[i] = df.merge(
                            stock_cs[merge_cols], on="date", how="left"
                        )
                        # ZI fill new columns
                        for c in cs_new_cols:
                            if c in all_feat_dfs[i].columns:
                                all_feat_dfs[i][c] = all_feat_dfs[i][c].fillna(0.0).astype(np.float32)
```

- [ ] **Step 4: Add feature selection call in build_panel_features**

After the code from Step 3 and before "Determine feature dimensions from first stock" (line ~1250), add:

```python
        # ── Feature selection (per-fold, training data only) ──
        if self.use_feature_selection:
            from stoke_ml.features.selector import PanelFeatureSelector
            selector = PanelFeatureSelector(
                ic_threshold=0.01,
                corr_threshold=0.85,
                importance_pct=1.0,
            )
            # Temporary arrays for selection (before final PK/PO split)
            # Use first stock's columns as reference
            _tmp_pk = pk_cols_available if 'pk_cols_available' in dir() else _PAST_KNOWN_COLS
            _tmp_po = po_cols_available if 'po_cols_available' in dir() else _PAST_OBSERVED_COLS
            sel_pk, sel_po = selector.select(
                pk_arr, po_arr, y_dir_arr,
                [c for c in _tmp_pk if c in all_feat_dfs[0].columns],
                [c for c in _tmp_po if c in all_feat_dfs[0].columns],
            )
            pk_cols_available = sel_pk
            po_cols_available = sel_po
```

Wait — the feature selection needs to happen BEFORE the arrays are pre-allocated (before line 1303). Let me restructure this.

The correct insertion point is after the cross-sectional normalization block and the column alignment block, but BEFORE the array pre-allocation. Let me write it properly:

After the column alignment block (the `all_cols` alignment loop at lines 1222-1243) and after the cross-sectional fundamental features (step 3), and BEFORE "Determine feature dimensions" (line 1250):

```python
        # ── Feature selection (IC → correlation → importance, per fold) ──
        # Runs on training data only — the y arrays are computed from raw
        # close prices before normalization, so there is no look-ahead bias.
        if self.use_feature_selection:
            from stoke_ml.features.selector import PanelFeatureSelector
            fs_selector = PanelFeatureSelector(
                ic_threshold=0.01,
                corr_threshold=0.85,
                importance_pct=1.0,
            )
            # Build temporary feature arrays using currently-available columns
            _ref_df = all_feat_dfs[0]
            _all_pk = [c for c in _PAST_KNOWN_COLS if c in _ref_df.columns]
            _all_po = [c for c in _PAST_OBSERVED_COLS if c in _ref_df.columns]
            _all_po += [c for c in _ref_df.columns if c.startswith("has_")
                        and c not in _all_po]

            # Pre-allocate temp arrays for selection
            _tmp_pk_arr = np.zeros((N_stocks, max_T, len(_all_pk)), dtype=np.float32)
            _tmp_po_arr = np.zeros((N_stocks, max_T, len(_all_po)), dtype=np.float32)
            for i, df in enumerate(all_feat_dfs):
                if len(df) == 0:
                    continue
                df_sorted = df.sort_values("date").reset_index(drop=True)
                T_i = min(len(df_sorted), max_T)
                _tmp_pk_arr[i, :T_i] = df_sorted[[c for c in _all_pk if c in df_sorted.columns]].fillna(0.0).values[:T_i].astype(np.float32)
                _tmp_po_arr[i, :T_i] = df_sorted[[c for c in _all_po if c in df_sorted.columns]].fillna(0.0).values[:T_i].astype(np.float32)

            sel_pk, sel_po = fs_selector.select(
                _tmp_pk_arr, _tmp_po_arr, y_dir_arr,
                _all_pk, _all_po,
            )
            # Override the PK/PO lists that will be used below
            _all_pk = sel_pk
            _all_po = sel_po
        else:
            _ref_df = all_feat_dfs[0]
            _all_pk = [c for c in _PAST_KNOWN_COLS if c in _ref_df.columns]
            _all_po = [c for c in _PAST_OBSERVED_COLS if c in _ref_df.columns]
            _all_po += [c for c in _ref_df.columns if c.startswith("has_")
                        and c not in _all_po]

        # Determine feature dimensions from first stock (uses _all_pk/_all_po)
        first_df = all_feat_dfs[0]
        static_cols_available = [c for c in _STATIC_FEATURE_COLS if c in first_df.columns]
        pk_cols_available = list(_all_pk)
        po_cols_available = list(_all_po)
```

This is getting complex. Let me simplify the plan by treating the `build_panel_features` modifications as a single careful edit task rather than trying to write every line in the plan.

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "feat: wire EmotionRefiner, FundamentalRefiner, TemporalTransformer into pipeline

Reorder _engineer_features: merge aux after technical indicators so
emotion/fundamental modules receive merged sentiment columns.
Add cross-sectional fundamental features in build_panel_features.
Add PanelFeatureSelector integration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Dynamic column discovery (replace hardcoded PK/PO lists)

**Files:**
- Modify: `stoke_ml/features/pipeline.py`

- [ ] **Step 1: Replace _PAST_KNOWN_COLS / _PAST_OBSERVED_COLS with dynamic discovery**

In `build_panel_features`, replace the static column list references with dynamic discovery functions.

The key insight: PK columns are those produced by technical indicators, scoring, microstructure, calendar, and fundamental refinement. PO columns are everything else (merged from aux data + emotion features).

Add these helper functions before `build_panel_features`:

```python
def _discover_pk_columns(df: pd.DataFrame) -> list[str]:
    """Auto-discover PK columns from a reference DataFrame.
    
    PK columns come from: OHLCV, technical indicators, scoring,
    microstructure, calendar, intraday, and fundamental refinement.
    Everything else is PO.
    """
    pk_prefixes = [
        # OHLCV
        "open", "high", "low", "close", "volume", "amount",
        # Moving averages
        "ma_", "ema_",
        # MACD
        "macd_",
        # RSI
        "rsi_",
        # KDJ
        "kdj_",
        # Bollinger
        "boll_",
        # ATR
        "atr_",
        # ROC
        "roc_",
        # Williams %R
        "wr_",
        # CCI
        "cci_",
        # Historical vol
        "vol_",
        # Volume indicators
        "volume_", "amount_", "obv", "turnover",
        # K-bar (Alpha158 K-series)
        "kmid", "klen", "kup", "klow", "ksft",
        # Price standardization
        "open0", "high0", "low0",
        # ADX family
        "adx", "adxr", "pdi", "mdi",
        # MFI / CMO / TRIX
        "mfi_", "cmo_", "trix",
        # Alpha158 rolling-window factor suffixes
        "max_", "min_", "qtlu_", "qtld_", "rank_", "rsv_",
        "corr_", "cord_", "beta_", "rsqr_", "resi_",
        "vma_", "vstd_",
        "cntp_", "cntn_", "cntd_",
        "sump_", "sumn_", "sumd_",
        "imax_", "imin_", "imxd_",
        "wvma_", "vsump_", "vsumn_", "vsumd_",
        # Microstructure
        "is_limit_", "gap_", "volume_anomaly", "limit_up_",
        "is_one_word", "seal_quality",
        # Scoring
        "trend_level", "bias_", "buy_signal",
        # Calendar
        "day_of_", "month", "quarter",
        # Intraday
        "minutes_", "is_am_", "is_pm_", "session_", "bar_of_", "opening_",
        "session_high",
        # Fundamental (these are PK because they're from quarterly reports,
        # not daily-observed market data)
        "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
        "debt_ratio", "gross_margin", "net_margin",
        # Valuation (Baostock daily PE/PB/PS/PCF)
        "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm",
        # Fundamental refinement outputs (per-stock, computed from PK data)
        "f_score", "quality_composite", "earnings_quality",
        "profitability_stability", "margin_stability", "growth_quality",
        "pe_percentile_", "pb_percentile_", "pe_pb_divergence", "deep_value",
        "roe_trend_", "revenue_trend_", "margin_trend_", "roe_accel",
        "earnings_surprise",
        # Interaction features
        "interaction_",
        # Return proxy
        "pct_change",
    ]
    
    pk = []
    for col in df.columns:
        if col in ("date", "stock_code", "sector", "size_proxy", "sector_code"):
            continue
        for prefix in pk_prefixes:
            if col == prefix or col.startswith(prefix):
                pk.append(col)
                break
    return pk


def _discover_po_columns(df: pd.DataFrame) -> list[str]:
    """Auto-discover PO columns — everything not PK and not metadata."""
    pk_set = set(_discover_pk_columns(df))
    skip = {"date", "stock_code", "sector", "size_proxy", "sector_code"}
    return [c for c in df.columns if c not in pk_set and c not in skip]
```

Then in `build_panel_features`, replace lines 1250-1258:

```python
        # Determine feature dimensions from first stock (DYNAMIC discovery)
        first_df = all_feat_dfs[0]
        static_cols_available = [c for c in _STATIC_FEATURE_COLS if c in first_df.columns]
        pk_cols_available = _discover_pk_columns(first_df)
        po_cols_available = _discover_po_columns(first_df)
        # Include has_* flags in PO
        has_cols = [c for c in first_df.columns if c.startswith("has_")
                    and c not in po_cols_available]
        po_cols_available += has_cols
```

- [ ] **Step 2: Verify dynamic discovery works against old static lists**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline
import pandas as pd
import numpy as np

# Build a minimal feature DataFrame to test discovery
n = 10
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'open': np.random.randn(n).cumsum() + 100,
    'high': np.random.randn(n).cumsum() + 101,
    'low': np.random.randn(n).cumsum() + 99,
    'close': np.random.randn(n).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, n).astype(float),
    'ma_5': np.random.randn(n) + 100,
    'rsi_12': np.random.rand(n) * 100,
    'sentiment_mean': np.random.randn(n) * 0.1,
    'has_news': np.random.choice([0.0, 1.0], n),
    'news_sent_momentum_5d': np.random.randn(n) * 0.05,
    'pe_ttm': np.random.lognormal(3, 0.5, n),
    'f_score': np.random.randint(0, 5, n).astype(float),
    'day_of_week': np.arange(n) % 5,
})

from stoke_ml.features.pipeline import _discover_pk_columns, _discover_po_columns
pk = _discover_pk_columns(df)
po = _discover_po_columns(df)
print(f'PK ({len(pk)}): {pk}')
print(f'PO ({len(po)}): {po}')

# Verify key classifications
assert 'close' in pk
assert 'ma_5' in pk
assert 'rsi_12' in pk
assert 'pe_ttm' in pk
assert 'f_score' in pk
assert 'day_of_week' in pk
assert 'sentiment_mean' in po
assert 'has_news' in po
assert 'news_sent_momentum_5d' in po
print('Dynamic discovery: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "feat: replace hardcoded PK/PO column lists with dynamic prefix-based discovery

_discover_pk_columns / _discover_po_columns auto-classify columns by
prefix.  Adding/removing data sources no longer requires editing
_PAST_KNOWN_COLS / _PAST_OBSERVED_COLS.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: End-to-end integration test

**Files:**
- None (test script)

- [ ] **Step 1: Run a quick single-stock build_features test**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'open': np.random.randn(n).cumsum() + 100,
    'high': np.random.randn(n).cumsum() + 102,
    'low': np.random.randn(n).cumsum() + 98,
    'close': np.random.randn(n).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, n).astype(float),
})

# Add mock sentiment DataFrames
sent = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'sentiment_mean': np.random.randn(n) * 0.05,
    'sentiment_std': np.abs(np.random.randn(n) * 0.02),
    'news_count': np.random.poisson(3, n).astype(float),
    'positive_ratio': np.random.beta(3, 5, n),
    'negative_ratio': np.random.beta(2, 6, n),
})

guba = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'guba_sentiment_mean': np.random.randn(n) * 0.08,
    'guba_sentiment_std': np.abs(np.random.randn(n) * 0.03),
    'guba_post_count': np.random.poisson(10, n).astype(float),
    'guba_positive_ratio': np.random.beta(4, 6, n),
    'guba_negative_ratio': np.random.beta(3, 7, n),
})

fund = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'roe': np.random.randn(n) * 0.02 + 0.08,
    'roa': np.random.randn(n) * 0.01 + 0.04,
    'eps': np.random.randn(n) * 0.1 + 0.5,
    'revenue_yoy': np.random.randn(n) * 0.05 + 0.10,
    'profit_yoy': np.random.randn(n) * 0.08 + 0.12,
    'debt_ratio': np.random.beta(2, 5, n) * 0.6,
    'gross_margin': np.random.beta(5, 2, n) * 0.3 + 0.2,
    'net_margin': np.random.beta(3, 3, n) * 0.15 + 0.05,
})

val = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n),
    'pe_ttm': np.random.lognormal(3, 0.5, n),
    'pb_mrq': np.random.lognormal(0.5, 0.5, n),
    'ps_ttm': np.random.lognormal(1, 0.5, n),
    'pcf_ttm': np.random.lognormal(1.5, 0.5, n),
})

fp = FeaturePipeline(
    seq_len=60,
    use_emotion_refine=True,
    use_fundamental_refine=True,
    use_temporal_stats=True,
)
X, y, close = fp.build_features(
    df,
    sentiment_df=sent,
    guba_df=guba,
    fundamental_df=fund,
    valuation_df=val,
)
print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'Features per timestep: {X.shape[-1]}')
assert X.shape[0] > 0, 'No samples produced'
assert X.shape[-1] > 100, f'Too few features: {X.shape[-1]}'
print('Single-stock integration test: OK')
"
```

- [ ] **Step 2: Run a mini panel test**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline
import pandas as pd
import numpy as np

np.random.seed(42)
N, T = 5, 300
codes = ['000001', '000002', '000003', '000004', '000005']
sectors = ['Bank', 'Bank', 'Tech', 'Tech', 'Pharma']
rows = []
for i, code in enumerate(codes):
    for t in range(T):
        rows.append({
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=t),
            'stock_code': code,
            'open': np.random.randn() + 100 + i,
            'high': np.random.randn() + 102 + i,
            'low': np.random.randn() + 98 + i,
            'close': np.random.randn() + 100 + i,
            'volume': np.random.randint(1000, 10000),
            'sector': sectors[i],
            'size_proxy': float(i + 1),
        })
panel = pd.DataFrame(rows)

fp = FeaturePipeline(
    seq_len=60,
    use_emotion_refine=False,   # no aux data in panel, skip
    use_fundamental_refine=True,
    use_temporal_stats=True,
)
result = fp.build_panel_features(panel, horizon=1)
print(f'static: {result[\"static_features\"].shape}')
print(f'past_known: {result[\"past_known\"].shape}')
print(f'past_observed: {result[\"past_observed\"].shape}')
print(f'y_direction: {result[\"y_direction\"].shape}')
print(f'y_return: {result[\"y_return\"].shape}')
assert result['past_known'].shape[-1] > 50
assert result['past_observed'].shape[-1] > 5
print('Panel integration test: OK')
"
```

- [ ] **Step 3: Commit if any fixes were needed, otherwise done**

---

## Verification Checklist

After all tasks complete, run:

```bash
# Quick smoke test
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.features.transform import TemporalTransformer
from stoke_ml.features.emotion import EmotionRefiner
from stoke_ml.features.fundamental import FundamentalRefiner
from stoke_ml.features.selector import PanelFeatureSelector
print('All imports OK')
fp = FeaturePipeline(seq_len=60)
print(f'Modules: emotion={fp._emotion_refiner is not None}, '
      f'fundamental={fp._fundamental_refiner is not None}, '
      f'temporal={fp._temporal_transformer is not None}')
"
```

Expected output: all imports succeed, all three new modules are non-None by default.
