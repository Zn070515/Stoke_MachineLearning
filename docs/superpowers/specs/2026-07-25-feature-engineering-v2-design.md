# Feature Engineering v2 — Design Spec

> **Status:** Draft  
> **Date:** 2026-07-25  
> **Goal:** Deepen feature construction by fully utilizing processed auxiliary data,
> adding financial-logic-driven refinement layers, and enriching all PO columns with
> temporal statistics.

---

## 1. Architecture

Feature pipeline insertion points — no structural change to `build_panel_features`:

```
_engineer_features(df):
  1. Technical indicators (Alpha158 + extras, unchanged)
  2. Trend scoring + microstructure (unchanged)
  3. Merge aux DataFrames (existing, ZI fill) — [EXPANDED] broad PO columns
  4. [NEW] Emotion refinement — sentiment momentum/reversal/disagreement/attention
  5. [NEW] Per-stock fundamental refinement — quality/trend/stability
  6. [NEW] Temporal statistics — rolling mean/std/accel/zscore on all PO columns
  7. Temporal (lag + calendar, existing)

build_panel_features(all_dfs):
  ... (existing cross-sectional normalization) ...
  [NEW] Cross-sectional fundamental features — sector-relative valuation,
        leverage warning (sector percentile), valuation composite z-score
```

Column lists replaced with dynamic discovery — code auto-detects available columns
in each stock's DataFrame. Adding or removing data sources no longer requires
touching `_PAST_KNOWN_COLS` / `_PAST_OBSERVED_COLS`.

**Final feature budget:**

| Category | Before | After |
|----------|--------|-------|
| PK (past known) | 221 | ~241 (+20 fundamental refined) |
| PO base (past observed) | 32 | ~137 (+70 broad + ~35 emotion refined) |
| PO temporal | 0 | ~200-270 (base × ~2.5 transform factor) |
| Dynamic has_* | ~16 | ~80+ (auto-generated per PO column) |
| **Total** | **~253** | **~580** |

---

## 2. Module A: Broad PO Expansion

**Which data:** Capital flow, board, industry ranking, block trade, shareholder,
lockup, dividend — all already preprocessed in `*_processed/` directories.

**What changes:** `_merge_*` methods in `pipeline.py` extend their `suffix_cols`
parameter lists to include all processed columns from each data source.

### 2.1 Capital flow (+15 columns)

Data: `capital_flow_processed/` (20 columns, 5197 stocks, daily)

New columns exposed: `main_net`, `small_net`, `mid_net`, `large_net`, `super_net`,
`super_ratio`, `large_ratio`, `mid_ratio`, `small_ratio`, `main_ratio`,
`is_extreme_flow`, `consecutive_inflow_5d`, `consecutive_inflow_10d`,
`consecutive_inflow_20d`, `large_minus_small`

### 2.2 Board (+11 columns)

Data: `board_processed/` (28 columns, 5530 stocks, daily)

New columns exposed: `is_dt`, `is_yzt`, `consecutive_zt`, `market_state_strong`,
`market_state_volatile`, `market_state_weak`, `market_state_ice`,
`market_state_frenzy`, `market_state_normal`, `net_zt_proportion`, `seal_success`

### 2.3 Industry ranking (+21 columns)

Data: `industry_ranking_processed/` (34 columns, 5530 stocks, daily)

New columns exposed: `change_pct`, `ret_std`, `n_stocks`, `rank`, `up_count`,
`down_count`, `leader`, `leader_change`, `momentum_5d`, `momentum_20d`,
`momentum_60d`, `momentum_252d`, `sector_rrg_y`, `sector_rrg_x`,
`sector_rrg_quadrant`, `sector_breadth_raw`, `sector_rank_change`,
`is_top5_sector`, `is_sector_leader`

### 2.4 Block trade (+2 columns)

Data: `block_trade_processed/` (12 columns, 5111 stocks, sparse)

New columns exposed: `total_volume`, `amount_ratio`

### 2.5 Shareholder (+5 columns)

Data: `shareholder_processed/` (7 columns, 5472 stocks, sparse)

New columns exposed: `holder_num`, `change_num`, `change_ratio`, `avg_shares`,
`consecutive_quarter_decline`

### 2.6 Lockup (+6 columns)

Data: `lockup_processed/` (6 columns, 5125 stocks, sparse)

New columns exposed: `free_type`, `free_shares`, `able_shares`, `free_ratio`,
`is_upcoming`, `is_vc_backed`

### 2.7 Dividend (+10 columns)

Data: `dividend_processed/` (10 columns, 5282 stocks, sparse) — **Currently 0 columns in PO.**

New columns exposed: `bonus_rmb`, `transfer_ratio`, `bonus_ratio`, `plan`,
`dividend_yield`, `days_since_last_ex_div`, `effective_yield`,
`has_recent_dividend`, `plan_stage_encoded`, `dividend_growth`

### 2.8 Dynamic Column Discovery

Replace hardcoded `_PAST_KNOWN_COLS` and `_PAST_OBSERVED_COLS` with dynamic
discovery. Columns from each stock's engineered DataFrame are automatically
classified as PK (from technical/scoring/temporal/fundamental modules) or
PO (from merge methods). The `has_*` auto-generation is extended to cover all
new PO columns.

---

## 3. Module B: Emotion Refinement

**New file:** `stoke_ml/features/emotion.py`

**Class:** `EmotionRefiner` — stateless, pure functions operating on a stock's
daily DataFrame.

**Input:** Merged DataFrame with news + guba Gold columns (sentiment_mean,
sentiment_std, news_count, positive_ratio, negative_ratio, etc.)

**Output:** ~35 new emotion features added to the DataFrame in-place.

### 3.1 News emotion features (~15)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `news_sent_momentum_5d` | ma5(sentiment_mean) | Short-term trend |
| `news_sent_accel` | ma5 - ma20 of sentiment_mean | Acceleration/deceleration |
| `news_sent_reversal_5d` | sentiment_mean - min(sentiment_mean, 5d) | Bounce from extreme pessimism |
| `news_disagreement` | sentiment_std / (|sentiment_mean| + eps) | Divergence → regime change signal |
| `news_attention_z` | (count - ma20(count)) / std20(count) | Abnormal attention spike |
| `news_sent_volume` | sentiment_mean × log(count + 1) | Broad vs narrow optimism |
| `news_net_bullish` | positive_ratio - negative_ratio | Direct directional signal |
| `news_sent_streak` | Consecutive days of same sign on sentiment_mean | Persistence |
| `news_sent_vol_ratio` | std5(sentiment_mean) / std20(sentiment_mean) | Sentiment convergence → breakout |
| `news_sent_extreme` | sentiment_mean > percentile80 OR < percentile20 (20d) | Extreme sentiment regime flag |
| `news_pos_momentum` | positive_ratio - ma5(positive_ratio) | Shift in bullish proportion |
| `news_neg_momentum` | negative_ratio - ma5(negative_ratio) | Shift in bearish proportion |
| `news_count_momentum` | count / ma5(count) | Volume acceleration |
| `news_sent_skew` | (mean - median) / std proxy | Sentiment distribution shape |
| `news_body_ratio` | News with body text / total news | Rich-content ratio (where available) |

### 3.2 Guba emotion features (~15)

Same formula set, independently computed from Guba columns. Guba sentiment
behavior differs from news (retail panic vs institutional tone), so separate
features let the model learn both dynamics.

### 3.3 Combined cross-source features (~5)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `news_guba_divergence` | news_sentiment - guba_sentiment | Institutional vs retail gap |
| `news_guba_ratio` | news_count / (guba_count + 1) | Information source mix |
| `total_attention` | news_count + guba_count | Combined attention level |
| `cross_source_agreement` | sign(news_sent) == sign(guba_sent) | Consensus signal |
| `retail_panic` | guba_neg_ratio > 0.7 AND news_sent neutral | Retail fear not in news |

---

## 4. Module C: Fundamental Refinement

**New file:** `stoke_ml/features/fundamental.py`

**Class:** `FundamentalRefiner` — split into two execution phases:

- **Per-stock** (in `_engineer_features`): Features that only need the stock's own
  time series — F-score, quality composite, earnings quality, profitability/margin
  stability, growth quality, ROE/revenue/margin trends, earnings surprise.
- **Cross-sectional** (in `build_panel_features`): Features that need same-date
  sector comparisons — `pe_sector_ratio`, `pb_sector_ratio`, `ps_sector_ratio`,
  `leverage_warning`, `valuation_composite_z`, `pe_pb_divergence`, `deep_value`.
  These are computed after all stocks' data is assembled into the panel, using
  groupby on `sector_code` to get sector medians and percentiles per date.

**Input:** Forward-filled fundamental columns (roe, roa, eps, revenue_yoy,
profit_yoy, debt_ratio, gross_margin, net_margin) + valuation (pe_ttm, pb_mrq,
ps_ttm, pcf_ttm) + industry ranking data.

**Output:** ~20 refined fundamental features.

### 4.1 Quality composite features (~6)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `f_score` | Piotroski 9-point (ROA>0, CFO>0, ΔROA>0, accrual<0, Δleverage<0, Δcurrent>0, no_new_shares, Δmargin>0, Δturnover>0) | Multi-dimensional quality |
| `quality_composite` | Average of z-score(roe), z-score(gross_margin), z-score(-debt_ratio) | Simple quality proxy for stocks missing Piotroski inputs |
| `earnings_quality` | profit_yoy - revenue_yoy | Profit without revenue growth → one-time gains |
| `profitability_stability` | 1 - std(roe, 4q) / (|mean(roe, 4q)| + eps) | Stable profitability → moat signal |
| `margin_stability` | 1 - std(gross_margin, 4q) / (|mean(gross_margin, 4q)| + eps) | Stable margins → pricing power |
| `growth_quality` | revenue_yoy × gross_margin | High growth + high margin = quality growth |

### 4.2 Valuation refinement (~8)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pe_percentile_252d` | Percentile rank of pe_ttm in last 252 trading days | PE relative to own history |
| `pb_percentile_252d` | Same for pb_mrq | PB relative to own history |
| `pe_sector_ratio` | pe_ttm / median(pe_ttm in same sector) | Cross-sectional relative value |
| `pb_sector_ratio` | Same for pb_mrq | Cross-sectional relative value |
| `ps_sector_ratio` | Same for ps_ttm | Revenue-based relative value |
| `valuation_composite_z` | Average of z-score(-pe_percentile), z-score(-pb_percentile) | Composite cheapness signal |
| `pe_pb_divergence` | pe_percentile - pb_percentile | Earnings vs book value disagreement |
| `deep_value` | pe_percentile < 20 AND pb_percentile < 20 | Value regime flag |

### 4.3 Growth & trend (~6)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `roe_trend_4q` | Slope of roe over last 4 quarters | Direction of profitability |
| `roe_accel` | roe_trend_current - roe_trend_prior | Second derivative |
| `revenue_trend_4q` | Slope of revenue_yoy over last 4 quarters | Growth trajectory |
| `margin_trend_4q` | Slope of gross_margin over last 4 quarters | Margin expansion/contraction |
| `leverage_warning` | debt_ratio > sector_80th_percentile | High relative leverage flag |
| `earnings_surprise` | eps - ma4(eps) (quarterly diffs on daily data) | Positive earnings momentum |

### 4.4 Execution split

**Per-stock phase** (`_engineer_features`): All quality composite, stability, and
trend features in 4.1 and 4.3. Plus `pe_percentile_252d` and `pb_percentile_252d`
(these only need the stock's own history).

**Cross-sectional phase** (`build_panel_features`): All sector-relative features
in 4.2 that need same-date sector medians. Implemented as a post-hoc step after
all stock DataFrames are assembled into the panel, before z-score normalization.
Uses `groupby(["date", "sector_code"])` to compute sector medians and percentiles,
then joins back to individual stock rows.

---

## 5. Module D: Temporal Statistics

**New file:** `stoke_ml/features/transform.py`

**Class:** `TemporalTransformer` — stateless transform on PO columns.

### 5.1 Operators

| Operator | Windows | Output pattern | Applicable to |
|----------|---------|----------------|---------------|
| `rolling_mean` | 5, 10, 20 | `{col}_ma5`, `_ma10`, `_ma20` | All continuous, ratio, boolean |
| `rolling_std` | 20 | `{col}_std20` | Continuous, ratio |
| `accel` | (5 vs 20) | `{col}_accel` = ma5 - ma20 | Continuous, ratio |
| `zscore` | 20 | `{col}_z20` = (v - ma20)/std20 | Continuous, ratio |

### 5.2 Column type classification

Columns are auto-classified by prefix/name pattern:

- **Boolean** (`is_*`, `has_*`) → `rolling_mean` only (becomes proportion/frequency)
- **Ratio** (`*_ratio`, `*_pct`, `*_proportion`) → all four (bounded [0,1], zscore still meaningful)
- **Continuous** (everything else) → all four

### 5.3 NaN handling

- `rolling_mean` with min_periods = max(5, window//2) for short-series robustness
- `zscore` uses expanding std for first 20 observations
- `accel` is NaN when either input MA is NaN

---

## 6. Integration Points

### 6.1 pipeline.py: Merge method expansion

Each `_merge_*` method's `suffix_cols` parameter gets the full column list from
section 2. No logic change — just passing more column names.

### 6.2 pipeline.py: _engineer_features insertion

```python
def _engineer_features(self, df, ...):
    # ... existing technical + scoring + microstructure ...
    
    # [NEW] Emotion refinement (runs after aux merge, before temporal)
    if self._emotion_refiner is not None:
        df = self._emotion_refiner.refine(df)
    
    # [NEW] Fundamental refinement
    if self._fundamental_refiner is not None:
        df = self._fundamental_refiner.refine(df)
    
    # ... existing merge aux ...
    
    # [NEW] Temporal statistics on all PO columns
    if self._temporal_transformer is not None:
        df = self._temporal_transformer.transform(df)
    
    # ... existing temporal (lag + calendar) ...
```

### 6.3 pipeline.py: Dynamic column discovery + cross-sectional features

In `build_panel_features`:
1. After assembling all stock DataFrames, compute cross-sectional fundamental
   features (sector-relative valuation, leverage warning).
2. Replace hardcoded `_PAST_KNOWN_COLS` / `_PAST_OBSERVED_COLS` with dynamic
   discovery.

```python
# PK: columns from technical, scoring, temporal, fundamental modules
# PO: columns from merge methods (identified by source-specific patterns)
# Static: calendar columns, stock-level attributes
```

Column classification via prefix registry in each module (e.g., `EmotionRefiner`
registers its output column prefixes).

### 6.4 FeaturePipeline constructor flags

New optional flags defaulting to True:
- `use_emotion_refine: bool = True`
- `use_fundamental_refine: bool = True`
- `use_temporal_stats: bool = True`

---

## 7. File Changes

| File | Action | Scope |
|------|--------|-------|
| `stoke_ml/features/emotion.py` | **New** | `EmotionRefiner` class |
| `stoke_ml/features/fundamental.py` | **New** | `FundamentalRefiner` class |
| `stoke_ml/features/transform.py` | **New** | `TemporalTransformer` class |
| `stoke_ml/features/pipeline.py` | Modify | Merge method column expansion, insertion points, dynamic column discovery, new constructor flags |
| `stoke_ml/features/__init__.py` | Modify | Export new classes (if needed) |

No changes to training scripts, model architecture, or data storage.

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Column explosion → overfitting | VSN does per-feature gating; at ~580 dims this is within known working range |
| Sparse data columns creating NaN floods | `has_*` flags for every PO column; ZI fill handles sparsity |
| Piotroski F-score needs CFO (cash flow from ops) | Fallback to quality_composite if CFO unavailable |
| Sector-relative valuation needs sector mapping | Use `industry_ranking_processed/sector_code` already in data |
| Temporal transform multiplies columns → memory | Only apply to PO columns, not PK; use sliding_window_view for speed |
