# Multi-Shape Preprocessing System Design

> **Goal:** Extend the preprocessing pipeline to handle 10 new data types (capital flow, limit-up boards, block trades, shareholder count, lockup expiry, dividends, industry rankings, concept blocks, Sina fund flow, Tencent quote) — each with fundamentally different raw-data shapes — by classifying them into 4 morphological categories and creating a dedicated preprocessing module per category.

**Architecture:** Each morphological category gets a subdirectory under `stoke_ml/preprocessing/` with 1-2 new `PreprocessingStep` implementations. These feed into the existing `OutlierDetector → MissingImputer → CrossSectionNormalizer` chain for post-processing. Results are cached to `MarketWideStorage` (parquet partitions by year/month/stock) and merged into `FeaturePipeline` via dedicated `_merge_*` methods — same pattern as existing sentiment/margin/northbound data.

**Strategy A (preprocess → cache → merge):** All new data types follow the established pattern: run preprocessing once → store in partitioned parquet → FeaturePipeline reads cached results via left-join merge. This decouples preprocessing from training and maximizes cache reuse.

**Tech Stack:** pandas, numpy, scipy (existing); no new dependencies.

**Research Foundation:** This design incorporates findings from 15+ web/GitHub searches covering sell-side quant research (光大, 华泰, 国信, 开源, 东方), academic literature (Cont-Kukanov-Stoikov OFI, ICLR 2024 MFN graphs), and open-source factor platforms (FactorHub, aurumq-rl, AlphaAgent, jaqs-fxdayu, a-stock-data V3.3.0).

**Design Principle — Three-Phase Pipeline (production quant standard):**
1. Source-specific cleaning & decomposition (this spec's new Steps)
2. Cross-sectional standardization (existing: `CrossSectionNormalizer`)
3. Risk-factor neutralization (existing: sector/size/adaptive stages)

---

## 1. Morphological Classification

### Shape A: `daily_continuous/` — Daily continuous
Data is per-stock daily granularity with continuous numeric values.

| Data type | Storage key | Columns |
|-----------|------------|---------|
| capital_flow | `capital_flow` | main_net, super_net, large_net, mid_net, small_net |
| sina_fund_flow | `sina_fund_flow` | net_amount, turnover |

### Shape B: `event_sparse/` — Event-sparse
Data is discrete events on sparse dates. Must be aggregated/forward-filled to daily granularity.

| Data type | Storage key | Frequency | Key columns |
|-----------|------------|-----------|-------------|
| block_trade | `block_trade` | Irregular | deal_price, close_price, premium_pct, volume, amount, buyer, seller |
| shareholder | `shareholder` | Quarterly | holder_num, change_num, change_ratio, avg_shares |
| lockup | `lockup` | Irregular | free_type, free_shares, able_shares, free_ratio |
| lockup_upcoming | `lockup_upcoming` | Future-dated | same as lockup + days_until |
| dividend | `dividend` | Irregular | bonus_rmb, transfer_ratio, bonus_ratio, plan |

### Shape C: `cross_sectional/` — Cross-sectional ranking
Data is market-wide daily (not per-stock). Must be broadcast to individual stocks.

| Data type | Storage key | Shape |
|-----------|------------|-------|
| limit_up_zt | `limit_up_zt` | Pool of stocks that hit limit-up today |
| limit_up_zb | `limit_up_zb` | Pool of stocks that busted (炸板) today |
| limit_up_dt | `limit_up_dt` | Pool of stocks that hit limit-down today |
| limit_up_yzt | `limit_up_yzt` | Yesterday's ZT pool — today's performance |
| limit_up_sentiment | `limit_up_sentiment` | Market-wide: break_rate, advance_rate, max_height |
| industry_ranking | `industry_ranking` | Per-industry: rank, change_pct, up_count, down_count, leader |

### Shape D: `categorical/` — Multi-label categorical
Data is per-stock set of tags (concept board memberships).

| Data type | Storage key | Shape |
|-----------|------------|-------|
| concept_blocks | `concept_blocks` | Per stock: list of (board_name, board_code, board_change_pct) |

---

## 2. Directory Structure

```
stoke_ml/preprocessing/
├── __init__.py
├── base.py              # PreprocessingStep / PreprocessingChain (existing)
├── config.py            # build_pipeline_from_config (extend)
├── pipeline.py          # PreprocessingPipeline (existing)
├── registry.py          # FeatureRegistry (existing)
│
├── text/                # Text sentiment chain (existing)
├── numeric/             # Numeric OHLCV chain (existing)
├── monitor/             # Quality + drift monitors (existing)
│
├── daily_continuous/    # 🆕 Shape A
│   ├── __init__.py
│   └── flow.py          # FlowDecomposer
│
├── event_sparse/        # 🆕 Shape B
│   ├── __init__.py
│   └── aggregator.py    # EventToDaily
│
├── cross_sectional/     # 🆕 Shape C
│   ├── __init__.py
│   ├── board.py         # BoardBroadcaster
│   └── sector.py        # SectorBroadcaster
│
└── categorical/         # 🆕 Shape D
    ├── __init__.py
    └── encoder.py       # ConceptBlockEncoder
```

---

## 3. PreprocessingStep Designs

### 3.1 `FlowDecomposer` (Shape A: daily_continuous)

**Purpose:** Decompose raw capital flow amounts into a multi-layer factor suite: ratios, OFI (Order Flow Imbalance), persistence, intensity, divergence, and purified alpha (return-contamination stripped).

**Research basis:** Cont-Kukanov-Stoikov OFI framework; 开源证券 residual flow strength (IR 2.05-3.18); 华泰证券 dual-branch Transformer (RankIC 10.96%); 光大证券 size-tier interaction analysis.

**`__init__` parameters:**
- `persistence_windows: tuple = (5, 10, 20)` — rolling windows for consecutive inflow count
- `intensity_windows: tuple = (20, 60)` — rolling windows for z-score intensity
- `divergence_window: int = 5` — window for price-flow divergence check
- `flow_halflife: int = 7` — decay half-life (days) for weighted moving average of flow z-scores
- `extreme_threshold: float = 1.8` — |z| > threshold flags extreme flow day
- `residualize: bool = True` — run cross-sectional regression to strip return contamination

**`transform(df)` logic — 5-layer decomposition:**

*Layer 1 — Size-tier ratios:*
1. `super_ratio = super_net / (|super|+|large|+|mid|+|small| + 1e-8)`, same for large/mid/small
2. `main_ratio = main_net / (|super|+|large|+|mid|+|small| + 1e-8)`

*Layer 2 — OFI intensity (z-score based, industry-neutral after):*
3. Per stock: `flow_z = (main_net - rolling_mean(20d)) / rolling_std(20d)`, clip at ±5
4. `flow_intensity = |flow_z|` — extreme flag when `|flow_z| > extreme_threshold`
5. `flow_momentum` = decay-weighted EMA of `flow_z` with half-life = `flow_halflife`

*Layer 3 — Persistence:*
6. Per stock, per window w ∈ persistence_windows: consecutive trading days count where `main_net > 0`

*Layer 4 — Divergence:*
7. `price_z_5d` = 5-day z-score of close price change
8. `flow_z_5d` = 5-day z-score of cumulative main_net
9. `flow_price_divergence` = 1 if sign(price_z_5d) ≠ sign(flow_z_5d) else 0 (confirmation failure)

*Layer 5 — Residualization (optional, if `residualize=True`):*
10. Cross-sectional regression each period: `ret_t = α + β * flow_imbalance_t + ε_t`
11. `flow_alpha_residual = ε_t` — purified alpha stripped of return synchronization

*Layer 6 — Size-tier spread:*
12. `large_minus_small = large_ratio - small_ratio` (large-ticket minus retail flow spread)
13. Research: large-ticket flow works best in CSI 300, small-ticket shows negative alpha in large-caps ("dumb money crowding"); spread should be neutralized within size quintiles

**Output columns added:** `super_ratio`, `large_ratio`, `mid_ratio`, `small_ratio`, `main_ratio`, `flow_z`, `flow_intensity`, `is_extreme_flow`, `flow_momentum`, `consecutive_inflow_{w}d`, `flow_price_divergence`, `flow_alpha_residual` (if residualize), `large_minus_small`

**Dependencies:** None (pure pandas/numpy).

---

### 3.2 `EventToDaily` (Shape B: event_sparse)

**Purpose:** Single class dispatching by `event_type`. Converts sparse events to daily per-stock features, with forward-fill and optional exponential decay.

**Research basis:** 光大证券 HN_z/IHN/LHRD factor definitions (2020); 华泰证券 7-factor lockup scoring card (72.5% win rate); 海通证券 event→factor 3 conditions; NYSE Stern dividend normalization pipeline; FINRA block trade price impact decomposition.

**`__init__(event_type: str, calendar: TradingCalendar)`**
- `event_type` ∈ {"block_trade", "shareholder", "lockup", "dividend"}

---

**`transform(df)` — block_trade:**

1. `groupby(["date", "stock_code"])` → aggregate:
   - `premium_pct_mean` = simple mean
   - `premium_pct_wavg` = Σ(premium_pct_i × amount_i) / Σ(amount_i) — VWAP-weighted (research: weighted > simple)
   - `total_amount` = Σ(amount_i)
   - `trade_count` = count of trades
   - `buyer_is_inst` = any(buyer contains "机构" or "专用" or "瑞银" or "沪股通" etc.)
2. Reindex to full trading calendar → forward-fill (max 5 days, block trades are opportunistic) → ZI fill for gaps beyond 5 days
3. Compute 6-day amount volatility (光大 composite factor, 19.54% annualized): `amount_vol_6d = rolling_std(total_amount, 6) / rolling_mean(total_amount, 6)`
4. Price impact decomposition (permanent vs temporary):
   - `permanent_impact = (close_price[t+1] - close_price[t-1]) / close_price[t-1]` — lasting price effect
   - `temporary_impact = premium_pct_wavg - permanent_impact` — transient liquidity effect
5. Counter-intuitive signal encoding (research finding): premium 0-5% → negative forward return; discount 8-10% → +14.3% in 60d. Binary feature: `is_deep_discount = premium_pct_mean < -8%`

**Output columns:** `premium_pct_mean`, `premium_pct_wavg`, `total_amount`, `trade_count`, `buyer_is_inst`, `amount_vol_6d`, `permanent_impact`, `temporary_impact`, `is_deep_discount`

---

**`transform(df)` — shareholder:**

1. Sort by END_DATE, groupby stock_code → keep latest per quarter
2. Forward-fill to daily: `holder_num`, `change_ratio`, `avg_shares`
3. Compute HN_z (光大 2020, IC -3.57%, ICIR 0.54):
   ```
   HN_z = (holder_num_t - rolling_mean(holder_num, 8Q)) / rolling_std(holder_num, 8Q)
   ```
   Note: sign reversed in final output (declining holders = positive factor)
4. Compute `consecutive_quarter_decline`: count of consecutive quarters with `change_ratio < 0`
5. Compute PCRC (人均持股占比变动, IR 2.6):
   ```
   PCRC = (avg_shares_t / avg_shares_{t-4Q}) - 1  // YoY change in per-capita holdings
   ```
6. Dual-concentration flag (18% annualized pattern):
   - `price_falling` = close < SMA(60)
   - `holders_falling` = change_ratio < 0
   - `dual_concentration_signal` = price_falling AND holders_falling (strongest buy signal)
7. Filter: `is_sub_new = days_since_listing < 252` (sub-new stocks excluded — their holder data is unreliable)

**Output columns:** `holder_num`, `change_ratio`, `avg_shares`, `HN_z`, `consecutive_quarter_decline`, `PCRC`, `dual_concentration_signal`, `is_sub_new`

---

**`transform(df)` — lockup:**

1. From history table: most recent lockup details
2. From upcoming table: days_until_unlock for each upcoming event
3. Aggregate overlapping events:
   ```
   unlock_pressure = Σ_i [ free_ratio_i × exp(-λ × days_until_i) ]
   where λ = ln(2) / 30  (30-day half-life for approaching unlocks)
   ```
4. `total_upcoming_ratio = Σ_i(free_ratio_i)` — cumulative ratio of float to be unlocked
5. `days_to_nearest_unlock` = min(days_until_i) across all upcoming events
6. `unlock_count_upcoming` = number of upcoming unlock events within 90 days
7. `is_vc_backed` = any(free_type contains "首发" or "IPO") — VC-backed unlocks have ~4x larger negative impact (-2.81% vs -0.62%)
8. For history: `unlock_return_30d` = close price change 30d after each historical unlock date (trailing signal)
9. Forward-fill daily, with `unlock_pressure` naturally decaying as days_until decreases

**Output columns:** `unlock_pressure`, `total_upcoming_ratio`, `days_to_nearest_unlock`, `unlock_count_upcoming`, `is_vc_backed`, `unlock_return_30d`, `free_type`, `free_ratio_latest`

---

**`transform(df)` — dividend:**

1. Sort by EX_DIVIDEND_DATE
2. Compute dividend yield at ex-div date:
   ```
   dividend_yield = bonus_rmb / close_price  (per-share dividend / price)
   ```
3. Compute `days_since_last_ex_div`
4. Forward-fill dividend_yield with exponential decay:
   ```
   effective_yield = dividend_yield × exp(-λ × days_since)
   where λ = ln(2) / 90  (90-day half-life: dividend info decays over ~3 months)
   ```
5. Normalization pipeline (NYSE Stern standard):
   a. Log-transform: `ln(1 + rolling_12m_total_dividend / price)`
   b. Double-winsorize at 1%/99%
   c. Cross-sectional percentile rank → map to inverse normal CDF
   d. Re-standardize to mean=0, std=1
   e. Orthogonalize against size (Cholesky decomposition)
6. `dividend_growth` = (current dividend - previous dividend) / previous dividend (if both exist)
7. `has_recent_dividend` = days_since_last_ex_div ≤ 90
8. `plan_stage_encoded` = ordinal encoding of ASSIGN_PROGRESS (预案→决案→实施)

**Output columns:** `dividend_yield`, `effective_yield`, `dividend_yield_normalized`, `days_since_last_ex_div`, `dividend_growth`, `has_recent_dividend`, `plan_stage_encoded`

---

**Dependencies:** `TradingCalendar` for reindex to trading days; optional close price join for dividend yield and lockup return calculation.

---

### 3.3 `BoardBroadcaster` (Shape C: cross_sectional)

**Purpose:** Convert market-wide limit-up pool membership into per-stock daily features, with seal strength scoring and market state classification.

**Research basis:** 国泰君安 5-factor sentiment timing model; GF Securities ULTIMATE-FUSION V12 6-dimensional scoring; `simonlin1212/a-stock-data` V3.3.0 reference implementation; A-share board sentiment empirical thresholds.

**`__init__` parameters:**
- `consecutive_lookback: int = 20` — window for board_height calculation
- `market_state_windows: dict = {"strong": (80, 0.15), "volatile": (None, 0.25), "weak": (20, None)}` — thresholds for state classification

**`transform(df, pools: dict[str, pd.DataFrame])` logic:**

*Layer 1 — Board membership booleans:*
1. For each date × stock:
   - `is_zt` = stock ∈ zt_pool
   - `is_zb` = stock ∈ zb_pool (炸板)
   - `is_dt` = stock ∈ dt_pool (跌停)
   - `is_yzt` = stock ∈ yzt_pool (昨日涨停今日表现)

*Layer 2 — Consecutive board tracking:*
2. Per stock, scan backward:
   ```
   IF is_zt[t] AND is_zt[t-1]: consecutive_zt += 1
   ELSE IF is_zt[t]: consecutive_zt = 1
   ELSE: consecutive_zt = 0
   ```
3. `board_height_20d` = max(consecutive_zt) in rolling 20 days

*Layer 3 — Seal strength classification (封板强度):*
4. Seal type from limit_up_zt pool's `seal_type` field:
   - `seal_type_one_price` = 一字板 (opened at limit, never traded below)
   - `seal_type_hand_change` = 换手板 (opened, traded heavily, then sealed)
   - `seal_type_t_shape` = T字板 (sealed, opened/broken, re-sealed)
5. Seal strength score (higher = stronger):
   ```
   seal_strength = base_score(seal_type)
   base_score = {一字板: 1.0, T字板: 0.7, 换手板: 0.5}
   × seal_time_factor  (earlier seal time → higher score: morning=1.0, afternoon=0.6)
   × seal_cycle_penalty (每多一次开板 → ×0.5)
   ```
6. `seal_success` = did NOT end up in zb_pool (炸板池) — boolean

*Layer 4 — Market-level sentiment indices (broadcast to all stocks):*
7. From limit_up_sentiment (or computed from pools):
   ```
   break_rate = |zb_pool[t]| / (|zt_pool[t]| + |zb_pool[t]| + 1)
   advance_rate = |yzt_pool[t] with consecutive_zt>=2| / |zt_pool[t-1]|
   net_zt_proportion = (|zt_pool[t]| - |dt_pool[t]|) / total_stocks
   zt_proportion = |zt_pool[t]| / total_stocks
   dt_proportion = |dt_pool[t]| / total_stocks
   max_board_height = max(consecutive_zt across all stocks)
   ```

*Layer 5 — Market state classification (broadcast to all stocks):*
8. `market_state` ∈ {"strong", "normal", "volatile", "weak", "ice"} based on:
   - strong: |zt_pool| > 80 AND break_rate < 15%
   - volatile: break_rate > 25%
   - weak: |zt_pool| < 20
   - ice (冰点): 4+ of: advancers<1000, max_height≤3, advance_rate<20%, break_rate>40%, yzt_return<0%
   - frenzy (高潮): |zt_pool| > 120 AND break_rate < 10% AND advance_rate > 60%
   - normal: otherwise
9. One-hot encode market_state into 6 boolean columns

**Output columns:** `is_zt`, `is_zb`, `is_dt`, `is_yzt`, `consecutive_zt`, `board_height_20d`, `seal_type_one_price`, `seal_type_hand_change`, `seal_type_t_shape`, `seal_strength`, `seal_success`, `break_rate`, `advance_rate`, `net_zt_proportion`, `zt_proportion`, `dt_proportion`, `max_board_height`, `market_state_strong`, `market_state_volatile`, `market_state_weak`, `market_state_ice`, `market_state_frenzy`, `market_state_normal`

**Dependencies:** Date alignment with OHLCV index; pools dict from LimitUpSource.

---

### 3.4 `SectorBroadcaster` (Shape C: cross_sectional)

**Purpose:** Broadcast industry ranking data to individual stocks via sector mapping, with RRG-style relative strength and industry momentum computation.

**Research basis:** Two-stage RFE+RNN sector rotation methodology; RRG (Relative Rotation Graph) 252-bar z-score framework; DFQ genetic programming automated factor mining (18.42% excess return); 申万 industry classification (L1: 31, L2: ~134, L3: ~346).

**`__init__` parameters:**
- `momentum_windows: tuple = (5, 20, 60, 252)` — rolling windows for sector momentum
- `breadth_normalize_window: int = 252` — window for breadth z-score
- `sector_mapping_source: str = "concept_blocks"` — source for stock→industry mapping

**`transform(df, industry_ranking: pd.DataFrame, sector_map: dict)` logic:**

*Layer 1 — Stock-to-sector join:*
1. Requires `sector_map`: dict `stock_code → industry_code` (built from concept_blocks `board_type="industry"` or 申万 static mapping)
2. For each date × stock, look up stock's industry → left-join industry_ranking columns

*Layer 2 — Sector-level features per stock:*
3. Direct join columns: `sector_rank`, `sector_change_pct`, `sector_breadth` (= up_count - down_count), `sector_leader_change`
4. `is_sector_leader` = (stock == leader_stock for its industry that day)

*Layer 3 — Sector momentum (multi-timeframe):*
5. Per industry, per window w ∈ momentum_windows: `sector_momentum_{w}d` = cumulative return over w trading days
6. RRG normalization (252-bar z-score):
   ```
   RS_ratio = (sector_price / benchmark_price)  — relative strength
   RS_momentum = RS_ratio - SMA(RS_ratio, 10)   — rate of change of RS
   sector_rrg_x = zscore(RS_momentum, 252)       — normalized momentum (JdK RS-Momentum)
   sector_rrg_y = zscore(RS_ratio, 252)          — normalized strength (JdK RS-Ratio)
   ```
7. RRG quadrant assignment: leading (x>0, y>0), weakening (x<0, y>0), lagging (x<0, y<0), improving (x>0, y<0)

*Layer 4 — Sector breadth normalization:*
8. Raw breadth: `raw_breadth = (up_count - down_count) / total_industry_stocks` → [-1, +1]
9. Cross-sectional z-score across all industries (window = `breadth_normalize_window`):
   ```
   sector_breadth_z = (raw_breadth - cross_sectional_mean) / cross_sectional_std
   ```
10. Winsorize at 1%/99% before z-scoring

*Layer 5 — Sector rotation signals:*
11. `sector_rank_change` = sector_rank[t-1] - sector_rank[t] (positive = improving)
12. `sector_relative_strength` = sector_change_pct - market_average_change_pct
13. `is_top5_sector` = sector_rank ≤ 5

**Output columns:** `sector_rank`, `sector_change_pct`, `sector_breadth`, `sector_breadth_z`, `sector_leader_change`, `is_sector_leader`, `sector_momentum_{w}d` (per window), `sector_rrg_x`, `sector_rrg_y`, `sector_rrg_quadrant`, `sector_rank_change`, `sector_relative_strength`, `is_top5_sector`

**Dependencies:** `sector_map` dict (can be built once from concept_blocks and cached); industry_ranking DataFrame from `IndustryRankingSource`.

---

### 3.5 `ConceptBlockEncoder` (Shape D: categorical)

**Purpose:** Encode multi-label concept board membership as multi-hot features + concept heat scoring + concept momentum + board co-occurrence features.

**Research basis:** 东吴证券 HIST hidden-concept discovery model; 国信证券 3D concept heat framework (volume_ratio × 0.4 + momentum × 0.4 + news × 0.2); concept dimension momentum effect (formation 6mo, holding 6mo, monthly excess 1.25%); FactorHub (50+ preset factors); aurumq-rl (296 factors with board-aware price limits).

**`__init__` parameters:**
- `top_n: int = 100` — number of most frequent concepts to encode (Dongwu uses 356, THS uses 1023; 100-200 balances coverage vs dimensionality)
- `min_stocks_per_board: int = 5` — filter out micro-boards with too few members
- `heat_decay: float = 0.4` — weight decay for lagged concept heat (国信 framework)
- `momentum_windows: tuple = (3, 6, 12)` — months for concept momentum (formation/holding pattern)

**`transform(df)` logic:**

*Layer 1 — Concept vocabulary construction:*
1. Collect all `board_name` across all dates
2. Filter: keep boards with ≥ `min_stocks_per_board` member stocks
3. Select top-N by total market cap or member count → build vocabulary `V` of size N
4. Sort V alphabetically for deterministic column ordering

*Layer 2 — Multi-hot encoding:*
5. For each date × stock: `mh_vector[j] = 1` if stock belongs to concept `V[j]`, else `0`
6. Column names: `cb_{j}` where j ∈ [0, N-1]

*Layer 3 — Derived per-stock features:*
7. `board_count` = number of concepts this stock belongs to (cardinality)
8. `board_momentum_mean` = mean of all boards' `change_pct` for this stock (equal-weight)
9. `board_momentum_max` = max board `change_pct` (strongest concept signal)
10. `board_momentum_wavg` = weighted average by concept market cap (if available)
11. `has_hot_board` = boolean: any of stock's boards in top 10% by `change_pct` that day

*Layer 4 — Concept heat score (国信 3D framework):*
12. Per concept per day, compute 3 dimensions:
    - `volume_score` = percentile(concept_volume_ratio, 252)
    - `momentum_score` = percentile(concept_momentum_{3,6,12}m, 252)
    - `news_score` = percentile(concept_news_count, 252) — optional, if news data available
13. `concept_heat = 0.4 × volume_score + 0.4 × momentum_score + 0.2 × news_score` (or 0.5/0.5 without news)
14. Per stock: `avg_concept_heat` = mean heat of all boards it belongs to

*Layer 5 — Concept momentum & rotation:*
15. Per concept, per window w ∈ momentum_windows (months):
    - `concept_return_{w}m` = cumulative return over w months, skip most recent 1 month
16. Per stock: `concept_momentum_{w}m` = mean of all its boards' returns (formation 6mo → holding 6mo pattern, monthly excess 1.25%)

*Layer 6 — Board co-occurrence features:*
17. `board_overlap_score` = for each stock, average Jaccard similarity with other stocks in its boards (captures "concept neighborhood density")
18. `is_concept_leader` = stock is the lead_stock for any of its boards that day

*Layer 7 — Missing dates:*
19. Forward-fill multi-hot vectors and derived features (concept membership changes slowly, quarterly rebalancing)
20. ZI fill for stocks with no concept data at all

**Output columns:** `cb_{0..N-1}` (multi-hot), `board_count`, `board_momentum_mean`, `board_momentum_max`, `board_momentum_wavg`, `has_hot_board`, `avg_concept_heat`, `concept_momentum_{w}m` (per window), `board_overlap_score`, `is_concept_leader`

**Dimensionality note:** At top_n=100, this adds ~115 columns. If too wide, apply LASSO pre-filtering (research: Rank IC 13.1% after LASSO selection) or keep only layers 3-6 (derived features, ~10 columns) without multi-hot.

**Dependencies:** None (pure pandas).

---

## 4. Pipeline Architecture — Production Quant Patterns

### 4.1 Three-Phase Processing (industry standard)

Each data source flows through three independent stages:

```
Raw Data → [Phase 1: Source Cleaner] → [Phase 2: Cross-Sectional Standardizer] → [Phase 3: Risk Neutralizer] → Factor Store
```

- **Phase 1** (source-specific): the new Steps defined in Section 3 — FlowDecomposer, EventToDaily, BoardBroadcaster, SectorBroadcaster, ConceptBlockEncoder
- **Phase 2** (cross-sectional, existing): `CrossSectionNormalizer` — sector/industry neutralization, size neutralization, rank normalization, z-score, adaptive volatility
- **Phase 3** (risk, existing): `CrossSectionNormalizer` adaptive stage — orthogonalization against size/value/momentum risk factors

This three-phase split is used by jaqs-fxdayu, aurumq-rl, and the Multi-Factor-Strategy-Development-Framework.

### 4.2 Feature Store Pattern

Results from each preprocessing chain are cached to `MarketWideStorage` as partitioned Parquet files. This mirrors the Factor Hub pattern used in production:

```
a_shares/
├── flow_processed/{year}/{month}/{stock}.parquet      # Shape A output
├── event_{type}_processed/{year}/{month}/{stock}.parquet  # Shape B output
├── board_processed/{year}/{month}/{stock}.parquet      # Shape C output
├── sector_processed/{year}/{month}/{stock}.parquet     # Shape C output
└── concept_processed/{year}/{month}/{stock}.parquet    # Shape D output
```

### 4.3 PIT (Point-in-Time) Discipline

Critical for all new data types:
- Same-day data availability: capital_flow and limit_up pools are same-day (post-close) → assigned to `next_trading_day` via `TradingCalendar`
- Quarterly data (shareholder): published with delay → `END_DATE` is the reporting date, `PUBLISH_DATE` is when it became knowable → merge on `PUBLISH_DATE + 1`
- Event data (block_trade, dividend, lockup): trade/event date is knowable same-day → PIT shift by 1 day
- All features lagged 1 trading day before entering FeaturePipeline (enforced in merge methods)

### 4.4 Config-Driven Assembly

The `build_pipeline_from_config()` function is the single entry point. Adding a new data source requires:
1. One new `PreprocessingStep` class in the appropriate shape directory
2. One entry in `_STEP_REGISTRY`
3. One chain assembly block in `build_pipeline_from_config()`
4. One config block in `config.yaml`
5. One `_merge_*` method in `FeaturePipeline`

No changes to `PreprocessingPipeline`, `PreprocessingChain`, or `PreprocessingStep` base classes — they are shape-agnostic.

---

## 5. Config Integration

Extend `stoke_ml/preprocessing/config.py`:

### New `_STEP_REGISTRY` entries

```python
from stoke_ml.preprocessing.daily_continuous.flow import FlowDecomposer
from stoke_ml.preprocessing.event_sparse.aggregator import EventToDaily
from stoke_ml.preprocessing.cross_sectional.board import BoardBroadcaster
from stoke_ml.preprocessing.cross_sectional.sector import SectorBroadcaster
from stoke_ml.preprocessing.categorical.encoder import ConceptBlockEncoder

_STEP_REGISTRY.update({
    "FlowDecomposer": FlowDecomposer,
    "EventToDaily": EventToDaily,
    "BoardBroadcaster": BoardBroadcaster,
    "SectorBroadcaster": SectorBroadcaster,
    "ConceptBlockEncoder": ConceptBlockEncoder,
})
```

### New chains in `build_pipeline_from_config()`

```python
# ── Shape A: daily_continuous chain "flow" ──
flow_cfg = pp_cfg.get("flow", {})
flow_chain = PreprocessingChain(name="flow")
flow_chain.add(FlowDecomposer())
oc = num_cfg.get("outlier", {})
flow_chain.add(OutlierDetector(threshold=oc.get("threshold", 5.0), clip=oc.get("clip", True)))
mc = num_cfg.get("missing", {})
flow_chain.add(MissingImputer(short_gap_max=mc.get("short_gap_max", 2),
                               medium_gap_max=mc.get("medium_gap_max", 10)))
cs = num_cfg.get("cross_section", {})
flow_chain.add(CrossSectionNormalizer(enabled=cs.get("enabled", True),
                                       stages=cs.get("stages", ["sector", "size", "adaptive"])))
pp.register_chain("flow", flow_chain)

# ── Shape B: event_sparse chains (one per event type) ──
for etype in ["block_trade", "shareholder", "lockup", "dividend"]:
    echain = PreprocessingChain(name=f"event_{etype}")
    ecfg = pp_cfg.get("event", {}).get(etype, {})
    echain.add(EventToDaily(event_type=etype))
    echain.add(MissingImputer(short_gap_max=ecfg.get("short_gap_max", 5),
                               medium_gap_max=ecfg.get("medium_gap_max", 60)))
    pp.register_chain(f"event_{etype}", echain)

# ── Shape C: cross_sectional chains ──
board_chain = PreprocessingChain(name="board")
board_chain.add(BoardBroadcaster())
pp.register_chain("board", board_chain)

sector_chain = PreprocessingChain(name="sector")
sector_chain.add(SectorBroadcaster())
pp.register_chain("sector", sector_chain)

# ── Shape D: categorical chain ──
concept_chain = PreprocessingChain(name="concept")
ccfg = pp_cfg.get("concept", {})
concept_chain.add(ConceptBlockEncoder(top_n=ccfg.get("top_n", 50)))
pp.register_chain("concept", concept_chain)
```

### Config YAML additions (config.yaml)

```yaml
preprocessing:
  # ... existing text:, numeric: sections ...
  
  flow:
    enabled: true
    decompose_ratios: true
    persistence_windows: [5, 10, 20]
  
  event:
    block_trade:
      short_gap_max: 5
      medium_gap_max: 60
    shareholder:
      short_gap_max: 90      # quarterly data
      medium_gap_max: 365
    lockup:
      short_gap_max: 5
      medium_gap_max: 90
    dividend:
      short_gap_max: 30
      medium_gap_max: 365
      decay_halflife_days: 90
  
  cross_sectional:
    board:
      enabled: true
    sector:
      enabled: true
      sector_mapping_source: "concept_blocks"  # or "static_csv"
  
  concept:
    enabled: true
    top_n: 50
```

---

## 6. FeaturePipeline Integration

Add merge methods to `stoke_ml/features/pipeline.py` following the existing `_merge_sentiment()` pattern:

```python
def _merge_capital_flow(self, df, stock_code, start_date, end_date):
    """Merge preprocessed capital flow features."""
    flow_df = self._market_storage.load("capital_flow", stock_code, start_date, end_date)
    if flow_df.empty:
        return self._zi_fill(df, ["flow_*"], prefix="capital_flow")
    return df.merge(flow_df, on="date", how="left")

def _merge_block_trade(self, df, stock_code, start_date, end_date):
    """Merge preprocessed block trade daily features."""
    ...

def _merge_shareholder(self, df, stock_code, start_date, end_date): ...
def _merge_lockup(self, df, stock_code, start_date, end_date): ...
def _merge_dividend(self, df, stock_code, start_date, end_date): ...
def _merge_board(self, df, stock_code, start_date, end_date): ...
def _merge_sector(self, df, stock_code, start_date, end_date): ...
def _merge_concept(self, df, stock_code, start_date, end_date): ...
```

Each follows the same pattern:
1. Load from `MarketWideStorage` (or a new dedicated storage instance)
2. If empty → ZI fill with `has_*` flags
3. Left-join merge on `date`
4. Lag 1 trading day (PIT anti-leakage)

New `use_*` flags in `FeaturePipeline.__init__`:
```python
use_capital_flow: bool = True,
use_block_trade: bool = True,
use_shareholder: bool = True,
use_lockup: bool = True,
use_dividend: bool = True,
use_board: bool = True,         # limit-up board features
use_sector: bool = True,        # industry ranking features
use_concept: bool = True,       # concept block features
```

---

## 7. Preprocessing Script

New script `scripts/preprocess_new_data.py`:

```bash
# Preprocess all new data types for all stocks
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type all

# Preprocess specific shapes
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type flow
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type event
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type board
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type concept

# Single stock, date range
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py \
    --type flow --stocks 600519 --start 2024-01-01
```

The script:
1. Loads raw data from `MarketWideStorage` (downloaded by `scripts/download_datacenter.py`)
2. Runs the appropriate preprocessing chain via `PreprocessingPipeline`
3. Saves preprocessed results back to a separate storage location (e.g., `a_shares/flow_processed/` or a `_preprocessed` suffix)

---

## 8. Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Raw data file missing for a stock | ZI fill — all feature values = 0, all `has_*` flags = False |
| EventToDaily receives empty event df | Return empty daily df — MissingImputer handles fill |
| BoardBroadcaster receives empty pool for a date | All `is_zt` etc. = False for that date |
| ConceptBlockEncoder encounters unseen board name | Map to closest top-N board by name similarity, or ignore |
| FlowDecomposer encounters all-zero flow day | Ratios = 0, intensity = 0 (no division by zero due to epsilon) |
| CrossSectionNormalizer runs on single stock | Skip sector/industry neutralization, fall back to rank/z-score only |

---

## 9. Testing Strategy

Per module: unit test with synthetic DataFrames covering:
- Normal operation (5-stock, 20-day panel)
- Empty input → empty output (no crash)
- Single stock → correct broadcast
- Missing intermediate dates → forward-fill verified
- Edge values: zero flow, 100% premium, negative holder change

Integration test: end-to-end for one stock from raw → preprocessed → merged into FeaturePipeline.

---

## 10. Research References

### 10.1 Academic & Sell-Side Research

**Capital Flow & OFI:**
- Cont, Kukanov, Stoikov (2014) — "The Price Impact of Order Book Events" (OFI framework, tick-level LOB deltas)
- ICLR 2024 — MFN co-occurrence graph + XGBoost for capital flow prediction
- 华泰证券 (2023) — Dual-branch Transformer on 30-min flow data, RankIC 10.96%
- 开源证券 (2024) — Residual fund flow strength factor (ML_C), IR 2.05-3.18, strips reversal exposure
- 光大证券 (2021) — Size-tier decomposition: large-ticket flow in CSI 300; retail flow negative alpha in large-caps

**Shareholder Concentration:**
- 光大证券 (2020) — HN_z factor definition (8Q lookback, IC -3.57%, ICIR 0.54) + IHN + LHRD factors
- 开源证券 — PCRC (人均持股占比变动), multi-short IR 2.6
- 中信建投 — Dual-concentration model (price + investor), 18% annualized; exclude sub-new stocks (<1yr listed)

**Block Trade:**
- 光大证券 — 6-day amount volatility composite factor (19.54% annualized)
- FINRA — Price impact decomposition: permanent vs temporary impact
- "溢+大=利" strategy: discount 8-10% → +14.3% in 60d; premium 0-5% → -8.7%

**Lockup Expiry:**
- 华泰证券 — 7-factor scoring card (72.5% win rate); event→factor 3 conditions (independent alpha, time-insensitive, sufficient samples)
- 海通证券 — Factorized lockup improves CSI 300 enhanced IR from 2.58→3.29
- Research consensus: ~60% of IPOs have negative abnormal returns at lockup expiry; VC-backed unlocks ~4x larger impact (-2.81% vs -0.62%); average excess volume +64.8%

**Dividend:**
- NYSE Stern — Dividend yield normalization pipeline: log-transform → double-winsorize → cross-sectional percentile → inverse normal CDF → re-standardize → Cholesky orthogonalize → industry/size neutralize
- Sell-side consensus: Event-specific cum-dividend features outperform generic technicals; recent 3 payments as strong baseline (30% MAPE vs 118% analyst MAPE)

**Limit-Up Board Sentiment:**
- 国泰君安 — 5-factor sentiment timing model (board strategy return, limit-down next-day return, % limit-up, % limit-down, net proportion)
- GF Securities — ULTIMATE-FUSION V12 6-dimensional scoring (momentum, seal strength, popularity, pattern premium, capital structure, theme driver)
- Market state classification: strong (>80 ZTs, break_rate<15%), volatile (break_rate>25%), weak (<20 ZTs), ice/frenzy boolean detectors

**Sector Rotation:**
- DFQ — Genetic programming automated factor mining, fused model 18.42% excess return
- Dual-branch GRU with 3-layer signal fusion: 23.44% annualized
- RRG (Relative Rotation Graph) — 252-bar z-score framework for RS-Ratio × RS-Momentum

**Concept/Theme:**
- 东吴证券 — HIST model: hidden-concept discovery via unsupervised dynamic information sharing across stocks
- 国信证券 (2021) — Concept dimension momentum: formation 6mo, holding 6mo, skip 1mo, monthly excess 1.25%
- 国信证券 — 3D concept heat framework: volume_ratio(0.4) + momentum(0.4) + news_index(0.2)

### 10.2 Open-Source Reference Implementations

| Repository | Relevance |
|-----------|-----------|
| [simonlin1212/a-stock-data](https://github.com/simonlin1212/a-stock-data) V3.3.0 | Most complete A-share board data toolkit: 40 endpoints, 13 sources, dedicated 打板 layer with streak/seal_type/board_height |
| [cn-vhql/FactorHub](https://github.com/cn-vhql/FactorHub) | Full-stack factor platform: Tushare concepts, 9 categories, 50+ preset factors, MyLanguage DSL |
| [yupoet/aurumq-rl](https://github.com/yupoet/aurumq-rl) | 296 price-volume factors + board-aware price limits, Polars-native, GPU training + ONNX inference |
| [RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent) | DSL-driven factor mining with SHAP analysis, emmap factor library, incremental panel PIT |
| [quantasset/factorset](https://github.com/quantasset/factorset) | Lightweight factor computation, clean API for factor definition |
| [AnQreiShikov/Multi-Factor-Strategy-Development-Framework](https://github.com/AnQreiShikov/Multi-Factor-Strategy-Development-Framework) | Three-phase processing reference: source cleaning → cross-sectional → neutralization |
| [teancake/alphasickle](https://github.com/teancake/alphasickle) | Modular factor pipeline with Airflow/Prefect orchestration for heterogeneous data sources |
| [edologgerbird/sfyr-data-pipeline](https://github.com/edologgerbird/sfyr-data-pipeline) | Medallion architecture (bronze/silver/gold) for quant data, dbt-style transformations |
| [jaqs-fxdayu](https://pypi.org/project/jaqs-fxdayu/) | A-share data toolkit: MAD de-extreme + industry/size neutralization as standard panel ops |
| [FinRL_DeepSeek_Crypto_Trading](https://github.com/Mattbusel/FinRL_DeepSeek_Crypto_Trading) | OFI implementation: roll-z normalization with clipping at ±5, decay-weighted flow momentum |

### 10.3 Key Formulas Quick Reference

```
# OFI Intensity
flow_z = (main_net - rolling_mean(20d)) / rolling_std(20d)  # clip at ±5
flow_momentum = EMA(flow_z, halflife=7d)

# HN_z (光大 2020)
HN_z = (holder_num_t - rolling_mean(holder_num, 8Q)) / rolling_std(holder_num, 8Q)

# PCRC (开源)
PCRC = (avg_shares_t / avg_shares_{t-4Q}) - 1

# Unlock Pressure
P = Σ_i [ free_ratio_i × exp(-ln(2)/30 × days_until_i) ]

# Seal Strength
strength = base_score(seal_type) × seal_time_factor × 0.5^(n_cycles - 1)

# Break Rate
break_rate = |zb_pool| / (|zt_pool| + |zb_pool| + 1)

# Sector Breadth Z
breadth_z = ((up - down) / total - cross_mean) / cross_std  # winsorize 1%/99%

# Concept Heat (国信 3D)
heat = 0.4 × pct(vol_ratio, 252) + 0.4 × pct(momentum, 252) + 0.2 × pct(news, 252)

# Dividend Yield Normalized (NYSE Stern)
y_norm = Φ⁻¹(percentile(ln(1 + sum_12m_dividend / price)))  # inverse normal CDF
```
