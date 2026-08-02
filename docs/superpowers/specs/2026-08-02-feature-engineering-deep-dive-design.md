# Feature Engineering Deep-Dive — Design Spec

> **Status:** Draft
> **Date:** 2026-08-02
> **Goal:** Based on a deep structural analysis of the actual `data/a_shares`
> datasets, design a strengthened feature engineering pipeline: a four-layer
> feature factory plus an evaluation/selection closed loop, built on top of the
> existing FE v2 modules and the 20-source merge methods already wired into
> `FeaturePipeline`.
>
> **Predecessor:** `2026-07-25-feature-engineering-v2-design.md`

---

## 1. Data Deconstruction Findings

Deep probe of all datasets under `data/a_shares` (column schemas, file counts,
and per-stock time coverage via parquet date-column scans).

### 1.1 Dataset inventory

| Layer | Dataset | Files | Sample columns | Notes |
|-------|---------|-------|----------------|-------|
| Core | `daily` | 5530 | open/high/low/close/volume/amount/turnover/pct_change | 10 cols, all to 2026-07-31 |
| Core | `minute` / `minute_flat` | 247308 / 20804 | 4 freqs × 5201 stocks | Recent history only |
| Fundamental | `fundamentals` | 5530 | roe/roa/eps/revenue_yoy/profit_yoy/debt_ratio/margins | Quarterly (report_date) |
| Fundamental | `fundamentals_daily` | 5530 | ffill to daily | **Uniform 2015 start** |
| Valuation | `valuation` | 5530 | pe_ttm/pb_mrq/ps_ttm/pcf_ttm | 5509 stocks to 2026 |
| Sentiment | `sentiment` (news Gold) | 5530 | sentiment_mean/std/count/pos/neg ratio | **Nearly all 2026-only** |
| Sentiment | `guba_sentiment` | 5524 | guba sentiment stats | **10y history (2015+)** |
| Sentiment | `comment_sentiment` | 5192 | comment_score | Only 2026-06+ (30 rows) |
| Fund flow | `capital_flow` / `_processed` | 5201 | main/small/mid/large/super net + ratios | 2022-08 start |
| Margin | `margin` | 4610 | margin_balance/buy/repay/short | 2024-09 start |
| Northbound | `northbound` | 3327 | north_hold_*/north_net_buy | **Stale since 2024-08** |
| Board | `board_processed` | 5530 | is_zt/zb/dt/yzt, seal_*, market_state_*, concept_* | 27 cols |
| Sector | `industry_ranking_processed` | 5530 | sector_code, momentum_*, sector_rrg_*, breadth | 34 cols |
| Sector | `concept_blocks` / `_processed` | 5530 | board membership snapshots | **Only a few days** |
| Sector | `industry` | 3 | industry_ranking_computed (222k rows) | 2015-2026 historical |
| Event | `block_trade_processed` | 5111 | premium, seats, impact | ffill to 2000, sparse |
| Event | `dividend_processed` | 5284 | yields, ex-div, dv_* | ffill to 2000, sparse |
| Event | `lockup_processed` | 5346 | free_ratio, unlock_return_30d | ffill to 2000, sparse |
| Event | `shareholder_processed` | 5485 | holder_num, HN_z, PCRC | sparse |
| Event | `pledge` | 3259 | pledge ratio, margin line | **Not yet in FE** |
| Event | `dragon_tiger` | 6316 | buy/sell/net amount, lhb_reason | Not seat-classified |
| Limit-up | `limit_up_zt/zb/dt/yzt` | 1236/671/598/1243 | seal_fund, first/last_seal, break_times, limit_days | **Not yet in FE** |
| Limit-up | `limit_up_sentiment` | 1 | zt/zb/dt/yzt count, break_rate, max_height, ladders | Market-level, 25 rows |
| Market | `market_breadth` | 2 | investor counts, total cap | **Not yet in FE** |
| Market | `index_constituents` | 1 | weights, membership | **Not yet in FE** |
| Macro | `macro` | 1 | 28 cols: shibor/fx/bond/cpi/m2 | Global |
| Text raw | `news_raw/silver`, `guba_raw/silver`, `announcements` | 5530 | titles/bodies + sentiment | News ~7mo trailing (2025-12+), guba 2021+ |

### 1.2 Time coverage matrix (per-stock start-year distributions)

| Dataset | History | Distribution |
|---------|---------|--------------|
| `daily` | 2000–2026 | 2000: 864 stocks; ~2600 stocks pre-2015; ~2900 post-2015; 80 in 2026 |
| `fundamentals_daily` | **2015–2026** | **Uniform: all 5530 stocks from 2015** |
| `guba_sentiment` | 2015–2026 | Broad: 625 (2015) … 880 (2020) … 506 (2025), 5524 stocks |
| `sentiment` (news) | **mostly 2026** | 4639 start in 2026, 807 in 2025, ~84 before → **~7 months usable (2025-12+)** |
| `valuation` | 2000–2026 | Mirrors listing year; 5509 stocks reach 2026 |
| `capital_flow` | 2022-08+ | 5201 stocks |
| `margin` | 2024-09+ | 4610 stocks |
| `northbound` | 2023-06–2024-08 | Stale (frozen 2024-08) |
| `concept_blocks` / `industry_ranking` | days | Snapshot-only |
| `block_trade/dividend/lockup/shareholder_processed` | ffill to 2000 | Underlying events sparse |

### 1.3 Core contradictions revealed

1. **Sentiment history is extremely lopsided.** News Gold (`sentiment/`) exists
   for only ~7 months (mostly 2025-12 onward) while guba Gold has 10 years. Raw
   news (`news_raw/silver`) is structurally capped by free-API pagination
   limits (~3-6 months per fetch), so reaching 2023 is **not feasible** — the
   Gold layer can only be rebuilt from the existing ~7-month raw window.
2. **Sector/board data is a shell.** `concept_blocks` and `industry_ranking`
   are a few-day snapshot; `industry_ranking_processed` inherits that. Only the
   `industry/industry_ranking_computed` file carries 2015–2026 history.
3. **Fund-flow features have hard start dates.** `capital_flow` 2022-08,
   `margin` 2024-09, `northbound` frozen 2024-08.
4. **FE v2 un-touched alpha sources.** `limit_up_*` (board play), `pledge`,
   `market_breadth`, `index_constituents`, and seat attributes of `dragon_tiger`
   are not consumed anywhere.

---

## 2. Architecture: Four-Layer Feature Factory + Evaluation Closed Loop

```
┌─────────────────────────────────────────────────────┐
│ L4 Evaluate/Select  IC report → corr dedup → importance → leakage check │
├─────────────────────────────────────────────────────┤
│ L3 Deepen            EmotionRefiner · FundamentalRefiner ·            │
│                      TemporalTransformer · NEW: MarketEnv features     │
├─────────────────────────────────────────────────────┤
│ L2 Fuse              20-source merge (existing)                      │
│                      + NEW 4 sources: limit-up / pledge / market-breadth │
│                        / dragon-tiger seats                          │
├─────────────────────────────────────────────────────┤
│ L1 Base              Technical (Alpha158+) · trend scoring ·          │
│                      microstructure · calendar                       │
└─────────────────────────────────────────────────────┘
```

### 2.1 Design decisions

- **Service object:** Not constrained to a single model family. Features are
  produced as a full-history daily panel per stock (`data/features/{code}.parquet`)
  and can be consumed as TFT/xLSTM panel sequences (PK/PO/static split) or flat
  XGBoost/LightGBM rows. Both consumption paths already exist.
- **Training timeline:** Medium-long window **2021+** as the primary training
  window (covers bull/bear, has guba + fundamentals + capital flow + events;
  margin/comment/news sparse → `has_*` flag). Full-history features are still
  built so long-horizon technical/fundamental signals remain available for
  warmup and validation.
- **Existing assets preserved:** all 20 `_merge_*` methods, FE v2 modules
  (`emotion.py`, `fundamental.py`, `transform.py`, `selector.py`),
  `build_features.py`, and TFT/xLSTM + XGBoost dual forms stay untouched.
  New sources are added following the existing merge pattern.

### 2.2 Data flow

```
data sources (50+) → preprocessed (processed/*) → FeaturePipeline four layers
→ data/features/{code}.parquet → IC/leakage reports → training slice (2021+)
```

---

## 3. New Feature Families

### 3.1 Limit-up / board-play ecology (A-share short-line alpha)

Sources: `limit_up_zt` (limit-up list, no stock_code → map via stock_name),
`limit_up_zb` (broken board), `limit_up_dt` (limit-down, has dt_days/open_times),
`limit_up_yzt` (already-limit-up chain), `limit_up_sentiment` (market level).

**Per-stock** (complements board_processed `is_zt/is_zb/is_dt/is_yzt`):
- `zt_first_seal_hour` — time of first seal (earlier = stronger; 09:25 vs 14:50)
- `zt_seal_fund_ratio` — seal amount / float cap (seal strength magnitude)
- `zt_break_times` — board broken count (was limit-up then broke)
- `zt_limit_days` / `dt_days` — consecutive limit-up / limit-down days
- `has_yzt` — is already-limit-up high-label stock

**Market level** (from `limit_up_sentiment` + full-market aggregation):
- `break_rate`, `advance_rate` (promotion rate), `max_height`
- `ladder_2..6plus` — chain-tier distribution
- `market_zt_ratio` — limit-up count / market count (money-making effect)
- `market_heat_z` — sentiment temperature (z-score of limit-up count)

### 3.2 Equity & capital behavior risk

**Dragon-tiger seats** (`dragon_tiger`, classify `lhb_reason`):
- `lhb_is_hot_money` — seat contains known hot-money / 敢死队
- `lhb_is_institution` — institutional-dedicated seat
- `lhb_is_north` — Stock-Connect seat
- `lhb_buy_ratio`, `lhb_count_5d` — buy share + 5d frequency

**Pledge** (`pledge`, 16 cols incl. 占所持股份比例 / 占总股本比例 / 预估平仓线):
- `pledge_ratio` — cumulative pledged / total shares
- `pledge_margin_dist` — latest price / est. margin line − 1 (negative = below line)
- `pledge_risk` — flag when within 20% of margin line

**Lockup**: merge with existing `unlock_return_30d` from `lockup_processed`
into an "equity risk warning block".

### 3.3 Market-level + macro

- **Market breadth** (`market_breadth`): new-investor count, total market cap,
  average account cap → incremental-money / sentiment.
- **Index membership** (`index_constituents`, 1900 rows): `is_index_member`
  (CSI300 / CSI500 / CSI1000), `index_weight`, recent 30d add/drop event.
- **Macro composite** (compress `macro` 28 cols → 6 factors):
  `shibor_1m_z` (funding), `bond_10y2y_spread` (term), `us_cn_10y_spread`,
  `fx_usd_cny_z` (fx pressure), `m1_m2_spread` (liquidity), `cpi_z` (inflation).
- **Market sentiment temperature**: market-wide advance/decline ratio,
  total-turnover z-score.

### 3.4 Feature evaluation landing

Runs before training as a standard gate (see §5).

---

## 4. Build Engineering

- Full-history parallel build of 5530 stocks into `data/features/{code}.parquet`.
  Current `build_features.py` is serial (5530 × 18 file reads each); upgrade to
  `multiprocessing` shards → estimated 20–40 min.
- **Idempotent + cached**: same input → same output; keep `--force` rebuild.
- **Training alignment**: train slices `start_date=2021-01-01`; long-history
  features used only for warmup / technical warm-up, not sample pollution.

---

## 5. Evaluation Closed Loop (L4)

**IC report** (mandatory pre-training):
- Per-feature Spearman RankIC vs forward-horizon return; report
  `IC / ICIR / IC>0 ratio / coverage / turnover`.
- Dual-window: full history + 2021+ primary window (avoid over-fitting recent style).
- Output `reports/feature_ic_report.csv`; grade features high-value / redundant / useless.

**Correlation dedup + importance**: reuse existing `selector.py`
(block-greedy dedup ρ<0.85 + LightGBM gain). No re-invention.

**Leakage check** (critical):
- Temporal-alignment audit: verify every feature at time t uses only t and prior
  (pipeline already lags 1 day; this re-audits all sources incl. new ones).
- Future-info scan: detect any feature contaminated by same-day post-close data.
- Output `reports/feature_leakage_report.csv`; hard-block on leakage.

---

## 6. Error Handling

- Sparse new sources → existing `has_*` flag + ZI fill; never blocks the build.
- `limit_up_zt/zb` lack `stock_code` → map via stock_name using
  `stock_sector_cache.csv`; unmapped stock-day skipped and logged.
- `pledge` Chinese column names → normalized to English.
- Single-stock failure (empty K-line / all-NaN) → skip + record, not abort
  (existing build_features try/except pattern).

---

## 7. Testing

1. **Feature completeness**: all 5530 stocks share identical column schemas;
   NaN rate below threshold (except sparse sources guarded by `has_*`).
2. **Leakage**: random feature sample — verify time-t value excludes t+1 info.
3. **IC correctness**: inject a known-signal factor and verify the computed IC.
4. **Parallel determinism**: single-thread vs multiprocess builds bit-identical.
5. **New-merge alignment**: limit-up / pledge / market-breadth merges produce
   correct column alignment and date index.

---

## 8. Deliverables

```
data/features/{code}.parquet        5530 full-history feature panels
reports/feature_ic_report.csv       feature value report
reports/feature_leakage_report.csv  leakage audit
logs/build_features_*.log           parallel build log
```

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| New sparse sources dilute signal | `has_*` flags; IC report grades each feature; correlation dedup |
| news sentiment only ~7 months | Rebuild news Silver+Gold from `news_raw` (2025-12+) for freshness; **no free-API route reaches 2023** |
| Sector shell (snapshot-only) | Backfill `industry_ranking_processed` from `industry/industry_ranking_computed` (2015+) |
| northbound frozen 2024-08 | Accept + document; substitute with margin/capital-flow north components if needed |
| Parallel build nondeterminism | Determinism test; per-shard isolation |
| Leakage from new merges | Leakage audit gate before training |

---

## 10. Open Items (data backfill)

Short-board items identified in deconstruction that may be backfilled before
full build (see separate execution task):
1. Rebuild news Silver+Gold from raw — **verified 2026-08: not a 3y backfill**.
   Free-API pagination caps depth at ~3-6 months per fetch (EastMoney search
   `beginTime`/`endTime` ignored, ~3 pages max; Sina AllNewsStock ~days; Sina
   roll keyword ~3-4 days). `news_raw` spans only 2025-12+ (89% of stocks start
   2025-12/2026). Rebuild refreshes Silver+Gold to the raw ceiling instead.
2. Backfill `industry_ranking_processed` sector history from
   `industry/industry_ranking_computed` — **done** (2015-2026).
3. Evaluate `capital_flow` / `margin` historical API availability —
   **done**: capital_flow reachable to 2010-03 (Sina, main_net only),
   margin reachable to 2012 (SSE/SZSE via AKShare). Both backfills running.
4. Accept-and-document `northbound` freeze.
5. Evaluate `concept_blocks` historical feasibility.
