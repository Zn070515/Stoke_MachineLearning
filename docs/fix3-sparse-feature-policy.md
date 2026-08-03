# Fix 3 — 稀疏特征族显式处理 (Sparse Feature Family Handling)

> **Date:** 2026-08-03
> **Scope:** 全修 item 3 — explicit handling of sparse / dead feature families in the
> FE v2 flat pipeline (`scripts/build_features.py` → `data/features/*.parquet`).
> The quantitative basis is `reports/feature_sparsity_report.csv` (full 5530-stock
> panel, regenerable via `scripts/feature_sparsity_report.py`).

---

## 1. Panel Statistics (5530 stocks)

| Metric | Count |
|--------|-------|
| Total numeric feature columns across panel | 3,916 |
| Dense (mean_nonzero_ratio ≥ 0.01) | 2,890 |
| **Event-sparse** (mean_nonzero_ratio < 0.01) | **1,026** |
| **Constant-on-≥90%-of-stocks** (dead columns) | **152** |

> Counts refreshed 2026-08-03 after the full feature rebuild; the constant set
> moved 133 → 152 → **157** (see finding 5 below, and §4 for the post-`main_ratio`
> propagation count).

Event-sparse ≠ dead. The 1,130 event-sparse features carry real cross-sectional
signal on the few dates they activate (news/announcement sentiment lags, lockup
pressure, dragon-tiger presence) and are consumed through their `has_*` mask
columns. The 133 constant columns are the ones that need explicit handling.

---

## 2. Dead-Column Classification (133 constant columns)

### 2A. Source-dead — the upstream data source does not provide this column (80)

| Family | # | Root cause |
|--------|---|-----------|
| `super_net` / `large_net` / `mid_net` / `small_net` + `_ratio` + ma/accel | 60 | EastMoney tiered capital-flow endpoint went offline 2026-07; Sina source zeroes all tier columns (`capital_flow_source.py`). `main_net` (the aggregate) is real. |
| `broad_main_net` + ma/accel | 5 | = super+large+mid (all zero) → constant zero |
| `large_minus_small` + ma/accel | 5 | = large_ratio − small_ratio (both zero) → constant zero |
| `comment_attention` / `comment_institution` / `comment_trend` + ma/accel | 15 | per-stock comment source (`stock_comment_detail_zhpj_lspf_em`) returns only `comment_score`; the 4-column schema exists only in the full-market cross-section snapshots, which are not wired to daily features |

### 2B. Snapshot-dead — built from a few-day snapshot instead of history (34)

| Family | # | Root cause |
|--------|---|-----------|
| `concept_momentum_3m/6m/12m` + `concept_board_height` + ma/accel | 20 | `concept_blocks_processed` is a few-day snapshot → momentum over a snapshot is constant |
| `is_concept_leader` + ma/accel | 4 | same concept snapshot |
| `sector_alpha` + `sector_relative_strength` + ma/accel | 10 | `industry_ranking_processed` inherits the snapshot-only `industry_ranking` (the 2015–2026 history lives in `industry/industry_ranking_computed` but is not wired here) |

### 2C. Legitimately sparse — rare regimes/events, correctly handled (≈19)

`market_state_frenzy/ice/strong` (+ ma/accel), `is_dt` (+ ma/accel),
`dividend_growth`, `plan_stage_encoded`, `transfer_ratio` (+ ma/accel). These
are constant on ≥90% of stocks because the underlying regime/event is genuinely
rare; they carry signal when active and must be kept (tree models gate them
naturally; gradient models see them as a near-constant offset).

---

## 3. Findings During Verification

1. **`main_ratio` was actively polluted (FIXED).** `_compute_ratios` in
   `flow.py` guarded division by zero with `+eps=1e-8`; with all tier columns
   zero, `main_ratio = main_net/1e-8 = main_net × 1e8` — values up to ±4e16 hit
   the feature files. Without normalization this is catastrophic scale
   pollution; with per-stock z-scoring it becomes an exact duplicate of
   `main_net`. **Fixed** to `denom = total.replace(0, np.nan)` so the ratio is
   undefined (NaN → ZI-fills to 0) when tiers are absent. Verified on a
   re-preprocessed 000001: `main_ratio` is now all-NaN. **Propagated
   2026-08-03**: flow re-run (13.46M rows) + full 5530-stock feature rebuild;
   `main_ratio` now reports `constant_stock_ratio=0.940506` in the sparsity
   report and the whole family (main_ratio + ma5/ma10/ma20/accel) enters the
   training drop set (§4).
2. **`margin_net` is partially dead (per-stock, not structural).**
   `margin_net = margin_buy − margin_repay`; `margin_repay`/`short_repay_vol`
   are all-NaN for a subset of stocks (000001: 100% NaN; 600519: 0%). Not a
   dead column family — a downloader completeness issue for affected stocks.
   Re-run the margin downloader for affected codes; flag in the gate.
3. **Capital-flow processed dirs can be stale.** Re-preprocessing 000001
   produced 3,981 rows vs the previous 2,803 (different `main_net` range too).
   The gate's `aux_close_aligned` check does not catch value-level staleness of
   `capital_flow_processed`.
4. **`unlock_return_30d` is a look-ahead leak (2026-08-03, new).** In
   `event_sparse/aggregator.py`, `unlock_return_30d[t] = close[t+30]/close[t] − 1`
   via `grp.shift(-30)` — a forward 30-day window that is NOT known at feature
   time.  It is constant on ≥90% of stocks (only unlocking stocks activate it),
   so the training-side drop mechanism removes it from the model input — the
   leak never reaches the learner.  Proper fix (backfill follow-up): compute a
   *realized* post-unlock drift with a rolling 30d window ending at t instead.
5. **The constant set expanded 133 → 152 after the feature rebuild.**  The 19
   new columns are `concept_zt_count/ratio` (10), `is_zb` (4), and
   `unlock_return_30d` (5) — all classified dead except `is_zb`/`concept_zt_*`
   are snapshot-dead (limit-up board pools not yet wired to daily features) and
   `unlock_return_30d` is leak-dead.  This confirms the drop mechanism must be
   report-driven, not a hard-coded list.

---

## 4. Model Policy

| Class | Model behavior | Action |
|-------|----------------|--------|
| Event-sparse, has_* gated (news/ann, lockup, dt, market_state) | Keep — real cross-sectional signal on activation dates | none; already gated |
| Source-dead constants (2A) | Tree models ignore; gradient models waste params on constant input | **Drop from feature set** — add to a denylist derived from the sparsity report (`constant_stock_ratio ≥ 0.9`), consumed by `train_lstm.py` / `train_baseline.py` |
| Snapshot-dead constants (2B) | Same as 2A | **Drop** until a real history source is wired |
| `main_ratio` family (post-fix) | NaN→0 constant → dead | Drop with 2A |

**Mechanism:** the sparsity report already outputs `constant_stock_ratio` per
column. Training should filter columns with `constant_stock_ratio ≥ 0.9` (and
optionally `mean_nonzero_ratio < threshold` for near-constant), rather than
hard-coding a list. This keeps the policy data-driven: as sources get fixed and
features rebuilt, columns fall out of the drop set automatically.

**Implemented (2026-08-03) in `stoke_ml/features/pipeline.py`:**
`FeaturePipeline._dead_features()` loads the sparsity report once (cached),
derives the drop set = `{constant_stock_ratio ≥ 0.9}` minus the protected
families (prefix list `SPARSE_KEEP_PREFIXES = market_state_, is_dt,
dividend_growth, plan_stage_, transfer_ratio`), and filters it out at both
sequence choke points — `_prep_feature_df` (LSTM / XGBoost flat-mode and
`build_features_from_panel`) and the PK/PO discovery in `build_panel_features`.
`save_features` / `engineer_features` (the parquet writers) are unchanged, so
canonical panels keep all columns and the drop is a training-time view.  Toggle
`drop_dead_features=False` to disable.  Current drop set: **123 columns**
(157 constant − 34 protected) after the post-`main_ratio`-fix rebuild.  The 5
new entries are exactly the `main_ratio` family (main_ratio + ma5/ma10/ma20/
accel): with tier columns zeroed by Sina, the fixed ratio is NaN→0 on 94.05% of
stocks (`constant_stock_ratio=0.940506`), so it and its rolling transforms now
qualify as dead.  `main_ratio_std20` stays (constant_stock_ratio=0.0 — it
carries real dispersion after main_net spikes).  Verified: 0 dead columns
survive sequencing, 287/287 protected columns survive; 48 feature tests pass;
full rebuild 5530/5530 exit 0; gate re-run green.

---

## 5. Source-Level Follow-ups (not in current scope)

1. **Capital-flow tiers**: restore EastMoney tiered endpoint (datacenter
   资金流向) OR drop the tier/broad_main/large_minus_small families from the
   feature spec (accepting `main_net` as the sole flow signal).
2. **Comment attention/institution/trend**: accumulate daily full-market
   snapshots into a per-stock daily series (`save_snapshot` already exists), or
   drop the three dead columns.
3. **Concept / sector history**: wire `industry/industry_ranking_computed`
   (2015–2026) into `industry_ranking_processed`; concept history requires a
   backfill source.
4. **margin_net**: re-run margin download for stocks with all-NaN
   `margin_repay`/`short_repay_vol`.
