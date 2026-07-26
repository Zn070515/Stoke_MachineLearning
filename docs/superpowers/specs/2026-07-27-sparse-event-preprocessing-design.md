# Sparse Event Preprocessing — Design Spec

> **Status:** Draft
> **Date:** 2026-07-27
> **Goal:** Replace hard-cutoff forward-fill + ZI z-score with decay-based persistence
> and event-time features, grounded in 2024–2025 quant finance research and our
> actual data characteristics (5530 stocks, 2000–2026, 4 sparse event types).

---

## 1. Problem Statement

### 1.1 Data Characteristics

| Dimension | Value |
|-----------|-------|
| Universe | 5530 A-shares |
| Time span | 2000-01-01 → 2026-07-27, ~6429 trading days |
| block_trade | ~0.03 events/stock/day, 150K total records |
| shareholder | Quarterly (4×/year), 37.3M rows |
| lockup | 1–3 events/stock/year, 36.4M rows |
| dividend | 1–2 events/stock/year |

### 1.2 Current Approach and Its Problems

```
Raw sparse events
  → _fill_to_daily: reindex to trading calendar, ffill(max_ffill=N)
  → ZI-fill after merge: missing days → 0.0 + has_* flag
  → TemporalTransformer: rolling z-score (_z20) on all PO columns
```

**Problem A — Information cliff:** `ffill(limit=N)` creates an artificial cutoff.
On day N+1, the signal abruptly drops to zero. Real information decays
smoothly — a block trade premium from 3 weeks ago is less relevant than
one from 3 days ago, but not zero.

**Problem B — Z-score on ZI-filled data is structurally broken:**

```
ZI-fill → long stretches of zeros → rolling_std_20 = 0 → division by zero → NaN
```

For block_trade (ffill=30) and lockup (ffill=90): even after the fix,
ZI-filled gaps between events have std=0, so `_z20` is always NaN there.
The columns are wasted — 100% NaN for block_trade `_z20`, ~95% for lockup.

**Problem C — `forward_fill_max` config was silently ignored.** All 4 event
types hardcoded their own `max_ffill` values, bypassing `config.yaml`.
Fixed in 2026-07-27 code review (aggregator.py now uses `self.forward_fill_max`).

### 1.3 Research Foundation

| Finding | Source | Implication |
|---------|--------|-------------|
| Forward fill inappropriate for structural sparsity; missingness IS signal (MNAR) | [MDPI 2025](https://www.mdpi.com/2504-4990/7/3/106) | Keep `has_*` flags; don't impute over structural zeros |
| Transformers handle zero-encoded missing values naturally via attention masking | [TARNet 2024](https://acta.sapientia.ro/content/docs/info16-1-01-324608.pdf) | Our xLSTM/TFT models can learn from sparse patterns if we preserve them |
| Exponential decay with half-life preferred over hard window cutoffs | [Entropy Pooling, Meucci](http://arxiv.org/pdf/1910.05555) | Replace `ffill(limit=N)` with `value × exp(-λ × days_since_event)` |
| Event-time features (days since, count in window, cumulative intensity) outperform daily z-scores for sparse events | [STDM, CIKM 2024](https://dl.acm.org/doi/10.1145/3627673.3679806) | Replace `_z20` with event-time features for sparse columns |
| Random Forest imputation outperforms mean/ffill for financial time series | [Goldani 2024](https://ecc.isc.ac/showJournal/39106/280604/3539132) | Long-term: consider ML-based imputation; short-term: decay + event-time |
| Linear regression imputation significantly outperforms adjacent filling for LSTM inputs (p < 0.001, Diebold-Mariano test) | [CSDN 2025](https://blog.csdn.net/luansj/article/details/155192299) | Confirms ffill is a weak baseline for DL models |

---

## 2. Design

### 2.1 Column Classification

Each column produced by `EventToDaily` is classified into one of three types,
determining its persistence strategy:

| Type | Examples | Persistence Strategy |
|------|----------|---------------------|
| **Continuous signal** | `premium_pct_wavg`, `HN_z`, `PCRC`, `unlock_pressure`, `dividend_yield`, `effective_yield` | Exponential decay with per-type half-life |
| **Discrete flag** | `is_deep_discount`, `buyer_is_inst`, `seller_is_hot_money`, `is_vc_backed`, `has_recent_dividend` | ffill(limit=half_life_days) + `has_*` flag |
| **Cumulative / count** | `trade_count`, `total_amount`, `unlock_count_upcoming`, `consecutive_quarter_decline` | ZI (0 = genuinely no events) + `has_*` flag |

### 2.2 Decay-Based Persistence (replaces hard ffill for Type 1)

```
For each event at date t₀ with value v:
  On day t (t ≥ t₀):
    decayed_value = v × exp(-ln(2) / halflife × (t - t₀))
```

Multiple events in the same window: **max** of all active decay curves
(strongest signal dominates, not sum — avoids inflation from event clustering).

| Event Type | Column | Half-life (trading days) | Rationale |
|------------|--------|--------------------------|-----------|
| block_trade | `premium_pct_wavg` | 10 | 2 weeks for premium info to decay |
| block_trade | `permanent_impact` | 20 | Price impact persists longer than premium |
| block_trade | `temporary_impact` | 5 | Temporary impact reverts quickly |
| block_trade | `amount_vol_6d` | 10 | Rolling stat, already decaying |
| shareholder | `HN_z` | 45 | ~2 months, bridges to next quarterly report |
| shareholder | `PCRC` | 45 | Same logic as HN_z |
| shareholder | `change_ratio` | 45 | Same logic |
| lockup | `unlock_pressure` | 60 | 3 months for unlock pressure to decay |
| lockup | `unlock_pressure_mcap` | 60 | Same logic |
| dividend | `dividend_yield` | 20 | ~1 month relevance |
| dividend | `effective_yield` | 20 | Already has decay, but half-life is config-driven |

### 2.3 Event-Time Features (replaces `_z20` for sparse columns)

**Principle:** Instead of forcing sparse events into a daily z-score mold,
compute features that describe the event regime directly.

For each sparse event column family, add:

| Feature | Formula | Semantics |
|---------|---------|-----------|
| `{prefix}_days_since` | Days since last non-zero value (0 if today has event) | How stale is the last signal? |
| `{prefix}_count_20d` | Count of non-zero values in past 20 trading days | How active is this event type lately? |
| `{prefix}_intensity_20d` | Sum of values in past 20 trading days | Cumulative signal strength |
| `{prefix}_last_value` | Most recent non-zero value (carried forward indefinitely) | What was the last signal? |
| `{prefix}_last_direction` | sign(last_value), 0 if no event in 60 days | Is the last signal bullish or bearish? |

**When to apply:** Only for columns where the underlying event frequency is
< 1 event per 5 trading days on average. This covers block_trade, lockup,
and dividend. Shareholder (quarterly) is borderline — its `_z20` works
because 90-day ffill bridges quarters, but event-time features still add value.

### 2.4 `_z20` Handling

| Column family | Keep `_z20`? | Reason |
|---------------|-------------|--------|
| K-line derived (OHLCV, technicals) | Yes | Dense data, z-score works |
| Sentiment (news, guba) | Yes | Daily data, ZI is genuinely "no news today" |
| Market-wide (margin, northbound) | Yes | Daily data |
| Flow (capital_flow) | Yes | Daily data |
| Board (limit-up pools) | Yes | Daily data |
| Sector | Yes | Daily data |
| block_trade | **No — replace with event-time** | Sparse, ZI z-score = NaN |
| lockup | **No — replace with event-time** | Sparse, ZI z-score ≈ 95% NaN |
| shareholder | **Keep both** | 90d ffill bridges quarters, z-score works; event-time adds value |
| dividend | **No — replace with event-time** | Sparse, ZI z-score = NaN |

### 2.5 Implementation Architecture

```
EventToDaily.transform(df, close_prices, trading_dates)
  → aggregate raw events by date+stock (unchanged)
  → _fill_to_daily with decay (NEW: _decay_to_daily replaces _fill_to_daily for Type 1)
  → compute event-time features (NEW: _add_event_time_features)
  → return daily DataFrame

TemporalTransformer.transform(df)
  → detect sparse columns (heuristic: >80% zeros OR mean interval > 5 days)
  → skip _z20 for sparse columns
  → compute event-time features for sparse columns (if not already present)
  → normal rolling stats for dense columns (unchanged)
```

**Backward compatibility:** Event-time columns are additive. Existing `_z20`
columns for non-sparse types remain unchanged. The merge methods in
`FeaturePipeline` use `_merge_daily_aux` which picks up all available columns
automatically — no changes needed there.

### 2.6 Config Changes

```yaml
preprocessing:
  event:
    block_trade:
      decay_halflife_days: 90    # unchanged (lockup decay)
      forward_fill_max: 30       # already fixed
      persistence_mode: "decay"  # NEW: "decay" | "ffill" | "zi"
      persistence_halflife:      # NEW: per-column half-life overrides
        premium_pct_wavg: 10
        permanent_impact: 20
        temporary_impact: 5
    shareholder:
      decay_halflife_days: 90
      forward_fill_max: 90       # already fixed
      persistence_mode: "decay"
      persistence_halflife:
        HN_z: 45
        PCRC: 45
        change_ratio: 45
    lockup:
      decay_halflife_days: 180
      forward_fill_max: 90       # already fixed
      persistence_mode: "decay"
      persistence_halflife:
        unlock_pressure: 60
        unlock_pressure_mcap: 60
    dividend:
      decay_halflife_days: 365
      forward_fill_max: 30       # already fixed
      persistence_mode: "decay"
      persistence_halflife:
        dividend_yield: 20
        effective_yield: 20     # overrides the config-level decay_halflife_days
```

---

## 3. Implementation Plan

### Phase 1: Decay Persistence in EventToDaily (P0)

**Files:**
- Modify: `stoke_ml/preprocessing/event_sparse/aggregator.py`
- Modify: `config.yaml`

**Changes:**
1. Add `_decay_to_daily()` method — vectorized exponential decay across the trading calendar
2. Add `_add_event_time_features()` method — days_since, count_20d, intensity_20d, last_value, last_direction
3. Per-event-type column classification dict (continuous/flag/cumulative)
4. Wire decay path in `_transform_block_trade`, `_transform_lockup`, `_transform_dividend`, `_transform_shareholder`
5. Keep `_fill_to_daily` for flag/cumulative columns (unchanged)
6. Config: add `persistence_mode` and per-column `persistence_halflife` overrides

### Phase 2: Sparse-Aware TemporalTransformer (P1)

**Files:**
- Modify: `stoke_ml/features/transform.py`

**Changes:**
1. Detect sparse columns (heuristic: zero fraction > 80% after ZI)
2. Skip `_z20` for sparse columns
3. Compute event-time features for sparse columns not already covered by EventToDaily

### Phase 3: Re-preprocess + Rebuild (P2)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type all --start 2000-01-01
PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py
```

---

## 4. Acceptance Criteria

1. block_trade `_z20` columns: zero NaN after rebuild (replaced by event-time features)
2. lockup `_z20` columns: zero NaN after rebuild (replaced by event-time features)
3. Decay curve visually verified: block trade premium 10 days ago ≈ 50% of original value
4. `has_*` flags correctly distinguish "no event" from "decayed to near-zero"
5. All existing tests pass (no regression on shareholder/dividend paths)
6. Feature count: +25–35 event-time columns, −16 _z20 columns (block_trade + lockup + dividend), net +10–20

---

## 5. References

- Meucci, A. "Entropy Pooling with Exponential Time Decay." arXiv:1910.05555
- Fang et al. "STDM: Spatio-Temporal Diffusion Model for Missing Financial Data." CIKM 2024
- Buza & Novák. "TARNet: Transformer for Missing-Value Time Series." Acta Univ. Sapientiae, 2024
- Goldani. "Comparative Analysis of Missing Values Imputation Methods in Financial Series." Iranian Journal of Finance, 2024
- Yang et al. "Semi-supervised TFFA-MCSSLSTM for Financial Time Series." Applied Soft Computing, 2024
- "The Alchemy of Factor Creation." xglamdring.com, 2025
