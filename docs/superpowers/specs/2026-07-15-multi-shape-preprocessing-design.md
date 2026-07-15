# Multi-Shape Preprocessing System Design

> **Goal:** Extend the preprocessing pipeline to handle 10 new data types (capital flow, limit-up boards, block trades, shareholder count, lockup expiry, dividends, industry rankings, concept blocks, Sina fund flow, Tencent quote) — each with fundamentally different raw-data shapes — by classifying them into 4 morphological categories and creating a dedicated preprocessing module per category.

**Architecture:** Each morphological category gets a subdirectory under `stoke_ml/preprocessing/` with 1-2 new `PreprocessingStep` implementations. These feed into the existing `OutlierDetector → MissingImputer → CrossSectionNormalizer` chain for post-processing. Results are cached to `MarketWideStorage` (parquet partitions by year/month/stock) and merged into `FeaturePipeline` via dedicated `_merge_*` methods — same pattern as existing sentiment/margin/northbound data.

**Strategy A (preprocess → cache → merge):** All new data types follow the established pattern: run preprocessing once → store in partitioned parquet → FeaturePipeline reads cached results via left-join merge. This decouples preprocessing from training and maximizes cache reuse.

**Tech Stack:** pandas, numpy, scipy (existing); no new dependencies.

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

**Purpose:** Decompose raw capital flow amounts into ratios, intensity, persistence, and divergence features.

**`transform(df)` logic:**
1. Empty guard: return immediately if `df.empty`
2. Compute size-tier ratios (each / sum of absolute values, + epsilon for zero-div):
   - `super_ratio`, `large_ratio`, `mid_ratio`, `small_ratio`
   - `main_ratio = main_net / (|super|+|large|+|mid|+|small| + 1e-8)`
3. Compute intensity: `flow_intensity = |main_net| / turnover` (if turnover column present)
4. Compute persistence per stock: consecutive trading days with `main_net > 0` (requires date sorting)
5. Compute divergence: `super_small_div = super_ratio - small_ratio`
6. Return df with original + derived columns

**Dependencies:** None (pure pandas).

---

### 3.2 `EventToDaily` (Shape B: event_sparse)

**Purpose:** Single class, dispatching by `event_type` parameter. Converts sparse events to daily per-stock features.

**`__init__(event_type: str, calendar: TradingCalendar)`**
- `event_type` ∈ {"block_trade", "shareholder", "lockup", "dividend"}

**`transform(df)` logic per event_type:**

**block_trade:**
1. `groupby(["date", "stock_code"])` → agg: premium_pct_mean, premium_pct_wavg (weighted by amount), total_amount, trade_count, buyer_is_inst (buyer column contains "机构" or "专用")
2. Reindex to full trading calendar → forward-fill (max 5 days) → ZI fill

**shareholder:**
1. Sort by END_DATE, groupby stock_code → keep latest per quarter
2. Forward-fill: holder_num, change_ratio, avg_shares
3. Compute HN_z: `(holder_num - rolling_mean(8Q)) / rolling_std(8Q)` per stock
4. Compute consecutive_quarter_decline (count of consecutive quarters with negative change_ratio)

**lockup (history + upcoming merged):**
1. From history: most recent lockup details (free_ratio, free_type)
2. From upcoming: days_until_unlock = (FREE_DATE - today).days
3. Compute unlock_pressure = free_ratio * 1/max(days_until_unlock, 1)
4. Forward-fill + decay as date approaches

**dividend:**
1. Sort by EX_DIVIDEND_DATE
2. Compute dividend_yield = bonus_rmb / close_price (requires close price join)
3. Compute days_since_last_ex_div
4. Forward-fill, with exponential decay: `value * exp(-λ * days_since)` where λ = ln(2)/90 (90-day half-life)

**Output:** Per-stock daily DataFrame with event_type-specific columns.

**Dependencies:** `TradingCalendar` for reindex; optional close price join for dividend yield.

---

### 3.3 `BoardBroadcaster` (Shape C: cross_sectional)

**Purpose:** Convert market-wide limit-up pool membership into per-stock daily features.

**`transform(df, pools: dict[str, pd.DataFrame])` logic:**
1. Takes the per-stock OHLCV DataFrame as primary input
2. Receives `pools` dict with keys: "zt", "zb", "dt", "yzt" — each is a DataFrame of pool members for each date
3. For each date × stock, set boolean columns:
   - `is_zt`, `is_zb`, `is_dt`, `is_yzt` (membership check)
4. Compute per stock: `consecutive_zt_days` (rolling count), `board_height_20d` (max consecutive ZT in last 20 days)
5. Broadcast sentiment: if `limit_up_sentiment` DataFrame provided with columns (break_rate, advance_rate, max_height, ladder), add these as market-level features to every stock row

**Output:** Per-stock daily DataFrame with board-related boolean + derived columns.

**Dependencies:** Date alignment with OHLCV index.

---

### 3.4 `SectorBroadcaster` (Shape C: cross_sectional)

**Purpose:** Broadcast industry ranking data to individual stocks via sector mapping.

**`transform(df, industry_ranking: pd.DataFrame, sector_map: dict)` logic:**
1. Requires `sector_map`: dict mapping stock_code → industry_code (from concept_blocks or a static mapping)
2. For each date × stock, look up the stock's industry → join industry_ranking columns:
   - `sector_rank`, `sector_change_pct`, `sector_breadth` (up_count - down_count), `sector_leader_change`
3. Compute rolling: `sector_momentum_5d`, `sector_momentum_20d`, `sector_rank_change`
4. `is_sector_leader`: boolean, True if stock is the leader stock for its industry that day

**Output:** Per-stock daily DataFrame with sector-level features.

**Dependencies:** Industry-to-stock mapping (can be built from concept_blocks `board_type="industry"` or a static CSV).

---

### 3.5 `ConceptBlockEncoder` (Shape D: categorical)

**Purpose:** Encode multi-label concept board membership as multi-hot features + derived statistics.

**`transform(df)` logic:**
1. Collect all `board_name` values across all dates → select top-N by frequency (N=50 default, configurable)
2. For each date × stock, construct multi-hot vector of length N:
   - `cb_{idx}` = 1 if stock belongs to board at index `idx`, else 0
3. Derived features:
   - `board_count`: total number of boards this stock belongs to
   - `board_momentum`: mean of all boards' `change_pct` for this stock
   - `has_hot_board`: boolean, True if any of the stock's boards is in top 5% by `change_pct` that day
4. Missing dates: forward-fill (board membership changes slowly)

**Output:** Per-stock daily DataFrame with N + 3 derived columns. Column names: `cb_0`, `cb_1`, ..., `cb_{N-1}`, `board_count`, `board_momentum`, `has_hot_board`.

**Dependencies:** None (pure pandas).

---

## 4. Config Integration

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

## 5. FeaturePipeline Integration

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

## 6. Preprocessing Script

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

## 7. Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Raw data file missing for a stock | ZI fill — all feature values = 0, all `has_*` flags = False |
| EventToDaily receives empty event df | Return empty daily df — MissingImputer handles fill |
| BoardBroadcaster receives empty pool for a date | All `is_zt` etc. = False for that date |
| ConceptBlockEncoder encounters unseen board name | Map to closest top-N board by name similarity, or ignore |
| FlowDecomposer encounters all-zero flow day | Ratios = 0, intensity = 0 (no division by zero due to epsilon) |
| CrossSectionNormalizer runs on single stock | Skip sector/industry neutralization, fall back to rank/z-score only |

---

## 8. Testing Strategy

Per module: unit test with synthetic DataFrames covering:
- Normal operation (5-stock, 20-day panel)
- Empty input → empty output (no crash)
- Single stock → correct broadcast
- Missing intermediate dates → forward-fill verified
- Edge values: zero flow, 100% premium, negative holder change

Integration test: end-to-end for one stock from raw → preprocessed → merged into FeaturePipeline.

---

## 9. Research References

- Capital flow: Huatai dual-branch Transformer on 30-min flow (RankIC 10.96%); Kaiyuan residual fund flow strength (IR 2.05-3.18); size-tier decomposition as standard quant factor
- Block trade: Counter-intuitive premium/discount pattern (discount 8-10% → +14.3% in 60d); 6-day volatility composite factor (19.54% annualized); "premium + large = profit" strategy
- Shareholder concentration: HN_z strongest predictor (IC -3.57%, ICIR 0.54); PCRC multi-short IR 2.6; exclude sub-new stocks (<1yr); falling price + falling holders = strongest signal
- Lockup expiry: Huatai 7-factor scoring card (72.5% win rate); factorized lockup improved CSI 300 enhanced IR from 2.58→3.29
- Dividend: Event-specific cum-dividend features outperform generic technicals; recent 3 payments as strong baseline
- Sector rotation: Two-stage RFE+RNN methodology; genetic programming for automated factor mining (18.42% excess)
