# TFT Panel Training — Architecture Redesign for RTX 4090

> **Goal:** Replace single-stock small-model training with Panel TFT (Temporal Fusion Transformer) on 798 A-share stocks × 252-day sequences, targeting 3-task output (direction/return/volatility), optimized for RTX 4090 24GB.

**Motivation:** Previous architecture (single-stock, ~350K params, seq_len=60, XGBoost baseline) was constrained by laptop hardware. RTX 4090 + good CPU removes these constraints. Design goal: maximize signal extraction, not maximize parameter count.

**Baseline:** MCC ≈ 0.07 with current XGBoost + 200 MI-selected features.

---

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Feature Pipeline                       │
│  14 aux dimensions → ZI merge → 200 core + raw features │
│  shape: (N_stocks, 252 seq, ~400 features)               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Cross-Section Normalizer                     │
│  Z-score per date across all stocks                      │
│  → (x - μ_day) / σ_day per feature                       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              TFT Model (15-25M params)                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  VSN (past)  │  │  LSTM Enc    │  │  Multi-Head   │  │
│  │  feature sel  │  │  (2 layers)  │  │  Attention    │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│         │                │                  │            │
│         └────────────────┴──────────────────┘            │
│                           │                               │
│                    ┌──────┴──────┐                        │
│                    │   Gating +  │                        │
│                    │   GRN × N   │                        │
│                    └──────┬──────┘                        │
│                           │                               │
│         ┌─────────────────┼─────────────────┐            │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐     │
│  │ Direction   │  │  Return %   │  │ Volatility  │     │
│  │ Head (CE)   │  │ Head (MSE)  │  │ Head (MSE)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Uncertainty Weighting                        │
│  loss = CE/2σ₁² + MSE_r/2σ₂² + MSE_v/2σ₃²              │
│       + log(σ₁) + log(σ₂) + log(σ₃)                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Training Loop                                │
│  AMP fp16 + torch.compile + grad_accum=2                 │
│  Purged walk-forward (train 2yr / purge 5d / val 3mo)   │
│  Early stopping: top-20 sim portfolio Sharpe             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. TFT Model Design

### 2.1 Input Types

TFT requires splitting features into three semantic categories:

| Type | Content | Dim |
|------|---------|-----|
| **Static** | Time-invariant attributes: industry, market-cap quantile, concept board membership | ~30 |
| **Past known** | Known-ahead temporal: price series, volume, technical indicators, calendar features | ~250 |
| **Past observed** | External temporal (look-back only): capital flow, sentiment, board events | ~120 |

### 2.2 Variable Selection Network (×3)

Each input type gets its own VSN. Inside each VSN: GRN + softmax gating, applied **per timestep independently** — Monday's attention may focus on capital flow, Friday's on technical levels.

Selected feature dimensions are unified to `hidden_dim`.

### 2.3 Temporal Encoder

```
selected past ──┬──► LSTM Encoder (2 layers, hidden=256) ──► h_t
selected obs   ──┘                                            │
                                                              ▼
selected static ──► Tile across time ──► enrich h_t ──► Multi-Head Attention (4 heads)
                                                                  │
                                                                  ▼
                                                            GRN × 3 layers
                                                                  │
                                                                  ▼
                                                            Decoder output (per-timestep)
```

### 2.4 Output Heads

```
                ┌─► Dropout(0.2) → Linear(hidden, 2) → Direction (logits)
decoder_out ────┤
  (hidden_dim)  ├─► Dropout(0.2) → Linear(hidden, 1) → Return % (float)
                │
                └─► Dropout(0.2) → Linear(hidden, 1) → Volatility (float, softplus)
```

- **Direction**: 2-class logits → CrossEntropyLoss (no sigmoid needed)
- **Return %**: raw float → MSE loss
- **Volatility**: softplus-gated positive float → MSE vs future 5-day realized volatility

### 2.5 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 256 | TFT paper default |
| `lstm_layers` | 2 | Balance sequence modeling vs VRAM |
| `attention_heads` | 4 | 252 steps needs 4 time-scale views |
| `grn_layers` | 3 | Non-linear depth matters more than LSTM depth |
| `dropout` | 0.15 | Conservative; panel mode provides implicit regularization |
| `static_dim` | ~30 | Industry, size quantile, concept encoding |
| `past_known_dim` | ~250 | Price, volume, technical, calendar |
| `past_obs_dim` | ~120 | Capital flow, sentiment, board events |
| **Total params** | **~20M** | Center of 15-25M target range |

---

## 3. Training Strategy

### 3.1 Purged Walk-Forward Split

```
Time ─────────────────────────────────────────────────►
2015    2017     2019     2021     2023     2025
├────────┼────────┼────────┼────────┼────────┤

Fold 1: [Train: 2015-01 ─► 2016-12]  Purge: 5d  [Val: 2017-01 ─► 2017-03]
Fold 2:            [Train: 2015-04 ─► 2017-03]  Purge: 5d  [Val: 2017-04 ─► 2017-06]
...
Fold ~36:                                          [Train...]  [Val: 2025-10 ─► 2026-01]
```

- Train window: 2 years, step: 3 months
- Purge gap: 5 trading days between train and val to prevent adjacent-day leakage
- Per fold: randomly sample 20% of stocks for validation (preserves temporal order, adds cross-sectional coverage)
- ~36 folds total, each independently trained

### 3.2 Batch Construction

```
One batch contains:
  - N dates (8-16), randomly sampled
  - M stocks per date (32-64), randomly sampled
  - Batch size = N × M = 8 × 64 = 512
  - Each stock has a full 252-step sequence

Gradient accumulation: 2 steps → effective batch = 1024
```

**Design rationale for grouping same-date stocks in a batch:** Cross-sectional relationships within a trading day (sector rotation, capital flow direction) are the strongest signals. Keeping same-day stocks together lets BatchNorm/GRN learn daily mean/variance patterns.

### 3.3 Optimizer & Scheduler

| Component | Config | Rationale |
|-----------|--------|-----------|
| Optimizer | AdamW, lr=1e-3, wd=1e-4 | TFT standard, weight decay regularization |
| Scheduler | OneCycleLR → CosineAnnealing | Warm up first 30% steps to 1e-3, then cosine decay to 1e-6 |
| Warmup | 1000 steps | Prevent gradient explosion in early batches |
| Gradient clipping | max_norm=1.0 | 252-step LSTM BPTT prone to explosion |
| Epochs | 100 max, early stop | Typically converges at 30-50 epochs |

### 3.4 Hardware Configuration

```
GPU: RTX 4090 24GB
  - batch_size=512, grad_accum=2 → effective 1024
  - fp16 AMP (torch.cuda.amp)
  - torch.compile(mode="reduce-overhead")
  - Estimated VRAM: ~16-18GB / 24GB
  - Estimated per-fold time: ~15-25 min
  - 36 folds total: ~12-15 hours

CPU:
  - DataLoader workers=8 (preload + ZI merge + normalize in parallel)
  - Pin memory on
```

### 3.5 Evaluation & Early Stopping

Every 5 epochs, run a simulated portfolio on the validation set:

1. Predict direction probability + expected return for all stocks in val set
2. Sort by expected return, select top-20
3. Equal-weight hold, compute Sharpe ratio over val period
4. No improvement in Sharpe for 10 consecutive epochs → early stop
5. Keep checkpoint with highest Sharpe

**Model selection criterion:** Top-20 portfolio Sharpe, not MCC/Loss. This is the same signal usage pattern as live trading — selection criteria must be consistent end-to-end.

---

## 4. Feature Pipeline Changes

### 4.1 Output Format

Current `FeaturePipeline.build_features()` returns `(X_flat, y)` for XGBoost. New pipeline needs:

```python
def build_panel_features(self, stock_list, start_date, end_date) -> PanelDataset:
    """Returns:
    - static_features: (N_stocks, static_dim)
    - past_known: (N_stocks, T, past_known_dim)  # T=252
    - past_observed: (N_stocks, T, past_obs_dim)
    - y_direction: (N_stocks, T, 1)
    - y_return: (N_stocks, T, 1)
    - y_volatility: (N_stocks, T, 1)  # 5-day realized vol
    - date_index: (T,)
    - stock_codes: (N_stocks,)
    """
```

### 4.2 Cross-Section Normalizer

New component in `stoke_ml/preprocessing/numeric/cross_section.py`:

```python
class CrossSectionNormalizer:
    """Z-score per date across all stocks."""
    def fit(self, df: pd.DataFrame) -> CrossSectionNormalizer:
        # Compute per-date μ, σ for each feature
        ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # (x - μ_day) / σ_day
        ...
```

### 4.3 Feature Split

Existing 14 auxiliary dimensions are classified into TFT input types:
- **Static**: industry, concept membership (from `ConceptBlockEncoder`), market-cap quantile
- **Past known**: OHLCV, technical indicators, calendar features, fundamental (forward-filled quarterly)
- **Past observed**: news sentiment, capital flow, block trade events, board events, ETF flow

---

## 5. Model Implementation

### 5.1 File Structure

```
stoke_ml/models/tft/
├── __init__.py
├── model.py              # TFTModel (nn.Module)
├── components.py         # VSN, GRN, GatedLinearUnit, TimeDistributed
├── attention.py          # InterpretableMultiHeadAttention
├── heads.py              # DirectionHead, ReturnHead, VolatilityHead
├── dataset.py            # PanelDataset, collate_fn
├── train.py              # Training loop, uncertainty weighting
├── evaluate.py           # Top-K portfolio simulation, Sharpe calc
└── config.py             # TFTConfig dataclass
```

### 5.2 Key Components

- `GRN`: Gated Residual Network — the core nonlinear block
- `VSN`: Variable Selection Network — per-input-type feature selection via softmax gating
- `InterpretableMultiHeadAttention`: Standard MHA with attention weight export (for interpretability in live trading)
- `UncertaintyLoss`: Trainable log-variance parameters, auto-balanced multi-task loss

### 5.3 Dependencies

Add to `requirements.txt`:
```
# TFT model (Phase 5)
pytorch-forecasting>=0.10   # Reference implementation, may adapt
```

Primary implementation will be custom (not `pytorch-forecasting` wrapper) for full control over panel batching and 3-task heads. `pytorch-forecasting` may be used as reference/fallback.

---

## 6. Migration Path

Existing code is preserved. New components are additive:

| What | Fate |
|------|------|
| `XGBoostBaseline` | Keep as baseline comparison |
| `LSTMModel`, `TransformerModel` | Keep for ablation vs TFT |
| `FeaturePipeline.build_features()` | Keep for XGBoost, add `build_panel_features()` for TFT |
| `train_baseline.py`, `train_lstm.py` | Keep unchanged |
| Existing preprocessing chain | Keep, TFT reads its output |
| `evaluation/` | Reuse walk-forward splitter, add top-K portfolio sim |

New training entry point: `scripts/train_tft.py`.

---

## 7. Success Criteria

| Metric | Baseline (XGBoost) | Target (TFT Panel) |
|--------|-------------------|--------------------|
| Direction MCC | 0.07 | > 0.10 |
| Return IC (rank) | — | > 0.03 |
| Top-20 Sharpe (annualized) | — | > 0.5 |
| Out-of-sample R² (return) | — | > 0.01 |
