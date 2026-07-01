# Research Findings

Benchmark experiments and their conclusions. Every claim is backed by reproducible scripts in `scripts/`.

---

## 1. Ablation — Data Source Contributions (2026-06-29)

95 stocks, 1000 bootstrap samples, XGBoost flat.

| Config | MCC | 95% CI | Δ vs technical |
|---|---|---|---|
| technical | 0.0136 | [-0.0035, 0.0312] | — |
| + sentiment | 0.0279 | [0.0095, 0.0464] | +0.0143 |
| + guba | 0.0219 | [0.0032, 0.0384] | +0.0084 |
| + comment | 0.0224 | [0.0045, 0.0408] | +0.0089 |
| ALL | 0.0261 | [0.0104, 0.0426] | +0.0125 |

- All text dimensions improve MCC vs technical-only (all CIs > 0)
- News sentiment has largest effect (+104% MCC)
- ALL config underperforms +sentiment alone (dimension explosion)
- Δ CIs all cross zero — need more data or stronger signal for significance

Script: `scripts/train_baseline.py` with ablation args.

---

## 2. Feature Selection — MI Filter vs Dimension Explosion (2026-07-01)

20 stocks, 99 folds, XGBoost flat, seq_len=60, step_months=6, max 5 folds.

| Config | MCC | Features |
|--------|-----|----------|
| technical | 0.0631 | 1605 |
| sentiment | 0.0589 | 1915 |
| all | 0.0478 | 4075 |
| **all_mi** | **0.0685** | **200** |

- **Dimension explosion confirmed**: ALL (4075 features) is WORSE than technical-only by -0.0153 MCC
- **MI filter is the best config**: 200 features achieves MCC=0.0685, +0.0054 vs baseline
- MI top scores range 0.07–0.11, median ~0.043 — moderate signal strength

Script: `scripts/benchmark_feature_selection.py`
CSV: `models/checkpoints/feature_selection_benchmark.csv`

---

## 3. Preprocessing — Old vs New Numeric Chain (2026-07-01)

20 stocks, 99 folds, XGBoost flat, technical+sentiment+guba+xueqiu only.

| Config | MCC |
|--------|-----|
| old (current pipeline) | 0.0690 |
| new_numeric (outlier→missing→robust_scaling) | 0.0449 |

- **Δ = -0.0241**: Numeric preprocessing DEGRADES XGBoost performance
- **Root cause**: Tree models don't need normalization; outlier clipping removes informative extremes; robust scaling distorts distribution
- **Design intent**: This preprocessing chain is built for DL models (LSTM/Transformer), not tree models
- RobustScaler had inf overflow bugs (fixed: clip ±1e4, nan_to_num)

Script: `scripts/benchmark_preprocessing.py`
CSV: `models/checkpoints/preprocessing_benchmark.csv`

---

## 4. Label Types — Absolute vs Sector-Relative (2026-07-01)

10 stocks, 50 folds, XGBoost flat.

| Label | MCC |
|-------|-----|
| abs (price[t+1] > price[t]) | 0.0603 |
| rel (stock_ret > sector_median_ret) | -0.0073 |

- **Δ = -0.0676**: Sector-relative outperformance is near-random with current features
- **Interpretation**: Our features capture market beta, not stock-specific alpha
- Sector-relative labels are well-balanced (~50/50) but unpredictable

Script: `scripts/benchmark_labels.py`
CSV: `models/checkpoints/label_benchmark.csv`

---

## Summary — MCC Ceiling Analysis

| Experiment | Best MCC | Key Takeaway |
|------------|----------|--------------|
| Feature Selection | 0.0685 | MI filter to 200 features is optimal |
| Preprocessing A/B | 0.0690 | Numeric chain hurts XGBoost (for DL only) |
| Label Types | 0.0603 | Alpha prediction needs alpha features |

**MCC ceiling with current features + XGBoost ≈ 0.07.** Next breakthroughs require:
- **DL models**: LSTM/Transformer can use the preprocessing chain (normalization is beneficial)
- **Alpha features**: Higher moments (skew/kurtosis), Amihud illiquidity, realized volatility surface
- **Cross-sectional training**: Panel-mode with per-date normalization across stocks
- **Richer text**: BERTopic topic modeling, body sentiment utilization
