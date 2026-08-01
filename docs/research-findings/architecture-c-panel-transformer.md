# 方向 C: Multi-Task Panel Transformer（类 Stockformer）深度分析

> **候选定位**: 🥉 远期储备方案
> **一句话**: 面板原生设计，小波分解+图嵌入+多任务，理论上最适配我们的数据形态，但实现复杂度最高

---

## 1. 理论与实证基础

### 1.1 Stockformer (Ma et al., 2025)

**论文**: "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks", *Expert Systems with Applications*, 2025

**GitHub**: Eric991005/Multitask-Stockformer (⭐109, Fork 31)

**核心架构**（5 个关键模块）:

```
Input: 360 price-volume factors × CSI 300 stocks
│
├─ 1. Discrete Wavelet Transform (DWT)
│   returns → DWT →
│     High-frequency component  # 短期波动、突发事件
│     Low-frequency component   # 长期趋势
│
├─ 2. Dual-Frequency Spatiotemporal Encoder
│   Low-freq → Temporal Self-Attention      # 长期依赖建模
│   High-freq → Dilated Causal Convolution   # 短期局部模式
│   Combined → Graph Attention Network (GAT) # 股票间关系
│
├─ 3. Graph Embedding (Struc2vec)
│   Stock similarity graph：基于价格形态的股票间关系
│   Industry graph：同行业股票关系
│   → GAT 编码空间-时间联合表示
│
├─ 4. Multi-Task Learning
│   Task 1: Return regression
│   Task 2: Trend direction (up/down) classification → 57.46% accuracy
│   Joint loss: λ₁ * MSE_return + λ₂ * CE_direction
│
└─ 5. TopK-Dropout Backtesting (via Qlib)
    Transaction costs considered
    Rolling 14-sub-period backtesting (2018-2024)
```

**关键实验结果**:
- **方向准确率**: 57.46% (CSI 300 成分股)
- **超越 10 个基线**: XGBoost, LSTM, Transformer, GRU, TCN, ALSTM, SFM, etc.
- **牛熊震荡市均稳定**: 没有明显的 regime 敏感性
- **开源完整**: 模型代码 + 数据处理 + 回测 + 预训练权重

### 1.2 MT-DNN-DAE (M6 竞赛第 4 名, 2025)

**论文**: "Robust Returns Ranking Prediction and Portfolio Optimization for M6", *International Journal of Forecasting*, 2025

**亮点**:
- 降噪自编码器 (DAE) 做自监督预训练 → 鲁棒特征表示
- Robust Feature Selection (RFS) 筛选高 SNR 特征
- 多任务联合优化
- **M6 全球竞赛第 4 名** — 实战验证

### 1.3 行为驱动 MLP (Luan, 2025)

**论文**: "Deep Learning for Short Term Equity Trend Forecasting: A Behavior Driven Multi Factor Approach"

**亮点**:
- 40 个行为金融因子 + 双任务 MLP
- **MLP 而非 Transformer** — 在低 SNR 环境下简单架构未必差
- IC, IR, 组合回测全面超越线性基线

---

## 2. 开源参考实现

### 2.1 Eric991005/Multitask-Stockformer (⭐109, Fork 31)

- **语言**: Python / PyTorch
- **最近更新**: 2025-05
- **目录结构**:
  ```
  Multitask_Stockformer_models.py   # 核心模型 (完整 Stockformer)
  MultiTask_Stockformer_train.py    # 训练脚本
  Stockformermodel/                 # 模型子模块
  data_processing_script/           # 数据处理 (360 factors 构建)
  Backtest/                         # Qlib 回测 + TopK-Dropout
  config/                           # 超参配置
  cpt/                              # 预训练权重
  ```
- **许可**: 未明确（学术仓库）
- **代码质量**: 单文件大模型 (~2000+ lines)，功能齐全但不太模块化

### 2.2 daniel-debrun/StockFormer

- **定位**: 非官方 PyTorch 重实现，更模块化
- **状态**: 开发中 (2025.08 活跃)

---

## 3. 与我们现有代码的衔接分析

### 3.1 可复用的模块

| 模块 | 复用程度 | 说明 |
|------|---------|------|
| 输出头 | ✅ 80% | Stockformer 用 2 任务 (return+direction)，我们需要加 volatility |
| UncertaintyLoss | ✅ 100% | 多任务 loss 框架直接可用 |
| PanelDataset + Collate | ⚠️ 50% | Stockformer 有自己的 dual-frequency data format |
| 训练循环 | ⚠️ 50% | 需适配 Qlib 回测框架 |
| 评估 | ⚠️ 60% | Stockformer 在 Qlib 里做，我们的独立评估保留 |

### 3.2 需要新写的模块

| 模块 | 复杂度 | 说明 |
|------|--------|------|
| DWT 小波分解 | 🟢 低 | `pywt` 库一行搞定 |
| Dual-Frequency Encoder (低频 Transformer + 高频 CNN) | 🔴 高 | 双路径设计，参数量大 |
| Graph 构建 (Struc2vec or 行业图) | 🟡 中 | 需要股票关系数据 |
| Graph Attention Network (GAT) | 🟡 中 | PyG 或手写 |
| Qlib 集成 | 🔴 高 | 我们的数据 pipeline vs Qlib 的数据格式差异大 |

### 3.3 不能复用的模块

| 模块 | 原因 |
|------|------|
| VSN | Stockformer 用自己的特征选择和因子体系，不用 VSN |
| 几乎所有 TFT 组件 | Stockformer 是完全不同的架构范式 |

---

## 4. 架构设计草案

```
Input: 488 stocks × 60 steps × features
│
├─ 1. Wavelet Decomposition (per stock, per return series)
│   past_returns (60,) → DWT(db4, level=2) →
│     cA2: low-freq trend  (15 steps)  # 低频成分
│     cD2 + cD1: high-freq (60 steps)  # 高频成分
│
├─ 2. Feature Encoding (per frequency)
│   Low-freq path:  cA2 → PositionalEncoding → TransformerEncoder(2 layers) → h_low
│   High-freq path: cD  → DilatedConv1d(dilation=[1,2,4,8]) → h_high
│   Fusion: h_fused = Gate(h_low, h_high)  # 自适应融合
│
├─ 3. Graph Construction
│   Stock similarity: 基于收益率相关性构建 adjacency matrix
│   每季度重建一次（保持行业结构稳定）
│   → GAT(h_fused, adj_matrix) → h_spatial (B, N, T, h)
│
├─ 4. Temporal Aggregation
│   h_spatial → temporal attention pooling → h_stock (B, h)
│
├─ 5. Multi-Task Heads
│   h_stock →
│     RegressionHead → return (B, 1)
│     DirectionHead  → direction (B, 3)  # 涨/平/跌
│     VolatilityHead → volatility (B, 1)
│
│   Loss = w₁ * AdjMSE_return + w₂ * CE_direction + w₃ * MSE_vol
│        + RankICLoss(return_pred, return_true)
```

### 关键设计抉择

1. **DWT vs 原始序列**: 小波分解是有信息损失的（降采样）。对于 60 步的短序列，level=2 后低频只剩 15 步。在金融数据上是否值得，需要 ablation
2. **Graph 依赖**: 需要维护股票关系图，增加工程复杂度。A 股行业分类可以从东方财富获取
3. **双路径参数量**: 低频 Transformer + 高频 CNN + GAT 三个模块，参数量可能超 20M，过拟合风险大

---

## 5. 优劣势评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 金融实证 | ⭐⭐⭐⭐ | Stockformer CSI 300 57.46%, M6 第 4 名, 行为 MLP IC/IR 双优 |
| 梯度稳定性 | ⭐⭐⭐ | 仍含 temporal self-attention，存在梯度坍塌风险（需加保护） |
| Panel 适配 | ⭐⭐⭐⭐⭐ | **原生 panel 设计**：图嵌入+跨股票 GAT+多任务，最适配 |
| 实现复杂度 | ⭐⭐ | 5 个模块全是新的，与现有代码几乎无复用 |
| 代码可参考性 | ⭐⭐⭐⭐ | Stockformer 官方 109⭐ 完整仓库，但单文件不模块化 |

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 实现复杂度太高，容易出 bug | 🔴 | 分模块逐一实现并单测 |
| 参数过多 → 过拟合 | 🟡 | 每个模块先用最小配置，ablation 验证必要性 |
| DWT 在短序列上的信息损失 | 🟡 | 先用原始序列做基线，再加 DWT 对比 |
| Qlib 依赖（回测框架） | 🟢 | 我们的评估模块可以替代 Qlib |
| 仍含 attention → 梯度问题可能复现 | 🟡 | 梯度监控 + 分层裁剪保护 |

---

## 6. 为什么放在第三优先级

Stockformer 在理论上最适配我们的数据形态（panel + 多任务 + A 股），但有几个硬伤：

1. **实现成本**: 5 个模块全新实现，预计 15+ 小时，且与现有代码几乎零复用
2. **仍含 attention**: 低频 Transformer 路径仍然有 temporal self-attention，梯度坍塌风险不能完全排除
3. **过度工程化风险**: 360 因子 + 小波分解 + 图嵌入 + 双路径 + 多任务，在金融低 SNR 环境下，简单模型往往比复杂模型更好（Oxford 基准的结论）
4. **Qlib 耦合**: Stockformer 官方代码深度依赖 Qlib 的数据格式和回测框架，脱离 Qlib 使用的适配成本高

**建议路径**: 先用方向 A (VSN+xLSTM) 跑通验证 → 如果效果不理想 → 考虑 B (Mamba) → 如果 A 和 B 都无法满意 → C 作为"终极方案"。

但 Stockformer 的很多**设计思想**值得借鉴（可以嫁接到 A 或 B 中）：
- 多任务联合损失设计 ✅ 我们已有
- 跨股票信息共享 ⬜ 方向 A/B 可加轻量 cross-stock pooling
- 行为金融因子 ⬜ 可以作为新的辅助特征维度
- TopK-Dropout 选股 ⬜ 可以在后处理阶段实现

---

## 7. 实施预估

| Phase | 工作内容 | 预估时间 |
|-------|---------|---------|
| C.1 | DWT 小波分解模块 | 1h |
| C.2 | Dual-Frequency Encoder (低频 Transformer) | 2.5h |
| C.3 | Dilated Conv1d (高频 CNN) | 1.5h |
| C.4 | Graph 构建 + GAT | 3h |
| C.5 | 多任务输出适配 (3 tasks) | 1h |
| C.6 | 端到端集成 + 与现有 pipeline 对接 | 3h |
| C.7 | 全量训练 + Qlib 替代评估 | 3h |
| **合计** | | **~15h + 训练时间** |

---

## 8. 参考文献

1. Ma et al., "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks", Expert Systems with Applications, 273, 126803, 2025
2. Ai, Liu & Lin, "Robust Returns Ranking Prediction and Portfolio Optimization for M6", International Journal of Forecasting, 41(4), 2025
3. Luan, "Deep Learning for Short Term Equity Trend Forecasting: A Behavior Driven Multi Factor Approach", 2025
4. GitHub: Eric991005/Multitask-Stockformer (⭐109, Fork 31)
5. GitHub: daniel-debrun/StockFormer (非官方重实现)
