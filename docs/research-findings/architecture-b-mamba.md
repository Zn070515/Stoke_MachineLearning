# 方向 B: Mamba 双尺度架构 深度分析

> **候选定位**: 🥈 备选方案
> **一句话**: 选择性状态空间 + 线性复杂度 + 双尺度微观/宏观分离，2025 年金融时序增速最快的研究方向

---

## 1. 理论与实证基础

### 1.1 从 S4 到 Mamba → Mamba-2 → Mamba-3

| 版本 | 时间 | 关键创新 |
|------|------|---------|
| **S4** (Gu et al.) | 2021 | HiPPO 初始化 + 结构化状态矩阵 + FFT 卷积 |
| **S4D** | 2022 | 对角 A 矩阵简化 |
| **Mamba / S6** (Gu & Dao) | 2023 | **选择性状态空间**：B, C, Δ 变成输入依赖 → 选择性关注 |
| **Mamba-2** (Dao & Gu) | 2024.05 | SSD (State Space Duality)：统一 SSM 与结构化注意力，2-8× 加速 |
| **Mamba-3** (Wang et al.) | 2025 | MIMO 多输入多输出 SSM，复数状态 |

**Mamba/S6 的核心优势**（是替代 TFT 的关键理由）:
- **线性 O(T) 复杂度** vs Transformer 的 O(T²)
- **输入依赖的选择性**：动态决定记住什么、忽略什么 → 天然适应市场 regime 切换
- **梯度稳定**：结构化状态空间，无 post-attention 梯度坍塌问题
- **可解释性**：通过状态转移矩阵可分析模型关注的时间点

### 1.2 CMDMamba (2025.07, Frontiers in AI)

**论文**: "CMDMamba: dual-layer Mamba architecture with dual convolutional feed-forward networks for efficient financial time series forecasting"

**核心架构**:
```
Input (OHLCV + indicators)
    │
    ├─ High-Sensitivity Mamba Layer ─→ 微观短期波动捕获
    │   • 快衰减率 (fast state decay)
    │   • 小感受野
    │
    └─ Low-Sensitivity Mamba Layer  ─→ 宏观长期趋势捕获
        • 慢衰减率 (slow state decay)
        • 大感受野
    │
    ├─ DConvFFN (Dual Convolutional FFN)
    │   • 捕获跨变量相关关系
    │   • 保持时序局部性
    │
    └─ Block-wise sequence partitioning
        • 增强语义粒度
```

**实验结果**:
- **10.4% 平均提升** vs SOTA (多变量预测)
- **高噪声、高波动条件**下表现最优
- **近线性时间复杂度**，匹配或超越 Transformer 准确率
- 测试覆盖：股票、指数、外汇、加密货币

### 1.3 GHOST (2025, Expert Systems with Applications)

**论文**: "GHOST: Sentiment-gated Mamba and Stock-wise Tokenization Attention"

**GitHub**: WHUT-zwj/GHOST (⭐24, Fork 11, MIT license)

**核心架构**:
```
1. Hierarchical Sentiment-Gated Layer
   News → GDELT sentiment extraction (30 features)
   → MLP gate → dynamic weight allocation
   → Time-varying gating structure

2. Intra-Stock Mamba Selection Layer  
   每只股票共享 Mamba 参数（正则化效果）
   O(T) 线性复杂度
   选择性状态空间 ≈ 自适应市场 regime 检测

3. Stock-wise Tokenization
   Temporal tokens (T) → Stock tokens (N)
   复杂度从 O(T²) 降到 O(N²)

4. Inter-Stock Attention
   股票 token 之间做 multi-head attention
   捕获跨股票相关关系
```

**实验结果**:
- 中美两个市场均超越最新基线
- 代码公开完整

### 1.4 MambaStock (2023/2024 基线)

**方法**: 第一个将 Mamba/S6 应用于股票预测的工作

**输入**: OHLCV + 财务比率 (P/E, P/B, 换手率等)

**输出**: 价格变化率 (tanh activation)

**结果**: 在 4 只中国银行股上 R² 0.8873-0.9733，超越 ARIMA、Kalman Filter、LSTM、BiLSTM、Transformer

---

## 2. 开源参考实现

### 2.1 WHUT-zwj/GHOST (⭐24, Fork 11)

- **语言**: Python / PyTorch
- **最近更新**: 2025-11
- **目录结构**:
  ```
  models/
    GHOST.py               # 主模型
  layers/
    Embed.py               # 嵌入层
    GatingMechanism.py     # 门控机制
    SelfAttention_Family.py # 自注意力族
    Transformer_EncDec.py  # Transformer 编解码器
  data_provider/           # 数据加载
  exp/                     # 实验配置
  run.py                   # 训练入口
  ```
- **质量评估**: 学术代码，模块清晰，有完整的数据加载和实验管理
- **直接可用性**: 中 — 需要适配我们的 panel 数据格式和 3-task 输出

### 2.2 其他相关 Mamba 实现

| 仓库 | 定位 | 相关度 |
|------|------|--------|
| state-spaces/mamba (official) | Mamba 官方实现 | 底层 SSM 算子参考 |
| alexandrehuat/mamba2-torch | Mamba-2 PyTorch 实现 | 如果要用 Mamba-2 |
| BigQuant/MambaStock | MambaStock 复现 | 简单，入门参考 |

---

## 3. 与我们现有代码的衔接分析

### 3.1 可复用的模块

| 模块 | 复用程度 |
|------|---------|
| VSN (标量路径) | ✅ 100% — 作为 Mamba 之前的特征选择 |
| 输出头 | ✅ 90% — 需加一个 aggregation 层 |
| UncertaintyLoss | ✅ 100% |
| PanelDataset + Collate | ✅ 100% |
| 训练循环 | ⚠️ 60% — Mamba 有自己的训练 trick |
| 评估 | ✅ 100% |

### 3.2 需要新写的模块

| 模块 | 复杂度 | 参考 |
|------|--------|------|
| Mamba SSM 核心算子 | 🟡 中 | mamba 官方 repo 的 `selective_scan` |
| 双尺度 Mamba 层 (High/Low sensitivity) | 🟡 中 | CMDMamba 论文 |
| Stock Tokenization 层 | 🟢 低 | GHOST 的 `Stock-wise Tokenization` |
| Cross-stock attention (轻量) | 🟢 低 | 标准 MHA |

---

## 4. 架构设计草案

```
Input: static (B,S), past_known (B,T,K), past_observed (B,T,O)

1. Feature Embedding
   past_known (B,T,K) → VSN → feat_known (B,T,h)
   past_observed (B,T,O) → VSN → feat_obs (B,T,h)
   concat + project → x (B,T,h)

2. Static Encoder → Mamba State Init
   static_features → GRN_h → h_0 (B,h)  # Mamba 初始状态
   static_features → GRN_c → c_0 (B,h)  # Mamba 初始状态（或 context）

3. Dual-Scale Mamba Backbone
   x →
     ├─ High-Sensitivity Mamba (fast decay)
     │   • 关注最近 5-10 天的短期波动
     │   • 选择性状态空间，动态过滤噪声
     │
     └─ Low-Sensitivity Mamba (slow decay)
         • 关注 30-60 天的中期趋势
         • 学习市场 regime 转换信号
     │
     ├─ Gate: α * high_out + (1-α) * low_out
     │   α 由 static context 动态控制（不同股票不同权重）
     │
     └─ DConvFFN: 跨变量交互 + 时序局部性保持

4. Cross-Stock Pooling (可选，轻量)
   (B, T, h) → reshape → (N_stocks, T, h) per date
   → mean/max pooling over stock dim → (B, h)
   → 捕捉市场整体情绪的跨股票信号

5. Output Heads
   → DirectionHead (3-class)
   → ReturnHead (scalar)
   → VolatilityHead (positive scalar)

Loss = UncertaintyLoss + RankICLoss
```

### 关键设计抉择

1. **是否需要双尺度**: 如果单 Mamba 层效果足够，可以先做简化版（单层 Mamba），验证后再加双尺度
2. **Stock Tokenization**: 488 stocks 的 cross-stock attention 是 488² ≈ 238K，可承受。如果用 GHOST 的方式做，T→N tokenization 可以再降复杂度
3. **状态维度**: Mamba 的 `d_state` 通常 16-64，结合我们的 hidden_dim=128，建议 `d_state=32`

---

## 5. 优劣势评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 金融实证 | ⭐⭐⭐⭐ | CMDMamba 10.4% 提升, MambaStock R² 0.97, GHOST 中美双市场 |
| 梯度稳定性 | ⭐⭐⭐⭐ | SSM 无 attention，结构化状态空间梯度更稳定 |
| Panel 适配 | ⭐⭐⭐⭐ | GHOST 的 Stock Tokenization 专门解决跨股票建模 |
| 实现复杂度 | ⭐⭐⭐ | Mamba 选择性扫描算子较复杂，双尺度 + cross-stock 增加量 |
| 代码可参考性 | ⭐⭐⭐ | GHOST (24⭐) 直接可用但小，mamba official 需要适配时间序列 |

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Mamba SSM 算子实现复杂，易出错 | 🟡 | 直接用 mamba 官方 repo 的 `selective_scan` 而非自写 |
| Mamba-2/Mamba-3 尚未在金融验证 | 🟡 | 先用 Mamba-1 (S6)，有 CMDMamba/MambaStock 实证 |
| 双尺度设计增加超参调优空间 | 🟢 | 先做单尺度验证，再加双尺度 |
| GHOST 仓库较小 (24⭐)，代码质量未知 | 🟡 | 用 CMDMamba 论文指导双尺度，GHOST 只参考 Stock Tokenization |

---

## 6. 实施预估

| Phase | 工作内容 | 预估时间 |
|-------|---------|---------|
| B.1 | 集成 mamba 官方 `selective_scan` 算子 | 1.5h |
| B.2 | 实现 Mamba Block（单层 S6 + FFN） | 2h |
| B.3 | 实现 Stock Tokenization + Cross-stock attention | 1.5h |
| B.4 | 组装 VSN → Mamba → Heads 主模型 | 1.5h |
| B.5 | 端到端集成测试 (10 stocks) | 0.5h |
| B.6 | 加双尺度层 (High/Low sensitivity) | 1.5h |
| B.7 | 全量训练 (488 stocks) | 训练时间 |
| **合计（单尺度）** | | **~7h + 训练时间** |
| **合计（双尺度）** | | **~8.5h + 训练时间** |

---

## 7. 参考文献

1. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
2. Dao & Gu, "Mamba-2: State Space Duality", 2024
3. CMDMamba: "Dual-layer Mamba with Dual Convolutional FFN for Financial Forecasting", Frontiers in AI, 2025
4. Zhu et al., "GHOST: Sentiment-gated Mamba and Stock-wise Tokenization Attention", Expert Systems with Applications, 2025
5. Shi, "MambaStock: Selective State Space Model for Stock Prediction", 2023
6. MambaDiffTS: "Mamba Diffusion Probabilistic Models for Time Series", EAAI, 2025
7. GitHub: state-spaces/mamba (Mamba 官方)
8. GitHub: WHUT-zwj/GHOST (⭐24, MIT)
