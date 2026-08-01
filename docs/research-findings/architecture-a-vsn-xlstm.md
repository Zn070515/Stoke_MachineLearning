# 方向 A: VSN + xLSTM 深度分析

> **候选定位**: 🥇 首选方案
> **一句话**: Oxford 2025 金融基准测试 Sharpe 第一，梯度天然稳定，VSN 代码可复用

---

## 1. 理论与实证基础

### 1.1 xLSTM 论文 (Beck et al., 2024)

**论文**: "xLSTM: Extended Long Short-Term Memory", arXiv:2405.04517

xLSTM 对传统 LSTM 做了两个关键改进：

| 改进 | sLSTM | mLSTM |
|------|-------|-------|
| 记忆类型 | 标量记忆 | 矩阵记忆（协方差更新） |
| 并行性 | 序列化（循环） | 完全可并行化 |
| 指数门控 | ✅ | ✅ |
| 记忆混合 | ✅ (新的 memory mixing) | ❌ |
| 因果卷积 | ✅ kernel=4 | ✅ kernel=4 |
| 典型扩展因子 | 4/3 | 2 |

**指数门控**是核心创新：传统 LSTM 用 sigmoid 门控，xLSTM 用指数门控 + 稳定化技巧，使得模型可以学习到更长期、更稳定的记忆模式。对于金融时序的低 SNR 环境，这个特性尤其重要。

### 1.2 Oxford 2025 金融基准测试

**论文**: "Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance", Saly-Kaufmann, Wood, Calliess, Zohren (Oxford-Man Institute)

**测试范围**: 2010-2025，15 年全球期货数据（商品、股票指数、债券、外汇）

**核心结论**:
- **VSN+LSTM (VLSTM)**: Sharpe 2.40, CAGR 26.3%, MaxDD -22.9% → **Sharp 最高**
- **VSN+xLSTM**: **下行保护最优**, 回撤控制最好
- **xLSTM**: **最大 breakeven 交易成本缓冲**（对摩擦最鲁棒）
- **线性模型 AR1x**: Sharpe 仅 0.77

**关键洞察**: "inductive bias trumps model scale" — 为稳定时序结构设计的模型，碾压泛用 Transformer 和状态空间模型（Mamba2）

### 1.3 xLSTM-TS: 专门用于股票预测 (González-Pérez et al.)

**论文**: "An Evaluation of Deep Learning Models for Stock Market Trend Prediction"

| 数据集 | 准确率 | F1 |
|--------|--------|----|
| S&P 500 日频 | 71.28% | 73.00% |
| EWZ (巴西) 日频 | 72.87% | 73.16% |
| S&P 500 小时频 | 71.42% | 66.67% |

对比基准：TCN, N-BEATS, **TFT**, N-HiTS, TiDE — **xLSTM-TS 全面胜出**

架构亮点：集成小波去噪 (DWT) 预处理 + xLSTM backbone，噪音环境下的预测稳定性显著优于 TFT。

---

## 2. 开源参考实现

### 2.1 myscience/x-lstm (⭐184, Fork 19)

- **定位**: 原始论文的 PyTorch Lightning 实现
- **质量**: 清晰、模块化，按论文结构组织
- **最近更新**: 2024-08（不太活跃，但代码完善）
- **关键结构**:
  ```
  xlstm/
    slstm.py       # sLSTM 单元 + block
    mlstm.py       # mLSTM 单元 + block
    block.py       # 混合 block 层
    model.py       # xLSTM 主模型
    llm.py         # LLM 应用示例
  ```
- **亮点**: `signature` 参数 `(m_num, s_num)` 控制每层 mLSTM:sLSTM 比例
- **许可**: MIT

### 2.2 gonzalopezgil/xlstm-ts (⭐46, Fork 14)

- **定位**: 专门为时间序列预测优化的 xLSTM
- **质量**: 包含完整训练/评估/可视化 pipeline
- **最近更新**: 2026-05（活跃维护）
- **目录结构**:
  ```
  src/
    ml/
      xlstm_model.py     # xLSTM 时间序列模型
      wavelet_denoise.py # 小波去噪预处理
    main.py              # 训练入口
  data/                  # S&P 500, EWZ 数据集
  notebooks/             # 分析和可视化
  ```
- **许可**: MIT
- **直接可用性**: 高 — 已经是股票预测场景，模型架构可以直接参考

### 2.3 PyxLSTM (PyPI 包)

- `pip install PyxLSTM`
- 模块化实现：`slstm.py`, `mlstm.py`, `block.py`, `model.py`
- 包含预训练权重和训练脚本

---

## 3. 与我们现有代码的衔接分析

### 3.1 可复用的模块

| 模块 | 文件 | 复用程度 |
|------|------|---------|
| VSN (标量路径) | `stoke_ml/models/tft/vsn.py` | ✅ 100% 复用 |
| 输出头 (Direction/Return/Vol) | `stoke_ml/models/tft/heads.py` | ✅ 100% 复用 |
| UncertaintyLoss | `stoke_ml/models/tft/loss.py` | ✅ 100% 复用 |
| PanelDataset + Collate | `stoke_ml/models/tft/dataset.py` | ✅ 100% 复用 |
| 训练循环框架 | `stoke_ml/models/tft/train.py` | ⚠️ 70% 复用（LR scheduler 部分改写） |
| 评估 (Sharpe/IC) | `stoke_ml/models/tft/evaluate.py` | ✅ 100% 复用 |
| 配置 (TFTConfig) | `stoke_ml/models/tft/config.py` | ⚠️ 重命名为 ModelConfig，调整参数 |

### 3.2 需要新写的模块

| 模块 | 理由 | 参考来源 |
|------|------|---------|
| xLSTM backbone (sLSTM + mLSTM) | 替代 TFT 的 LSTM+Attention | myscience/x-lstm + xlstm-ts |
| Static Encoder (4 上下文向量) | TFT weakness analysis 的结论 | pytorch-forecasting |
| Model 主类 | 组装 VSN → xLSTM → Heads | 我们的 model.py 改 |

### 3.3 需要废弃的模块

| 模块 | 文件 | 原因 |
|------|------|------|
| InterpretableMultiHeadAttention | `attention.py` | 不用的 temporal attention |
| TFTModel | `model.py` | 替换为新架构 |
| GRN + GLU + TimeDistributed | `components.py` | xLSTM 有自己的 gating，不需要 GRN |

---

## 4. 架构设计草案

```
Input: static (B,S), past_known (B,T,K), past_observed (B,T,O)

1. Static VSN
   static_vars → VSN_scalar → static_embedding (B, h)

2. Static Encoder (4-GRN)
   static_embedding → 
     GRN_vs → c_s (B, h)   # VSN context
     GRN_h  → c_h (B, h)   # xLSTM hidden init  
     GRN_c  → c_c (B, h)   # xLSTM cell init
     GRN_e  → c_e (B, h)   # post-xLSTM enrichment

3. Temporal VSN (encoder + decoder 共享)
   past_known (B,T,K,1) → VSN(past_known, context=c_s) → feat_known (B,T,h)
   past_observed (B,T,O,1) → VSN(past_observed, context=c_s) → feat_obs (B,T,h)
   feat = concat(feat_known, feat_obs) → project → (B,T,h)

4. xLSTM Backbone
   feat → xLSTM(c_h, c_c) → temporal_features (B,T,h)
   # sLSTM + mLSTM 交替堆叠
   # 无 attention → 无梯度坍塌风险

5. Static Enrichment
   temporal_features → GRN_enrich(context=c_e) → enriched (B,T,h)

6. Multi-Head Output (取最后时间步)
   enriched[:,-1,:] →
     DirectionHead → (B, 3)    # 涨/平/跌
     ReturnHead   → (B, 1)     # 预期收益率
     VolatilityHead → (B, 1)   # 预期波动率

Loss = UncertaintyLoss(CE_direction, AdjMSE_return, MSE_volatility)
      + λ * RankICLoss(pred_return, true_return)
```

### 关键设计决策

1. **sLSTM:mLSTM 比例**: 金融数据偏序列化，用 `sLSTM` 偏多的 signature，如 `(m_num=1, s_num=2)` 或 `(m_num=0, s_num=3)`
2. **隐藏维度**: xLSTM 论文用 128-256，结合我们 RTX 4090 和 488 stocks panel，建议 `hidden_dim=128`
3. **层数**: 2-3 层 xLSTM block，多了容易过拟合（金融 SNR 低）
4. **VSN 位置**: 在 xLSTM 之前做特征选择，比在 LSTM 之后更有效（Oxford VLSTM 的做法）

---

## 5. 优劣势评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 金融实证 | ⭐⭐⭐⭐⭐ | Oxford 基准 Sharpe 第一，xLSTM-TS 在 S&P 500 上 71.28% 准确率 |
| 梯度稳定性 | ⭐⭐⭐⭐⭐ | xLSTM 指数门控天然稳定，无 attention = 无坍塌风险 |
| Panel 适配 | ⭐⭐⭐⭐ | VSN 提供 per-stock 个性化，需额外加跨股票信息共享层 |
| 实现复杂度 | ⭐⭐⭐⭐ | 可复用 VSN/Heads/Loss/Dataset，只需新写 xLSTM backbone |
| 代码可参考性 | ⭐⭐⭐⭐ | myscience/x-lstm (184⭐) + xlstm-ts (46⭐) 两个高质量参考 |

### 风险

| 风险 | 缓解 |
|------|------|
| xLSTM 论文较新 (2024.05)，社区积累不如 LSTM | PyxLSTM 有 pip 包，xlstm-ts 已在股票上验证 |
| sLSTM 序列化，训练速度可能慢于 Transformer | 488 stocks × 60 steps 序列不长，RTX 4090 可承受 |
| 无内置跨股票信息共享 | Phase 2 加 cross-sectional pooling 层 |

---

## 6. 实施预估

| Phase | 工作内容 | 预估时间 |
|-------|---------|---------|
| A.1 | 实现 sLSTM + mLSTM 模块 (参考 myscience/x-lstm) | 2h |
| A.2 | 实现 Static Encoder (4 GRN) | 1h |
| A.3 | 组装 VSN → xLSTM → Heads 主模型 | 1.5h |
| A.4 | 调整训练循环（LR warmup + 分层梯度裁剪） | 1h |
| A.5 | 端到端集成测试 (10 stocks) | 0.5h |
| A.6 | 全量训练 (488 stocks) | 训练时间取决于 epoch |
| **合计** | | **~6h + 训练时间** |

---

## 7. 参考文献

1. Beck et al., "xLSTM: Extended Long Short-Term Memory", arXiv:2405.04517, 2024
2. Saly-Kaufmann, Wood, Calliess, Zohren, "Deep Learning for Financial Time Series: A Large-Scale Benchmark", Oxford-Man Institute, arXiv:2603.01820, 2025
3. González-Pérez et al., "An Evaluation of Deep Learning Models for Stock Market Trend Prediction", 2024-2026
4. GitHub: myscience/x-lstm (PyTorch Lightning, 184⭐)
5. GitHub: gonzalopezgil/xlstm-ts (时间序列优化, 46⭐)
6. PyPI: PyxLSTM (pip installable)
