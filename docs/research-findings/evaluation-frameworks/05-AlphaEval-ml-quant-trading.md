# AlphaEval & ml-quant-trading 评估框架

> 来源: arXiv 2508.13174 (AlphaEval), arXiv 2507.07107 (ml-quant-trading), GitHub

## AlphaEval — Alpha 挖掘评估 (2025)

免回测、可并行的 alpha 质量评估框架，五维度:

| 维度 | 内容 |
|------|------|
| **Predictive Power** | IC, RankIC |
| **Temporal Stability** | 跨时间段信号质量一致性 |
| **Robustness** | 抗噪声和市场体制变化 |
| **Financial Logic** | alpha 公式的逻辑自洽性和透明度 |
| **Diversity** | 生成 alpha 之间的结构多样性 |

与 8 种流行 alpha 挖掘模型 (遗传编程、RL、LLM) 对标。评估一致性可媲美完整回测，效率更高。

## ml-quant-trading — ML 多因子研究栈 (2025)

PyTorch 端到端 ML 多因子交易研究栈:

### 核心组件
- **213 因子维度** (9 Alpha101 + 204 传统手工因子)
- **GPU 向量化张量因子引擎** (masked primitives)
- **偏差修正** (涨跌停/停牌)
- **截面 Markowitz 组合优化** (shrunk covariance)
- **向量化回测引擎** → Sharpe / IC / IR / Drawdown
- MLP & Transformer 模型基线

### 评估指标
- Sharpe Ratio
- Information Coefficient (IC)
- Information Ratio (IR)
- Max Drawdown
- 换手率

### 对比点
- 涨跌停偏差修正 (与本项目 limit-up/down bias correction 同源)
- 截面标准化
- 纯量 PyTorch 因子引擎
