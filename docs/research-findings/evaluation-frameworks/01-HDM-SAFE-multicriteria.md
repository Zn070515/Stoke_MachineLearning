# HDM & SAFE 多维度评估框架

> 来源: IEEE TEM 2025 (HDM), Mathematics 2025 (SAFE), Semantics Scholar 2025 (Walk-Forward)

## 1. HDM 层次决策模型 (IEEE TEM, 2025)

五维度 360° 量化策略评估:

| 视角 | 评估内容 |
|------|---------|
| **经济金融基础** | 底层经济逻辑的合理性 |
| **数据视角** | 数据质量、完整性、治理 |
| **特征视角** | 预测因子/alpha 因子的质量和相关性 |
| **建模视角** | ML/DS 方法论的严谨性 |
| **绩效视角** | 结果的统计和经济有效性 |

## 2. SAFE 框架 (Mathematics, Jan 2025)

专为黑盒 ML 模型设计，四维度:

- **Sustainability** — 不同市场体制下的长期可行性
- **Accuracy** — 预测性能和与观测数据的对齐
- **Fairness** — 伦理考量和偏差检测
- **Explainability** — 模型决策的可解释性

验证数据集: IBM 股票，对比传统 ML (LR, SVM) vs DL (RNN, LSTM, GRU)

## 3. Walk-Forward 验证框架 (2025)

解决量化金融"再现性危机"的关键创新:
- **Deflated Sharpe Ratio (DSR)**: 修正选择偏差、回测过拟合、非正态性
- **迭代滚动优化/测试周期** (Walk-Forward Analysis)
- 可扩展到 LLM 驱动的假设生成器

## 4. 7 大验证支柱

1. Walk-Forward Analysis — WFE ≥ 50%, MDD < 40%
2. Monte Carlo 模拟 — 概率性压力测试
3. 下行风险指标 — Sortino + Calmar (非仅 Sharpe)
4. 偏差消除 — 系统审计 look-ahead/survivorship/confirmation
5. 真实摩擦建模 — 滑点、市场冲击、佣金
6. 数据质量标准与治理
7. 纸交易/前向测试
