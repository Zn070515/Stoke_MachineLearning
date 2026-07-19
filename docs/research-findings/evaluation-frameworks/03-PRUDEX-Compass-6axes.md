# PRUDEX-Compass 六轴评估体系

> 来源: NTU, arXiv 2302.00586, TradeMaster-NTU/TradeMaster (GitHub)

## 6 轴 (PRUDEX 缩写)

| 轴 | 英文 | 中文 | 评估内容 |
|----|------|------|---------|
| **P** | Profitability | 盈利能力 | 总回报、alpha 衰减、权益曲线分析 |
| **R** | Risk-Control | 风险控制 | 波动率、最大回撤、极端市场表现 |
| **U** | Universality | 普适性 | 跨国家、跨资产类型、跨时间尺度 |
| **D** | Diversity | 多样性 | t-SNE、熵、相关性、多样性热力图 |
| **E** | Reliability | 可靠性 | 多随机种子、超参数敏感性、滚动窗口 |
| **X** | eXplainability | 可解释性 | LIME/SHAP（规划中） |

## 两层结构

- **内层 (轴级)**: 星形图，0-100 归一化分数，50=市场平均 (等权组合)，100=超越市场平均 20%
- **外层 (指标级)**: 17 项具体指标追踪报告完整性

## 17 项具体指标

### 盈利能力 (Profitability)
1. Total Return
2. Alpha Decay
3. Equity Curve

### 风险控制 (Risk-Control)
4. Volatility
5. Max Drawdown
6. Extreme Market Performance (黑天鹅)

### 普适性 (Universality)
7. Cross-Country
8. Cross-Asset
9. Cross-TimeScale

### 多样性 (Diversity)
10. t-SNE Visualization
11. Entropy
12. Correlation Matrix
13. Diversity Heatmap

### 可靠性 (Reliability)
14. Performance Profile (多种子)
15. Variability Analysis
16. Rolling Window Test
17. Rank Comparison

## Benchmark 算法
AlphaMix+: MoE + risk-sensitive Bellman backup，在所有 6 轴上超越市场平均

## 开源
github.com/TradeMaster-NTU/PRUDEX-Compass
