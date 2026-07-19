# 截面/Panel ML 评估方法论

> 来源: Review of Finance 2024 (Han, He, Rapach & Zhou), Intl J Forecasting 2024 (Akgun et al.)

## Han et al. (2024) — 截面收益预测三指标

Fama-MacBeth 回归在 ML 时代的扩展，三个专用评估指标:

### 1. 截面 OOS R²
将传统时间序列 OOS R² 推广到截面设置。任何低于截面均值基准的模型视为有问题。

### 2. Encompassing Test (包容性检验)
形式化统计检验，比较两个竞争模型截面预测的信息内容，判断一个模型是否"包容"(dominate) 另一个。

### 3. 图形化一致性评估
可视化工具，评估预测准确性随时间推移的稳定性。

### 补充指标
- **Long-Short 对冲组合 Sharpe**: E-LASSO 年化 1.65 vs 市场 0.47 (1970-2021)
- **累计组合收益**: 特别在经济衰退期表现强势

## Akgun et al. (2024) — Panel 预测评估检验

### Cₙₜ⁽³⁾ 统计量
检验两预测在**截面聚类内**是否等精度，同时允许精度**跨聚类**不同。
- 适应强截面相依性
- 实现于 R 包 `pEPA`

### 未知聚类扩展
- Panel Kmeans + Selective Inference (多面体方法)
- 截断 χ² 分布检验统计量 (后聚类有效)
- HAC 方差估计 (对任意截面相依形式鲁棒)

## 2024 关键方法论贡献

| 方法 | 评估方式 |
|------|---------|
| E-LASSO | 截面 R² + 包容检验 + 图形一致性 + Sharpe |
| Kernel Joint Mean-Cov | OOS Sharpe ratio |
| Sparse-group LASSO (混频) | 优于分析师预测 |
| Panel Kmeans + 选择性推断 | 截断 χ² + 逐对聚类比较 |

## 鲁棒性最佳实践
1. OOS 测试是强制的 — 仅 IS 拟合不足
2. 预测组合和包容检验提升鲁棒性
3. 时间一致性检查 (可预测性可能集中)
4. 聚类感知推断 (截面异质性)
5. 凸优化保证可复现性
