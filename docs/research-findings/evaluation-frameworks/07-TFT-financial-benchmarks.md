# TFT 金融时序评估基准

> 来源: IEEE 2025, CFE-CMStatistics 2025, AppliedMath 2025, GitHub aman-720/sp500-tft-forecasting

## TFT vs 经典模型

| 研究 | 对比模型 | TFT 表现 |
|------|---------|---------|
| Maheshwari et al. (2025) | ARIMA, LSTM, Autoformer, Informer, PatchTST, NHITS | **TFT 最低 RMSE 1.41%** (20 NSE 股票, 14年) |
| Fleury (2025) | N-BEATS | TFT 持续优于 N-BEATS，尤其在波动期 |
| TFT-GNN Hybrid (2025) | SARIMA, ETS, TFT | TFT-GNN 混合最优 |

## TFT vs DL 模型

| 研究 | 发现 |
|------|------|
| IEEE AFRICON 2025 | GRU-Optuna 最优; TFT-Nevergrad 最差 — 超参优化方法显著影响 TFT |
| IEEE Bandung 2025 | TCN 最低 MAPE 4.51%; TFT 表现依赖上下文 |
| Garuda (Solana) | TFT 跨体制劣化 218% vs LSTM 1576%; **TFT 计算效率高 62%** |

## 关键发现与警示

### 1. 梯度坍缩 (Gradient Collapse)
S&P 500 基准 (450+ 实验):
- TFT 输出层梯度在 **5 epochs 内坍缩 83%**
- 同时编码器梯度增加 279%
- 注意力学到了有意义的表示，但输出层无法翻译成稳定的方向预测

### 2. Fixed-Split vs Rolling 评估
- Fixed-split 收益 (+1.6-2.1% 超额方向准确率) **在滚动评估中不泛化**
- 滚动评估中 TFT 超额准确率接近零

### 3. 周频优势
- 周频方向准确率最高 (59.1±8.3%)
- 但这主要反映周线数据的正向偏差，而非真正的预测能力

### 4. 2022 压力测试
- **所有模型在 2022 年美联储紧缩期普遍失败** (~40% 准确率)
- 无架构或配置例外

## 对 VSN+xLSTM Panel 模型的启示
- VSN+xLSTM 替代 TFT 避免了 TFT 的梯度坍缩问题
- 但仍需警惕 Rolling vs Fixed-Split 评估差异
- xLSTM 的指数门控 vs TFT 的 LSTM 编码器是根本区别
