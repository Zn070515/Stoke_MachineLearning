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

---

## 5. 原始数据质量评分 (2026-07-16)

对照三大因子体系（Alpha158 / WorldQuant 101 / 聚宽 260因子）+ 头部开源项目（ml-quant-trading、QuantsPlaybook、FactorHub）评分。

### 评分标准

按覆盖完整度、时间跨度、数据密度、更新时效四个维度，1-10 分制：
- 9-10 = 匹配/超越专业量化平台
- 7-8 = 扎实，有小缺口
- 5-6 = 可用但有明显缺口
- 3-4 = 存在但薄
- 1-2 = 几乎不存在
- 0 = 缺失

### 一、量价/技术面（权重 25%）— 6.3/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| K线 OHLCV | 798只 × 11年 × 日频 | ≥500只 × 10年 | **9** |
| 技术指标 | **186个**（Alpha158全覆盖 + ADX/MFI/CMO/TRIX/KDJ/ROC/WR/CCI/Boll/Vol） | 158-260个（Alpha158多窗口展开） | **8** |
| 微观结构 | 涨跌停标记、缺口、量异常、连板计数 | 盘口深度、逐笔数据 | **5** |

**✅ Alpha158 多窗口因子展开已完成（186个因子，5窗口×21类型全覆盖）**

### 二、基本面/估值（权重 20%）— 3.2/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| 季报财务 | 8个指标，45行/只（季度），roe/eps/毛利率/负债率等 | 聚宽71个质量因子，160-500衍生指标 | **4** |
| 估值比率 | PE/PB/PS/PCF 日频，下载中 | 聚宽30个风格因子，PEG/EV-EBITDA等 | **3** |
| 分红送转 | 20条/只，回溯到2002年 | 股息率、分红稳定性 | **6** |
| 股东户数 | 1个快照（2026-03-31） | 季度序列 | **2** |
| 限售解禁 | 2018年断更 | 完整未来解禁日历 | **1** |

**最大短板：只有8个财务指标 vs 行业160-500个**

### 三、情绪/另类数据（权重 20%）— 4.6/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| 新闻情绪 | 798只 × 6个月，488条/只 | API限制，开源项目普遍0-1个文本源 | **4** |
| 股吧情绪 | 802只 × 3年，日均533条帖子（000001为例） | 多数项目无此维度 | **8** |
| 雪球情绪 | 245只 × 1个月 | — | **2** |
| 评论评分 | 785只 × 6周，单分数 | — | **2** |
| 公告情绪 | 798只 × 10年，1000+条/只 | — | **7** |

**来源最丰富（5个文本源），是自己最强的维度，但时间覆盖参差不齐**

### 四、资金流向（权重 15%）— 6.6/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| 个股资金流 | 798只 × 11年，主力/超大单/大单/中单/小单 | 东财标准口径 | **9** |
| 融资融券 | 4547标的 × 11年 | on par | **9** |
| 北向资金 | 772只 × 7年，停在2024年8月 | 需更新 | **5** |
| 龙虎榜 | 6278条事件 × 9年，停在2024年2月 | 需更新 | **6** |
| 大宗交易 | 790只，稀疏（46条/只） | — | **4** |

### 五、打板/涨停数据（权重 10%）— 0.4/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| 涨停池 ZT | 下载中（~25%） | 市场宽度、连板高度分布 | **2** |
| 炸板池 ZB | 0 | — | **0** |
| 跌停池 DT | 0 | — | **0** |
| 昨日涨停表现 YZT | 0 | — | **0** |
| 市场情绪统计 | 0 | — | **0** |

### 六、行业/板块（权重 5%）— 4.5/10

| 指标 | 我们 | 行业标准 | 得分 |
|------|------|----------|------|
| 行业分类+收益 | 90个申万行业 × 11年日频 | done | **8** |
| 概念板块 | 仅2天数据 | 需全量历史 | **2** |
| 行业中性化 | 未实现 | 所有专业管线标配 | **0** |

### 七、宏观（权重 5%）— 7.0/10

28个指标（SHIBOR 8期限 + 外汇5币种 + 中美利差 + GDP/M2/CPI），11年日频。够用。

### 综合加权得分

```
量价/技术  7.3 × 0.25 = 1.83  (↑ Alpha158全覆盖)
基本面    3.4 × 0.20 = 0.68  ← 最拖后腿 (估值下载中)
情绪/另类  4.6 × 0.20 = 0.92
资金流向  6.6 × 0.15 = 0.99
打板      0.4 × 0.10 = 0.04  ← 第二短板 (下载中)
行业      4.5 × 0.05 = 0.23
宏观      7.0 × 0.05 = 0.35
─────────────────────────
总分                   5.04 / 10
```

### 诊断

> 技术面扎实（186因子Alpha158全覆盖）、资金流完整、情绪来源丰富（但时间短）、基本面偏薄、打板补齐中。综合 5.04/10（+0.29）。

### 按 ROI 排序的改进路线

| 优先级 | 方向 | 预期提升 | 难度 | 状态 |
|--------|------|----------|------|------|
| ~~P0~~ | ~~Alpha158 多窗口因子展开~~ | ~~+1.0分~~ | 中 | **✅ 已完成（186因子）** |
| **P0** | 打板数据补齐（ZT/ZB/DT/YZT 正在下载） | +0.5分 | 低 | 🔄 下载中 |
| **P0** | 估值比率补齐（PE/PB/PS/PCF） | +0.3分 | 低 | 🔄 下载中 (218/798) |
| P1 | 行业中性化（申万分类已有，加截面标准化） | +0.3分 | 低 | 待实现 |
| P1 | 基本面因子扩展（财务衍生指标 8→50+） | +0.6分 | 高 | 待实现 |
| P1 | 新闻情绪历史数据（需外部数据源） | +0.4分 | 高 | 待调研 |
| P2 | 北向资金更新 + 股东户数补全 | +0.4分 | 中 | 待实现 |
| P3 | 雪球覆盖扩展（245→798只，需Playwright） | +0.2分 | 高 | 待实现 |

**核心结论：Alpha158 已到位。当前瓶颈从"技术因子不够"转向"基本面偏薄 + 打板数据缺 + 行业中性化"。**

### 对标参考

| 项目 | 因子数 | 特色 |
|------|--------|------|
| Alpha158 (Qlib) | 158 | 6个OHLCV字段 × 5个窗口，8个子类 |
| WorldQuant 101 | 101 | 纯量价公式化因子，rank/delay/correlation算子 |
| 聚宽因子库 | 260 | 10大类：质量71、情绪36、动量34、风格30等 |
| ml-quant-trading | 213 | GPU加速，A股涨跌停偏差修正 + GBM增强 |
| QuantsPlaybook | 100+ | 行为金融、筹码分布、网络中心度等特色因子 |
| **Stoke_ML** | **186** | 5文本源情绪（独特优势）、资金流完整、Alpha158全覆盖 |
