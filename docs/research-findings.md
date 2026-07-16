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

---

## 6. 新数据预处理最佳实践调研 (2026-07-16)

对照 4 大搜索方向（涨停板特征工程、资金流预处理、事件因子、行业/概念轮动）交叉验证现有 `stoke_ml/preprocessing/` 下的 5 个模块。

### 6.1 资金流向 (FlowDecomposer)

**现状：** 6 层分解 — 规模分层比率 / OFI 强度（Z-score+EMA）/ 持续性 / 量价背离 / 残差化（截面回归）/ 大小单差。

| 来源 | 核心发现 |
|------|---------|
| [Kang (2025)](https://ideas.repec.org/p/arx/papers/2601.07131.html) 韩国 2439 只实证 | **市值标准化资金流的简单线性模型 Sharpe=1.30，完整 ICA-Wavelet-LSTM 仅 0.07** |
| [开源证券 (2024)](http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/789007305546/index.phtml) | 资金流天然暴露反转效应，需截面回归剥离；合并大中小单为"广义主力资金"提升 Alpha |
| [国盛证券 (2025)](https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/11/rptid/792664341934/index.phtml) | 10 万因子→50 精选，等权 Top10 月频 RankIC=0.110，ICIR=4.24 |

| 对齐/缺失 | 详情 |
|-----------|------|
| ✅ | L1-L6 覆盖核心方法论，L5 残差化已实现截面回归 |
| 🔴 P0 | **市值标准化缺失** — `main_net / market_cap`，不同市值不可直接比较 |
| 🔴 P1 | **广义主力资金** — 研究建议合并 large+mid 为"broad_main" |
| 🟡 P2 | 资金流相似图（CashCoIn/CashCoOut）、溢出因子 |

### 6.2 大宗交易 (EventToDaily block_trade)

**现状：** trade_count / premium_pct_mean / total_amount / buyer_is_inst / VWAP 溢价 / impact / 6d 波动率。

| 来源 | 核心发现 |
|------|---------|
| [光大证券](https://stock.finance.sina.com.cn/stock/view/paper.php?symbol=sh000001&reportid=744566192026) | **成交金额比率（大宗成交额/当日总成交额）是最强 Alpha**，"高比率+低波动"组合年化 19.54%，超额 16.41% |

| 对齐/缺失 | 详情 |
|-----------|------|
| ✅ | VWAP 溢价、6d 金额波动率、折溢价率 |
| 🔴 P0 | **成交金额比率缺失** — 最强单因子完全没实现 |
| 🔴 P1 | 席位分类仅检查"机构"字符串匹配，研究建议三分法（机构/游资/普通）+ 卖方分析 |
| 🟡 P2 | 价格分位数、折价后趋势跟踪 |

### 6.3 股东户数 (EventToDaily shareholder)

**现状：** HN_z（8Q Z-score）/ consecutive_quarter_decline / PCRC（YoY）/ dual_concentration_signal。

✅ 核心因子覆盖良好（HN_z + PCRC 是标准方法），无需 P0/P1 改动。P2 可选：前十大持股占比、户数变化加速度。

### 6.4 限售解禁 (EventToDaily lockup)

**现状：** unlock_pressure（指数衰减）/ total_upcoming_ratio / is_vc_backed / 30d 解禁后收益。

| 对齐/缺失 | 详情 |
|-----------|------|
| 🔴 P1 | **市值标准化缺失** — `free_ratio` 绝对值未除市值 |
| 🟡 P1 | 解禁类型仅 IPO/非IPO 二分，研究建议首发/定增/股权激励/其他 |
| 🟡 P2 | 解禁前累计超额收益（CAR） |

### 6.5 分红 (EventToDaily dividend)

✅ 股息率 / 衰减有效收益率 / 预案阶段编码 / 分红增长率全覆盖。无 gap。

### 6.6 涨停板 (BoardBroadcaster)

**现状：** 5 层 — 池成员 / 连板跟踪 / 封板质量（类型×时间×回封次数）/ 情绪广播 / 市场状态分类。

| 来源 | 核心发现 |
|------|---------|
| [同花顺量化](https://quant.10jqka.com.cn/view/article/8F9RSG6HNQ1582620HY53IE988) | 封单强度=封单金额/流通市值是封板质量最重要指标；封板时间早晚盘分类 |
| [ml-quant-trading](https://github.com/initial-d/ml-quant-trading) | 内置涨跌停偏差校正模块 `features.bias`，213 因子体系 |

| 对齐/缺失 | 详情 |
|-----------|------|
| ✅ | L1-L5 覆盖同花顺框架，seal_strength（类型×时间×回封）设计良好 |
| 🔴 P0 | **封单强度缺失** — `seal_amount / market_cap` 是最重要的封板质量指标 |
| 🔴 P1 | **同概念联动缺失** — 概念内涨停家数/连板率，判断题材强度的关键 |
| 🟡 P2 | 市场状态阈值（80/20/120）是 magic number，需自适应分位数 |

### 6.7 行业板块 (SectorBroadcaster)

**现状：** 5 层 — sector join / sector features / 多窗口动量 / RRG 框架 / 轮动信号。

| 来源 | 核心发现 |
|------|---------|
| [华泰证券 (2024)](https://m.sohu.com/a/997408608_122014422/) | **残差动量**（剥离市场+风格因子）年化超额 12.90%；**拥挤度预警** 4 指标>95%分位成功预警 2026 年风险 |
| [东方证券 DFQ](http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/785320315165/index.phtml) | GP 多目标优化 Top5 行业年化超额 18.42%，夏普 1.77 |

| 对齐/缺失 | 详情 |
|-----------|------|
| ✅ | RRG 框架（RS-Ratio × RS-Momentum）先进，多窗口动量完整 |
| 🔴 P1 | **拥挤度缺失** — 成交额波动率/成交量波动率/行业换手率 Z-score，2024 年最重要研究方向 |
| 🔴 P1 | **残差动量缺失** — 需剥离市场 Beta + 风格因子 |
| 🟡 P2 | 风格中性化在行业因子合成前未做 |

### 6.8 概念板块 (ConceptBlockEncoder)

**现状：** 6 层 — 词汇表 / multi-hot / 衍生 / 热度 / 动量 / 联动。覆盖面好。

🟡 P2：龙头→跟风时间差特征；100 维 multi-hot 可选 PCA 降维。

---

### 跨模块 Gap：市值标准化

资金流向、大宗交易、限售解禁、封板强度 **4 个模块**都缺市值标准化。方案：各预处理步骤加可选 `market_cap` 参数，从 FeaturePipeline 的 K-line 数据计算流通市值后传入。

### 行动清单

#### P0 — 立即修复

| # | 模块 | 改动 | 依据 |
|---|------|------|------|
| 1 | FlowDecomposer | 加 `flow_market_cap_adj = main_net / market_cap` | Kang (2025) 实证 Sharpe=1.30 |
| 2 | EventToDaily block_trade | 加 `amount_ratio = block_amount / daily_volume` | 光大 年化超额 16.41% |
| 3 | BoardBroadcaster | 加 `seal_intensity = seal_amount / market_cap` | 同花顺框架核心 |

#### P1 — 重要增强

| # | 模块 | 改动 |
|---|------|------|
| 4 | FlowDecomposer | `broad_main_net = super + large + mid` |
| 5 | EventToDaily block_trade | 席位三分法 + 卖方分析 |
| 6 | EventToDaily lockup | 解禁市值/流通市值标准化 |
| 7 | SectorBroadcaster | 拥挤度（成交额波动率/换手率 Z-score） |
| 8 | SectorBroadcaster | 残差动量（剥离 Beta） |
| 9 | BoardBroadcaster | 同概念涨停统计 |

#### P2 — 锦上添花

| # | 模块 | 改动 |
|---|------|------|
| 10 | FlowDecomposer | 资金流相似图/溢出因子 |
| 11 | EventToDaily shareholder | 前十大持股占比、户数加速度 |
| 12 | EventToDaily lockup | 解禁前 CAR |
| 13 | ConceptBlockEncoder | 龙头→跟风滞后、PCA 降维 |
| 14 | BoardBroadcaster | 市场状态阈值自适应 |

### 6.9 GitHub 开源项目参考 (gh CLI 调研)

通过 `gh search repos` + `gh api` 调研了头部 A 股量化开源项目的预处理实践：

| 项目 | Stars | 与我们相关的模块 |
|------|-------|-----------------|
| [ml-quant-trading](https://github.com/initial-d/ml-quant-trading) | — | `bias.py` 涨跌停偏差校正 + `neutralize.py` 截面/行业中性化 |
| [qlib-factor-zoo](https://github.com/JustinF8/qlib-factor-zoo) | 6 | 1000+ 因子库（Alpha360/158/101 + GTJA191 + TDXGS + JQ110） |
| [FactorMiner](https://github.com/CharlesJ-ABu/FactorMiner) | 126 | 四范式因子挖掘（LLM+GP+RL+DL），DiversityFilter 去重 |
| [QuantMind](https://github.com/qusong0627/QuantMind) | 441 | 基于 Qlib 的量化平台，146 维因子训练 |
| [AlphaGPT_Tushare](https://github.com/LilianaMajamay/AlphaGPT_Tushare) | 157 | DeepSeek AI + Tushare 因子生成与回测 |

---

#### ml-quant-trading: 涨跌停偏差校正

该项目的 `bias.py` 模块是最直接可借鉴的实现：

```python
# 两种 regime：
# 1. 真实涨跌停价格可用 → close >= limit_up - eps 视为不可交易
# 2. 无涨跌停数据 → |return| > 9.8% 代理判断
# 两种路径统一返回 bool mask，训练时排除被 mask 的日期
```

**与我们的关系：** 我们的 BoardBroadcaster 已经标记了 `is_zt/zb/dt`，但**没有在训练标签中使用这些标记做偏差校正**。TFT 训练时应该排除涨跌停日的标签（这些日期的收益不可交易）。

---

#### ml-quant-trading: 截面中性化

```python
# neutralize_cs: 每日期截面 Z-score，mask-aware
# neutralize_industry: 每日对行业 one-hot + log_market_cap 做 OLS 残差化
# 残差再做截面 Z-score 标准化
```

**与我们的关系：** 我们的 `CrossSectionNormalizer` 实现了截面 Z-score，但**缺少行业中性化（OLS 回归残差法）**。华泰和东方证券的研报反复强调行业+市值中性化是因子有效的前提。

---

#### qlib-factor-zoo: 因子库参考

六大因子库总计 1000+ 因子，其中与我们最相关的：
- **Alpha158** — 我们已实现 186 个因子（对齐）
- **GTJA191** — 国泰君安 191 短周期 Alpha，我们的缺项
- **JQ110** — 聚宽 110 因子（动量/情绪/技术/风险/风格），部分缺失

**与我们的关系：** Alpha158 已覆盖。GTJA191 和 JQ110 可作为 P2 扩展参考。

---

#### FactorMiner: 工程参考

- **DiversityFilter**: MD5 哈希去重，避免同质化因子浪费计算——我们的因子库增长后也会面临这个问题
- **FactorMetadata**: 因子"灵魂与肉体分离"——公式存为 AST/脚本，时序数据存 Parquet——这是好的工程实践
- **Config-Driven + IoC**: 声明式配置替代硬编码——我们 config.yaml 的 preprocessing 段已经是这个方向

---

#### 从 GitHub 项目新增的 Gap

| # | 来源 | 缺失项 | 影响 |
|---|------|--------|------|
| 15 | ml-quant-trading `bias.py` | **训练时涨跌停偏差校正** — 排除涨跌停日的训练标签 | 防止模型学到不可交易的假 Alpha |
| 16 | ml-quant-trading `neutralize.py` | **OLS 行业中性化**（行业 one-hot + log_market_cap 回归残差） | 当前只有截面 Z-score，缺行业+市值正交 |
| 17 | qlib-factor-zoo | **GTJA191** (191 短周期 Alpha) + **JQ110** (动量/情绪/技术/风险/风格) | P2 扩展可选 |

---

> **参考资料：** [ml-quant-trading](https://github.com/initial-d/ml-quant-trading) · [qlib-factor-zoo](https://github.com/JustinF8/qlib-factor-zoo) · [FactorMiner](https://github.com/CharlesJ-ABu/FactorMiner) · [QuantMind](https://github.com/qusong0627/QuantMind) · [同花顺量化](https://quant.10jqka.com.cn/view/article/8F9RSG6HNQ1582620HY53IE988) · [华泰金工](https://m.sohu.com/a/997408608_122014422/) · [东方 DFQ](http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/785320315165/index.phtml) · [光大 大宗交易](https://stock.finance.sina.com.cn/stock/view/paper.php?symbol=sh000001&reportid=744566192026)
