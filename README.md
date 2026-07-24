# Stoke-ML · 深度学习股票预测系统

A股（沪深300 + 中证500）三阶段深度学习预测系统：反封锁爬虫 → 数据管道 → 特征工程 → 模型训练 → 评估回测。

> A-share (CSI 300 + CSI 500) deep learning stock prediction: anti-block crawler → data pipeline → feature engineering → model training → backtesting.

---

## 项目结构

```
data/a_shares/
├── daily/{year}/{month}/{stock}.parquet         日K线 (OHLCV, 798 stocks)
├── news_raw/{stock}.parquet                     原始新闻 + 情感分 (63,448 articles)
├── news_silver/{stock}.parquet                  PIT时点对齐后新闻
├── news_sentiment/{year}/{month}/{stock}.parquet 日聚合新闻情感
├── guba_raw/{stock}.parquet                     股吧原始帖 (已弃用, body覆盖率14.3%)
├── guba_sentiment/{year}/{month}/{stock}.parquet 股吧日聚合情感
├── comment/{year}/{month}/{stock}.parquet       AKShare市场评论分
├── dragon_tiger/{year}/{month}/{stock}.parquet  龙虎榜数据
├── margin/{year}/{month}/{stock}.parquet        融资融券数据
├── northbound/{year}/{month}/{stock}.parquet    北向资金数据
├── fundamentals/{year}/{quarter}/{stock}.parquet 季度基本面 (ROE/PE/PB/EPS等)
├── etf_flow/{year}/{month}/sector_{name}.parquet 行业ETF资金流
├── capital_flow/{year}/{month}/{stock}.parquet  资金流向 (主力净流入, 新浪财经)
├── block_trade/{year}/{month}/{stock}.parquet   大宗交易记录
├── shareholder/{year}/{month}/{stock}.parquet   股东户数变化
├── lockup/{year}/{month}/{stock}.parquet        限售解禁日历
├── lockup_upcoming/{year}/{month}/{stock}.parquet 即将解禁预告
├── dividend/{year}/{month}/{stock}.parquet      分红送转历史
├── limit_up_zt/{year}/{month}/{stock}.parquet   涨停池 (每日)
├── limit_up_zb/{year}/{month}/{stock}.parquet   炸板池 (每日)
├── limit_up_dt/{year}/{month}/{stock}.parquet   跌停池 (每日)
├── limit_up_yzt/{year}/{month}/{stock}.parquet  昨日涨停表现池
├── limit_up_sentiment/{year}/{month}/{stock}.parquet 打板情绪汇总
├── industry_ranking/{year}/{month}/{stock}.parquet 行业排名 (申万一级)
└── concept_blocks/{year}/{month}/{stock}.parquet 概念板块成员

stoke_ml/
├── crawler/          6层反封锁爬虫 (TLS/指纹/代理池/限速/熔断/Playwright降级)
├── data/             数据源、存储引擎、故障切换、交易日历、Medallion管道
├── features/         技术指标、趋势评分、时序特征、新闻NLP、FeaturePipeline
├── preprocessing/    模块化预处理系统 (4种数据形态 × 独立预处理链)
│   ├── text/         QualityFilter → Bipolar → TimeDecay → TopicModeler → DailyAggregator
│   ├── numeric/      Outlier(MAD) → Missing(Kalman) → CrossSection → RobustScaler → HigherOrder
│   ├── daily_continuous/  FlowDecomposer (主力资金流分解)
│   ├── event_sparse/      EventToDaily (稀疏事件→日频聚合)
│   ├── cross_sectional/   BoardBroadcaster + SectorBroadcaster (打板/行业广播)
│   ├── categorical/       ConceptBlockEncoder (概念板块6层编码)
│   └── monitor/           CoverageMonitor + DriftMonitor (KS-test漂移检测)
├── models/           TFT Panel联合训练、XGBoost基线、LSTM、Transformer、SimpleAttention
├── evaluation/       Walk-Forward拆分、MCC、夏普、金融指标
└── config.py         YAML配置加载 (OmegaConf)
```

## 环境配置

```bash
pip install -r requirements.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 数据管道

**Always use venv Python:**
```bash
PYTHONPATH=. ./.venv/Scripts/python <script>
# NEVER use bare `python` — resolves to Anaconda which lacks dependencies.
```

### K线数据 (798只股票, 2015–2026)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_data.py
```

### 新闻 + 情感 (3-source aggregation)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_news.py --source all --max-pages 5
```

### 论坛情感 (Guba)

```bash
# 股吧 (词典情感 fallback — body被WAF拦截)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --max-pages 10
```

### 市场数据 (龙虎榜/融资融券/北向资金/ETF资金流)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_market_data.py --type all
```

### 基本面数据 (季度财报)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_fundamentals.py
```

### 数据中心数据 (资金流向/大宗交易/股东户数/解禁/分红/打板/行业/概念)

```bash
# 下载全部12种数据类型
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type all

# 按类型下载
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type capital_flow
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type block_trade
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type limit_up     # 全部打板数据
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type industry_ranking
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type concept_blocks

# 指定日期范围 + 单只股票测试
PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type block_trade --start 2024-01-01 --stocks 600519
```

### 数据预处理 (4种形态 → 统一日频特征)

```bash
# 全部类型预处理
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type all

# 按形态预处理
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type flow
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type event --event-type block_trade
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type board
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type sector
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type concept --stocks 600519,000001
```

### 四源故障切换

| 优先级 | 数据源 | 接口 | 反封锁 |
|--------|--------|------|--------|
| 1 | 东方财富 | 直接HTTP | curl-cffi 模拟 Chrome 120 |
| 2 | AKShare | 新浪财经 | — |
| 3 | Tushare | Tushare Pro | 需token |
| 4 | Baostock | 宝信证券 | 免费免认证 |

## 特征维度

FeaturePipeline 支持 **14个辅助维度** (全部左连接 + ZI填充 + PIT lag(1))：

| 维度 | 开关 | 列数 | 密度 | 数据源 |
|------|------|------|------|--------|
| sentiment (新闻) | `use_sentiment` | 6 | 中 | 东方财富+新浪+同花顺 |
| guba (论坛) | `use_guba` | 6 | 高(post)/低(body) | 东财股吧 |
| comment (评论) | `use_comment` | 5 | 中 | AKShare |
| announcement (公告) | `use_announcements` | 6 | 低 | AKShare |
| margin (融资融券) | `use_margin` | 4 | 高 | AKShare |
| northbound (北向) | `use_northbound` | 2 | 中 | AKShare |
| dragon_tiger (龙虎榜) | `use_dragon_tiger` | 3 | 低 | AKShare |
| fundamental (基本面) | `use_fundamental` | 8 | 低(季频) | AKShare |
| ETF flow (资金流) | `use_etf_flow` | 2 | 高(行业级) | 东方财富 |
| capital flow (资金流向) | `use_capital_flow` | 6 | 高 | 新浪财经 |
| block trade (大宗交易) | `use_block_trade` | 5 | 低 | 东方财富 |
| board (打板) | `use_board` | 9 | 低 | 东方财富 |
| concept (概念板块) | `use_concept` | 100+ | 中 | 东方财富 |

**ALL config: ~600+ features × 60 seq_len = 36,000 flat dimensions (XGBoost模式).**

**TFT Panel 格式**: 221 PastKnown + 29 PastObserved + 4 Static = 254 features × 60 seq_len. 跨股票截面归一化 (per-date z-score).

## 模型训练

### TFT Panel 联合训练 (主力模型, RTX 4090)

```bash
# 500只股票 Panel 联合训练 (多任务: 方向+涨跌幅+波动率)
PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stocks 500 --horizon 5 --epochs 20 --max-folds 3

# 少量股票快速测试
PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stocks 50 --epochs 5 --max-folds 1

# 指定股票训练
PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stock-list 600519,000001,000858
```

TFT (Temporal Fusion Transformer) 配置:
- **输入**: Static(4) + PastKnown(221) + PastObserved(29) × seq_len=60
- **架构**: 2×LSTM + 4-head Multi-Head Attention + Variable Selection Network + GRN
- **多任务**: CrossEntropy(方向3类) + AdjMSE(涨跌幅) + MSE(波动率) + RankICLoss(截面排序)
- **损失加权**: UncertaintyLoss (Kendall et al. 2018) 自适应任务权重
- **验证**: Purged Walk-Forward (504天训练 / 63天验证 / 63天步长 / 60天purge)
- **评估指标**: 年化Sharpe + Spearman Rank IC (截面排序能力)

### XGBoost 基线

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/train_baseline.py --stock 000001
PYTHONPATH=. ./.venv/Scripts/python scripts/train_baseline.py  # 全量
```

### LSTM / Transformer / SimpleAttention (PyTorch Lightning)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/train_lstm.py --stock 000001 --epochs 50
```

### 消融实验

```bash
# XGBoost: technical-only vs +sentiment vs +guba vs ALL
PYTHONPATH=. ./.venv/Scripts/python scripts/train_baseline.py --ablation
```

## 消融结果 (95 stocks, 1000 bootstrap)

| Config | MCC | 95% CI | Δ vs technical |
|---|---|---|---|
| technical | 0.0136 | [-0.0035, 0.0312] | — |
| + sentiment | 0.0279 | [0.0095, 0.0464] | +0.0143 (+104%) |
| + guba | 0.0219 | [0.0032, 0.0384] | +0.0084 |
| + comment | 0.0224 | [0.0045, 0.0408] | +0.0089 |
| ALL | 0.0261 | [0.0104, 0.0426] | +0.0125 |

- 所有文本维度均提升MCC (所有CI > 0)
- 新闻情感效应最大 (+104% MCC)
- ALL受维度爆炸影响，不如+sentiment单独使用
- Δ CIs全部跨零 — 统计显著性需要更多数据

## 预处理系统

模块化预处理架构，按4种数据形态独立设计预处理链：

```
Shape A (daily_continuous):  FlowDecomposer → 主力/超大/大/中/小单分解
Shape B (event_sparse):      EventToDaily → 稀疏事件→日频聚合 + 衰减
Shape C (cross_sectional):   BoardBroadcaster / SectorBroadcaster → 打板/行业广播
Shape D (categorical):       ConceptBlockEncoder → 6层概念编码 (multi-hot→heat→动量→共现)

文本链 (Phase 3):   QualityFilter → BipolarClassifier → TimeDecayWeighter → TopicModeler → DailyAggregator
数值链 (Phase 4):   OutlierDetector(MAD) → MissingImputer(Kalman) → CrossSectionNormalizer → RobustScaler → HigherOrder
监控 (Phase 5):     CoverageMonitor + DriftMonitor (KS-test漂移检测)
```

### 管道对比

```bash
# 对比新旧预处理的模型表现
PYTHONPATH=. ./.venv/Scripts/python scripts/compare_pipelines.py --stock 000001
```

- 设计文档: `docs/superpowers/specs/2026-07-01-preprocessing-redesign-design.md`
- 实施计划: `docs/superpowers/plans/2026-07-01-preprocessing-redesign-plan.md`

## 评估体系

- **主要指标**：Matthews Correlation Coefficient (MCC)
- **分类指标**：准确率、精确率、召回率、F1
- **金融指标**：夏普比率、最大回撤、胜率、盈亏比
- **验证方法**：Walk-Forward滚动验证 (2年训练/3月验证)，**严格时序拆分，无shuffle**

## 配置说明

`config.yaml` 关键参数：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `features.seq_len` | 60 | 回看窗口 (LSTM) |
| `features.flat_seq_len` | 5 | 回看窗口 (XGBoost flat) |
| `features.target_horizon` | 1 | 预测次日涨跌 |
| `training.validation.train_years` | 2 | 训练窗口 |
| `training.validation.val_months` | 3 | 验证窗口 |
| `model.params.max_depth` | 6 | XGBoost树深度 |
| `preprocessing.text.time_decay.halflife_days` | 7 | 文本时间衰减半衰期 |
| `preprocessing.numeric.scaling.winsorize_sigma` | 3.0 | ±3σ截尾 |
| `preprocessing.numeric.outlier.threshold` | 5.0 | MAD异常值阈值 |
| `preprocessing.numeric.missing.medium_gap_method` | kalman | 中空档填充方法 (linear/kalman) |
| `preprocessing.cross_sectional.board.consecutive_lookback` | 20 | 连板回看天数 |
| `preprocessing.concept.top_n` | 100 | 概念板块编码数量 |

## 已知问题

| 问题 | 状态 |
|------|------|
| Guba post body不可用 (详情页SPA, WAF拦截) | 由雪球论坛源替代 |
| 雪球IP被WAF封禁 (aliyun) | Playwright可绕过, API仍受保护 |
| ALL config维度爆炸 (~36,000维) | 用+sentiment替代ALL |
| FinBERT首次加载需网络或预缓存模型 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| 消融Δ CIs跨零 (需>100 stocks或更强信号) | TFT panel联合训练解决 |
| EastMoney push2his 资金流向API下线 (2026-07) | 已切换至新浪财经 (仅总净额, 无分层) |
| BERTopic 依赖链 (umap/hdbscan/sentence-transformers) | 可选安装, 缺失时降级为TF-IDF |
| Playwright browser 在WAF绕过时可能无限挂起 | `threading.Timer(timeout, os._exit)` 硬杀 |
| TFT UncertaintyLoss 收敛后 train loss 变负 | 正常 (log_var → ln(task_loss)), 已收紧clamp |
| TFT multi-horizon Sharpe 年化虚高 | 已修复: stride=horizon去重叠 + sqrt(252/horizon)年化 |

## 设计文档

- [股票预测系统设计](docs/superpowers/specs/2026-06-19-stock-prediction-design.md)
- [TFT Panel 联合训练设计](docs/superpowers/specs/2026-07-15-tft-panel-training-design.md)
- [TFT Panel 实施计划](docs/superpowers/plans/2026-07-15-tft-panel-training-plan.md)
- [预处理系统重构设计](docs/superpowers/specs/2026-07-01-preprocessing-redesign-design.md)
- [预处理重构实施计划](docs/superpowers/plans/2026-07-01-preprocessing-redesign-plan.md)
- [新闻管道最佳实践研究](docs/research/news-pipeline-best-practices.md)
