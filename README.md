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
├── xueqiu_raw/{stock}.parquet                   雪球论坛原始帖 (245 stocks, 66,044 posts)
├── xueqiu_sentiment/{year}/{month}/{stock}.parquet 雪球日聚合情感
├── comment/{year}/{month}/{stock}.parquet       AKShare市场评论分
├── dragon_tiger/{year}/{month}/{stock}.parquet  龙虎榜数据
├── margin/{year}/{month}/{stock}.parquet        融资融券数据
├── northbound/{year}/{month}/{stock}.parquet    北向资金数据
├── fundamentals/{year}/{quarter}/{stock}.parquet 季度基本面 (ROE/PE/PB/EPS等)
└── etf_flow/{year}/{month}/sector_{name}.parquet 行业ETF资金流

stoke_ml/
├── crawler/          6层反封锁爬虫 (TLS/指纹/代理池/限速/熔断/Playwright降级)
├── data/             数据源、存储引擎、故障切换、交易日历、Medallion管道
├── features/         技术指标、趋势评分、时序特征、新闻NLP、FeaturePipeline
├── preprocessing/    [NEW] 模块化预处理系统 (text/numeric/monitor/registry)
├── models/           XGBoost基线、LSTM、Transformer、SimpleAttention
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

### 论坛情感 (雪球/Guba)

```bash
# 雪球论坛 (Playwright绕过WAF, Guba替代方案)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --max-pages 20

# 股吧 (已弃用 — body被WAF拦截, 仅保留历史数据)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --max-pages 10
```

### 市场数据 (龙虎榜/融资融券/北向资金)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_market_data.py --type all
```

### 基本面数据 (季度财报)

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/download_fundamentals.py
```

### 四源故障切换

| 优先级 | 数据源 | 接口 | 反封锁 |
|--------|--------|------|--------|
| 1 | 东方财富 | 直接HTTP | curl-cffi 模拟 Chrome 120 |
| 2 | AKShare | 新浪财经 | — |
| 3 | Tushare | Tushare Pro | 需token |
| 4 | Baostock | 宝信证券 | 免费免认证 |

## 特征维度

FeaturePipeline 支持 **10个辅助维度** (全部左连接 + ZI填充 + PIT lag(1))：

| 维度 | 开关 | 列数 | 密度 | 数据源 |
|------|------|------|------|--------|
| sentiment (新闻) | `use_sentiment` | 6 | 中 | 东方财富+新浪+同花顺 |
| guba (论坛) | `use_guba` | 6 | 高(post)/低(body) | 东财股吧 |
| comment (评论) | `use_comment` | 5 | 中 | AKShare |
| xueqiu (论坛) | `use_xueqiu` | 6 | 中 | 雪球 (Playwright) |
| announcement (公告) | `use_announcements` | 6 | 低 | AKShare |
| margin (融资融券) | `use_margin` | 4 | 高 | AKShare |
| northbound (北向) | `use_northbound` | 2 | 中 | AKShare |
| dragon_tiger (龙虎榜) | `use_dragon_tiger` | 3 | 低 | AKShare |
| fundamental (基本面) | `use_fundamental` | 8 | 低(季频) | AKShare |
| ETF flow (资金流) | `use_etf_flow` | 2 | 高(行业级) | 东方财富 |

**ALL config: ~405 features × 60 seq_len = 24,300 flat dimensions.**

## 模型训练

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

## 预处理系统 (NEW — 开发中)

替换 ZI填零+简单均值 为模块化预处理链：

```
文本链: Quality → Bipolar → TimeDecay → Topic(BERTopic) → Aggregation
数值链: Outlier(MAD) → Missing(Kalman/线性) → CrossSection(行业/市值/自适应) → RobustScaler → HigherOrder
监控:  输入层 → 变换层 → 输出层(KS-test漂移检测)
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
| `preprocessing.text.time_decay.halflife_days` | 7 | 时间衰减半衰期 |
| `preprocessing.numeric.scaling.winsorize_sigma` | 3.0 | ±3σ截尾 |

## 已知问题

| 问题 | 状态 |
|------|------|
| Guba post body不可用 (详情页SPA, WAF拦截) | 由雪球论坛源替代 |
| 雪球IP被WAF封禁 (aliyun) | Playwright可绕过, API仍受保护 |
| ALL config维度爆炸 (24,300维) | 用+sentiment替代ALL |
| FinBERT首次加载需网络或预缓存模型 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| 消融Δ CIs跨零 (需>100 stocks或更强信号) | 活跃研究中 |

## 设计文档

- [股票预测系统设计](docs/superpowers/specs/2026-06-19-stock-prediction-design.md)
- [预处理系统重构设计](docs/superpowers/specs/2026-07-01-preprocessing-redesign-design.md)
- [预处理重构实施计划](docs/superpowers/plans/2026-07-01-preprocessing-redesign-plan.md)
- [新闻管道最佳实践研究](docs/research/news-pipeline-best-practices.md)
