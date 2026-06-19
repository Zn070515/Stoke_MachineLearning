# Stoke-ML · 深度学习股票预测系统

A股（沪深300 + 中证500）三阶段深度学习预测系统：反封锁爬虫 → 数据管道 → 特征工程 → 模型训练 → 评估回测。

> A-share (CSI 300 + CSI 500) deep learning stock prediction: anti-block crawler → data pipeline → feature engineering → model training → backtesting.

---

## 项目结构 · Project Structure

```
data/a_shares/
├── daily/{year}/{month}/{stock}.parquet     日K线 (OHLCV)
├── news_raw/{stock}.parquet                 原始新闻 + 情感分
├── news_silver/{stock}.parquet              PIT时点对齐后新闻
└── sentiment/{year}/{month}/{stock}.parquet 日聚合情感特征

stoke_ml/
├── crawler/         6层反封锁爬虫 (TLS伪装/浏览器指纹/代理池/限速/熔断)
├── data/            数据源、存储引擎、故障切换、交易日历、新闻管道
├── features/        技术指标、时序特征、趋势评分、新闻NLP
├── models/          XGBoost基线、LSTM (PyTorch Lightning)
├── evaluation/      Walk-Forward拆分、MCC、夏普、金融指标
└── config.py        YAML配置加载 (Hydra/OmegaConf)
```

## 环境配置 · Setup

```bash
pip install -r requirements.txt
# 如有GPU，安装CUDA版PyTorch：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 数据管道 · Data Pipeline

### 下载日K线数据（798只股票，2015–2026年）

```bash
PYTHONPATH=. python scripts/download_data.py
# 约650MB Parquet，按年/月/股票代码分区存储
```

### 下载新闻 + 情感分析（三层存储架构）

```bash
PYTHONPATH=. python scripts/download_news.py --max-pages 5 --sleep 2
# 原始层(Bronze) → 对齐层(Silver) → 聚合层(Gold)
# L1: SnowNLP (离线中文NLP)，L2: FinBERT (计划中)
```

### 四源故障切换

| 优先级 | 数据源 | 接口 | 反封锁 |
|--------|--------|------|--------|
| 1 | 东方财富 | 直接HTTP | curl-cffi 模拟 Chrome 120 |
| 2 | AKShare | 新浪财经 | — |
| 3 | Tushare | Tushare Pro | 需token |
| 4 | Baostock | 宝信证券 | 免费免认证 |

## 模型训练 · Training

### XGBoost 基线（展平特征）

```bash
# 单只股票
PYTHONPATH=. python scripts/train_baseline.py --stock 000001

# 全部798只股票
PYTHONPATH=. python scripts/train_baseline.py
```

### LSTM（序列模型）

```bash
PYTHONPATH=. python scripts/train_lstm.py --stock 000001 --epochs 50
```

### 特征集

| 类别 | 示例 |
|------|------|
| 技术指标 | 均线、RSI、MACD、布林带、ATR、OBV |
| 时序特征 | 滞后项(1/2/3/5/10/20日)、滚动统计(5/10/20/60日) |
| 日历特征 | 星期几、月份、季度 |
| 情感特征 | 日均情感/标准差、正面/负面占比、是否有新闻 |

### 评估体系

- **主要指标**：Matthews Correlation Coefficient (MCC)
- **分类指标**：准确率、精确率、召回率、F1
- **金融指标**：夏普比率、最大回撤、胜率、盈亏比
- **验证方法**：Walk-Forward滚动验证（2年训练/3月验证），严格时序拆分

## 配置说明 · Configuration

`config.yaml` 关键参数：

| 配置项 | 参数 | 默认值 | 说明 |
|--------|------|--------|------|
| `markets.a_shares` | `stock_universe` | `[csi300, csi500]` | 指数成分股 |
| `features` | `seq_len` | `60` | 回看窗口（交易日） |
| `features` | `use_sentiment` | `true` | 是否使用新闻情感特征 |
| `features` | `target_horizon` | `1` | 预测次日涨跌方向 |
| `training.validation` | `train_years` | `2` | 训练窗口年数 |
| `training.validation` | `val_months` | `3` | 验证窗口月数 |
| `model.params` | `max_depth` | `6` | XGBoost树深度 |

## 当前进度 · Current Status

| 模块 | 状态 |
|------|------|
| 反封锁爬虫 | ✅ 6层架构，冒烟测试通过 |
| 日K线数据 | ✅ 798只股票，2015–2026，9.2万文件 |
| 新闻管道 | 🔄 下载中 325/798 (41%)，预计22分钟 |
| XGBoost基线 | ✅ 已训练，初始MCC ~0.05–0.15 |
| LSTM模型 | ✅ 就绪，待全量跑 |
| FinBERT L2情感 | 📋 计划中 |
| 回测引擎 | 📋 计划中 |

## 设计文档 · Design Docs

- [设计规格书](docs/superpowers/specs/2026-06-19-stock-prediction-design.md)
- [实施计划](docs/superpowers/plans/2026-06-19-stock-prediction-plan.md)
- [新闻管道最佳实践研究](docs/research/news-pipeline-best-practices.md)
