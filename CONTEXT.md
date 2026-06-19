# CONTEXT.md

领域术语表与命名约定。本文件服务于：
- 工程师在讨论设计/代码时使用统一语言
- Agent 在写 issue、PR、commit message 时使用正确术语
- `/grill-with-docs` 的词汇锚点

---

## 市场与数据

| 术语 | 英文 | 含义 |
|------|------|------|
| A股 | A-shares | 中国沪深交易所上市股票 |
| 沪深300 | CSI 300 | 沪市+深市市值最大300只，指数代码 000300 |
| 中证500 | CSI 500 | 除沪深300外市值最大500只，指数代码 000905 |
| 股票代码 | stock_code | 6位数字字符串，如 `000001`（平安银行）、`600519`（贵州茅台） |
| 交易日 | trading day | 周一至周五，排除 A 股节假日（2015-2028 硬编码于 `calendar.py`） |
| 收盘时间 | market close | **15:00 CST** — A 股每日收盘时刻 |
| 日K线 | daily K-line | OHLCV 日线数据 |

## 股票代码规则

| 首字母 | 交易所 | 前缀示例 |
|--------|--------|----------|
| 6 | 上海 (SH) | `sh.600519` / `600519.SH` |
| 0 / 3 | 深圳 (SZ) | `sz.000001` / `000001.SZ` |
| 4 / 8 | 北京 (BJ) | `bj.430047` / `430047.BJ` |

## 数据字段

| 字段 | 含义 | 单位 |
|------|------|------|
| open / high / low / close | 开盘价/最高价/最低价/收盘价 | 元 |
| volume | 成交量 | 手 |
| amount | 成交额 | 元 |
| pct_change | 涨跌幅 | % |
| turnover | 换手率 | % |
| amplitude | 振幅 | % |

---

## 存储架构

### Medallion 三层架构

| 层 | 目录 | 分区策略 | 含义 |
|----|------|----------|------|
| **Bronze** (原始层) | `data/a_shares/news_raw/{stock_code}.parquet` | 按股票 | 爬取即存，追加模式，去重(title+date) |
| **Silver** (对齐层) | `data/a_shares/news_silver/{stock_code}.parquet` | 按股票 | PIT 时点对齐后，去重(title+aligned_date) |
| **Gold** (聚合层) | `data/a_shares/sentiment/{year}/{month}/{stock_code}.parquet` | 按年/月 | 日聚合情感特征，ZI 填充 |

### K线存储

`data/a_shares/daily/{year}/{month}/{stock_code}.parquet` — 与 Gold 层相同的年/月分区。

### 格式

全链路 Parquet（列存，压缩，pandas 原生读写）。

---

## 情感分析

| 术语 | 含义 |
|------|------|
| 情感 (sentiment) | 新闻文本的正负面倾向，范围 [-1, 1]。**统一用"情感"，不用"情绪"** |
| sentiment_title | 新闻标题的情感分 |
| sentiment_body | 新闻正文的情感分（可选） |
| sentiment_mean | 当日所有新闻情感分的均值 |
| sentiment_std | 当日所有新闻情感分的标准差 |
| news_count | 当日新闻条数 |
| positive_ratio | 正面新闻占比（sentiment > 0.2） |
| negative_ratio | 负面新闻占比（sentiment < -0.2） |
| has_news | 当日是否有新闻（bool） |

**情感阈值**: > 0.2 为正面，< -0.2 为负面。

**模型层级**:
- L1: **SnowNLP** — 离线中文 NLP，得分 [0,1] 映射到 [-1,1]
- L2: **FinBERT Chinese** — 计划中（HuggingFace 在大陆受限）

---

## PIT（时点对齐）

> PIT = Point-In-Time。防止未来信息泄露的核心机制。

| 规则 | 说明 |
|------|------|
| 收盘后新闻 → 下一交易日 | 15:00 CST 之后的新闻归属 T+1 |
| ZI 方法 | Zeros & Imputation：无新闻的交易日情感特征填 0，`has_news=False` |
| 无时间戳时 | 当前所有新闻仅有日期无时间戳，视为同日新闻 |

---

## 特征工程

| 术语 | 含义 |
|------|------|
| seq_len | 回看窗口长度，**60 个交易日** |
| target_horizon | 预测目标，**1** = 次日涨跌方向 |
| flat_mode | XGBoost 模式：将 (60, n_features) 展平为 (60*n_features,) |
| 技术指标 (technical) | MA/EMA/MACD/RSI/Bollinger/ATR/OBV/volume_ratio |
| 趋势评分 (scoring) | 规则型 trend_level（0-6）/ buy_signal（0-5）/ bias |
| 时序特征 (temporal) | 滞后项 lag(1/2/3/5/10/20) + 滚动统计 rolling(5/10/20/60) + 日历特征 |
| 情感特征 (sentiment) | SENTIMENT_COLS 全部加入 lag 和 rolling |

**特征工程顺序**（不可改变）：
1. 合并情感列（左连接 date）
2. ZI 填充缺失情感日
3. 技术指标
4. 趋势评分
5. 时序特征（滞后+滚动+日历）

---

## 模型

| 术语 | 含义 |
|------|------|
| XGBoost baseline | 展平特征 + 梯度提升树，Phase 1 |
| LSTM | 2层单向 LSTM + PyTorch Lightning，Phase 2 |
| class_weight | 处理涨跌样本不均衡，自动计算 neg/pos |

---

## 评估

| 术语 | 含义 |
|------|------|
| **MCC** (Matthews Correlation Coefficient) | **主要评估指标**，适用于不平衡二分类 |
| Walk-Forward 验证 | 扩展窗口滚动验证，严格时序拆分，**绝不打乱** |
| Sharpe Ratio | 年化夏普比率 = (日均收益/日收益标准差) × sqrt(252) |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 = 正收益交易占比 |
| Profit Factor | 盈亏比 = 总盈利/总亏损 |

**Walk-Forward 参数**: 2年训练 / 3月验证 / 3月步长。

---

## 故障切换

| 术语 | 含义 |
|------|------|
| Failover | 4源优先级链：Efinance → AKShare → Tushare → Baostock |
| Circuit Breaker | 熔断器：连续 15 次失败后暂停该源 300 秒 |
| curl-cffi | TLS 指纹伪装库，模拟 Chrome 120 的 JA3/JA4 |
| Impersonate | TLS 层面的浏览器身份模拟 |

---

## 命名约定

- 股票代码变量统一用 `stock_code`（不用 `ticker` / `symbol`）
- 情感分析统一用 `sentiment`（不用 `emotion` / `情绪`）
- 对齐后的日期用 `aligned_date`（区别于原始的 `date`）
- 特征 DataFrame 统一用 `feats` / `df`
- 目标变量统一用 `y`（0=下跌, 1=上涨）
- 模型输出统一用 `preds` / `probs`

## 关键常量

| 常量 | 值 | 位置 |
|------|-----|------|
| seq_len | 60 | config.yaml → features.seq_len |
| target_horizon | 1 | config.yaml → features.target_horizon |
| 情感正面阈值 | > 0.2 | news_nlp.py |
| 情感负面阈值 | < -0.2 | news_nlp.py |
| 涨跌幅限制 | ±11% | cleaner.py (含容差) |
| efinance 重试次数 | 3 | efinance_source.py → MAX_RETRIES |
| efinance 退避基数 | 2.0s | efinance_source.py → RETRY_BACKOFF |
| 熔断冷却时间 | 300s | failover.py / rate_limiter.py |
| 熔断失败阈值 | 15 (failover), 5 (rate_limiter) | |
| 请求基础延迟 | 2.0s | config.yaml → crawler.rate_limit.base_delay_sec |
| Session Pool 上限 | 50 | config.yaml → crawler.session_pool.max_sessions |
| Walk-Forward train | 2 年 | config.yaml → training.validation.train_years |
| Walk-Forward val | 3 月 | config.yaml → training.validation.val_months |

---

## 反模式 (Anti-Patterns)

- ~~随机打乱时序数据~~ → 必须用 WalkForwardSplitter，按时间顺序拆分
- ~~用收盘价预测收盘价~~ → 预测的是次日涨跌**方向**（0/1），不是价格
- ~~在全部数据上 fit StandardScaler~~ → 只在训练窗口上 fit，验证窗口仅 transform
- ~~"情绪分析"~~ → 统一用"情感分析"（sentiment）
- ~~裸 `python`~~ → 必须 `PYTHONPATH=. ./.venv/Scripts/python`（系统 Anaconda 缺依赖）
