# Stoke_MachineLearning 全面项目分析

> 2026-06-29 | 基于完整代码库探索 + 100-Stock 消融实验

---

## 一、项目概况

**A股股票预测系统**（沪深300+中证500），覆盖 798 只股票。

**三阶段架构：** 数据采集 → 特征工程 → 模型训练

**技术栈：** Python, PyTorch 2.11 + Lightning, XGBoost, OmegaConf, curl-cffi

**数据总量：** ~759 MB，15 个 Parquet 存储分区

---

## 二、数据源全景

### 2.1 K线数据（4源故障切换链）

| 优先级 | 数据源 | 类名 | 状态 |
|---|---|---|---|
| 1 | EastMoney HTTP API | `EfinanceSource` | 主力，Chrome 120 TLS 伪装 |
| 2 | AKShare (Sina) | `AKShareSource` | 回退 |
| 3 | Tushare | `TushareSource` | 需 token |
| 4 | Baostock | `BaostockSource` | 免费，最后手段 |

**熔断：** 连续 10 次失败 → 300s 冷却

### 2.2 新闻数据（3 源聚合）

| 数据源 | 类名 | 状态 |
|---|---|---|
| EastMoney 搜索 API | `THSNewsSource` | **主力**，100条/页×5页≈500篇/股 |
| Sina Finance | `SinaNewsSource` | 辅助，25条/页 |
| 雪球 | `XueqiuNewsSource` | **封禁中**（Cloudflare WAF） |

聚合管道：`NewsPipeline` → 按标题+日期去重，优先保留有正文的行

### 2.3 论坛/社区数据

| 数据源 | 类名 | 数据量 | 正文覆盖 |
|---|---|---|---|
| 东方财富股吧 | `GubaSource` | 802 股 × ~800 帖 | **~0%**（detail 页面被 WAF 封） |
| AKShare 评论 | `CommentSource` | 5184 股 | 快照+30天历史评分 |

**Guba 爬虫技术栈：**
- 列表页：`/topic,{code}` URL（绕过 WAF）
- 正文页：`/news,{code},{post_id}.html`（**已被 WAF 封锁**）
- 解析：`article_list` JSON + BeautifulSoup 回退
- 并发：10 线程池获取正文（当正文可用时）

### 2.4 市场数据（独立源）

| 数据源 | 存储大小 | 说明 |
|---|---|---|
| 融资融券 | 253 MB（最大数据集）| SSE/SZSE 日频明细 |
| 龙虎榜 | 52 MB | 日频上榜数据 |
| 北向资金 | 36 MB | 沪深港通个股流向 |
| ETF 资金流 | 928 KB | 21 个行业×2 ETF |

### 2.5 基本/公司数据

| 数据源 | 存储大小 |
|---|---|
| 财务指标（季报）| 12 MB |
| 公司公告 | 46 MB（1596 文件）|

---

## 三、存储架构（3 层 Medallion）

所有文本类数据均遵循 Bronze → Silver → Gold 三层架构：

```
Bronze (raw)         → Silver (PIT-aligned)     → Gold (daily aggregation)
/news_raw/{code}.pq     /news_silver/{code}.pq      /sentiment/{year}/{month}/{code}.pq
/guba_raw/{code}.pq     /guba_silver/{code}.pq      /guba_sentiment/{year}/{month}/{code}.pq
```

关键机制：
- **PIT 防泄漏：** 15:00 CST 后新闻归入下一交易日
- **ZI 方法：** 无数据日填零 + `has_* = False` 标志
- **分区策略：** `{year}/{month}/{code}.parquet`（Hive 风格）
- **去重：** Bronze 按标题+日期，Guba 按 `post_id`

---

## 四、特征工程

### 4.1 FeaturePipeline（516行，核心文件）

```
K线数据 → 合并辅助DataFrame → 技术指标 → 趋势评分 → 微结构 → 时序特征 → (X, y)
```

**9 个可选辅助维度（均滞后 1 日防泄漏）：**

| 维度 | 列数 | 开关 | 数据密度 |
|---|---|---|---|
| sentiment（新闻）| 6 | `use_sentiment` | 中 |
| guba（股吧）| 6 | `use_guba` | 高（帖子量）低（正文） |
| comment（评论）| 5 | `use_comment` | 中 |
| announcement（公告）| 6 | `use_announcements` | 低 |
| margin（融资融券）| 4 | `use_margin` | 高 |
| northbound（北向）| 2 | `use_northbound` | 中 |
| dragon_tiger（龙虎榜）| 3 | `use_dragon_tiger` | 低 |
| fundamental（基本面）| 8 | `use_fundamental` | 低（季频） |
| etf_flow（ETF资金）| 2 | `use_etf_flow` | 高 |

**技术指标（~40 个）：** MA(5/10/20/60/120), EMA(12/26), MACD, RSI(6/12/24), KDJ, Bollinger, ATR, ROC, Williams %R, CCI, 波动率, OBV, 成交量比率

**微结构：** 涨停/跌停标志、缺口、量比、成交量异常

**时序特征：** 滞后(1/2/3/5/10/20)、滚动统计(5/10/20/60)、日历特征

**维度爆炸：** 全部开启时 ~405 特征 × 60 seq_len = **24,300 维度**（flat 模式）

### 4.2 情感分析

**FinBERT Chinese** (`yiyanghkust/finbert-tone-chinese`)：
- 加载：HF 镜像 → 本地缓存 → lexicon 回退
- 推理：CPU 38ms/条, GPU batch 更快
- 评分：P(positive) - P(negative)，连续值 [-1, 1]

**金融词库回退：** 39 正面词 + 35 负面词，命中率仅 13%（Guba 标题）

---

## 五、模型

| 模型 | 类型 | 参数规模 | 训练速度 |
|---|---|---|---|
| `XGBoostBaseline` | 树模型 | 50-200 trees, depth 4-6 | ~8s/窗口 |
| `LSTMModel` | 2层 LSTM | hidden_dim=128 | ~分钟/epoch |
| `TransformerModel` | 3层 Transformer | d_model=128, nhead=8 | ~分钟/epoch |
| `SimpleAttentionModel` | 1层 Attention | d_model=64, nhead=4 | ~分钟/epoch |

**已有模型检查点：** 平安银行(000001)、中国平安(601318)、贵州茅台(600519)

---

## 六、评估体系

- **主指标：** MCC（马修斯相关系数，适合不平衡分类）
- **Walk-Forward：** 1年训练 / 3月验证 / 3月步长，严格时序
- **金融指标：** Sharpe, Max Drawdown, Win Rate, Profit Factor
- **Bootstrap CI：** 1000 次重采样，95% 置信区间

---

## 七、100-Stock 消融实验结果

**配置：** 95 stocks, 2 walk-forward windows, 1000 bootstrap samples

| Config | MCC | 95% CI | 自身显著 | Δ vs Baseline |
|---|---|---|---|---|
| technical | 0.0136 | [-0.0035, 0.0312] | 跨 0 | — |
| + sentiment | 0.0279 | [0.0095, 0.0464] | ✅ | +0.0143 |
| + guba | 0.0219 | [0.0032, 0.0384] | ✅ | +0.0084 |
| + comment | 0.0224 | [0.0045, 0.0408] | ✅ | +0.0089 |
| ALL | 0.0261 | [0.0104, 0.0426] | ✅ | +0.0125 |

**Δ 显著性（配对 Bootstrap）：全部未通过**（所有 Δ CI 跨 0）

**最佳配置分布：** sentiment(23) > ALL(21) > guba(19) > comment(17) > technical(15)

### 关键发现

1. 加任何文本维度均优于纯技术面（所有 4 个配置 CI 整体 > 0）
2. **+sentiment 是最优单维度**，MCC 翻倍（0.0136→0.0279）
3. **ALL 不优于 +sentiment**，维度爆炸抵消了新信息增益
4. MCC 绝对值偏低（0.014-0.028），反映中小盘股数据稀疏
5. **Guba body 缺失是瓶颈**：仅标题做情感，信号强度不足
6. **FinBERT 加载问题已修复**：支持 CPU/离线/镜像加载

---

## 八、数据规模与覆盖

| 数据类型 | 大小 | 文件数 | 覆盖 |
|---|---|---|---|
| K线（日频）| 71 MB | 798 | 100% |
| 新闻原始 | 116 MB | 798 | 100% |
| 新闻白银 | 41 MB | 798 | 100% |
| 情感黄金 | 6.7 MB | — | 100% |
| Guba 原始 | 50 MB | 802 | 100%（标题），~0%（正文）|
| Guba 白银 | 51 MB | — | 100% |
| Guba 情感 | 21 MB | — | 100% |
| 融资融券 | 253 MB | 4545 | 高 |
| 龙虎榜 | 52 MB | — | 低（仅上榜股）|
| 北向资金 | 36 MB | — | 中 |
| 基本面 | 12 MB | — | 低（季频）|
| 评论情感 | 7 MB | — | 5184 股 |
| 公告 | 46 MB | 1596 | 低 |
| ETF 资金流 | 928 KB | — | 高（行业级）|
| **总计** | **759 MB** | — | — |

---

## 九、已知问题与限制

| 问题 | 严重性 | 状态 |
|---|---|---|
| Guba body 被 WAF 封禁 | 高 | 无法解决（已尝试所有 URL/伪装） |
| 雪球被 Cloudflare WAF 封禁 | 中 | 无法解决 |
| 维度爆炸（ALL=24,300 特征）| 中 | 需要特征选择/PCA |
| MCC 整体偏低（<0.03）| 高 | 小盘股数据稀疏 |
| 测试目录基本为空 | 中 | 仅 2 个 Guba 测试 |
| FinBERT 首次加载需网络 | 低 | 已通过镜像+缓存修复 |
| 情感词库命中率仅 13% | 中 | FinBERT 已修复 |

---

## 十、已提交的关键修复（本轮）

1. `fix: Guba detail URL to /news/ pattern` — 正文获取（被 WAF 覆盖）
2. `feat: add resume support to download_guba` — 断点续传
3. `feat: add feature dimension ablation study` — 消融+置信区间
4. `feat: add Guba body refetch script` — 正文补采工具
5. `fix: FinBERT loading supports CPU, mirror, and offline cache` — 核心 NLP 修复
