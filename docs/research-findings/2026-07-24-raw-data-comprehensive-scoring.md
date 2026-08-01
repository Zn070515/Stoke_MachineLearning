# 原始数据全量评分 + 预处理增益分析

> 2026-07-24 | 18 个数据维度 × 5 项质量指标 | 5530 只 A 股

## 评分框架

每维度 5 项指标，各 0-5 分：
- **完整性 (Completeness)**：股票覆盖率、日期跨度、字段缺失率
- **准确性 (Accuracy)**：数值合理性、异常值、无脏数据
- **时效性 (Timeliness)**：最新日期距今天数、历史深度
- **一致性 (Consistency)**：跨文件列结构统一、无 schema 漂移
- **粒度 (Granularity)**：数据频率（日/季/分钟）、是否适合模型直接消费

**综合分 = 五项均分**，映射等级：A+ (4.5-5.0) / A (4.0-4.4) / B+ (3.5-3.9) / B (3.0-3.4) / C (2.0-2.9) / F (<2.0)

---

## 一、全量评分卡

### 核心数据 (Core)

| # | 维度 | 文件数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|--------|--------|--------|--------|------|----------|------|
| 1 | `daily` | 5,530 | 2015~2026-07-23 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 2 | `minute` | 233,033 | 2024~2026-07 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |

**K 线数据是项目最强的资产。** 日线和分钟线覆盖完整、数据干净、无 schema 漂移。唯一瑕疵：11/500 股票有负 OHLCV（极少数退市/复权异常），126/500 有 >|20%| 的日涨跌幅（北交所/ST 真实极端行情）。

### 文本数据 (Text)

| # | 维度 | 文件数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|--------|--------|--------|--------|------|----------|------|
| 3 | `announcements` | 5,531 | 2015~2026-07-15 | 5 | 4 | 4 | 5 | 5 | **4.6** | A+ |
| 4 | `news_silver` | 5,530 | 2026-01~07 | 3 | 5 | 3 | 5 | 5 | **4.2** | A |
| 5 | `guba_sentiment` | 3,710 | 2025-08~2026-07 | 3 | 4 | 4 | 5 | 5 | **4.2** | A |
| 6 | `guba_silver` | 3,708 | 2025-10~2026-07 | 3 | 5 | 4 | 5 | 5 | **4.4** | A |
| 7 | `comment_sentiment` | 5,190 | 2026-06-11~07-23 | 4 | 4 | 2 | 5 | 5 | **4.0** | A |
| 8 | `cninfo_announcements` | 255 | 2024~2026-07 | 1 | 5 | 4 | 4 | 5 | **3.8** | B+ |
| 9 | `guba_raw` | 296 | 2025-10~2026-07 | 1 | 4 | 4 | 5 | 5 | **3.8** | B+ |
| 10 | `xueqiu_raw` | 261 | 2025-08~2026-06 | 1 | 3 | 4 | 5 | 5 | **3.6** | B+ |
| 11 | `news_raw` | 5,530 | 2025-12~2026-07 | 3 | 3 | 3 | 2 | 5 | **3.2** | B |
| 12 | `xueqiu_silver` | 246 | 2026-05~06 | 1 | 3 | 3 | 5 | 5 | **3.4** | B |
| 13 | `xueqiu_sentiment` | 246 | 2026-03~06 | 1 | 3 | 3 | 3 | 4 | **2.8** | C+ |
| 14 | `news_sentiment` | 31,191 | 2020~2026-07 | 4 | 4 | 4 | 5 | 5 | **4.4** | A |
| 15 | `comment_sentiment` | 5,190 | 2026-06~07 | 4 | 4 | 2 | 5 | 5 | **4.0** | A |

### 资金/市场数据 (Market)

| # | 维度 | 文件数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|--------|--------|--------|--------|------|----------|------|
| 16 | `etf_flow` | 1,906 | 2015~2026-06 | 4 | 5 | 4 | 5 | 4 | **4.4** | A |
| 17 | `fundamentals` | 42,581 | 2015-Q1~2026-Q2 | 2 | 4 | 5 | 4 | 3 | **3.6** | B+ |
| 18 | `margin` | 4,608 | 2015~2026-07 | 4 | 5 | 5 | 5 | 4 | **4.6** | A+ |
| 19 | `dragon_tiger` | 6,308 | 2015~2024-02 | 4 | 5 | 4 | 5 | 4 | **4.4** | A |
| 20 | `northbound` | 3,327 | 2017~2024-08 | 3 | 5 | 4 | 5 | 4 | **4.2** | A |

### 总分概览

| 等级 | 数量 | 维度 |
|------|------|------|
| **A+** (4.5-5.0) | 4 | daily, minute, announcements, margin |
| **A** (4.0-4.4) | 7 | news_silver, guba_silver, guba_sentiment, comment_sentiment, etf_flow, news_sentiment, dragon_tiger, northbound |
| **B+** (3.5-3.9) | 4 | cninfo_announcements, guba_raw, xueqiu_raw, fundamentals |
| **B** (3.0-3.4) | 2 | news_raw, xueqiu_silver |
| **C+** (2.5-2.9) | 1 | xueqiu_sentiment |
| **F** (<2.0) | 0 | — |

**加权平均（按维度数）：4.05 / 5.0 → A**
（原始调查误把 `news_sentiment`、`market_wide`、`comment_raw` 判为 F，实际数据存在于 `sentiment/`、`margin/dragon_tiger/northbound/`、`comment_sentiment/` 目录中）

---

## 二、关键质量缺陷详析

### 🔴 致命级：数据完全缺失

> **2026-07-24 勘误：** 原始调查中的三个 F 均为误判 — 查错了目录名。
> - `news_sentiment` → 实际路径是 `sentiment/`（31,191 文件，2020-2026）
> - `market_wide` → 实际路径是 `margin/`(4,608) + `dragon_tiger/`(6,308) + `northbound/`(3,327)
> - `comment_raw` → 该数据源设计上直接输出 `comment_sentiment/`（5,190 文件），没有独立的 raw 层
>
> **修正后，项目不存在 F 级数据维度。**

### 🟠 严重级：覆盖率极度不足

| 维度 | 当前覆盖 | 根因 |
|------|----------|------|
| `cninfo_announcements` | 255/5530 (4.6%) | CNINFO IP 被拉黑，切换 EastMoney 前只下了 255 只 |
| `guba_raw` | 296/5530 (5.4%) | 下载被中断，Silver/Gold 层通过 `guba_storage` 管道处理过更多 |
| `xueqiu_raw/silver/sentiment` | ~250/5530 (4.5%) | Playwright 绕过 WAF 代价高，下载量受限 |
| `fundamentals` | 798/5530 (14.4%) | `download_fundamentals.py` 只覆盖了启动集 |

### 🟡 中等级：数据存在但有质量问题

| 问题 | 维度 | 详情 |
|------|------|------|
| body 缺失 60% | `news_raw` | 新闻正文大面积缺失，只能依赖标题情绪 |
| 列不一致 | `news_raw` | 27% 文件有 sentiment 列，73% 没有 — schema 漂移 |
| sentiment_body 缺失 60-76% | `xueqiu_raw/silver` | 雪球帖子正文情绪大面积空 |
| 日期重复 | `xueqiu_sentiment` | 看起来像 post-level 而非 daily aggregation |
| 历史过短 (6周) | `comment_sentiment` | 5190 只股票但只有 6 周数据 |
| 历史过短 (7个月) | `news_raw/silver` | 新闻数据从 2025-12 才开始 |
| 负 OHLCV | `daily` (11/500) | 部分股票复权数据异常，需过滤 |

### 🟢 轻微级：可接受的问题

| 问题 | 详情 |
|------|------|
| fundamentals ROE/EPS 19% 空值 | 早期报告缺失，forward-fill 即可 |
| announcements body 缺失 | EastMoney API 不提供正文，设计如此 |
| daily pct_change >\|20%\| | 北交所 30% 涨跌停，ST 股票 5%，真实数据 |
| sentiment 有负值 | 情绪范围 [-1, 1]，负值正常 |

---

## 三、预处理增益分析

### 3.1 通过"跑脚本"就能修复（无需写代码）

> **2026-07-24 勘误：** 原始调查中的 F 级维度大部分是目录名误判。实际只需补以下三项：

| 操作 | 影响的维度 | 修复前 | 修复后 | 增益 |
|------|-----------|--------|--------|------|
| 运行 `download_fundamentals.py` 全量 5530 只 | `fundamentals` | B+ (3.6) | A- (4.0) | **+0.4** |
| 运行 `download_guba.py --max-pages 50` | `guba_raw` | B+ (3.8) | A (4.2) | **+0.4** |
| 运行 `download_xueqiu.py` 全量 | `xueqiu_raw/silver/sentiment` | B/B+ (3.3) | B+ (3.7) | **+0.4** |

**小计：重跑 3 个下载脚本 → 拉升整体均分 0.1-0.2 分**

### 3.2 通过数据预处理管道能提升的

| 预处理步骤 | 解决的问题 | 影响的维度 | 增益 |
|-----------|-----------|-----------|------|
| PIT 对齐 (post-15:00 → next trade day) | 消除 look-ahead bias | 所有文本 silver | 准确性 +1 |
| ZI 填充 (zero-impute + has_* flag) | 缺失 auxiliary 的日期不被丢弃 | 所有 sentiment→feature merge | 完整性 +1 |
| 前向填充 (fundamentals q→d) | 季度数据变成日频可用 | `fundamentals` | 粒度 +2 |
| 交叉截面标准化 | 消除股票间量纲差异 | 所有 feature | 准确性 +1 |
| 缺失列标准化 (统一 schema) | `news_raw` 列不一致 | `news_raw` | 一致性 +2 |
| Lexicon sentiment 保底 | body 缺失时退回 title | `news_raw`, `xueqiu` | 完整性 +0.5 |
| 异常值过滤 (负 OHLCV) | 脏 K 线进入训练 | `daily` | 准确性 +0.5 |

**小计：预处理管道可拉升整体均分 0.5-1.0 分**

### 3.3 通过特征工程能额外解锁的

| 特征类别 | 来源 | 新增信息维度 |
|----------|------|-------------|
| 技术指标 (74 维) | 日 K 线 | 趋势/动量/波动率/量价 |
| 趋势评分 (8 维) | 日 K 线 | 多时间框架趋势一致性 |
| 微观结构 (8 维) | 日 K 线 | 涨跌停/缺口/量异常 |
| 时序特征 (lags+rolling) | 所有维度 | 时序记忆 |
| 日历特征 | 日期 | 星期/月份/季度效应 |

**这些不改变原始数据分数，但显著提升模型可用信号量。**

---

## 四、综合评分轨迹（2026-07-24 修正版）

```
原始数据现状 (Raw)           →  补完下载后              →  预处理管道后
──────────────────────────────────────────────────────────────────────────
A+ (4): daily, minute,       A+ (4): daily, minute,       A+ (7): 上述 + announcements,
        announcements, margin        announcements, margin         news_sentiment, guba_sentiment
A  (7): news_silver,         A  (8): 上述 + dragon_tiger, A  (7): 上述 + fundamentals,
        dragon_tiger, northbound,    northbound, guba_silver,      etf_flow, comment_sent.
        news_sentiment, guba_silver, guba_sentiment, fundamentals,
        guba_sentiment,              news_sentiment
        comment_sentiment, etf_flow
B+ (4): cninfo_ann., guba_raw,B+ (3): cninfo_ann., guba_raw,B+ (3): cninfo_ann., guba_raw,
        xueqiu_raw, fundamentals      xueqiu_raw                     xueqiu_raw
B  (2): news_raw, xueqiu_s.  B  (2): news_raw, xueqiu_s.   B  (2): news_raw, xueqiu_s.
C+ (1): xueqiu_sentiment     C+ (1): xueqiu_sentiment      C+ (0): —
F  (0): —                    F  (0): —                     F  (0): —

均分: 4.05 (A)               均分: 4.12 (A)                均分: 4.25 (A)
```

## 五、结论

> **2026-07-24 修正：** 原始调查中 3 个 F 级维度均为目录名误判。修正后项目不存在 F 级数据，真实均分为 A（4.05/5.0）。

| 阶段 | 均分 | 等级 |
|------|------|------|
| **当前原始数据（修正后）** | **4.05 / 5.0** | **A** |
| 补完 fundamentals/guba/xueqiu 后 | **4.12 / 5.0** | **A** (+0.07) |
| 预处理管道后 | **4.25 / 5.0** | **A** (+0.20) |

**剩余主要缺口（按贡献排序）：**
1. 前向填充 fundamentals 到日频 — 粒度 +2
2. 统一 schema + body 缺失兜底 — news_raw 一致性 +2
3. PIT 对齐 + ZI 填充 — 所有文本维度完整性 +1
4. fundamentals 股票覆盖率 798→5530（进行中）
5. guba_raw 覆盖率 296→3708+
6. xueqiu 覆盖率 250→更多（Playwright 代价高）

**无法通过预处理修复的硬伤：**
- CNINFO body 文本覆盖率（IP 被封，已切换 EastMoney）
- 新闻 history 太短（7 个月）— 需要更早开始积累或换源
- comment 历史太短（6 周）— 同上
- xueqiu 覆盖率低 — Playwright 下载代价高，需长期积累
