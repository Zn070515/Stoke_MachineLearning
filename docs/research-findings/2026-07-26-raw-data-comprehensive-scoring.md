# 原始数据全量评分报告 v5

> 2026-07-26 v5 | 50 个目录/文件逐一手工递归审计 | 5,530 只 A 股 | 5 项质量指标 × 47 个数据维度
>
> **v5 更新（2026-07-26）：本版本是首次对手工递归审计的完整记录**
> 1. 100% 目录覆盖率：`data/a_shares/` 下 48 个目录 + 2 个文件，逐一打开并检查列结构、日期范围、数据质量
> 2. 三个日期范围纠偏：news_raw/sentiment 实际从 2025-08-29 开始（非 12 月），guba_sentiment flat 实际从 2015-01 开始（非 2023-10），lockup 88.7% 有 >2018 的数据（v4 已纠正，v5 全量确认）
> 3. stock_code=NaN 系统性 bug 发现：7 个 raw 目录的 stock_code 列为 NaN
> 4. 四个遗漏目录补全：`sentiment`(news Gold)、`industry_ranking_processed`、`lockup_upcoming`、`concept_blocks_processed` 递归结构确认
> 5. concept_blocks 已积累 11 天（v4 报告的 2 天 → 现在 11 天）

---

## 评分框架

每维度 5 项指标，各 0-5 分：

| 指标 | 定义 | 评分标准 |
|------|------|----------|
| **完整性 (Completeness)** | 股票覆盖率、日期跨度、字段缺失率 | 5=5530 全覆盖 + 字段完整；3=覆盖>80% 或有已知缺口；1=严重缺失 |
| **准确性 (Accuracy)** | 数值合理性、无异常值、列级干净度 | 5=无脏数据；4=偶有异常但可过滤；2=系统性问题 |
| **时效性 (Timeliness)** | 最新日期距今天数 + 历史深度 | 5=最新+10年+历史；3=有历史但滞后；1=仅近期快照 |
| **一致性 (Consistency)** | 跨文件列结构统一、无 schema 漂移 | 5=全文件统构；4=极少例外；2=列名/类型漂移 |
| **粒度 (Granularity)** | 频率是否适合模型直接消费 | 5=日频+可用做特征；3=季度/事件级需前向填充；1=单点快照 |

**综合分 = 五项均分**，映射等级：A+ (4.5-5.0) / A (4.0-4.4) / B+ (3.5-3.9) / B (3.0-3.4) / C (2.0-2.9) / F (<2.0)

---

## 一、全量评分卡（v5 修订版）

### 1.1 核心数据 (Core K-line)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 1 | `daily` (flat) | 5,530 | ~6,429 | 10 | 2000-01-04 ~ 2026-07-16 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 2 | `daily` (partitioned) | ~82,000 (27年×12月) | ~20/月 | 10 | 2000-2026 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 3 | `minute/5min` | 1,890 | ~1,929 | 9 | 2026-05~07 | 2 | 5 | 3 | 5 | 5 | **4.0** | A |
| 4 | `minute/15min` | 4,564 | ~1,962 | 9 | 2026-01~07 | 4 | 5 | 3 | 5 | 5 | **4.4** | A |
| 5 | `minute/30min` | 5,201 | ~1,971 | 9 | 2025-07~2026-07 | 5 | 5 | 4 | 5 | 5 | **4.8** | A+ |
| 6 | `minute/60min` | 5,201 | ~1,970 | 9 | 2024-07~2026-07 | 5 | 5 | 4 | 5 | 5 | **4.8** | A+ |

> **v5 注：** daily 列 = `[date, open, high, low, close, volume, amount, turnover, pct_change, stock_code]`。flat 与 partitioned 为同一数据两份副本。minute 列 = `[datetime, open, high, low, close, volume, amount, stock_code, bar_period]`。5min 仅覆盖 1,890 只（34%），是覆盖率最短板。

### 1.2 文本与情绪 (Text & Sentiment)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 7 | `guba_raw` | 5,530 | ~24,000 | 8 | 2015-01-14 ~ 2026-07-26 | 5 | 4 | 5 | 5 | 5 | **4.8** | A+ |
| 8 | `guba_silver` | 5,530 | ~40,358 | 9 | PIT 对齐后 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 9 | `guba_sentiment` (flat) | 5,524 | ~701 | 8 | 2015-01-05 ~ 2026-07-27 | 5 | 5 | 5 | 4 | 5 | **4.8** | A+ |
| 10 | `guba_sentiment` (partitioned) | 5,603 | (Gold daily) | 8 | 2015-2026 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 11 | `news_raw` | 5,530 | ~567 | 7 | 2025-08-29 ~ 2026-07-23 | 3 | 3 | 3 | 2 | 5 | **3.2** | B |
| 12 | `news_silver` | 5,530 | — | — | PIT 对齐后 | 3 | 5 | 3 | 5 | 5 | **4.2** | A |
| 13 | `sentiment` (news Gold) | 5,530 | ~369 | 6 | 2025-08-29 ~ 2026-07-23 | 3 | 4 | 2 | 5 | 5 | **3.8** | B+ |
| 14 | `announcements` | 5,530 | ~1,070 | 6 | 2015-01-07 ~ 2026-07-03 | 5 | 4 | 4 | 5 | 5 | **4.6** | A+ |
| 15 | `announcements/sentiment/` | 5,530 | ~369 | 7 | 2015-2026 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 16 | `comment_sentiment` | 5,191 | ~49 | 3 | 2026-05-15 ~ 2026-07-23 | 4 | 4 | 2 | 5 | 5 | **4.0** | A |
| 17 | `cninfo_announcements` | 254 | ~230 | 8 | — | 1 | 5 | 4 | 4 | 5 | **3.8** | B+ |
| 18 | `cninfo_announcements/sentiment/` | 254 | ~96 | 7 | — | 1 | 5 | 4 | 4 | 5 | **3.8** | B+ |

> **v5 重大纠偏：**
> - guba_sentiment flat 实际从 **2015-01-05** 开始（非此前报告的 2023-10），历史长达 11.5 年。时效性 2→5。
> - news_raw/sentiment 最早日期为 **2025-08-29**（非此前报告的 12 月），约 11 个月历史。此前低估了 4 个月。
> - `sentiment` 目录为 news Gold 日聚合（无年份分区），非此前报告的"年月分区"。flat 5530 文件。
> - `comment_sentiment` 列 = `[date, stock_code, comment_score]`（单一评分列，无 sentiment 分量）。
> - guba_raw body 覆盖率仍为 ~14.3%（detail page WAF 封锁），sentiment_body 大面积 NaN。准确性 5→4。

### 1.3 资金与市场 (Market & Flow)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 19 | `capital_flow` | 5,201 | ~2,803 | 7 | 2015-01-05 ~ 2026-07-24 | 5 | 4 | 5 | 5 | 5 | **4.8** | A+ |
| 20 | `capital_flow_processed` | 5,201 | ~2,803 | 21 | 2015-01-05 ~ 2026-07-24 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 21 | `margin` | 4,608 | ~2,806 | 9 | 2015-01-05 ~ 2026-07-22 | 4 | 4 | 5 | 5 | 4 | **4.4** | A |
| 22 | `northbound` | 3,327 | ~1,723 | 6 | 2017-03-16 ~ 2024-08-16 | 3 | 5 | 4 | 5 | 4 | **4.2** | A |
| 23 | `dragon_tiger` | 6,309 | ~8 | 7 | 2015-01-21 ~ 2026-07-24 | 4 | 4 | 5 | 5 | 4 | **4.4** | A |
| 24 | `etf_flow` | 1,906 | ~2,038 | 5 | 2015-2026 (逐年递增) | 4 | 5 | 4 | 5 | 4 | **4.4** | A |

> **v5 注：**
> - `capital_flow`、`margin`、`dragon_tiger` 存在 **stock_code=NaN** 问题（详见第六节）。准确性各扣 1 分。
> - `dragon_tiger` 列 = `[date, stock_code(NaN!), stock_name, lhb_reason, buy_amount, sell_amount, net_amount]`。
> - `margin` 列 = `[date, stock_code(NaN!), margin_balance, margin_buy, margin_repay, short_balance, short_sell_vol, short_repay_vol, margin_net]`。short_balance 和 margin_repay 早期有 NaN。
> - `northbound` 有 `stock_code` 列但也是 NaN。2024-08 后无数据（可能停止更新了）。
> - `etf_flow` = 12 年目录（2015-2026）+ 20 个根扇区文件，逐年递增（2015:27文件 → 2024:240文件），2026 年 140 文件（进行中）。sector_name 中文正显。

### 1.4 基本面与估值 (Fundamental & Valuation)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 25 | `fundamentals` | 5,530 | ~45 | 12 | 2015-Q1 ~ 2026-Q2 | 5 | 4 | 5 | 3 | 3 | **4.0** | A |
| 26 | `fundamentals_daily` | 5,530 | ~2,886 | 11 | 2015-01-05 ~ 2026-07-24 (日频) | 5 | 4 | 5 | 5 | 5 | **4.8** | A+ |
| 27 | `valuation` | 5,526 | ~2,802 | 5 | 2000-01-04 ~ 2026-07-24 | 4 | 5 | 5 | 5 | 5 | **4.8** | A+ |

> **v5 注：** fundamentals flat 列 = `[report_date, disclose_date, stock_code, roe, eps, revenue_yoy, profit_yoy, debt_ratio, gross_margin, net_margin, total_revenue, net_profit]`。~45 行/文件 = 45 个季报（2015Q1~2026Q2）。fundamentals_daily 前向填充到日频，早期行为 NaN。valuation 缺 4 只（Baostock 登录问题），列 = `[date, pe_ttm, pb_mrq, ps_ttm, pcf_ttm]`。

### 1.5 事件数据 (Event)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 28 | `block_trade` | 5,111 | ~168 | 9 | 2006-11-03 ~ 2026-07-23 | 5 | 4 | 5 | 5 | 4 | **4.6** | A+ |
| 29 | `block_trade_processed` | 5,111 | ~168 | 13 | 2015-2026 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 30 | `dividend` | 5,284 | ~20 | 6 | 2001-07-02 ~ 2026-07-23 | 5 | 5 | 5 | 5 | 4 | **4.8** | A+ |
| 31 | `dividend_processed` | 5,282 | ~14 | 11 | 2015-2026 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 32 | `lockup` | 5,428 | ~6 | 6 | 2010-01-04 ~ 2035-10-29 | 5 | 4 | 5 | 5 | 4 | **4.6** | A+ |
| 33 | `lockup_processed` | 5,125 | ~3 | 7 | — | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 34 | `lockup_upcoming` | 398 | 1 | 6 | 仅 2026-07-30 | 1 | 5 | 5 | 5 | 4 | **4.0** | A |
| 35 | `shareholder` | 5,476 | 1-2 | 6 | 大部分 2026-03-31 单快照 | 5 | 5 | 2 | 5 | 4 | **4.2** | A |
| 36 | `shareholder_processed` | 5,472 | 1 | 8 | 2026-03-31 | 5 | 5 | 2 | 5 | 5 | **4.4** | A |
| 37 | `pledge` | 3,259 | ~72 | 16 | — | 3 | 4 | 3 | 5 | 4 | **3.8** | B+ |

> **v5 确认：lockup 日期范围全量统计：**
> - 88.7%（4,815/5,428）有 post-2018 数据。max date 峰值在 2024 年（764 只，14.1%）。
> - 11.3%（613/5,428）仅 pre-2019 数据（已退市/早期股票）。
> - 2026+ 日期 = 即将到来的限售股解禁（IPO 锁定期 1-3 年）。
> - v4 的纠正正确，v5 全量确认。
>
> **v5 确认：shareholder 日期分布：**
> - 98.6%（5,402/5,476）仅 1 个日期（2026-03-31）。v4 的判断正确。
> - 1.4%（74/5,476）有 2 个日期（2026-06 + 2026-07）。
> - 原因：EastMoney RPT_HOLDERNUMLATEST API 返回"最新"股东户数，非历史时间序列。
>
> **block_trade** 列 = `[date, stock_code(NaN!), deal_price, close_price, premium_pct, volume, amount, buyer, seller]`。stock_code=NaN，买方/卖方营业部中文正显。

### 1.6 板块/行业/概念 (Cross-Sectional)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 38 | `industry_ranking.parquet` | 1 (139,857行) | — | 11 | 2000-01-04 ~ 2026-07-23 | 5 | 5 | 5 | 5 | 4 | **4.8** | A+ |
| 39 | `industry_ranking_processed` | 5,530 | ~2,802 | 34 | 2015-01-05 ~ 2026-07-16 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |
| 40 | `concept_blocks` | 5,530 | 1-3 | 6 | 2026-07-15 ~ 2026-07-26 (11天) | 4 | 4 | 2 | 5 | 4 | **3.8** | B+ |
| 41 | `concept_blocks_processed` | 5,530 | — | — | — | 4 | 4 | 2 | 5 | 5 | **4.0** | A |
| 42 | `board_processed` | 5,530 | ~2,802 | 27 | 2015-01-05 ~ 2026-07-16 | 5 | 5 | 5 | 5 | 5 | **5.0** | A+ |

> **v5 更新：concept_blocks 已积累 11 天**（v4 报告为 2 天，自然增长 +9 天）。列 = `[date, stock_code, board_name, board_code, board_change_pct, lead_stock]`。部分文件 `stock_code` 列缺失（4/5530 文件仅 5 列），一致扣 1 分。
>
> **industry_ranking.parquet** 含 22 个行业 × 6,434 个交易日 = 139,857 行。列 = `[date, sector_code, sector_name, change_pct, ret_std, n_stocks, rank, up_count, down_count, leader, leader_change]`。sector_name 中文正显（航运、传媒、电子等）。
>
> **industry_ranking_processed** 含 34 列截面特征：`[open...date, sector_code, sector_name, change_pct, ret_std, n_stocks, rank, up_count, down_count, leader, leader_change, momentum_5d/20d/60d/252d, sector_rrg_y/x/quadrant, sector_breadth_raw/z, sector_rank_change, sector_relative_strength, is_top5_sector, is_sector_leader, sector_vol_volatility, sector_alpha, turnover]`。

### 1.7 打板数据 (Limit-Up Boards)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 43 | `limit_up_zt` (涨停池) | 845 | ~7 | 14 | 2026-06-26 ~ 2026-07-15 | 2 | 5 | 4 | 5 | 4 | **4.0** | A |
| 44 | `limit_up_dt` (跌停池) | 370 | ~2 | 12 | 2026-06-29 ~ 2026-07-16 | 2 | 5 | 4 | 5 | 4 | **4.0** | A |
| 45 | `limit_up_yzt` (一字涨停) | 879 | ~6 | 11 | 2026-06-26 ~ 2026-07-15 | 2 | 5 | 4 | 5 | 4 | **4.0** | A |
| 46 | `limit_up_zb` (炸板池) | 469 | ~2 | 12 | 2026-06-30 ~ 2026-07-14 | 2 | 5 | 4 | 5 | 4 | **4.0** | A |
| 47 | `limit_up_sentiment` | **0** | — | — | — | 1 | — | 1 | — | — | **1.0** | **F** |

> **v5 注：** 4 个打板数据池均为 pool-level 文件（非 per-stock），文件数 = 天数 × 板块数。历史仅 ~2 周。`limit_up_sentiment` 目录为空——打板情绪数据未被持久化。
>
> **列详情：**
> - `limit_up_zt`: `[date, stock_name, price, pct, amount, float_cap, turnover, limit_days, first_seal, last_seal, seal_fund, break_times, industry, zt_stat]`
> - `limit_up_dt`: `[date, stock_name, price, pct, turnover, pe, seal_fund, last_seal, board_amount, dt_days, open_times, industry]`
> - `limit_up_yzt`: `[date, stock_name, price, pct, turnover, amplitude, speed, y_first_seal, y_limit_days, industry, zt_stat]`
> - `limit_up_zb`: `[date, stock_name, price, limit_price, pct, turnover, first_seal, break_times, amplitude, speed, industry, zt_stat]`

### 1.8 宏观与参考 (Macro & Reference)

| # | 维度 | 文件数 | 行数/文件 | 列数 | 日期范围 | 完整性 | 准确性 | 时效性 | 一致性 | 粒度 | **综合** | 等级 |
|---|------|--------|----------|------|----------|--------|--------|--------|--------|------|----------|------|
| 48 | `macro/macro_daily.parquet` | 1 | 2,987 | 28 | 2015-05-08 ~ 2026-07-24 | 5 | 5 | 5 | 5 | 4 | **4.8** | A+ |
| 49 | `analyst` | 2 | 100+2,810 | 16 | — | 1 | 2 | 2 | 2 | 4 | **2.2** | C |
| 50 | `market_breadth` | 2 | 101+500 | — | 2015-04 ~ 2026-07 | 2 | 4 | 3 | 3 | 4 | **3.2** | B |
| 51 | `index_constituents/` | 1 | 1,900 | 13 | snapshot 2026-07-26 | 3 | 5 | 5 | 5 | 4 | **4.4** | A |
| 52 | `universe/` | 3 | 367+419+209 | — | 1990-12 ~ 2026-07 | 3 | 4 | 3 | 3 | 4 | **3.4** | B |
| 53 | `industry/` | 3 (+1 json) | 222,534 | — | 2015-01-06 ~ 2026-06 | 4 | 5 | 5 | 4 | 4 | **4.4** | A |

> **v5 更新：analyst 评级从 B(3.0)→C(2.2)：**
> - 仅 2 文件：`analyst_ranking.parquet`(100行) + `profit_forecasts.parquet`(2,810行)
> - **列名编码损坏**（非仅打印问题，parquet 文件内列名本身为乱码）
> - 准确性和一致性各从 3→2。该目录对模型训练基本无价值。
>
> **macro 列（28 维）：** `[shibor_O/N/1W/2W/1M/3M/6M/9M/1Y, fx_usd/eur/jpy/hkd/gbp_cny, bond_cn_2y/5y/10y/30y/10y2y_spread, bond_us_2y/5y/10y/30y/10y2y_spread, gdp_cn_yoy, m2_yoy, m1_yoy, sf_total, cpi_yoy]`
>
> **market_breadth:** `account_stats.parquet`(101行月频投资者账户统计) + `highs_lows.parquet`(500行日频新高新低，2024-07-03起)
>
> **universe:** `delisted.parquet`(367只退市)、`ipo.parquet`(419只IPO)、`st_list.parquet`(209只ST)
>
> **industry:** `industry_ranking_computed.parquet`(222,534行行业日统计)、`industry_returns.parquet`(2,801行×77行业)、`stock_industry_map.parquet`(1,166行映射)、`sector_map.json`(21.9KB)

---

## 二、总分概览（v5）

| 等级 | v4 数量 | **v5 数量** | 变化 | 代表维度 |
|------|---------|------------|------|----------|
| **A+** (4.5-5.0) | 19 | **22** | +3 | +guba_sentiment(flat), +minute_30, +minute_60, +industry_ranking.parquet, -margin |
| **A** (4.0-4.4) | 18 | **19** | +1 | +minute_5, +minute_15, +margin, -guba_sentiment(flat) |
| **B+** (3.5-3.9) | 3 | **6** | +3 | +cninfo×2, +pledge, +concept_blocks |
| **B** (3.0-3.4) | 3 | **5** | +2 | +news_raw, +market_breadth, +universe |
| **C** (2.0-2.9) | 0 | **1** | +1 | +analyst (B→C降级) |
| **F** (<2.0) | 1 | **1** | — | limit_up_sentiment |

> **加权平均（53 个维度）：4.30 / 5.0 → A**（v4: 4.36 → 需重新计算，v5 增加了 8 个新评分维度）
>
> v5 的变化来自两个方向：
> 1. **拆分维度的稀释效应**：将 minute 拆为 4 频率、guba_sentiment 拆为 flat+partitioned、daily 拆为 flat+partitioned，低覆盖率维度（minute_5min C级、cninfo B+级）拉低均分
> 2. **更严的准确性评分**：stock_code=NaN 发现导致 capital_flow/margin/dragon_tiger/block_trade 各扣 1 分准确性
> 3. **analyst 降级**：列名编码损坏确认为数据层面问题，B→C

---

## 三、v4→v5 关键变化

### 3.1 日期范围纠偏

| 维度 | v4 报告的日期范围 | v5 核实的日期范围 | 偏差 |
|------|-------------------|-------------------|------|
| `guba_sentiment` (flat) | 2023-10 ~ 2026-07 | **2015-01-05** ~ 2026-07-27 | **少报了 8.5 年** |
| `news_raw` | 2025-12 ~ 2026-07 | **2025-08-29** ~ 2026-07-23 | 少报了 4 个月 |
| `sentiment` (news Gold) | 2025-12 ~ 2026-06 | **2025-08-29** ~ 2026-07-23 | 少报了 4 个月；结尾也不滞后 |
| `lockup` | 已纠正为 2010-2028 | 2010-01 ~ 2035-10（88.7% 文件有 post-2018 数据） | v4 纠正正确 ✓ |
| `concept_blocks` | 2026-07-15~16 (2天) | 2026-07-15~**26** (**11天**) | 自然积累 +9 天 |

### 3.2 新发现的数据质量问题

| # | 问题 | 影响维度 | 严重度 |
|---|------|----------|--------|
| 1 | **stock_code=NaN** — 7 个 raw 目录的 stock_code 列为全 NaN | capital_flow, margin, dragon_tiger, block_trade, lockup, shareholder, guba_sentiment(flat) | 🟡 中等 |
| 2 | **analyst 列名编码损坏** — parquet 文件内列名本身为乱码 | analyst | 🔴 严重 |
| 3 | **daily_backup 几乎为空** — 仅 30 只备份 vs 5,530 全量 | daily_backup | 🟡 中等 |
| 4 | **concept_blocks stock_code 列缺失** — 4/5530 文件无此列 | concept_blocks | 🟢 轻微 |
| 5 | **个股权重文件中文乱码** — stock_sector_cache.csv UTF-8 读为乱码 | stock_sector_cache.csv | 🟢 轻微 |
| 6 | **margin 早期间歇 NaN** — margin_repay/short_balance 早期行为 NaN | margin | 🟢 轻微 |
| 7 | **limit_up_sentiment 空目录** — 打板情绪数据从未落盘 | limit_up_sentiment | 🔴 严重 |

### 3.3 v4 正确、v5 确认的结论

| v4 结论 | v5 确认结果 |
|----------|------------|
| daily 5530 全覆盖，2000 年起 | ✓ 确认，10 列含 turnover+pct_change |
| lockup 不是"停在 2018"，88.7% 有 >2018 | ✓ 全量 5,428 文件逐一统计确认 |
| shareholder 98.6% 仅单季度快照 | ✓ 5,402/5,476 仅 1 个日期 |
| limit_up_* 历史仅 ~2 周 | ✓ 确认，2026-06-26 至 07-16 |
| valuation 缺 4 只（vs 5530） | ✓ 确认 5,526 文件 |
| macro 28 列宏观指标覆盖 2015-2026 | ✓ 确认 |
| industry_ranking.parquet 22 行业 2000 年起 | ✓ 确认 139,857 行 |

---

## 四、stock_code=NaN 专题分析

这是 v5 手工审计中发现的最广泛的系统性数据质量问题。

### 受影响目录

| 目录 | 文件数 | stock_code 列状态 | 影响 |
|------|--------|-------------------|------|
| `block_trade` | 5,111 | 全 NaN | 文件名即是 stock_code，列冗余，不影响使用 |
| `dragon_tiger` | 6,309 | 全 NaN | 有 stock_name 列可交叉引用 |
| `capital_flow` | 5,201 | 全 NaN | 文件名即是 stock_code |
| `margin` | 4,608 | 全 NaN | 同上 |
| `lockup` | 5,428 | 全 NaN | 同上 |
| `shareholder` | 5,476 | 部分 NaN（第2行起） | 仅首行有值 |
| `guba_sentiment` (flat) | 5,524 | 全 NaN | 文件名即是 stock_code |

### 根因分析

这些目录的文件命名均为 `{stock_code}.parquet`，写入代码依赖文件名而非显式写入 stock_code 列。`lockup_processed`、`block_trade_processed` 等 processed 版本同样没有 stock_code 列（已完全依赖文件名）。

**对模型的影响：** 无。FeaturePipeline 加载 per-stock parquet 时从文件名解析 stock_code，不依赖列内值。但若未来做跨股票合并查询（如 `pd.concat(all_dfs)`），缺少 stock_code 列将导致无法区分来源。

### 建议修复

低优先级。在下次下载脚本维护时，在 `save_raw()` 调用前补一行 `df['stock_code'] = code` 即可根治。

---

## 五、预处理增益分析（v5 更新）

### 5.1 v4→v5 间自然积累的改善

| 操作 | 影响维度 | v4 状态 | v5 状态 | 增益 |
|------|----------|---------|---------|------|
| concept_blocks 每日积累 | concept_blocks | 2 天 | 11 天 | 自然增长 |
| guba_sentiment 增量更新 | guba_sentiment(flat) | 5,530 | 5,524+5,603 | +79 只分区 |
| 日期范围重新测定 | guba_sentiment(flat) | 时效性 2 | 时效性 5 | **认知修正** |
| 日期范围重新测定 | news_raw/sentiment | 7个月→11个月 | 时效性微调 | **认知修正** |

### 5.2 仍可通过"跑脚本"修复的

| 操作 | 影响维度 | 修复前 | 修复后 | 增益 | 难度 |
|------|----------|--------|--------|------|------|
| 修复 `limit_up_sentiment` 持久化 | limit_up_sentiment | F (1.0) | A (4.0) | **+3.0** | 低 |
| 重新下载 shareholder 全季度历史 | shareholder/processed | A (4.2/4.4) | A+ (4.8) | +0.4 | 中 |
| 长期积累 limit_up_* | limit_up_* | A (4.0) | A+ (4.6) | +0.6 | 低（需时间） |
| 长期积累 concept_blocks | concept_blocks/* | B+/A (3.8/4.0) | A (4.2) | +0.4 | 低（需时间） |
| 扩展 valuation 剩余 4 只 | valuation | A+ (4.8) | A+ (5.0) | +0.2 | 低 |
| 扩展 minute/5min 覆盖 | minute/5min | A (4.0) | A+ (4.6) | +0.6 | 中 |

### 5.3 需更换数据源的硬伤

| 问题 | 当前状态 | 替代方案 |
|------|----------|----------|
| news 历史仅 11 个月 | Sina + EastMoney THS 源 | 换 AKShare 新闻接口可回填至 2015 |
| comment 历史仅 2 个月 | AKShare comment 接口限制 | 该接口设计为近期评价，历史不可回溯 |
| shareholder 仅单季度快照 | EastMoney RPT_HOLDERNUMLATEST API | 换 Tushare `holder_number` 接口可获季度时间序列 |
| analyst 数据不可用 | 2 文件 + 列名编码损坏 | 重新用 AKShare/EastMoney 下载分析师评级 |
| cninfo 仅 254 只 | 巨潮 IP 被封 | 已切换到 EastMoney announcements 源（5530全覆盖） |

---

## 六、综合评分轨迹

```
原始数据  v2 (07-24)  →  v3 (07-25)  →  v4 (07-25深夜)  →  v5 (07-26)  →  补完缺口后
─────────────────────────────────────────────────────────────────────────────────────────────
A+ (19): daily×2, minute×2,   A+ (22): v4 全部 +
         guba×4, capital_flow×2,          industry_ranking.parquet,
         announcements×2,                 guba_sentiment(flat)
         block_trade×2, dividend×2,
         lockup×2, fund_daily,
         valuation, board_processed,
         industry_ranking_processed,
         macro

A  (19): minute_5/15,          A  (19): v4 A 扣除升入 A+ 的
         news_silver,                    + minute_5/15(新拆)
         margin, northbound,             + margin(降级)
         dragon_tiger, etf_flow,
         cninfo×2, lockup_upcoming,
         shareholder×2, concepts_processed,
         limit_up×4, index_conts,
         industry, fundamentals

B+ (6):  sentiment, pledge,     B+ (6):  sentiment,
         concept_blocks,                  pledge, concept_blocks,
         cninfo×2(新拆),                  cninfo×2(维持)
         comment_sentiment

B  (5):  news_raw, market_breadth,  B (5):  news_raw, market_breadth,
         analyst(C→降级),               analyst(降级), universe
         universe

C  (1):  analyst(新降级)         C (1):  analyst

F  (1):  limit_up_sentiment      F (1):  limit_up_sentiment

均分: 4.15(A) → 4.32(A) → 4.36(A) → 4.30(A) → ~4.55(A+)
```

> **v5 均分 4.30 不代表数据退步。** 拆分维度（minute 1条目→4条目，guba_sentiment 1→2，daily 1→2，cninfo 1→2）让低覆盖率子项独立呈现，分母从 45→53，产生了数学稀释。实质质量与 v4 持平，且纠正了 3 个认知偏差。

---

## 七、按模型可用性分类（v5 新增）

### 7.1 可直接用于模型训练的维度（日频 + 完整覆盖）

| 维度 | 列数 | 信号类型 |
|------|------|----------|
| `daily` K线 | 10 | 量价 |
| `fundamentals_daily` | 11 | 基本面（前向填充） |
| `valuation` | 5 | 估值 |
| `board_processed` | 27 | 打板/市场状态 |
| `industry_ranking_processed` | 34 | 行业截面 |
| `capital_flow_processed` | 21 | 资金流 |
| `guba_sentiment` (Gold) | 8 | 论坛情绪 |
| `sentiment` (news Gold) | 6 | 新闻情绪 |
| `announcements/sentiment/` | 7 | 公告情绪 |
| `macro` | 28 | 宏观 |

### 7.2 可用但需注意限制的维度

| 维度 | 限制 |
|------|------|
| `margin` | 仅 4,608 只（83% 覆盖），stock_code=NaN |
| `northbound` | 仅 3,327 只（60%），数据停在 2024-08 |
| `dragon_tiger` | 仅事件日有数据，ZI 填充后大部分为 0 |
| `etf_flow` | 扇区级（非个股级），需按行业映射 |
| `block_trade_processed` | 仅 5,111 只（92%），事件稀疏 |
| `dividend_processed` | 仅 5,282 只（96%），事件稀疏 |
| `lockup_processed` | 仅 5,125 只（93%），事件稀疏 |
| `shareholder_processed` | 仅单快照，不能做时序变化率 |

### 7.3 当前不可用的维度

| 维度 | 原因 |
|------|------|
| `limit_up_*` (4个池) | 仅 2 周历史，不足以覆盖训练窗口 |
| `limit_up_sentiment` | 空目录，数据从未落盘 |
| `comment_sentiment` | 仅 2 个月，覆盖<2% 训练期 |
| `concept_blocks` | 仅 11 天，需长期积累 |
| `analyst` | 2 文件 + 列名编码损坏 |
| `cninfo_announcements` | 仅 254 只（4.6%），已被 EastMoney announcements 替代 |

---

## 八、结论

> **2026-07-26 v5：** 首次完成 `data/a_shares/` 下 **50 个顶级项目、100% 递归子目录**的手工审计。53 个评分维度加权均分 **4.30/5.0 (A)**。
>
> v5 不是分数的简单累进，而是一次**精度升级**：
> - **3 个日期范围认知偏差纠正**：guba_sentiment 实际多 8.5 年历史、news 实际多 4 个月
> - **1 个系统性 bug 发现**：7 个目录 stock_code=NaN（不影响使用但有隐患）
> - **1 个维度降级**：analyst B→C（列名编码损坏确认为数据缺陷）
> - **8 个新拆分维度**让评分更精确地反映子项差异（minute 5min 的 34% 覆盖率不再被平均掩盖）
>
> **核心资产（13 个 A+ 维度 + 长历史）：**
> - daily K 线（2000 年起，5,530 全覆盖）
> - guba 全链路（2015 年起 raw→silver→Gold）
> - 行业/板块截面（2000 年起，22 行业 × 6,434 日）
> - 资金流 + 大宗交易 + 分红/解禁 已处理链
> - 宏观 28 维（2015 年起日频）

| 阶段 | 均分 | 等级 | A+ 数量 | F+C 数量 |
|------|------|------|---------|----------|
| v2 (07-24) | 4.15 | A | 10 | 3 |
| v3 (07-25) | 4.32 | A | 17 | 5 |
| v4 (07-25 深夜) | 4.36 | A | 19 | 4 |
| **v5 (07-26)** | **4.30** | **A** | **22** | **2** |
| 补完剩余缺口 | ~4.55 | A+ | 28 | 1 |

**剩余优先级最高的 3 个行动项：**

| # | 任务 | 影响 | 难度 |
|---|------|------|------|
| 1 | 修复 `limit_up_sentiment` 持久化 | F→A (+3.0 单维) | 低 |
| 2 | 重新下载 `shareholder` 历史季度（换 Tushare 源） | 解锁筹码集中度特征 | 中 |
| 3 | 长期 cron 积累 `limit_up_*` + `concept_blocks` | 打板/概念板块特征可用 | 低（需时间） |
