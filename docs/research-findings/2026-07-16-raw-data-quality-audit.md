# 原始数据质量全面审计

> 2026-07-16 | 对 `data/a_shares/` 下 25 个数据目录的完整性、时效性、API 可用性进行全面审查
> 包含 API 实际测试验证结果

## 审计方法

1. 遍历所有 25 个数据目录，统计文件数、行数、日期范围、列结构
2. 对照 8 个数据源实现 (`stoke_ml/data/sources/a_shares/*.py`)，追溯根因
3. 逐一测试 API 端点可用性（历史日期支持、分页限制、字段完整性）
4. 交叉验证 Web 搜索的 API 文档和已知 issue

## 问题分类（经 API 实测验证）

### 🔴 严重：数据源基本不可用

| # | 数据源 | 现状 | 实测根因 |
|---|--------|------|----------|
| 1 | industry_ranking | 0 文件 | API 超时未响应（待重新测试） |
| 2 | lockup_upcoming | 0 文件 | API 可用（90 只股票有 upcoming），下载脚本未执行到此处 |
| 3 | limit_up_zt/dt/yzt/zb | 仅 ~2 周 | **API 仅支持近期 ~2 周**，push2ex 不返回 2026-01 之前的数据 |
| 4 | concept_blocks | 仅 2 天 | **API `slist/get` 只返回实时快照**，无历史支持 |
| 5 | shareholder | 每股票 1 行 | `RPT_HOLDERNUMLATEST` 只返回最新一期，4 个替代 reportName 全部返回 0 |
| 6 | northbound | 2024-08 停止 | 监管变更（2024-08-19 起停批每日净流量），不可修复 |

### 🟡 中等：数据可用但质量不足

| # | 数据源 | 现状 | 根因 |
|---|--------|------|------|
| 7 | capital_flow | 无 tier 分层 | EastMoney push2his 下线，Sina 只提供总净额；AKShare 连接被拒 |
| 8 | lockup | 历史稀疏 | API `RPT_LIFT_STAGE` 返回数据有限（部分股票 0 条） |

### 🟢 健康：覆盖完整

daily / dividend / valuation / fundamentals / margin / dragon_tiger / announcements / macro / guba / news / xueqiu / industry(2文件，设计如此)

## API 实测记录

### limit_up push2ex API

| 测试日期 | 结果 | 说明 |
|----------|------|------|
| 2026-07-14 | 81 rows | 正常 |
| 2026-07-01 | 152 rows | 正常 |
| 2026-01-05 | **0 rows** | 不支持 |
| 2025-07-15 | **0 rows** | 不支持 |
| 2024-07-15 | **0 rows** | 不支持 |
| 2023-07-17 | **0 rows** | 不支持 |

**结论**：EastMoney push2ex 涨停池接口仅支持近期约 2 周数据。历史数据需要 Tushare `limit_list_ths`（需 8000+积分，从 2023-11-01 起）。

### AKShare stock_zt_pool_em

| 测试日期 | 结果 |
|----------|------|
| 2026-07-14 | 81 rows（正常） |
| 2025-01-06 到 2024-01-02 | 全部 0 rows |

**结论**：AKShare 封装了相同的 EastMoney 底层 API，限制一致。

### shareholder RPT_HOLDERNUMLATEST

尝试的替代 reportName（全部返回 0）：
- `RPT_HOLDERNUMLIST` → 0
- `RPT_HOLDERNUMLISTHIST` → 0
- `RPT_HOLDERNUMSH` → 0
- `RPT_F10_SHAREHOLDER` → 0

**结论**：EastMoney datacenter 无公开的历史股东户数 reportName。需要 Tushare Pro 或 AKShare `stock_zh_a_gdhs`。

### lockup RPT_LIFT_STAGE

- 历史数据（无日期过滤）：正常返回
- Upcoming（FREE_DATE='2026-07-16'~'2026-10-14'）：90 只股票有 upcoming 数据
- API 完全可用，下载脚本只需重新运行

### concept_blocks slist/get

- API 返回正常（2026-07-16: 798 股票各有归属板块）
- 但 `date` 硬编码为 `pd.Timestamp.now()`，无法查询历史
- 唯一的解决办法：每日定时运行，积累历史面板

## 修复可行性与方案

### 立即可修复（无需外部依赖）

| # | 问题 | 方案 |
|---|------|------|
| 1 | lockup_upcoming 为空 | 重跑 `download_datacenter.py --type lockup` |
| 2 | industry_ranking 为空 | 测试 API → 修正参数 → 重跑 `--type industry_ranking` |
| 3 | 涨停特征缺失 | 从 K 线计算 `is_limit_up`（日收益 > 9.5%），替代 BoardBroadcaster 对 pool 数据的依赖 |

### 需要 Tushare Pro 令牌

| # | 问题 | Tushare 接口 | 积分要求 |
|---|------|-------------|----------|
| 4 | 涨停历史 | `limit_list_ths` | 8000+ |
| 5 | 股东户数历史 | `stk_holdernumber` | 2000+ |
| 6 | 资金流分层 | `moneyflow` | 2000+ |
| 7 | 概念板块历史成分 | `concept_detail` | 500+ |

### 需持续运营

| # | 问题 | 方案 |
|---|------|------|
| 8 | concept_blocks 仅 2 天 | 每日 cron 运行积累 |
| 9 | northbound 停更 | 用 QFII 季度持仓 / 南向资金作为代理变量 |

## K 线替代涨停特征（无需外部数据）

以下特征可直接从 daily K 线计算，不依赖 limit_up pool 数据：

| 特征 | 计算方式 |
|------|----------|
| `is_limit_up` | `pct > 9.5` (主板) / `pct > 19.5` (科创/创业) |
| `limit_up_streak` | 连续 `is_limit_up` 天数 |
| `seal_strength` | `(high == close) & is_limit_up` → 一字板强度 |
| `gap_up_pct` | `(open - prev_close) / prev_close` > 2% |
| `limit_up_count_20d` | 20 日涨停次数 |

## 修复记录

| 日期 | 问题 | 操作 | 结果 |
|------|------|------|------|
| 2026-07-16 | 6 项 | API 实测 + 审计完成 | 见上表 |
| 2026-07-16 | industry_ranking | API 改用 urllib 取代 curl-cffi | 修复后返回 100 行/日 |
| 2026-07-16 | 涨停特征 | 新增 4 个 K 线特征 + TFT 路由 | is_one_word_board, seal_quality, limit_up_count_5d/20d |
| 2026-07-16 | lockup_upcoming | 重跑 download_datacenter --type lockup | 进行中 (800 stocks) |
