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
| 交易日 | trading day | 周一至周五，排除 A 股节假日（2001-2026 已验证官方公告；2027-2028 为前瞻估计，见 `calendar.py` `VERIFIED_UNTIL`） |
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

> **pct_change 语义**：当日值（`pct_change[t]` = t 日收盘已知的当日涨跌幅，**不 lag**）。它是 PK 列（`_discover_pk_columns`），`_engineer_features` 在 technical 计算前后保留/恢复 K 线 pct_change，`_merge_daily_aux` 的 skip 集显式排除 `pct_change`/`vol_change`——K 线派生列永不来自 aux 注入。

---

## 存储架构

### Medallion 三层架构

| 层 | 目录 | 分区策略 | 含义 |
|----|------|----------|------|
| **Bronze** (原始层) | `data/a_shares/news_raw/{stock_code}.parquet` | 按股票 | 爬取即存，追加模式，去重(title+date) |
| **Silver** (对齐层) | `data/a_shares/news_silver/{stock_code}.parquet` | 按股票 | PIT 时点对齐后，去重(title+aligned_date) |
| **Gold** (聚合层) | `data/a_shares/sentiment/{year}/{month}/{stock_code}.parquet` | 按年/月 | 日聚合情感特征，ZI 填充 |

### K线存储

`data/a_shares/daily/{code}.parquet` — 每股票一个完整文件的唯一 canonical 布局（v7 §一）。前复权 qfq 序列，每文件携带 `{code}.manifest.json` 契约 sidecar，正式读取要求 manifest 验证通过（`require_valid_manifest=True`）。

### 预构建特征存储

`data/features/{stock_code}.parquet` — `scripts/production/build_features.py` 一次性构建的全市场特征：5530 只 × 3744 列/股（516 基础 + 3228 时序展开），109GB。训练脚本直接读盘，特征工程与训练解耦。

### 数据契约与特征缓存

**数据契约** `stoke_ml/data/contract.py`（v6 §十六）— 每个数据集（daily_equity / margin / northbound / dragon_tiger / fundamentals）用一份冻结的 `DataContract` 描述 schema、主键、单位、复权口径、时区、日历与源优先级，下载器 / 存储 / 质量门禁 / 特征构建器共享同一份契约。质量门禁的 `contract_schema` 检查按 `DAILY_EQUITY` 验证每个 daily 文件（必需列、主键唯一、日期规则、单位符号），任一违反即 fail。

**特征缓存 manifest** `stoke_ml/features/cache_manifest.py`（v6 §十二 / v10 §十一 / v11 §十.1）— 每个预构建特征 parquet 携带 sidecar JSON manifest（`data/features/.manifests/{code}.json`），记录 git_commit、config_hash、feature_schema_hash、horizon、seq_len、panel_mode 与逐通道源文件指纹。`config_hash` 覆盖完整特征相关 config.yaml 区块（features / preprocessing / universe / fundamental，含技术指标开关、缺失处理、阈值、横截面归一化参数与源 effective-date/persistence 策略），同一 commit 下改 config 也会使缓存失效；start/end 单独以源文件 range 校验。`build_features.py` 缓存命中必须比较这些 hash；训练侧 `FeaturePipeline` 对 prebuilt 目录做 lineage 校验，`--require-feature-manifest`（默认开）时缺失 / stale（schema 漂移、git commit 不一致、config 漂移）直接 fail。v11 §十.1 起 manifest 额外指纹化**市场级共享输入**（macro_daily / market_env_daily / industry_returns / stock_sector_cache.csv / exchange_calendar/a_shares.parquet / 整个 etf_flow 目录，目录按内容整体哈希），任一变化即失效旧特征缓存；旧版缺共享指纹的 manifest 一律视为 stale 强制重建。面板构成工件（universe/ipo、universe/delisted、index_constituents_hist/membership）不在此列——它们改变进入面板的股票/日期而非每股特征值，由 fold 级 universe/membership hash（train_panel FoldResearchContext）覆盖。

### Formal Asset protocol（§十七 / §十九-9，T9 收口）

文件级资产治理协议 —— `stoke_ml/data/asset_contract.py` 为辅助数据通道提供 `DataAssetContract`（文件级契约）+ 原子 manifest sidecar（`{parquet}.manifest.json`）。每份契约钉住七个方面：

1. **Content identity** — `schema_hash`：列名 + dtype + 值的校验和，parquet 往返读写稳定；文件被改则不再匹配。
2. **Source identity** — `source`：取自 `df.attrs["source"]`（未声明则 `"unknown"`），仅 provenance，不 cross-check。
3. **Coverage** — `start`/`end`：extent 列（`date` / `report_date` / `announce_date`…）的 min/max；DatetimeIndex 广播文件（industry / market_env）回退到 index。
4. **Effective-date policy** — `effective_date_policy`：`record_date` / `event_date` / `post_close_next_trading_day` / `index_date`。
5. **Vintage status** — `vintage_source` / `vintage_transform` / `vintage_pit`：`contract_for_channel` 从 `channel_vintage` 声明自动填入，manifest 携带与训练 admission 相同的标签。
6. **Schema** — `column_contract`（仅 provenance；强制是另一 gate）。
7. **Atomic commit** — temp + `os.replace` 原子写；parquet 与 manifest 两次 rename 之间崩溃留下的 stale 对由下次 validated read 捕获。

**Formal read**（`require_valid_manifest=True`，对应 `train_panel_panel._enforce_formal_manifests`）：manifest 缺失或失配即拒绝 —— `check_asset_read` 抛错；默认（lenient）读：无 manifest = legacy 文件照读（debug 日志），有 manifest 但失配 = warning + 照读。

**通道采用状态（T9 收口）**：
- `industry_returns.parquet`（`INDUSTRY_ASSET`）：`download_industry.py` 原子写 + manifest；formal gate 校验。
- `market_env_daily.parquet`（`MARKET_ENV_ASSET`）：`build_market_env.py` 原子写 + manifest（含 price/account 分列 `parts`）；formal gate 校验。
- `forecasts.parquet` / `express.parquet`（`EARNINGS_ASSET`，`download_earnings.py` `_accumulate` 对两个文件写 manifest）：**governance-only** —— earnings 是 `latest_revised` 源（revision-safe 不要求，manifest 只记录 provenance / 统一治理），不进 formal required 集。
- `stock_industry_map.parquet`：无 manifest（无 consumer / formal gate 要求，T9 不改）。
- **cninfo announcement-sentiment 路径**（`a_shares/cninfo_announcements/sentiment/`）：无 storage/manifest 支持 —— formal 模式下显式拒绝（`use --prebuilt or add a DataAssetContract writer`），T9 维持显式排除，不补 writer。

### 行业分类真 PIT（v19 P0#1 + 行业链 provenance closeout，CNINFO 事件溯源）

`scripts/production/download_sector_membership.py` 拉取 CNINFO
`stock_industry_change_cninfo` 逐股行业分类变更事件（列：新证券简称 / 行业
门类 / 机构名称 / 证券代码 / 变更日期 / 分类标准…），构建真 PIT 的
`a_shares/sector_membership.parquet`（列 `[date, stock_code, sector_code,
sector_name]`，CSRC 门类 A-S 级别，合并证监会 2001 / 2012 / 中国上市公司协会
改名；逐事件区间边界 + dropna 后建边界，不做 present-backfill）。逐股缓存
在 `a_shares/sector_membership_pit/_stocks/{code}.json`，支持断点续爬
（§十九：缓存携带 `cache_version`/`parser_hash`/`source`/区间，不匹配即整体重爬一次；
§十八：CNINFO 空结果仅认 `KeyError('变更日期')` 签名，其它列索引 KeyError 一律 re-raise）。
run manifest 记录 `coverage_by_year`——**活动股分母**（§十六，读每只 daily
manifest 的 start/end 算当年 active 股票数），早期年份远高于旧 universe 分母
（2002+ 均 ≥ 97.8%）。`complete` 仅在 fetch + 正式 daily 读取
（`require_valid_manifest=True`）+ 展开全部成功后授予（§十五）。

`download_industry_ranking.py` 优先读这份 membership（date+stock_code INNER
join → 诚实剔除未分类日），读取前用 `validate_asset_manifest` 校验（§十三）。
**缺 membership 时默认 fail-closed（P0.2）**：必须显式传
`--allow-snapshot-sector-fallback` 才回退历史快照，且产物 manifest 标
`pit_alignment="proxy"`（绝不进 strict headline）。产物写 `INDUSTRY_RANKING_ASSET`
manifest（§十四）：`upstream_roots.daily`（daily manifest-root Merkle digest）+
`upstream_roots.sector_membership`（文件 sha256）+ transform_code/config_hash。
formal gate 用同一 `compute_lineage` 重算比对（§十四）——`sector_membership`
变了但 industry_ranking 没重建 → `upstream_roots.sector_membership` 翻转 →
STALE → market_env 链 fail（Tuesday bug 兜住）。`market_adv_ratio` 是
**证监会门类广谱涨跌比率**（§二十，A-S 门类等权正收益占比，非单行业指标）；
`build_market_env` 消费时用 `check_asset_read(require_valid_manifest=True)`
校验 ranking，并把 pit 声明进 `parts.price.industry_advance_pit`（manifest 缺
key 时保守标 `proxy`，绝不静默升级为 `verified`）。formal 的 `market_env` 运行
还要求每年 sector 活动股覆盖 ≥ `SECTOR_COVERAGE_THRESHOLD=0.80`（§十七，
缺失年份按 0.0 fail-closed）。

注意：`industry` 通道的 `channel_vintage` 声明仍标 `pit_alignment=proxy`
（v18 §二十-9，代码未改）；数据已 PIT，正式 vintage 标签升级留待后续独立决策。

### v19 日线库迁移 + aux 通道重建（#85 / #88 / #100 / #103）

**迁移**（#85 Phase 1/2）：daily 库归一化手/股混合单位（逐行 ×100）并把
provenance 统一为 qfq。迁移改变价格与 `pct_change`，使所有**内嵌 daily 价格**
的下游 aux `*_processed` 通道与预构建特征全部失效（board / block_trade /
industry_ranking / dividend / lockup / shareholder 等）。

**重建**（#88 / #100）：用 `preprocess_new_data.py --type <ch>` 重跑各通道。
- pandas 3.0 日期 dtype 严格化（ms/us merge `MergeError`）在 read 层修复
  （`stoke_ml/data/date_normalize.py::as_date_us`）：daily / aux 读盘后统一
  `datetime64[us]`，flat 优先路径与 year/month 分区路径一致。
- 结构性 schema 演化（如 SectorBroadcaster 以派生指标替换原始排行列
  `leader_change/n_stocks/ret_std/sector_name`）触发 replace_range 的
  dropped_cols 拒绝 → `--force` 显式绕过；date-loss 审计全过（重建产物
  missing_vs_daily=0，完整覆盖 canonical daily 日历；manifest missing_dates
  均为旧文件独有行、无消费方）。
- 事件通道（board/block_trade/dividend/lockup/shareholder）结构性稀疏，
  formal 默认 strict 质量门全拦 → `--allow-degraded` 写降级产物。

**Gate aux_close 语义**（#103）：`check_aux_close_aligned` 的 close 比较分两类
——per-date-price 通道（block_trade/board/sector/shareholder）**逐行**比较；
forward-filled-close 通道（dividend/lockup）的 close 是事件日 close 前向填充的
state byproduct（`aggregator._fill_to_daily`），只在**真实事件行**
（`dv_days_since==0` / `lu_days_since==0`，事件时间特征）比较——事件行 close
与 daily 精确相等，basis-drift canary 保留。

**重建后全量质量门**（2026-08-10，`data_quality_gate.py` 全扫，报告
`reports/data_quality_gate.json`）：datasets / daily_internal /
`aux_pct_aligned`（max_diff=0，board + sector 全量）/ `aux_close_aligned`
（max_abs_diff=0，6 通道全量）/ sparsity / ohlc_sanity / contract_schema /
manifest 全 PASS；**唯一 FAIL 是 `feature_pct`**（25 只，与迁移前同一批代码、
max_diff 逐字一致——特征层未重建）。

**已知残留**（#89 pending）：109GB 预构建特征（`data/features/` 全量 +
`data/features_panel/` 面板）未重建 → gate `feature_pct` 仍 FAIL（特征层
pct_change 与 daily 不一致来自迁移前旧特征）。#89 重建特征面板后收口。

### 正式研究 Prebuilt 主线（§P2-16，T14 收口）

正式研究的 canonical 流程（各阶段产物在括号中）：

**Raw Assets → Formal Asset Gate → Prebuilt Feature Artifact → Formal Feature Manifest → Streaming PanelStore → Training**

即：原始数据下载（`download_*.py`）→ 质量门报告（`reports/data_quality_gate.json`，formal 必检）→ `build_features.py --panel-mode` 一次性构建面板特征（`data/features_panel/`）→ 每份特征 parquet 携带 sidecar manifest（`--require-feature-manifest` 默认开）→ 可选 `--panel-store` 流式固化 → 训练直接读 `--prebuilt data/features_panel`（`train_panel.py`）。面板特征只构建一次，训练循环不再在线做特征工程。

门禁规则（`_require_prebuilt_mainline`，v19 P0#3 收口，取代 T14 的 >1000 阈值规则）：**formal 研究**（quality gate 强制 且 非 `--no-formal`）一律要求 `--prebuilt`（prebuilt 特征主线）或完整 `--panel-store`，不再有股票数阈值逃生口——缺则启动即拒绝，提示先构建 prebuilt 特征。`--universe all` 无论模式一律拒绝（5530 只无法在 RAM 内做特征工程，§七-P0）。在线特征工程降级为 debug / smoke / 探索性路径（逃生口 `--no-formal` / `--no-require-quality-gate`）。minute 模式无 prebuilt 产物，豁免。

### Formal 训练主线（v19 P0#3 收口，取代 v18 §二十-4 的 >1000 阈值声明）

Formal 研究统一走 Prebuilt Feature → PanelStore → Training 主线
（Raw Assets → Formal Asset Gate → Prebuilt Feature Artifact → Feature
Manifest → Streaming PanelStore → Train）。Live FeaturePipeline 保留为
Feature 开发 / Debug / Smoke / 探索性快速实验路径。

- v18 曾以 `_PREBUILT_MAINLINE_THRESHOLD`（解析后股票数 > 1000）作为 formal
  的阈值逃生口；v19 P0#3 收口——formal 研究**一律**要求 prebuilt / 完整
  store，无股票数阈值。live 数学输入（raw → in-memory transform）与 prebuilt
  （raw → preprocess_new_data → processed → build_features）不是同一套输入，
  formal 大实验不维护两套 Feature Source 语义。
- 逃生口是 `--no-formal`（探索性）/ `--no-require-quality-gate`（dev smoke），
  不是股票数阈值。`--universe all` 无论模式一律拒绝。
- 不删除 live 路径——只做门禁 + 文档。

### Lockbox 声明：revision-safe headline_v1 含 industry proxy（v18 §二十-9）

`headline_v1`（默认 formal profile）是 **revision-safe** 档，不是
headline-strict：其 required channel 里的 `industry` 历史行业分类带
`pit_alignment=proxy`（源：immutable_snapshot 行业快照 × formula_versioned
分类算法）。正式 headline_v1 结论的 feature set 内含该 industry proxy。
Lockbox 最终主结论二选一：(a) 新建 headline_strict_v1 剔除 industry；(b)
沿用 headline_v1 并在研究报告中显式声明该 proxy。当前按 (b) 声明，不建新
profile。

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
| ZI 方法（通道区分，§九-4） | 事件型通道（新闻/公告/龙虎榜/大宗...）缺失日填 0 + `has_*=False`；状态型通道（两融余额/北向持仓/估值/基本面/宏观利率/市场环境/股东/质押）缺失日向前填充 ffill（状态缺失=未变，绝不归零） |
| 无时间戳时 | 当前所有新闻仅有日期无时间戳，视为同日新闻 |

---

## 特征工程

| 术语 | 含义 |
|------|------|
| seq_len | 回看窗口长度，**60 个交易日** |
| target_horizon | 预测目标，XGBoost=1 (次日涨跌)，Panel=5 (5日涨跌) |
| flat_mode | XGBoost 模式：将 (60, n_features) 展平为 (60*n_features,) |
| panel_mode | Panel 模式：保持 (N_stocks, T, D) 三维Panel结构，截面归一化 |
| 技术指标 (technical) | MA/EMA/MACD/RSI/Bollinger/ATR/OBV/volume_ratio |
| 趋势评分 (scoring) | 规则型 trend_level（0-6）/ buy_signal（0-5）/ bias |
| 时序特征 (temporal) | 滞后项 lag(1/2/3/5/10/20) + 滚动统计 rolling(5/10/20/60) + 日历特征 |
| 情感特征 (sentiment) | SENTIMENT_COLS 全部加入 lag 和 rolling |
| 股权质押 (pledge) | PLEDGE_COLS 5列：pledge_ratio / pledge_margin_dist / pledge_risk / pledge_count_20d / has_pledge。公告日期 keyed，PIT lag(1) |
| 指数成分 (index_membership) | INDEX_MEMBER_COLS 3列：is_index_member / n_indexes / idx_change_30d。无 index_weight（Baostock 无历史权重） |
| 市场环境 (market_env) | MARKET_ENV_COLS 7列：high_low_ratio / mkt_cap_total_z / avg_account_cap_z / investor_new_num / investor_new_z / market_adv_ratio / market_turnover_z。全市场日频，无涨停温度列 |
| 宏观制度 (macro regime) | MarketEnvRefiner 产出 menv_* 49列：shibor_z / fx_z / cpi_z / 期限利差 / M1-M2 扩散 / regime_z |
| 龙虎榜席别 (seat) | DRAGON_TIGER_SEAT_COLS 4列：lhb_is_wave / lhb_is_sustained / lhb_is_drop / lhb_count_5d，属 `use_dragon_tiger` 维度 |

**特征工程顺序**（不可改变）：
1. 合并情感列（左连接 date）
2. 按通道策略填充缺失日（事件型 ZI，状态型 ffill）——§九-4
3. 技术指标
4. 趋势评分
5. 时序特征（滞后+滚动+日历）
6. （FE v2）新家族（pledge / index_membership / market_env / macro regime）与其它辅助维度同一管道：步骤 1 左连接合并，步骤 2 按通道策略填充（事件型 ZI，状态型 ffill），统一 PIT lag(1)

---

## 模型

| 术语 | 含义 |
|------|------|
| Panel Model (VSN + xLSTM) | 主力模型：Panel联合训练，多任务学习 (方向+涨跌幅+波动率)，RTX 4090 |
| XGBoost baseline | 展平特征 + 梯度提升树，Phase 1 |
| Panel 基线 (Ridge / LightGBM / MLP / naive momentum) | 同 inner_val 日历口径的截面对照基线 (v8 四-2)；评估器版本 `evaluator_version 2026-08-05` |
| LSTM | 2层单向 LSTM + PyTorch Lightning，Phase 2 |
| class_weight | 处理涨跌样本不均衡，自动计算 neg/pos |

### Panel Model 架构组件

| 术语 | 含义 |
|------|------|
| VSN (Variable Selection Network) | 变量选择网络 — 在每个时间步对输入特征做软特征选择 (GRN + softmax) |
| GRN (Gated Residual Network) | 门控残差网络 — 基础构建块，ELU + GLU + 残差连接 + LayerNorm |
| GLU (Gated Linear Unit) | 门控线性单元 — `(X·W₁ + b₁) ⊙ σ(X·W₂ + b₂)`，控制信息流通 |
| sLSTM (scalar LSTM) | 指数门控 + memory mixing，序列处理，适用于短序列金融数据 |
| mLSTM (matrix LSTM) | 矩阵记忆 + 协方差更新，并行化处理全局模式 |
| Static Encoder | 静态特征通过4个GRN编码为 c_e/c_h/c_vs 上下文向量，分别注入时序编码和特征选择 |

### Panel Model 多任务输出

| 任务 | 损失函数 | 说明 |
|------|----------|------|
| 方向分类 (3类) | CrossEntropyLoss | 下跌(0) / 横盘(1) / 上涨(2)，阈值 ±0.003×√horizon |
| 涨跌幅回归 | AdjMSELoss (γ=0.1) | 符号感知MSE：符号错误惩罚11倍，符号正确仅0.1倍权重 |
| 波动率回归 | MSE | 未来horizon日波动率 (std of daily returns) |
| 截面排序 | PairwiseRankingLoss (tau=0.1, rank_weight=0.1) | 可微成对排序 hinge 损失（同日期内成对比较）+ spread 惩罚 |

### Panel Model 损失加权

| 术语 | 含义 |
|------|------|
| UncertaintyLoss | Kendall et al. 2018 — `0.5 × Σ( task_loss/exp(log_var) + log_var )`，自适应多任务权重 |
| log_var | 每个任务的可学log-方差参数，σ大→权重小，clamp in [-3, 10] |

### Panel 数据格式

| 术语 | 含义 |
|------|------|
| Panel | (N_stocks, T_timesteps, D_features) 三维数组，区别于单stock (T, D) |
| Static features | PIT 静态特征 (9维)：price_60d_q / amt_60d_q（60日滚动均值的截面分位）/ listing_days / board_*（6维交易所板块 one-hot：sh_main/star/sz_main/chinext/bse/unknown）。全部可由决策日已知数据推导（v8 §三-2）。~~industry_code~~ 已排除：唯一可用的 stock→industry 映射是当前快照（sector_map.json / stock_sector_cache.csv），无 PIT 成分历史，作静态特征会把今日分类回填到历史行 |
| Past Known (PK) | 已知历史特征 (255维)：价格、技术指标、情感、资金流等，含close用于target计算 |
| Past Observed (PO) | 观测历史特征 (1418维)：换手率、振幅、涨跌幅等，不含close。维度由当前预构建特征面板动态决定 |
| Cross-sectional normalization | 跨股票截面归一化：按日期 groupby → z-score，解决不同股票量纲差异 |
| Per-stock target normalization | 按股票z-score归一化回归target，使各股票在MSE loss中等权重 |

### TFT 训练配置

| 术语 | 含义 |
|------|------|
| Growing-Window Walk-Backward | 训练窗口从 [0, val_start−purge) 逐 fold 增长，非重叠验证 fold（step=val_len），训练和验证之间有 seq_len 的 purge gap（从最新数据向过去倒排 fold + 末尾 lockbox 保留） |
| horizon | 前向回报窗口（交易日），默认5天。方向阈值缩放 √horizon |
| Grad Accum | 梯度累加 (默认4步)，等效增大batch size |
| AMP (Automatic Mixed Precision) | 混合精度训练，BF16/FP16前向+FP32权重 |
| ReduceLROnPlateau | 监控 val_loss (非 train_loss)，factor=0.5, patience=10 |

---

## 评估

| 术语 | 含义 |
|------|------|
| **MCC** (Matthews Correlation Coefficient) | 主要评估指标，适用于不平衡二分类 (XGBoost/LSTM) |
| **IC** (Information Coefficient) | Spearman Rank IC — 截面排序能力，Panel Model的主要评估指标。每日计算 pred vs actual 的秩相关，取均值 |
| **PairwiseRankingLoss** | 可微成对排序 hinge 损失：同日期内 pairwise 比较 sign(Δtarget)·Δpred，hinge(max(0, margin−…)) + spread 惩罚（防预测坍缩） |
| 方向分类 (3类) | Panel: 下跌(0) / 横盘(1) / 上涨(2)，阈值 ±0.003×√horizon |
| Walk-Forward 验证 | 固定窗口滑动验证，严格时序拆分，**绝不打乱** |
| Growing-Window Walk-Backward | Panel: 训练窗口增长 / 非重叠 fold（step=val_len）/ inner_val 选择最佳 epoch + outer_test 单次评估 / seq_len purge gap |
| Sharpe Ratio | 年化夏普 = (期均收益/期收益标准差) × √(252)，sleeve 账户按日历日收益年化（交错 sleeve 去重叠） |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 = 正收益交易占比 |
| Profit Factor | 盈亏比 = 总盈利/总亏损 |
| Top-K Portfolio | Panel评估方法：每日按预测收益排序选top-K (默认20)，等权组合，逐日再平衡 |

**Walk-Forward 参数**: 
- XGBoost/LSTM: 2年训练 / 3月验证 / 3月步长
- Panel: 增长式训练窗口（[0, val_start−purge) 逐 fold 增长）/ 非重叠 fold（step=val_len）/ inner_val 选择最佳 epoch + outer_test 单次评估 / purge=seq_len（walk-backward，从最新数据倒排 + 末尾 lockbox 保留）
- **Lockbox 前（P1 precondition）**：对每个 PanelStore 主动执行一次完整 chunk verification —— `PYTHONPATH=. ./.venv/Scripts/python scripts/production/verify_panel_store.py <panel-store-dir>`，exit 0 方可作为 lockbox 训练输入；失败 exit 非 0，store 需重建（§十九-12）
- Panel OOS 连续账户：把全部 fold 的 OOS 预测按时间排序接到一个账户上重放（fold 边界处切换模型），最终 Sharpe/MDD/CAGR 只取该连续账户 → 产出 `oos_continuous.parquet` / `oos_continuous_ledger.parquet`（替代逐 fold 各自归一的 NAV）
- 多重试验修正（§十五-1）：PSR（Probabilistic Sharpe，非正态 SR 显著性）/ DSR（Deflated Sharpe，按项目累计实验数 N 与试验 Sharpe 离散度折价）/ Block-bootstrap max-mean reality check（best-of-K 相对基准的重采样 p 值；非完整 Hansen SPA，§十二.4），报告同时输出 `long_psr / long_dsr / ls_psr / ls_dsr / bbmm_*`，连续 OOS 账户也带 `psr / dsr / dsr_n_trials`。实验注册表落在仓库根 `reports/experiments/experiment_registry.json`（每次训练原子追加一行，带 `experiment_signature` = SHA1(data_manifest_hash, feature_schema_hash, universe_hash, model_hash, horizon, objective)；同签名同 outdir 去重，异 outdir 各计一次；baseline 各占一个 `baseline-{name}` trial；损坏时抛错而非静默复位）。DSR 的 N = 历史注册表中**不同实验签名**的计数（当前签名不重复计），DSR 的试验 Sharpe 离散度来自历史注册表 OOS Sharpe 分布（不足 2 条时退化为文档化的 block-bootstrap 代理），来源以 `dsr_trial_variance_source` 标注

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
| target_horizon | 1 (XGBoost), 5 (Panel default) | config.yaml / PanelConfig.horizon |
| Panel hidden_dim | 128 | PanelConfig.hidden_dim |
| Panel xlstm_num_blocks | 2 | PanelConfig.xlstm_num_blocks（train_panel CLI 默认） |
| Panel batch_size | 128 | PanelConfig.batch_size（train_panel CLI 默认） |
| Panel lr_warmup_epochs | 5 | PanelConfig.lr_warmup_epochs |
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
| use_pledge / use_market_env / use_index_membership / use_market_env_refine | 默认 True | pipeline.py 构造器；build_features.py CLI 控制 |
| PLEDGE_COLS / INDEX_MEMBER_COLS / MARKET_ENV_COLS / DRAGON_TIGER_SEAT_COLS | 5 / 3 / 7 / 4 列 | pipeline.py |
| use_limit_up | False（deferred） | pipeline.py 构造器 |
| market_env 源文件 | market_breadth/market_env_daily.parquet | pipeline.py → _merge_market_env |
| macro 源文件 | macro/macro_daily.parquet | pipeline.py → _merge_macro |

---

## 环境方案（Windows CUDA 开发 / Linux CPU CI，§十三-3）

依赖唯一事实源是 `pyproject.toml`；`uv.lock`（`uv lock` 生成，universal 锁，覆盖 win32/linux/emscripten × python 3.12/3.13）提交入库，`uv lock --check` 校验新鲜度。改任何依赖后必须 `uv lock` 重新生成。`.python-version` = 3.12 固定 uv 的 Python 版本（本地、CI 一致），避免 `uv sync` 自动挑到 3.13。

- **Windows CUDA 开发机**：本地 `.venv`（Python 3.12.10，`torch==2.11.0` CUDA 版，RTX 4090）。所有命令经 `PYTHONPATH=. ./.venv/Scripts/python`，禁裸 `python`（Anaconda 缺依赖）。本地 CI 镜像：`PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py`。
- **Linux CPU CI（GitHub Actions）**：各 job 用 `astral-sh/setup-uv@v4` + `actions/setup-python@v5`(3.12)，`uv sync --frozen --extra <组>` 从提交的 `uv.lock` 装对应依赖组，`uv run python -m pytest ...` 跑测试。组 → extras 映射见 `.github/workflows/ci.yml`（core-fast=dev；storage-parquet=+data-adapters；ml=+ml/nlp/ta/data-adapters；slow-nightly=全组；optional-online=+online/data-adapters）。
- `requires-python = ">=3.12"`：声明的 `pandas-ta>=0.3` 需要 >=3.12，声明 3.10 会让依赖集形式不可满足而阻塞 `uv lock`；本机 venv 与 CI 均为 3.12。

---

## 反模式 (Anti-Patterns)

- ~~随机打乱时序数据~~ → 必须用 WalkForwardSplitter，按时间顺序拆分
- ~~用收盘价预测收盘价~~ → 预测的是次日涨跌**方向**（0/1），不是价格
- ~~在全部数据上 fit StandardScaler~~ → 只在训练窗口上 fit，验证窗口仅 transform
- ~~在全量历史上算 normalization 再拿去训练早期窗口~~ → 已审计（v8 §三-1）：所有 scaler/normalizer 均为 PIT 安全——CrossSectionNormalizer 按日期截面、RobustScaler/OutlierDetector 按滚动/回溯窗口、MissingImputer 因果插值，fit 均无状态；每个 `PreprocessingStep` 记录 `fit_start`/`fit_end`（fit 输入日期范围），`tests/preprocessing/test_pit_fit_range.py` 用「追加未来行不改过去输出」的截断不变性测试兜底
- ~~static 特征用"现在回填历史"的值~~ → 已审计（v8 §三-2）：static 9 维全部由决策日已知数据推导（price_60d_q/amt_60d_q=60日滚动均值→截面分位、listing_days、board_* 交易所板块 one-hot 6 维），财务/估值列（pe/pb/roe…）经 disclose_date 前向填充 + `_batch_fill_shift` 滞后 1 日 + 滚动 252 日分位（`FundamentalRefiner`），均为 PIT；`industry_code` 因无 PIT 成分历史源（`sector_map.json`/`stock_sector_cache.csv` 只是当前快照）已从 `_PIT_STATIC_COLS` 移除，`tests/features/test_static_feature_pit.py` 用截断不变性 + 滞后性测试兜底
- ~~"情绪分析"~~ → 统一用"情感分析"（sentiment）
- ~~裸 `python`~~ → 必须 `PYTHONPATH=. ./.venv/Scripts/python`（系统 Anaconda 缺依赖）
- ~~用 `use_limit_up=True`~~ → limit-up 生态 19 列已定义但未接线（`use_limit_up=False`），勿在训练中开启
- ~~让 aux 注入 K 线派生列~~ → `technical.compute_all` 会丢弃 pct_change/vol_change 中间列；若不恢复，`_merge_daily_aux` 会把 aux（如 board/industry ranking）自带的同名 stale 列注入当个股涨跌。主 K 线可派生的列，合并前必须在主 df 上恢复，并在 merge skip 集显式排除
