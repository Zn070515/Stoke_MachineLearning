# 东方财富股吧情感管道 — Design Spec

> 独立维度的散户论坛情感信号，遵循现有 3 层奖章 + FeaturePipeline merge 模式。

## Architecture

```
GubaSource (爬虫) → GubaStorage (存储) → FeaturePipeline (特征) → 模型
  ↓                    ↓                      ↓
列表页→详情页        三层奖章               guba_sentiment_*
80条/页,全量历史    Bronze→Silver→Gold      独立6列
```

股吧情感作为**独立特征维度**，不与现有新闻/公告合并。FeaturePipeline 新增 `use_guba` 开关和 `guba_df` 参数。无 guba 数据时优雅降级（全零列）。

---

## Component 1: GubaSource

**文件**: `stoke_ml/data/sources/a_shares/guba_source.py`

**类**: `GubaSource`

**公开方法**:
- `fetch_posts(stock_code, start_date=None, end_date=None, max_pages=10, fetch_bodies=True) -> DataFrame`

**实现细节**:
- 列表页 URL: `https://guba.eastmoney.com/list,{code}.html`，分页 `list,{code}_{N}.html`
- 每页 80 条帖子，`data-postid` 属性提取 post_id
- 详情页 URL: `https://guba.eastmoney.com/news,{code},{post_id}.html`
- 翻到最早日期或 `max_pages` 上限（取先到者）
- HTTP: curl-cffi + impersonate="chrome120"，复用现有 `SessionPool`
- 反爬: 复用现有 rate limiter (1.0s base + jitter)
- 时间戳保留 HH:MM:SS，存储为 `time` 列

**输出列**: `date, time, title, body, post_id, url`

**日期范围**: 按 `start_date/end_date` 过滤，当帖子日期早于 start_date 时停止翻页。

---

## Component 2: GubaStorage

**文件**: `stoke_ml/data/guba_storage.py`

**类**: `GubaStorage`

遵循和 `NewsStorage` 完全相同的接口和 3 层奖章物流：

| 层 | 路径 | 逻辑 |
|---|------|------|
| Bronze | `guba_raw/{code}.parquet` | 追加写入，按 (title, date) 去重 |
| Silver | `guba_silver/{code}.parquet` | PIT 对齐：datetime > 15:00 CST → next_trading_day() |
| Gold | `guba_sentiment/{year}/{month}/{code}.parquet` | 日度聚合 + ZI 填充无帖子日 |

**方法**:
- `save_raw(stock_code, df)` — 追加到 Bronze
- `load_raw(stock_code) -> DataFrame`
- `bronze_to_silver(stock_code) -> DataFrame` — PIT 对齐
- `save_silver(stock_code, df)` / `load_silver(stock_code)`
- `silver_to_gold(stock_code, analyzer=None) -> DataFrame` — 日度聚合
- `save_daily_sentiment(df)` — 分区写入 Gold
- `load_daily_sentiment(stock_code, start_date, end_date) -> DataFrame`

**PIT 对齐**: 帖子有精确时间戳，`datetime > 15:00` 的帖子通过 `TradingCalendar.next_trading_day()` 归入下一交易日。无时间戳时回退到自然日。

**日度聚合**: FinBERT 评分 body（无 body 时用 title），然后和 `NewsStorage.silver_to_gold` 相同的 6 列聚合逻辑。阈值 ±0.2 区分 pos/neg。

**ZI 填充**: 日度聚合后，用 `TradingCalendar.get_trading_days()` 补齐所有交易日，无帖子日填 0 + `has_guba_post=False`。

---

## Component 3: FeaturePipeline 扩展

**文件**: `stoke_ml/features/pipeline.py` (修改)

**新增常量**:
```python
GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]
```

**`__init__` 新增参数**: `use_guba: bool = True`

**`build_features` 新增参数**: `guba_df: pd.DataFrame | None = None`

**新增方法**: `_merge_guba(df, guba_df) -> DataFrame`
- 和 `_merge_sentiment` 相同模式：左连接 → fillna(0/False) → shift(1) 防泄漏
- guba 列纳入 `_engineer_features` 中的 temporal 特征生成（lags + rolling stats）

**降级**: `guba_df=None` 或 `use_guba=False` 时跳过，无 guba 特征的模型仍可训练。

---

## Component 4: 下载脚本

**文件**: `scripts/download_guba.py`

```bash
# 单只测试
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --stocks 600519 --max-pages 5

# 全量
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --max-pages 10
```

**CLI 参数**: `--stocks`, `--start`, `--end`, `--max-pages` (默认 10), `--sleep` (默认 1.0), `--skip-sentiment`, `--concurrent`

**流程**: 对每只股票 → `GubaSource.fetch_posts()` → `GubaStorage.save_raw()` → `bronze_to_silver()` → `silver_to_gold()` (含 FinBERT)

---

## Data Flow

```
GubaSource.fetch_posts()
  → 列表页爬取标题+时间+post_id
  → 详情页爬取正文
  → 返回 DataFrame[date, time, title, body, post_id, url]

GubaStorage.save_raw()  →  Bronze
  → 追加写入 guba_raw/{code}.parquet
  → 跨次去重 (title+date)

GubaStorage.bronze_to_silver()  →  Silver
  → 解析 datetime (date + time)
  → PIT 对齐: 15:00 后 → 下一交易日

GubaStorage.silver_to_gold()  →  Gold
  → FinBERT 评分 body (或 title 作为 fallback)
  → 日度聚合 (mean/std/count/pos_ratio/neg_ratio)
  → ZI 填充无帖子交易日

FeaturePipeline._merge_guba()
  → 左连接 on date
  → fillna(0.0) / fillna(False)
  → shift(1) 防泄漏
  → 产生 lag/rolling 特征
```

---

## Testing

- **单元测试**: `GubaSource._fetch_list_page()` 解析 HTML，返回正确列数和类型
- **集成测试**: 下载 600519 前 2 页 → 检查 Bronze/Silver/Gold 各层数据完整
- **特征测试**: `FeaturePipeline.build_features()` 含 guba_df → 验证 GUBA_COLS 存在且无 NaN
- **PIT 测试**: 15:01 的帖子 → aligned_date 应为下一交易日
