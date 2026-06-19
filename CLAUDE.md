# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always** use the venv Python and set `PYTHONPATH=.`:

```bash
PYTHONPATH=. ./.venv/Scripts/python <script>
# NEVER use bare `python` — it resolves to Anaconda which lacks dependencies.
```

### Data Pipeline

```bash
# Download K-line for all CSI 300+500 stocks (798 stocks, 2015–2026)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_data.py

# Download news + sentiment for all stocks on disk
PYTHONPATH=. ./.venv/Scripts/python scripts/download_news.py --max-pages 5 --sleep 2

# Single stock test
PYTHONPATH=. ./.venv/Scripts/python scripts/download_news.py --stocks 600519 --max-pages 3
```

### Training

```bash
# XGBoost baseline (flat features, walk-forward validation)
PYTHONPATH=. ./.venv/Scripts/python scripts/train_baseline.py --stock 000001
PYTHONPATH=. ./.venv/Scripts/python scripts/train_baseline.py  # all stocks

# LSTM (sequence model, PyTorch Lightning)
PYTHONPATH=. ./.venv/Scripts/python scripts/train_lstm.py --stock 000001 --epochs 50
```

### Testing

Tests directory exists but is empty (`tests/{crawler,data,evaluation,features,models}/`). No pytest config yet. `pytest` is available in the system Anaconda but tests must run under `.venv/Scripts/python -m pytest`.

## Architecture

### Three-Phase Design

```
Phase 1: Data Acquisition → Phase 2: Feature Engineering → Phase 3: Model Training
```

### Data Layer (`stoke_ml/data/`)

**4-source failover chain** for A-share K-line data (`failover.py` → `AShareDownloader`):
1. Efinance (EastMoney direct HTTP, curl-cffi Chrome 120 impersonation)
2. AKShare (Sina Finance wrapper)
3. Tushare (needs token)
4. Baostock (free, last resort)

Each source implements `AShareSourceBase` and has a `SOURCE_NAME` string. The downloader tracks failure counts and opens a circuit breaker (5 min cooldown) after 15 consecutive failures for a source.

**3-layer news medallion** (`news_storage.py` → `NewsStorage`):
- Bronze: `news_raw/{stock}.parquet` — raw as-fetched, append-only
- Silver: `news_silver/{stock}.parquet` — PIT-aligned (post-15:00 CST news → next trading day)
- Gold: `sentiment/{year}/{month}/{stock}.parquet` — daily aggregation, same partition as K-line

**K-line storage** (`storage.py` → `DataStorage`): Parquet partitioned `daily/{year}/{month}/{stock}.parquet`.

**Trading calendar** (`calendar.py` → `TradingCalendar`): Hardcoded A-share holidays 2015–2028. `get_trading_days()`, `is_trading_day()`, `next_trading_day()`.

### Feature Layer (`stoke_ml/features/`)

`FeaturePipeline.build_features(df, sentiment_df=None)` returns `(X, y, aligned_close)`:

1. Merge sentiment columns (if `sentiment_df` provided and `use_sentiment=True`) — ZI method: missing days get zeros + `has_news=False`
2. Technical indicators (`technical.py`): MA, MACD, RSI, Bollinger, ATR, OBV, volume ratios
3. Trend scoring (`scoring.py`): Rule-based trend labels
4. Temporal features (`temporal.py`): lags (1/2/3/5/10/20), rolling stats (5/10/20/60), calendar features
5. Sequence creation: `seq_len=60` trading-day windows → `(n_samples, seq_len, n_features)` or flat `(n_samples, n_features*seq_len)` for XGBoost

`SENTIMENT_COLS = ["sentiment_mean", "sentiment_std", "news_count", "positive_ratio", "negative_ratio", "has_news"]`

**News NLP** (`news_nlp.py`):
- L1: SnowNLP — offline Chinese sentiment, maps [0,1]→[-1,1]
- `compute_raw_sentiment(df, analyzer)` scores titles + bodies → adds `sentiment_title`, `sentiment_body` columns
- `aggregate_daily_sentiment(titles)` returns dict of daily stats
- L2 upgrade path: FinBERT Chinese (model download blocked in mainland China)

### Model Layer (`stoke_ml/models/`)

- `XGBoostBaseline`: Flat mode classifier, sklearn-compatible `fit/predict/save`
- `LSTMModel` + `StockLightningModule`: PyTorch Lightning sequence model, class-weighted for imbalance

### Evaluation (`stoke_ml/evaluation/`)

- `WalkForwardSplitter`: Expanding window with chronological splits only (NO shuffle). Default: 2yr train / 3mo validation / 3mo step.
- `compute_classification_metrics(y_true, y_pred)`: MCC (primary), accuracy, precision, recall, F1
- `compute_financial_metrics(close_prices, predictions)`: Sharpe, max drawdown, win rate, profit factor
- `aligned_close` in pipeline output has `n_samples+1` elements to produce `n_samples` returns matching `n_samples` predictions

### Crawler (`stoke_ml/crawler/`)

6-layer anti-block: TLS impersonation (curl-cffi Chrome 120) → browserforge headers → session pool (50 max, 30min TTL) → proxy rotation → rate limiter (2s base + jitter) → circuit breaker (5min cooldown). Fallback to Playwright with stealth JS when curl-cffi fails.

## Configuration

`config.yaml` at project root, loaded via OmegaConf (`stoke_ml/config.py` → `load_config()`). Relative paths (`data_dir`, `model_dir`) are resolved relative to project root automatically.

Key settings: `features.seq_len=60`, `features.target_horizon=1`, `training.validation: train_years=2, val_months=3`, `evaluation.primary_metric=mcc`.

## Key Conventions

- **No shuffle in time series**: Walk-forward splits only, chronological order preserved
- **PIT anti-leakage**: Post-close news (15:00 CST) assigned to next trading day via `TradingCalendar.next_trading_day()`
- **ZI method**: Days without news get zero-filled sentiment + `has_news=False` flag
- **FeaturePipeline graceful degradation**: `sentiment_df=None` → trains on technical features only
- **Data partitioned by year/month/stock**: Enables loading date ranges without scanning all files
- **`PYTHONPATH=.` mandatory**: All scripts import from `stoke_ml` package relative to project root

## Agent skills

### Issue tracker

Issues live as GitHub Issues in `Zn070515/Stoke_MachineLearning` — use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: one `CONTEXT.md` + `docs/adr/` at root. See `docs/agents/domain.md`.
