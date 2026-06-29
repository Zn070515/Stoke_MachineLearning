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

# Download news + sentiment (multi-source: EastMoney THS + Sina)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_news.py --source all --max-pages 5

# Download Guba forum posts + sentiment (802 stocks)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --max-pages 10

# Download AKShare comment sentiment (5184 stocks)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_comment.py

# Download Xueqiu forum posts (Guba alternative, Playwright required)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --max-pages 20

# Download market data (margin/northbound/dragon_tiger)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_market_data.py --type all

# Download fundamental data (quarterly financials)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_fundamentals.py

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
1. Efinance (EastMoney direct HTTP, curl-cffi Chrome 146 impersonation)
2. AKShare (Sina Finance wrapper)
3. Tushare (needs token)
4. Baostock (free, last resort)

Each source implements `AShareSourceBase` and has a `SOURCE_NAME` string. Circuit breaker: 10 consecutive failures → 300s cooldown.

**3-layer medallion architecture** — all text data sources follow this pattern:
- Bronze: `*_raw/{stock}.parquet` — raw as-fetched, append-only
- Silver: `*_silver/{stock}.parquet` — PIT-aligned (post-15:00 CST → next trading day)
- Gold: `*_sentiment/{year}/{month}/{stock}.parquet` — daily aggregation

**Storage classes and their data:**
- `DataStorage` — K-line, `daily/{year}/{month}/{stock}.parquet` (also flat `daily/{code}.parquet`)
- `NewsStorage` — news articles (3-source aggregation via `NewsPipeline`)
- `GubaStorage` — forum posts, dedup by `post_id`, columns: `guba_sentiment_mean/std/count/positive_ratio/negative_ratio/has_guba_post` (body coverage: 14.3%, detail page blocked)
- `XueqiuStorage` — Xueqiu forum posts (Guba alternative), Playwright WAF bypass, columns: `xueqiu_sentiment_mean/std/count/positive_ratio/negative_ratio/has_xueqiu_post`
- `CommentStorage` — AKShare comment ratings, `build_features()` returns daily ZI-filled features
- `AnnouncementStorage` — company announcements + sentiment
- `MarketWideStorage` — dragon_tiger/margin/northbound, partitioned `{type}/{year}/{month}/{stock}.parquet`
- `FundamentalStorage` — quarterly financials, forward-filled to daily
- `ETFStorage` — sector ETF flows, `etf_flow/{year}/{month}/sector_{name}.parquet`

**Trading calendar** (`calendar.py` → `TradingCalendar`): Hardcoded A-share holidays 2015–2028. `get_trading_days()`, `is_trading_day()`, `next_trading_day()`.

### Feature Layer (`stoke_ml/features/`)

`FeaturePipeline.build_features(df, **aux_dfs)` returns `(X, y, aligned_close)`:

**CRITICAL: All `use_*` flags default to `True`** in FeaturePipeline constructor. When running ablation, you MUST explicitly set unused dimensions to `False`:
```python
FeaturePipeline(seq_len=60, use_sentiment=True, use_announcements=False,
                use_guba=False, use_comment=False)
```

**9 auxiliary dimensions** (all lagged 1 day to prevent leakage, merged via left-join ZI):
| Dimension | switch | Columns | Data density |
|---|---|---|---|
| sentiment (news) | `use_sentiment` | 6 | medium |
| guba (forum) | `use_guba` | 6 | high (posts), low (body) |
| comment (ratings) | `use_comment` | 5 | medium |
| xueqiu (forum) | `use_xueqiu` | 6 | medium (Playwright) |
| announcement | `use_announcements` | 6 | low |
| margin trading | `use_margin` | 4 | high |
| northbound | `use_northbound` | 2 | medium |
| dragon tiger | `use_dragon_tiger` | 3 | low |
| fundamental | `use_fundamental` | 8 | low (quarterly) |
| ETF flow | `use_etf_flow` | 2 | high (sector-level) |

Pipeline steps:
1. Merge all auxiliary DataFrames (ZI fill for missing days/lags)
2. Technical indicators (`technical.py`): MA(5/10/20/60/120), EMA(12/26), MACD, RSI(6/12/24), KDJ(9/14), Bollinger %b, ATR(14), ROC, Williams %R, CCI, OBV, volume ratios
3. Trend scoring (`scoring.py`): trend_level (0-6), bias indicators, buy_signal (0-5)
4. Microstructure: is_limit_up/down, gap_up/down_pct, volume_anomaly, limit_up_streak
5. Temporal features (`temporal.py`): lags (1/2/3/5/10/20), rolling stats (5/10/20/60), calendar features
6. Sequence creation: `seq_len=60` windows → `(n, seq_len, n_features)` or flat `(n, n_features*seq_len)` for XGBoost
7. ALL config dimensionality: ~405 features × 60 seq_len = 24,300 flat dimensions

**News NLP** (`news_nlp.py`) — 3-tier sentiment:
- L1: FinBERT Chinese (`yiyanghkust/finbert-tone-chinese`) via HF mirror (`hf-mirror.com`) or local cache
- L2: FinBERT offline (`local_files_only=True`)
- L3: Financial lexicon fallback (39 positive + 35 negative Chinese financial terms)
- CPU inference: ~38ms/text; GPU: ~2ms/text with batching
- `compute_raw_sentiment(df, analyzer)` adds `sentiment_title` + `sentiment_body` columns
- `aggregate_daily_sentiment(titles)` returns dict of daily stats

### Model Layer (`stoke_ml/models/`)

- `XGBoostBaseline` (`models/baseline/`): Flat mode classifier, sklearn-compatible `fit/predict/save`
- `LSTMModel` (`models/dl/`): 2-layer LSTM, hidden_dim=128, dropout=0.3
- `TransformerModel` (`models/dl/`): 3-layer Transformer encoder, d_model=128, nhead=8
- `SimpleAttentionModel` (`models/dl/`): Single self-attention + learnable query pooling, d_model=64
- `StockLightningModule`: PyTorch Lightning wrapper, class-weighted CrossEntropyLoss, ReduceLROnPlateau, records val_mcc

Existing checkpoints: `lstm_000001_final.ckpt`, `lstm_601318_final.ckpt`, `xgboost_000001_best.json`, `xgboost_600519_best.json`

### Evaluation (`stoke_ml/evaluation/`)

- `WalkForwardSplitter`: Fixed-size sliding window with chronological splits only (NO shuffle). Default: 2yr train / 3mo validation / 3mo step.
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
- **ZI method**: Days without data get zero-filled values + `has_*=False` flag
- **Sentiment lag**: All text sentiment columns lagged 1 trading day to prevent same-day information leakage
- **FeaturePipeline defaults**: ALL `use_*` flags default to `True` — must explicitly disable for ablation
- **Data partitioned by year/month/stock**: Enables loading date ranges without scanning all files
- **`PYTHONPATH=.` mandatory**: All scripts import from `stoke_ml` package relative to project root
- **Flat parquet fallback**: Storage classes check flat `{code}.parquet` before partitioned path

## Known Issues

| Issue | Status |
|---|---|
| Guba post bodies unavailable (detail page SPA, WAF-blocked) | Replaced by Xueqiu forum source |
| Xueqiu news source (Cloudflare WAF) | Resolved — Playwright bypass works, used as forum data
| ALL config dimension explosion (24,300 features) | Use +sentiment instead |
| FinBERT first load needs network or pre-cached model | Use `HF_ENDPOINT=https://hf-mirror.com` |
| Ablation Δ CIs cross zero (need >100 stocks or stronger signal) | Active research |

## Ablation Results (95 stocks, 1000 bootstrap samples)

| Config | MCC | 95% CI | Δ vs technical |
|---|---|---|---|
| technical | 0.0136 | [-0.0035, 0.0312] | — |
| + sentiment | 0.0279 | [0.0095, 0.0464] | +0.0143 |
| + guba | 0.0219 | [0.0032, 0.0384] | +0.0084 |
| + comment | 0.0224 | [0.0045, 0.0408] | +0.0089 |
| ALL | 0.0261 | [0.0104, 0.0426] | +0.0125 |

- All text dimensions improve MCC vs technical-only (all CIs > 0)
- News sentiment has largest effect (+104% MCC)
- ALL config underperforms +sentiment alone (dimension explosion)
- Δ CIs all cross zero — need more data or stronger signal for significance

Full analysis: `docs/project-analysis-2026-06-29.md`

## Agent skills

### Issue tracker

Issues live as GitHub Issues in `Zn070515/Stoke_MachineLearning` — use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: one `CONTEXT.md` + `docs/adr/` at root. See `docs/agents/domain.md`.
