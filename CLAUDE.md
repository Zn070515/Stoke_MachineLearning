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
# Download K-line for all A-shares (5530 stocks, 2000–2026)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_data.py

# Download news + sentiment (multi-source: EastMoney THS + Sina)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_news.py --source all --max-pages 5

# Download Guba forum posts + sentiment (802 stocks)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_guba.py --max-pages 10

# Download AKShare comment sentiment (5184 stocks)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_comment.py

# Download market data (margin/northbound/dragon_tiger)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_market_data.py --type all

# Download fundamental data (quarterly financials)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_fundamentals.py

# Single stock test
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_news.py --stocks 600519 --max-pages 3
```

### Training

```bash
# XGBoost baseline (flat features, walk-forward validation)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_baseline.py --stock 000001
PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_baseline.py  # all stocks

# Prebuild features once (decouples feature engineering from training)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_features.py                 # flat → data/features/ (5530 × ~3744 cols)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_features.py --panel-mode    # panel → data/features_panel/ (cross-sectional z-score)

# Panel (VSN+xLSTM, main model) — read prebuilt features directly
PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --stocks 500 --prebuilt data/features_panel --epochs 30 --max-folds 3

# Docs-vs-code drift guard (exit 1 on mismatch)
PYTHONPATH=. ./.venv/Scripts/python scripts/production/check_docs_consistency.py
```

### Testing

~53 test files under `tests/{features,models,preprocessing,data,evaluation}/`. Run via the venv interpreter:

```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -q                    # fast smoke (default, excludes slow/network)
PYTHONPATH=. ./.venv/Scripts/python -m pytest -m "slow or network" tests/ -q   # slow + network only
PYTHONPATH=. ./.venv/Scripts/python -m pytest -m "" tests/ -q               # everything
```

Tests that train a model or run a full end-to-end pipeline chain carry the
`slow` marker; tests that hit live external APIs carry `network`. Both are
excluded from the default smoke run (pyproject.toml addopts) — the default
`pytest` is the fast smoke (~30-60s), `-m ""` overrides it to run everything.

Docs drift is guarded by `scripts/production/check_docs_consistency.py` (see Commands).

## Workflow (mandatory)

**需求明确并创建 Task 后，必须 fan out SubAgent 完成（保证工程质量）。**

- 一旦需求澄清、Task 已建好，就用 `Agent` 工具把每个 Task 派给独立的 subagent 执行（优先 `superpowers:subagent-driven-development` 流程），**禁止**在主会话里逐个内联实现。
- 每个 Task 的标准流程：dispatch implementer subagent → spec 合规审查 → 代码质量审查 → 两轮都通过后标记完成；审查发现问题就回 implementer 修复再复审，直到通过。
- 目的：每个 subagent 用全新上下文执行（不继承主会话、不互相污染），强制双阶段审查，质量不过关不放行。
- 唯一例外：被明确判定为「简单到不需要 fan out」的任务（单行修复、纯读/纯查询）才允许内联；拿不准时**默认 fan out**。

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
- `DataStorage` — K-line, flat `daily/{code}.parquet` (唯一 canonical 布局, 前复权 qfq) with per-stock `{code}.manifest.json` contract sidecar; formal reads enforce `require_valid_manifest=True` (v9 §九-1)
- `NewsStorage` — news articles (3-source aggregation via `NewsPipeline`)
- `GubaStorage` — forum posts, dedup by `post_id`, columns: `guba_sentiment_mean/std/count/positive_ratio/negative_ratio/has_guba_post` (body coverage: 14.3%, detail page blocked)
- `CommentStorage` — AKShare comment ratings, `build_features()` returns daily ZI-filled features
- `AnnouncementStorage` — company announcements + sentiment
- `MarketWideStorage` — dragon_tiger/margin/northbound, partitioned `{type}/{year}/{month}/{stock}.parquet`
- `FundamentalStorage` — quarterly financials, forward-filled to daily
- `ETFStorage` — sector ETF flows, `etf_flow/{year}/{month}/sector_{name}.parquet`

**Trading calendar** (`calendar.py` → `TradingCalendar`): externalized artifact `exchange_calendar/{market}.parquet` with `verified_until` (a_shares: 2026-12-31) — dates past that are forward estimates, and strict calendars fail rather than answer them. `get_trading_days()`, `is_trading_day()`, `next_trading_day()`.

### Feature Layer (`stoke_ml/features/`)

`FeaturePipeline.build_features(df, **aux_dfs)` returns `(X, y, aligned_close)`:

**CRITICAL: All `use_*` flags default to `True`** in FeaturePipeline constructor. When running ablation, you MUST explicitly set unused dimensions to `False`:
```python
FeaturePipeline(seq_len=60, use_sentiment=True, use_announcements=False,
                use_guba=False, use_comment=False)
```

**26 `use_*` data dimensions** (25 active + `use_limit_up` deferred; all lagged 1 day to prevent leakage, merged via left-join ZI):
| Dimension | switch | Columns | Data density |
|---|---|---|---|
| sentiment (news) | `use_sentiment` | 6 | medium |
| guba (forum) | `use_guba` | 6 | high (posts), low (body) |
| comment (ratings) | `use_comment` | 5 | medium |
| announcement | `use_announcements` | 6 | low |
| margin trading | `use_margin` | 4 | high |
| northbound | `use_northbound` | 2 | medium |
| dragon tiger (+ seat) | `use_dragon_tiger` | 7 (3+4) | low |
| fundamental | `use_fundamental` | 8 | low (quarterly) |
| earnings forecast | `use_earnings` | 6 | low |
| valuation | `use_valuation` | 4 | high |
| ETF flow | `use_etf_flow` | 2 | high (sector-level) |
| capital flow | `use_capital_flow` | 9 | high |
| block trade | `use_block_trade` | 7 | low |
| shareholder | `use_shareholder` | 6 | low |
| lockup | `use_lockup` | 5 | low |
| dividend | `use_dividend` | 3 | low |
| board (打板) | `use_board` | 12 | low |
| sector (行业) | `use_sector` | 5 | high |
| concept (概念) | `use_concept` | 5 | medium |
| industry ranking | `use_industry` | 9 | high |
| macro | `use_macro` | 28 | high (daily) |
| pledge | `use_pledge` | 5 | medium |
| index membership | `use_index_membership` | 3 | low |
| market env | `use_market_env` | 7 | high (all-market) |
| macro regime refine | `use_market_env_refine` | 49 | high (daily) |
| limit-up ecology | `use_limit_up` | 20 | low — **DEFERRED** |

(Plus pipeline-switch flags that toggle processing rather than data: `use_technical`, `use_scoring`, `use_temporal`, `use_interaction`, `use_feature_selection`, `use_new_preprocessing`, `use_emotion_refine`, `use_fundamental_refine`, `use_temporal_stats`.)

Pipeline steps:
1. Merge all auxiliary DataFrames (ZI fill for missing days/lags)
2. Technical indicators (`technical.py`): MA(5/10/20/60/120), EMA(12/26), MACD, RSI(6/12/24), KDJ(9/14), Bollinger %b, ATR(14), ROC, Williams %R, CCI, OBV, volume ratios
3. Trend scoring (`scoring.py`): trend_level (0-6), bias indicators, buy_signal (0-5)
4. Microstructure: is_limit_up/down, gap_up/down_pct, volume_anomaly, limit_up_streak
5. Temporal features (`temporal.py`): lags (1/2/3/5/10/20), rolling stats (5/10/20/60), calendar features
6. Sequence creation: `seq_len=60` windows → `(n, seq_len, n_features)` or flat `(n, n_features*seq_len)` for XGBoost
7. Prebuild: `scripts/production/build_features.py` engineers the full market once (5530 × ~3744 cols, 109GB flat; `features_panel/` for panel z-score mode) — training reads prebuilt parquet instead of re-engineering in-loop. Flat ALL-mode dimensionality (~24,300) is why panel mode + IC pre-filtering is preferred.

**News NLP** (`news_nlp.py`) — 3-tier sentiment:
- L1: FinBERT Chinese (`yiyanghkust/finbert-tone-chinese`) via HF mirror (`hf-mirror.com`) or local cache
- L2: FinBERT offline (`local_files_only=True`)
- L3: Financial lexicon fallback (39 positive + 35 negative Chinese financial terms)
- CPU inference: ~38ms/text; GPU: ~2ms/text with batching
- `compute_raw_sentiment(df, analyzer)` adds `sentiment_title` + `sentiment_body` columns
- `aggregate_daily_sentiment(titles)` returns dict of daily stats

### Model Layer (`stoke_ml/models/`)

- `PanelModel` (`models/panel/`): VSN (Variable Selection Network) + xLSTM backbone (sLSTM+mLSTM), multi-task heads (direction/return/volatility). Trained via `scripts/production/train_panel.py` — reads prebuilt panel features (`--prebuilt data/features_panel`) so feature engineering runs once offline, not in the training loop.
- `XGBoostBaseline` (`models/baseline/`): Flat mode classifier, sklearn-compatible `fit/predict/save`
- Panel baselines (`models/baseline/panel_baselines.py`): Ridge / LightGBM / MLP / naive momentum, evaluated on the same inner_val schedule as the main model (evaluator_version 2026-08-05)

Existing checkpoints: `xgboost_000001_best.json`, `xgboost_600519_best.json`

### Evaluation (`stoke_ml/evaluation/`)

- `WalkForwardSplitter`: Fixed-size sliding window with chronological splits only (NO shuffle). Default: 2yr train / 3mo validation / 3mo step.
- `compute_classification_metrics(y_true, y_pred)`: MCC (primary), accuracy, precision, recall, F1
- `compute_financial_metrics(close_prices, predictions)`: Sharpe, max drawdown, win rate, profit factor
- `bootstrap_ci(values, statistic="mean", n_boot=2000)`: bootstrap 95% CI for per-stock metric arrays
- `aligned_close` in pipeline output has `n_samples+1` elements to produce `n_samples` returns matching `n_samples` predictions

### Crawler (`stoke_ml/crawler/`)

6-layer anti-block: TLS impersonation (curl-cffi Chrome 120) → browserforge headers → session pool (50 max, 30min TTL) → proxy rotation → rate limiter (2s base + jitter) → circuit breaker (5min cooldown). Fallback to Playwright with stealth JS when curl-cffi fails.

## Configuration

`config.yaml` at project root, loaded via OmegaConf (`stoke_ml/config.py` → `load_config()`). Relative paths (`data_dir`, `model_dir`) are resolved relative to project root automatically.

Key settings: `features.seq_len=60`, `features.target_horizon=1`, `training.validation: train_years=2, val_months=3`, `evaluation.primary_metric=mcc`.

### Dependency management (§十三-3)

`pyproject.toml` is the single dependency source; `uv.lock` (generated by `uv lock`, checked by `uv lock --check`) is committed and pins all packages universally. CI installs per-job via `uv sync --frozen --extra <group>` (`.github/workflows/ci.yml`); the local mirror is `scripts/maintenance/current/ci.py`. After any dependency change run `uv lock` to regenerate the lockfile. `.python-version` pins Python 3.12 so `uv sync` always targets the same interpreter as CI. Environment scheme: Windows CUDA dev uses the local `.venv` (Python 3.12, `torch==2.11.0`); Linux CPU CI installs from the committed lockfile (details in CONTEXT.md §环境方案).

## Key Conventions

- **No shuffle in time series**: Walk-forward splits only, chronological order preserved
- **PIT anti-leakage**: Post-close news (15:00 CST) assigned to next trading day via `TradingCalendar.next_trading_day()`
- **ZI method (channel-specific, §九-4)**: Event channels (news/announcements/LHB/block-trade...) zero-fill missing days + `has_*=False` flag; state channels (margin balance, northbound holdings, valuation, fundamentals, macro rates, market-env breadth, shareholder, pledge) forward-fill gaps instead — a missing state day means "unchanged", never zero
- **Text sentiment effective-date**: News/guba post-close events are PIT-mapped to the next trading day at the storage layer (`effective_trade_date`); the feature layer does NOT shift these again (nor earnings, also effective-date-mapped) — review v9 §十二-7 removes the double lag. Same-day pre-close text is usable at the next day's open via the decision-column convention.
- **FeaturePipeline defaults**: ALL `use_*` flags default to `True` — must explicitly disable for ablation
- **Data partitioned by year/month/stock**: Enables loading date ranges without scanning all files
- **`PYTHONPATH=.` mandatory**: All scripts import from `stoke_ml` package relative to project root
- **Flat parquet fallback**: Storage classes check flat `{code}.parquet` before partitioned path

## Known Issues

| Issue | Status |
|---|---|
| Guba post bodies unavailable (detail page SPA, WAF-blocked) | Using lexicon-based sentiment fallback for body text |
| ALL config dimension explosion (24,300 features) | Use +sentiment instead |
| FinBERT first load needs network or pre-cached model | Use `HF_ENDPOINT=https://hf-mirror.com` |
| Ablation Δ CIs cross zero (need >100 stocks or stronger signal) | Active research |
| Playwright browser can hang indefinitely during WAF bypass | Use `threading.Timer(timeout, lambda: os._exit(1))` as hard kill-switch |

## Quick-Eval Stock Basket (15 stocks, cross-sector)

`000001`(银行) `600519`(白酒) `000725`(科技) `600276`(医药) `000651`(家电) `601318`(保险) `600900`(电力) `002415`(海康) `000858`(五粮液) `600036`(招行) `002594`(比亚迪) `601088`(神华) `300750`(宁德) `688981`(中芯) `002493`(荣盛)

Use for fast model comparison before full-scale training.

Benchmark findings & research: `docs/research-findings.md`

## Agent skills

### Issue tracker

Issues live as GitHub Issues in `Zn070515/Stoke_MachineLearning` — use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: one `CONTEXT.md` + `docs/adr/` at root. See `docs/agents/domain.md`.
