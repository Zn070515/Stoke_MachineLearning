# Stock Prediction with Deep Learning — Design Spec

> 2026-06-19 | Status: Draft | Author: User + Claude

## 1. Overview

**Purpose**: Build a deep learning system for stock prediction, serving dual goals of (a) learning DL in quantitative finance, and (b) personal trading assistance.

**Scope**: A-shares (沪深) + US stocks (S&P 500), daily and weekly/monthly predictions, multi-dimensional targets starting from price direction and expanding to volatility and turning points.

**Constraints**: Local consumer GPU (RTX 5060, 8GB VRAM), DL beginner, total data ~10-20GB.

**Reference projects analyzed**:
- `Stock-Prediction-Models` — 18 DL model catalog, stacking ensembles, sentiment consensus technique
- `MediaCrawler` — Playwright CDP, stealth.js, proxy pool, captcha solving, request signing
- `crawlee-python` — TLS fingerprint spoofing (curl-cffi), browserforge, session pool, tiered proxy
- `daily_stock_analysis` — Multi-source data failover, technical analysis scoring, LLM synthesis

## 2. Architecture

Five-layer modular pipeline. Each layer communicates through well-defined data contracts and can be developed, tested, and replaced independently.

```
LAYER 0 — Anti-Block Crawler Layer (NEW)
  TLS fingerprint spoofing + browserforge headers + proxy pool + session pool
  → Pluggable backends: HTTP (curl-cffi) → Playwright+stealth (fallback)

LAYER 1 — Data Layer
  Multi-source market data + News crawlers + Fundamentals
  → Failover chain: primary → secondary → fallback (circuit breaker on each)
  → Stores: Parquet (price) + SQLite (metadata) + JSON (raw news)

LAYER 2 — Feature Layer
  Technical indicators (ta-lib) + Rule-based scoring + NLP embeddings + Temporal features
  → Output: Aligned feature matrix (samples × seq_len × features)

LAYER 3 — Model Layer
  XGBoost baseline → LSTM/GRU → CNN-Seq2Seq → Dilated CNN → Transformer → Multi-task
  → Framework: PyTorch + PyTorch Lightning

LAYER 4 — Prediction Layer
  Direction (binary) → Magnitude (regression) → Volatility → Turning points
  → Independent prediction heads, added incrementally

LAYER 5 — Evaluation Layer
  Walk-forward backtest + Multi-metric evaluation + Sentiment consensus analysis + Equity curves
```

### Design Principles

- **Modular & replaceable**: each layer has a clear interface; internal changes don't break consumers
- **Start small, iterate**: run through full pipeline with the simplest configuration first
- **Progressive complexity**: ML baseline → LSTM → CNN-Seq2Seq → Transformer → multi-task
- **Data-driven**: unified data formats; models consume standardized feature matrices
- **Anti-fragile crawling**: defense in depth — multiple layers of anti-detection, graceful degradation

### Explicitly Out of Scope

- Real-time trading execution interface
- Automated hyperparameter search (manual tuning first)
- Distributed / multi-GPU training
- Large multimodal models (GPT-scale)

## 3. Anti-Block Crawler System (LAYER 0)

This is the **foundation** for all data acquisition. The crawler must operate without any practical restrictions — no IP bans, no rate-limit walls, no CAPTCHA dead-ends.

### Defense-in-Depth Architecture

```
                    Request In
                         │
            ┌────────────▼────────────┐
            │  Layer A: TLS Fingerprint │
            │  curl-cffi (Chrome TLS   │
            │  impersonation)          │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Layer B: HTTP Headers   │
            │  browserforge (consistent│
            │  UA + sec-ch-ua + Accept)│
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Layer C: Session Pool   │
            │  Per-session cookies,    │
            │  error scoring, auto-retire│
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Layer D: Proxy Rotation │
            │  Tiered proxy pool,      │
            │  auto failover on error  │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Layer E: Rate Limiting  │
            │  Random jitter delays,   │
            │  adaptive backoff        │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │  Layer F: Browser Fallback│
            │  Playwright + stealth.js │
            │  + CDP real-browser mode │
            │  (when HTTP is blocked)  │
            └────────────┬────────────┘
                         │
                    Response Out
```

### Layer A: TLS Fingerprint Spoofing

The most critical anti-detection layer. Most anti-bot systems fingerprint the TLS handshake — Python's default `ssl` library is trivially identifiable.

**Solution**: `curl-cffi` (curl-impersonate Python bindings). It patches libcurl to mimic browser TLS stacks exactly:

```python
from curl_cffi import requests
# Mimics Chrome 120's TLS fingerprint (JA3/JA4 fingerprint)
session = requests.Session(impersonate="chrome120")
```

Supported impersonation targets: `chrome110/116/120/123/124`, `safari15_5/17_0`, `firefox`, `edge99/101/110`.

**Fallback**: When `curl-cffi` is insufficient (heavy JS-rendered pages), escalate to Layer F.

### Layer B: Browser Fingerprint Consistency

Anti-bot systems check that HTTP headers are **internally consistent** — a Chrome 120 UA with Firefox `sec-ch-ua` is flagged.

**Solution**: `browserforge` library generates self-consistent header sets:

- `User-Agent` matches `sec-ch-ua` version
- `Accept-Language` matches OS locale
- `sec-ch-ua-platform` matches declared OS
- Screen resolution consistent with device type

```python
from browserforge.headers import HeaderGenerator
headers = HeaderGenerator(browser="chrome", device="desktop", os="windows")
```

Generated headers are cached and reused per session (not regenerated each request — real browsers don't change headers between requests).

### Layer C: Session Pool with Error Scoring

Each "session" simulates a distinct user with its own cookie jar, header set, and usage history.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_age` | 30 min | Session expires, forces refresh |
| `max_usage` | 30 requests | Cap per session lifecycle |
| `max_error_score` | 3.0 | Retire on excessive errors |
| `error_score_decrement` | 0.5 | Good responses reduce score |
| `blocked_status_codes` | [401, 403, 429] | Auto-retire trigger |

Session pool size: up to 100 sessions per data source. Sessions are randomly selected per request. Retired sessions are automatically replaced.

**Cookie persistence**: Sessions survive script restarts via SQLite-backed cookie store. After a restart, sessions resume with warm cookies (reduces login frequency).

### Layer D: Tiered Proxy Pool

**Free tier** (default for learning):
- Public proxy lists (scraped from proxylist sites, validated on startup)
- Local proxies from `github.com/.../hosts` reference

**Paid tier** (when reliability matters):
- Residential proxy services (e.g., KuaiDaili, BrightData)
- Automatic provider failover

**Tiered routing logic**:
```
primary tier → secondary tier (on error) → no proxy (last resort)
```
Error scores accumulated per proxy-domain pair. Proxies that fail on `finance.sina.com.cn` may still work on `eastmoney.com` — tracked separately.

### Layer E: Adaptive Rate Limiting

- **Random jitter**: `base_delay * (0.5 + random())` between requests
- **Exponential backoff**: on 429/503, delay doubles each retry up to 5 min cap
- **Circuit breaker**: after N consecutive failures on a domain, stop for `cooldown_seconds` (default 300s)
- **Daily quota tracking**: per-domain request counts, soft cap warnings

### Layer F: Browser-Based Fallback

When HTTP layers fail (CAPTCHA wall, JS challenge):

1. **Playwright + stealth.js**: Injects `stealth.min.js` to patch `navigator.webdriver`, `chrome.runtime`, etc.
2. **CDP mode** (strongest): Connects to user's **real** Chrome/Edge browser via DevTools Protocol — inherits real browser profile, extensions, cookies. Undetectable by fingerprinting. Used for critical or heavily-protected endpoints.
3. **Captcha solving**: OpenCV template matching for slider CAPTCHAs, with human-like mouse trajectories (acceleration→deceleration physics model, easing functions).

### Anti-Block Configuration

```yaml
crawler:
  tls_impersonate: "chrome120"       # curl-cffi target
  browserforge:
    browser: "chrome"
    device: "desktop"
    os: "windows"
  session_pool:
    max_sessions: 50
    max_age_minutes: 30
    max_usage: 30
    max_error_score: 3.0
  proxy:
    enabled: true
    tier: "free"                      # free | paid
    validate_on_startup: true
    max_proxies: 20
  rate_limit:
    base_delay_sec: 2.0
    jitter_factor: 0.5
    max_backoff_sec: 300
    circuit_breaker_cooldown_sec: 300
    daily_quota_per_domain: 10000
  browser_fallback:
    enabled: true
    engine: "playwright"              # playwright | cdp
    stealth_js: true
    captcha_solver: "opencv"          # opencv | manual
```

## 4. Data Pipeline (LAYER 1)

### Multi-Source Market Data with Failover

Inspired by `daily_stock_analysis`, each market has multiple data sources with automatic failover:

**A-shares priority chain**:

| Priority | Source | Type | Notes |
|----------|--------|------|-------|
| 0 | Efinance (东方财富) | API | Fast, reliable, preferred |
| 1 | AKShare | Scraping wrapper | Comprehensive, rate-limited |
| 2 | Tushare Pro | API (token) | Professional, quota-limited |
| 3 | Baostock | API | Free, limited history |

**US stocks priority chain**:

| Priority | Source | Type | Notes |
|----------|--------|------|-------|
| 0 | yfinance | API | Free, Yahoo backend |
| 1 | Finnhub | API (free tier) | 60 calls/min |
| 2 | Alpha Vantage | API (free tier) | 25 calls/day |
| 3 | Polygon.io | API (free tier) | 5 calls/min |

**Failover logic**: Try source 0 → on error/unavailable → try source 1 → ... → raise if all fail. Each source wrapped with circuit breaker (cooldown after N consecutive failures).

**Data normalization**: All sources output a unified schema:

```
[date, stock_code, open, high, low, close, volume, amount, pct_change]
```

Additional fields per market: `adj_factor`, `is_st`, `is_suspended`, `limit_up_price`, `limit_down_price` (A-shares).

### Market Data Pipeline

| Step | Tool | Output |
|------|------|--------|
| Download | Multi-source failover chain (above) | Raw OHLCV, full history + incremental |
| Clean | Custom cleaners | Fill missing, remove outliers, adjust prices |
| Store | Parquet (partitioned by year/month) | ~5-10 GB total |

### News Data Pipeline

| Step | Tool | Output |
|------|------|--------|
| Collect | Anti-block crawler → 财联社, 东方财富, NewsAPI, RSS | Raw text with timestamps |
| Process | jieba, dedup, stock name linking | Clean text aligned to trading days |
| Vectorize | FinBERT (EN), BERT-wwm-chinese (CN) | Sentiment scores + [CLS] embeddings (768d) |

### Stock Universe

- **A-shares**: CSI 300 + CSI 500 (~800 stocks)
- **US**: S&P 500
- **Filters**: exclude ST, suspended, IPO < 1 year

### Data Alignment

- Trading calendar as the standard timeline (separate calendars for A-shares and US)
- Weekly/monthly: aggregate from daily
- News: map to the next trading day after publication
- Feature matrix: all features merged by (date, stock_code)

### Storage Estimates

| Type | Size |
|------|------|
| Market data (price) | 5–10 GB |
| News raw text | 2–5 GB |
| Feature matrices | 1–3 GB |
| **Total** | **10–20 GB** |

## 5. Feature Engineering (LAYER 2)

### Technical Indicators

Standard indicators via `ta-lib` / `pandas-ta`:
- **Trend**: MA(5/10/20/60/120), EMA(12/26), MACD(DIF/DEA/histogram)
- **Momentum**: RSI(6/12/24), KD(J), WR, CCI
- **Volatility**: BOLL(upper/mid/lower), ATR(14), historical volatility
- **Volume**: OBV, volume ratio (vs 5-day avg), volume shrinkage/heavy flags

### Rule-Based Scoring (from daily_stock_analysis)

Extract structured signals from technical indicators:

```python
# Trend status: 7-level classification
trend_levels = {
    0: "strong_bull",   # MA5 > MA10 > MA20 > MA60
    1: "bull",          # MA5 > MA10 > MA20
    2: "mild_bull",     # price > MA20 but MA not aligned
    3: "neutral",       # price oscillating around MA20
    4: "mild_bear",
    5: "bear",
    6: "strong_bear",
}

# Composite buy signal: 6-level (strong_buy → strong_sell)
# based on: bias, volume ratio, MA support proximity, MACD divergence
```

Key thresholds:
- `BIAS_THRESHOLD = 5.0%` — price too far above MA5 signals danger
- `VOLUME_SHRINK_RATIO = 0.7` — volume < 70% of 5-day average
- `VOLUME_HEAVY_RATIO = 1.5` — volume > 150% of 5-day average
- `MA_SUPPORT_TOLERANCE = 0.02` — 2% tolerance for MA support/resistance

These scores serve as **input features** to the model (not standalone signals).

### NLP Features (Phase 3)

Per stock per trading day:
- `sentiment_score`: FinBERT/BERT-wwm sentiment (0-1)
- `news_count`: number of news articles
- `news_embedding`: [CLS] token embedding (768d)
- `title_sentiment`: headlines-only sentiment (more timely)

### Temporal Features

- **Lag features**: price/volume/indicator values at t-1, t-2, t-3, t-5, t-10, t-20
- **Rolling statistics**: 5/10/20/60-day mean/std/min/max of key indicators
- **Calendar features**: day_of_week, day_of_month, month, quarter, days_to_earnings

### Feature Pipeline Output

Shape: `(n_samples, seq_len, n_features)` — 3D tensor for DL models
Flat variant: `(n_samples, n_features * seq_len)` — for ML baselines

## 6. Model Evolution (LAYER 3)

### Phase 1 — ML Baseline (2–3 weeks)

- **Input**: 30–50 technical indicators + rule-based scores, flattened (no temporal structure)
- **Model**: XGBoost / LightGBM
- **Target**: Next-day price direction (binary classification)
- **Goal**: Establish end-to-end pipeline, get AUC > 0.55, F1 > 0.52

### Phase 2 — DL Sequential (4–6 weeks)

★ **Starting point for deep learning**

- **Input**: Sliding window sequences (60–120 days), shape (B, T, F)
- **Model candidates** (in priority order):
  1. 2-layer LSTM/GRU (hidden=128/256, ~500K params) — simplest, most proven
  2. Bidirectional LSTM — captures forward+backward context
  3. CNN-Seq2Seq with Gated Linear Units — good at local patterns
  4. Dilated CNN-Seq2Seq — best performer in reference (95.86% normalized accuracy)

  Start with LSTM. Only move to CNN variants if LSTM underperforms baseline.
- **Training**: PyTorch Lightning, batch=512, lr=1e-3, dropout=0.3
- **VRAM**: < 4 GB
- **Goal**: Beat XGBoost baseline on A-share daily direction

### Phase 3 — Multi-modal Multi-task (8–12 weeks)

- **Architecture**: Transformer encoder with cross-attention fusion of price features + news embeddings
- **Multi-task**: Shared backbone + 3 task-specific heads:
  - Head 1: Direction (binary classification, Focal Loss)
  - Head 2: Volatility (regression, Huber Loss)
  - Head 3: Turning point (binary, temporal label smoothing)
- **Multi-market**: Market embedding vector to differentiate A-shares vs US
- **VRAM**: < 7 GB (with gradient checkpointing, mixed precision)
- **Goal**: News-enhanced predictions outperforming price-only models on both markets

### Model Catalog (for future exploration)

From the `Stock-Prediction-Models` reference, architectures available for experimentation:

| Category | Models | Best Performer |
|----------|--------|----------------|
| Basic RNN | LSTM, GRU, Bi-LSTM, Bi-GRU, 2-Path variants | Bi-LSTM (93.8%) |
| Seq2Seq | LSTM-S2S, GRU-S2S, Bi-LSTM-S2S | LSTM-S2S-VAE (95.4%) |
| CNN | CNN-S2S, Dilated-CNN-S2S | **Dilated-CNN-S2S (95.9%)** |
| Attention | Transformer (8-head, 2-encoder) | 94.3% |
| Stacking | RNN+ARIMA+XGB, Autoencoder+Ensemble+XGB | Autoencoder+Ensemble (best ensemble) |

*Note: accuracy numbers are normalized RMSE, not classification accuracy. They are model-vs-model rankings, not absolute quality signals.*

### Stacking Ensemble (Phase 3+)

For robust final predictions, combine multiple model types:
```
Dilated CNN (price patterns)
    + LSTM (temporal dynamics)    → Meta-learner (XGBoost) → Final prediction
    + Transformer (price+news)
```
Autoencoder can be used for dimension reduction before meta-learner.

## 7. Training Methodology (LAYER 4)

### Walk-Forward Validation

Time series data cannot be randomly split. Rolling window approach:

```
|---- Train 2yr ----|-- Val 3mo --|
     |---- Train 2yr ----|-- Val 3mo --|
          |---- Train 2yr ----|-- Val 3mo --|
```

N windows (typically 5-8), metrics averaged across all windows.

### Class Imbalance Handling

Financial time series naturally have ~50% up/down split, but:
- Market regimes create prolonged bull/bear imbalances
- Use class weights proportional to inverse frequency
- Focal Loss (γ=2) for hard example mining
- Primary metric: MCC (Matthews Correlation Coefficient), not accuracy

### Regularization

- Early stopping: patience=5 on validation loss
- ReduceLROnPlateau: factor=0.5, patience=3
- Dropout: 0.3-0.5 on all FC layers
- Weight decay: 1e-4

### GPU Budget

| Phase | Model | VRAM | Fit on RTX 5060 8GB? |
|-------|-------|------|----------------------|
| P1 | XGBoost | CPU only | N/A |
| P2 | LSTM | < 4 GB | Easily |
| P2 | Dilated CNN-S2S | < 5 GB | Yes |
| P3 | Transformer (small) | < 7 GB | Yes, with gradient checkpointing |
| P3 | Multi-task Transformer | < 7 GB | Yes, with mixed precision |

### Monitoring

- **wandb** (primary): loss curves, metrics, gradient histograms, hyperparameter tracking
- **TensorBoard** (alternative): if wandb unavailable
- **Local logs**: JSON lines for offline analysis

## 8. Evaluation & Backtesting (LAYER 5)

### Model Metrics

**Classification (direction)**:
- Accuracy, Precision, Recall, F1
- **MCC** (primary — balanced, works with imbalanced data)
- AUC-ROC, confusion matrix

**Regression (magnitude, volatility)**:
- MAE, RMSE, MAPE
- R², explained variance

**Financial metrics** (what actually matters):
- Sharpe ratio (annualized)
- Maximum drawdown
- Win rate, profit factor
- Calmar ratio

### Walk-Forward Backtest

From backtesting, produce:
- Cumulative return curve vs benchmark
- Annual returns breakdown
- Drawdown plot
- Monthly heatmap

### Sentiment Consensus Analysis

Technique from `Stock-Prediction-Models`:
1. Train model with price + sentiment features
2. Run 3 inference simulations:
   - (a) Original: use real sentiment values
   - (b) Positive consensus: force all sentiment = 1.0
   - (c) Negative consensus: force all sentiment = 0.0
3. Compare (b) vs (c) vs (a) to understand sentiment impact on model behavior

This reveals whether the model is over-relying on sentiment vs price patterns.

## 9. Project Structure

```
stoke-ml/
├── crawler/              # LAYER 0: Anti-block crawler
│   ├── tls.py            # curl-cffi TLS impersonation
│   ├── fingerprint.py    # browserforge header generation
│   ├── session_pool.py   # Session management with error scoring
│   ├── proxy_pool.py     # Tiered proxy rotation
│   ├── rate_limiter.py   # Adaptive delay + circuit breaker
│   ├── browser_fallback.py  # Playwright + stealth.js + CDP
│   └── captcha.py        # OpenCV slider solver
├── data/                 # LAYER 1: Data layer
│   ├── sources/          # Multi-source downloaders
│   │   ├── a_shares/     # Efinance, AKShare, Tushare, Baostock
│   │   └── us/           # yfinance, Finnhub, Alpha Vantage
│   ├── cleaners/         # Clean, adjust prices, handle anomalies
│   ├── news/             # News crawlers + NLP vectorization
│   └── calendar.py       # Trading calendar (A-shares + US)
├── features/             # LAYER 2: Feature layer
│   ├── technical.py      # ta-lib technical indicators
│   ├── scoring.py        # Rule-based trend/buy scoring
│   ├── temporal.py       # Lag/rolling/calendar features
│   └── pipeline.py       # Feature pipeline orchestration
├── models/               # LAYER 3: Model layer
│   ├── baseline/         # XGBoost, LightGBM
│   ├── dl/               # LSTM, GRU, CNN-S2S, Dilated-CNN
│   ├── transformer/      # Transformer encoder
│   ├── multitask/        # Multi-task shared+heads
│   └── stacking/         # Ensemble meta-learner
├── evaluation/           # LAYER 5: Evaluation layer
│   ├── metrics.py        # Classification, regression, financial metrics
│   ├── backtest.py       # Walk-forward backtesting engine
│   ├── sentiment_consensus.py  # Forced sentiment analysis
│   └── viz.py            # Equity curves, heatmaps, dashboards
├── configs/              # YAML config files per experiment
├── notebooks/            # Exploratory data analysis
├── scripts/              # Download, train, evaluate entry points
├── tests/                # Unit tests
├── config.yaml           # Global configuration
└── requirements.txt
```

## 10. Tech Stack

| Category | Choices |
|----------|---------|
| Core | Python 3.10+, PyTorch 2.x, PyTorch Lightning |
| ML Baselines | XGBoost, LightGBM |
| Data Processing | pandas/polars, ta-lib/pandas-ta |
| NLP | transformers (FinBERT, BERT-wwm-chinese), jieba |
| Anti-Block Crawler | curl-cffi, browserforge, Playwright+stealth.js |
| Data Sources | AKShare, yfinance, Efinance, Tushare, Baostock, Finnhub |
| Storage | Parquet, SQLite, JSON |
| Config | Hydra / OmegaConf |
| Monitoring | wandb / TensorBoard |
| Code Quality | pytest, ruff |

## 11. Data Contracts

### Raw Price DataFrame Schema

Required: `[date, stock_code, open, high, low, close, volume, amount, pct_change]`
Optional: `[adj_factor, is_st, is_suspended, limit_up_price, limit_down_price]`

### Feature Matrix Schema

Input: `(n_samples, seq_len, n_features)` — 3D tensor
Labels: `(n_samples, n_tasks)` — multi-task label matrix

### Model Interface

```python
model.predict(features) -> predictions
model.train(train_loader, val_loader) -> metrics_dict
```

## 12. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| AKShare/Efinance API changes / rate limits | Multi-source failover chain; cache all raw data locally on first download |
| IP bans / anti-bot detection | 6-layer defense: TLS spoofing → fingerprint → sessions → proxies → rate limiting → browser fallback |
| Free proxies unreliable | Tiered architecture; paid tier ready to activate; proxy validation on startup |
| Free news sources incomplete / delayed | Start without news; only integrate if baseline performance is established |
| Price adjustment accuracy | Cross-validate with multiple sources; log all adjustments |
| Overfitting on financial data (low signal/noise) | Walk-forward validation; keep models small; focus on stable signals |
| GPU OOM on larger models | Gradient checkpointing; reduce batch/seq_len; mixed precision training |
| Model predicting noise not signal | Sentiment consensus analysis as diagnostic; financial metrics as truth metric | 
