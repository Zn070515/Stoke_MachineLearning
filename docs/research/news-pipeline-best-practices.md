# News Pipeline Best Practices — Research Summary

> Compiled 2026-06-19 from academic papers (ACL, Springer, MDPI 2024-2025),
> open-source projects (Qlib, FinNLP, TradingAgents-CN), and industry
> reference architectures (Databricks, Numerai).

## 1. Storage: Medallion Architecture (Bronze → Silver → Gold)

Industry standard across Databricks/numerai/pfeed:

```
Bronze (raw, immutable)    Silver (cleaned, aligned)    Gold (features, ML-ready)
─────────────────────────  ──────────────────────────  ──────────────────────────
News HTML/JSON as-is       PIT-aligned by trading day   Daily sentiment aggregates
OHLCV raw from API         Dedup + quality checks       Joined price+sentiment rows
                           Timezone-normalized          Ready for train/test split
```

**Format**: Parquet throughout. Columnar, compressed, pandas/Polars/DuckDB native.
**Partition**: By date (year/month) for time-series; by ticker for entity-level data.

## 2. Sentiment Models

| Tier | Model | Accuracy | Use Case |
|------|-------|----------|----------|
| L1 (lexicon) | VADER, SnowNLP, TextBlob | ~65% | Fast baseline, offline |
| L2 (transformer) | **FinBERT** (ProsusAI) | 85-90% | De facto standard |
| L2 (Chinese) | Erlangshen-RoBERTa (110M) | 88-92% | A-share news, beats GPT-3.5 |
| L3 (LLM) | GPT-4, Gemini | 90%+ | Explanation, not prediction |

**Key findings**:
- **Ensemble > single model**: FinBERT + RoBERTa + VADER weighted vote improves 2-4% over best single model
- **Erlangshen-RoBERTa-110M outperformed GPT-3.5 (175B)** on Chinese financial sentiment
- **Domain adaptation is essential**: general BERT 56% → FinBERT 90%+ on financial text

## 3. Date Alignment (PIT) — Most Critical Step

> "Date alignment rigor is far more important than model architecture choice."

Three iron rules from ACL/MDPI papers:

1. **Post-close news → next trading day**: News after 15:00 CST must be deferred to T+1
2. **Gap day**: When predicting T+N, leave N-1 gap days between train window and test point
3. **ZI method** (Zeros & Imputation): Missing-news days get zero-filled sentiment + binary `has_news` flag

## 4. Feature Engineering

### Structured (price/volume) — already implemented
SMA/EMA crossovers, RSI, MACD, Bollinger Bands, ATR, OBV, volume ratios

### Unstructured (news/sentiment) — need to add
- `sentiment_mean`, `sentiment_std` (daily aggregation)
- `positive_ratio`, `negative_ratio`
- `news_count`, `has_news` (binary indicator)
- Optional: Rolling 5-day sentiment mean, sentiment-volatility

### Anti-leakage rules
- All features computed from today's close or earlier
- StandardScaler fit on training window only
- Chronological splits only (no shuffle)

## 5. Model Selection

| Model | When to Use | Notes |
|-------|-------------|-------|
| XGBoost/LightGBM | Structured features baseline | Best for tabular data |
| LSTM/GRU | Pure price sequences | Simple, well-understood |
| Transformer + Cross-Attention | Price + text fusion | Best but complex |
| Mamba/SSM | Long sequences (>500) | O(T) vs Transformer O(T²) |

## 6. Evaluation

- **MCC** (Matthews Correlation Coefficient) — preferred for imbalanced directional prediction
- **Directional Accuracy** — 67-70% is competitive on CMIN datasets
- **Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor** — financial metrics
- **Walk-forward validation** with chronological splits and gap days
- **Per-era evaluation** (Numerai): compute metrics per time period, then average

## 7. Chinese A-Share Specific Resources

| Resource | Description |
|----------|-------------|
| FinNLP (AI4Finance) | Multi-source Chinese financial data SDK |
| StockSentCN | 9.23M labeled Chinese stock comments |
| EFSA (ACL 2024) | Event-level financial sentiment, 12K Chinese news |
| TradingAgents-CN | Multi-agent LLM framework for A-shares |
| CKIP BERT-base-Chinese | Base model for Chinese FinBERT pre-training |

## 8. Our Implementation Decisions

1. **Storage**: 3-layer Parquet medallion (news_raw → news_silver → sentiment)
2. **Sentiment L1**: SnowNLP (offline, fast, already working) → L2 upgrade path to FinBERT Chinese
3. **PIT**: Post-close (15:00 CST) news → next trading day via TradingCalendar
4. **Missing data**: ZI method (zeros + has_news flag)
5. **Integration**: Left-join daily sentiment onto K-line features by (date, stock_code)
