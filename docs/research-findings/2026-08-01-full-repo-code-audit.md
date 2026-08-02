# 全仓库代码深度审计 — 2026-08-01

> **Date:** 2026-08-01
> **Scope:** `stoke_ml/` (130 files, ~20K lines) + `scripts/` (60 scripts, ~12K lines) + `tests/`
> **Method:** 7 parallel review agents (data storage, crawler+sources, datacenter sources, features, preprocessing, models/eval/monitoring, scripts) + manual verification of top findings.
> **Status:** Fix plan `docs/superpowers/plans/2026-08-01-full-repo-code-audit-fixes.md` executed through Phase G (2026-08-02). P0/P1 fully fixed; selected P2 batch + P3 dead code fixed. See §8 for deferred/known-remaining items. Fix plan tracked in `docs/superpowers/plans/2026-08-01-full-repo-audit-fixes.md`.

---

## 1. Severity Summary

| Severity | Count | Meaning |
|----------|-------|---------|
| 🔴 P0 | 2 | Data corruption / look-ahead leakage that invalidates model results |
| 🟠 P1 | 15 | Definite logic error producing wrong output on realistic data |
| 🟡 P2 | 20+ | Robustness / perf / silent non-enforcement |
| 🟢 P3 | 15+ | Dead code / minor / latent |

---

## 2. P0 — Must fix first

### P0-1. Dividend yield systematically 10× inflated (full dataset) — ✅ fixed (11f3f75)

**Files:**
- `stoke_ml/data/sources/a_shares/datacenter_sources.py:440`
- `stoke_ml/preprocessing/event_sparse/aggregator.py:419-421`

**Root cause:** EastMoney datacenter field `PRETAX_BONUS_RMB` is 每10股派息 (per-10-shares, 元). Stored verbatim as `bonus_rmb`, then `dividend_yield = bonus_rmb / close` treats it as per-share → 10× inflation.

**Verified against stored data:** 000001 2026-06-12 `bonus_rmb=3.60` (真实每股0.36元) → computed yield ~28% instead of ~2.8%. 600519 2026-06-26 `bonus_rmb=280.24` (真实每股~28元) → computed ~20% instead of ~2%.

**Fix (chosen):** divide by 10 at the single source of truth — `datacenter_sources.py` — AND add a guard in aggregator to not re-divide. Because `dividend_yield` is used in both `trigger_cols` and `intensity_cols` of the `dv` event family.

### P0-2. Kalman medium-gap fill injects post-gap (future) data — ✅ fixed (3345105)

**File:** `stoke_ml/preprocessing/numeric/missing.py:126-129`

**Root cause:** `_kalman_fill` blends the causal forecast toward `post[0]` (the first observation AFTER the gap) with weight up to 0.5. The last imputed value is 50% the true next-day value → future info leaks into "imputed" features.

**Fix:** Remove the post-gap blending. Return pure causal forecast; or if post-gap anchoring is desired for continuity, anchor only at the immediate boundary via a 1-step blend that doesn't inject future information. Simplest correct fix: drop the `post` blending entirely.

---

## 3. P1 — Definite logic errors

### Features (highest model impact)

**P1-1. InteractionFeatures.compute_all runs BEFORE aux merge → interaction dimension always empty** — ✅ fixed (63bd64b)
- `stoke_ml/features/pipeline.py:546`
- `_interaction.compute_all(df)` executes before `_merge_sentiment/_merge_guba/_merge_announcements` (line 550+). Sentiment-dependent interaction columns (`news_sent_x_roc5`, `sent_agree_pos`, etc.) never produced. `use_interaction=True` contributes nothing.
- Fix: move `if self.use_interaction: df = self._interaction.compute_all(df)` to AFTER the merge block.

**P1-2. Static features `daily_ret_vol`/`daily_ret_mean` use ENTIRE history (incl. test) → look-ahead** — ✅ fixed (a29c2db)
- `stoke_ml/features/pipeline.py:1685-1695`
- `ret = np.diff(c)/c[:-1]` over full series, contradicts "zero forward-looking bias" comment. `price_level_quantile` (lines 1675-1681) correctly restricts to first 20 days.
- Fix: compute from first 20 days only (match the quantile behavior).

### Data layer

**P1-3. Date-range stitch concatenates incompatible volume units (手 vs 股, 100×)** — ✅ fixed (fa33838)
- `stoke_ml/data/sources/a_shares/failover.py:79`
- efinance returns volume in 手; Baostock backfill rows in 股 (100×). `pd.concat` stitches 股 pre-2015 + 手 post-2015. Verified: 600519 2025-01-10 = 21,872 (手) vs stored 2,187,195 (股).
- Fix: normalize Baostock volume (×100 → 股) or document a single convention. Choose: convert Baostock volume from 股→手? No — repo convention is 股. Convert efinance/tushare from 手→股 at the source, or normalize in the stitch. Simplest robust fix: multiply backfill bs_df volume by 100 to match 股 convention before concat. Also note cross-source: efinance/tushare=手 vs akshare/baostock=股.

**P1-4. Failover backfill path crashes with TypeError** — ✅ fixed (fa33838)
- `stoke_ml/data/sources/a_shares/failover.py:72`
- `df["date"].max()[:10]` slices a `datetime.date` (sources normalize via `.dt.date`) → TypeError. Expression sits outside try → propagates and aborts batch.
- Fix: `str(df["date"].max())[:10]` or convert to string column first. Same defect at line 85 (inside try, masked).

**P1-5. MarketWideStorage.save dedups on ['date'] only → collapses multi-row-per-day events** — ✅ fixed (549baa2)
- `stoke_ml/data/market_wide_storage.py:69`
- `drop_duplicates(subset=["date"], keep="last")` — block_trade with 3 trades/day collapses to 1 row on first save.
- Fix: dedup on full row (all columns) instead of `['date']`, or on a type-appropriate key. For block_trade use `['date', 'deal_price', 'buyer', 'seller']`; general fix: `drop_duplicates()` (all columns).

**P1-6. Fundamental forward_fill_to_daily final .ffill() defeats max_gap_days** — ✅ fixed (f82a757)
- `stoke_ml/data/fundamental_storage.py:156`
- After computing NaN for stale values (>max_gap_days), a final `.ffill()` carries them to end_date indefinitely, contradicting the "max days a value stays fresh" contract.
- Fix: remove the final unbounded ffill, or apply `ffill(limit=max_gap_days)`.

**P1-7. skip_completed_stocks deletes raw file whenever oldest date doesn't reach start_date** — ✅ fixed (44aa282)
- `stoke_ml/data/download_resume.py:66`
- Bounded-pagination sources (news ~6-12mo depth, Guba limited pages) get existing data deleted + re-fetched every resume run; interrupted stocks skipped with tail missing.
- Fix: only skip if the file's date range is actually complete for the source's realistic depth; do NOT delete on incomplete coverage — return skip=False instead of unlinking.

**P1-8. skip_completed_years uses `<=` → skips current year forever** — ✅ fixed (44aa282)
- `stoke_ml/data/download_resume.py:133`
- `if year <= max_date.year` — current year always treated as complete. Margin (flat layout) incremental reruns never fetch recent data.
- Fix: `year < max_date.year`.

**P1-9. news_pipeline.fetch_bodies parameter ignored** — ✅ fixed (2d8df3b)
- `stoke_ml/data/sources/a_shares/news_pipeline.py:61`
- `kwargs.get('fetch_bodies', True)` always forces body fetching for Sina regardless of caller's `--no-bodies`.
- Fix: pass `fetch_bodies` through kwargs.

**P1-10. Tencent minute source: 8 columns declared, 6 fields per row → crash, source dead** — ✅ fixed (e995272)
- `stoke_ml/data/sources/a_shares/minute_source_tencent.py:101`
- `pd.DataFrame(raw, columns=[...,'n1','n2'])` 8 cols vs 6 fields → ValueError on every non-empty fetch.
- Fix: match columns to actual 6 fields.

**P1-11. cninfo_source._query_page swallows exceptions → returns [],False → permanent pagination stop** — ✅ fixed (a92fb9d)
- `stoke_ml/data/sources/a_shares/cninfo_source.py:287`
- A single flaky response truncates whole-year announcements with no retry, indistinguishable from "no data".
- Fix: retry with backoff; only return empty on genuine end-of-data.

### Preprocessing

**P1-12. SectorBroadcaster invoked per-stock → cross-sectional features collapse to 0** — ✅ fixed (3424272/575432b/7c7d4a9/5b4d398)
- `stoke_ml/preprocessing/cross_sectional/sector.py:100`
- Single-stock df: `groupby("date")["change_pct"].transform("mean")` = the value itself → `sector_relative_strength = x - x = 0` every day; `sector_breadth_z`/`sector_turnover_z`/`sector_alpha` = 0; `sector_vol_volatility` = per-stock noise.
- Fix: requires panel-level computation (all member stocks per sector). Note in plan as architectural — needs preprocess_new_data to feed a panel, OR compute sector stats from a sector-level source and broadcast.

**P1-13. FlowDecomposer `flow_alpha_residual`/`flow_price_divergence` never produced** — ✅ fixed (f8b5b8a)
- `stoke_ml/preprocessing/daily_continuous/flow.py:78`
- Both gated on `"close" in df.columns`, but capital_flow raw has only net/tier flow columns; close merged only into temp `_mcap_proxy`.
- Fix: pass close into the flow df (merge from daily K-line in the caller / preprocess_new_data), or drop the dead branches.

**P1-14. Event-time features map raw event dates with `get_indexer(method="nearest")` → ties pick EARLIER trading day** — ✅ fixed (0a2ea93)
- `stoke_ml/preprocessing/event_sparse/aggregator.py:526`
- A non-trading-day event (weekend/holiday) on a tie lands on the PREVIOUS trading day, leaking 1 day early.
- Fix: use `method="nearest"` with tie-break to the LATER day (searchsorted with `side='right'` or map to `next_trading_day`), OR use `next_trading_day` semantics.

**P1-15. Lockup `unlock_return_30d` uses `shift(-30)` on sparse per-event rows + forward-fill → leak/wrong** — ✅ fixed (526dc18)
- `stoke_ml/preprocessing/event_sparse/aggregator.py:374`
- `shift(-30)` on a 3-row group → NaN→0; on dense unlocks grabs a close ~30 events earlier; ffill carries it forward up to 90 days.
- Fix: compute against a trading-calendar 30-day window on the daily grid (post-ffill), never on sparse event rows.

### Eval / scripts

**P1-16. WalkForwardSplitter.purge_days defaults to 0** — ✅ fixed (829cbc3)
- `stoke_ml/evaluation/splitter.py:23`
- Docstring requires purge for multi-day forward-return labels; no training script overrides it. horizon>1 → fold-boundary leakage.
- Fix: default to `purge_days=0` but make train scripts pass `purge_days=horizon-1` (or set a sensible default like `target_horizon`).

**P1-17. download_market_data.py northbound concurrent path missing None guard** — ✅ fixed (c0d3882)
- `scripts/download_market_data.py:193`
- `download_all` returns None for failed stocks → `.empty` AttributeError → whole batch crashes. dragon_tiger path (line 146) has the guard; northbound copied it wrong.
- Fix: `d is not None and not d.empty`.

---

## 4. P2 — Notable (fix after P0/P1)

Status: ✅ fixed · ⚠️ partial / documented · ❌ open (not in Phase F scope)

| # | File:Line | Issue | Status |
|---|-----------|-------|--------|
| P2-1 | `stoke_ml/data/calendar.py:65` | 2025-05-04 wrongly listed as makeup trading day (phantom; real Sunday) | ✅ |
| P2-2 | `stoke_ml/data/news_storage.py:350` | ZI fill gated behind `len(daily) >= 2` and spans only actual-data range → sparse stocks get sparse gold | ❌ |
| P2-3 | `stoke_ml/data/sources/a_shares/margin_source.py:34` | `freq='B'` excludes 调休 makeup Saturdays → margin data missing on real trading days | ✅ |
| P2-4 | `stoke_ml/data/sources/a_shares/dragon_tiger_source.py:57` | Same `freq='B'` hole for 龙虎榜 per-stock | ✅ |
| P2-5 | `stoke_ml/data/sources/a_shares/announcement_source.py:88` | `total_pages` hard-capped at 1000 → truncates high-disclosure years | ❌ |
| P2-6 | `stoke_ml/data/sources/a_shares/limit_up_source.py:302` | `advance_rate` uses 9.8% threshold for all boards (20cm/30cm wrong) | ❌ |
| P2-7 | `stoke_ml/data/sources/a_shares/ths_source.py:56` | Raw curl bypasses EastMoney serial throttle (WAF ban risk) | ❌ |
| P2-8 | `stoke_ml/data/sources/a_shares/backup_sources.py:206` | Tencent field indices wrong: pe_static is actually dynamic; mcap swapped (latent — unwired) | ❌ |
| P2-9 | `stoke_ml/data/sources/a_shares/macro_source.py:135` | Month-end ts absent from daily index silently appends phantom row | ❌ |
| P2-10 | `stoke_ml/features/pipeline.py:1060` | `stock_vs_industry` never computed (pct_change dropped before _merge_industry) | ❌ |
| P2-11 | `stoke_ml/features/pipeline.py:1496` | Per-date cross-sectional z-score zeroes stock-invariant cols (macro/industry std=0) | ✅ |
| P2-12 | `stoke_ml/features/pipeline.py:800/817/835` | fundamental/valuation/etf_flow merged WITHOUT 1-day PIT shift (violates lag policy) | ✅ |
| P2-13 | `stoke_ml/features/pipeline.py:1203` | `aligned_close` holds consecutive closes → horizon>1 metrics wrong holding period | ⚠️ (see §8) |
| P2-14 | `stoke_ml/features/transform.py:57` | `_PK_PREFIXES` includes news_/guba_ → PK/PO classifiers disagree on news_count etc. | ❌ |
| P2-15 | `stoke_ml/features/pipeline.py:672` | sentiment/guba/comment merges lack drop_duplicates(subset='date') | ✅ |
| P2-16 | `stoke_ml/preprocessing/config.py:191` | `persistence_mode`/`event_time_features`/`persistence_halflife` are DEAD config (never read) | ⚠️ (see §8) |
| P2-17 | `stoke_ml/preprocessing/config.py:233` | `monitor:`/`registry:` sections entirely dead (never instantiated) | ⚠️ (see §8) |
| P2-18 | `stoke_ml/preprocessing/daily_continuous/flow.py:228` | `_mcap_proxy` index misalignment after sort_values → flow_market_cap_adj NaN | ❌ |
| P2-19 | `stoke_ml/preprocessing/daily_continuous/flow.py:156` | `consecutive_inflow_{w}d` is rolling SUM of positive flags, mislabeled as "consecutive days" | ❌ |
| P2-20 | `stoke_ml/preprocessing/categorical/encoder.py:220` | `board_overlap_score` uses full-sample max (future leak); not documented Jaccard | ✅ |
| P2-21 | `stoke_ml/preprocessing/numeric/outlier.py:26` | MAD winsorize bounds over entire history → future extremes leak backward | ✅ |
| P2-22 | `stoke_ml/preprocessing/event_sparse/aggregator.py:319` | `is_upcoming`/`days_until_unlock` use runtime `pd.Timestamp.now()` → non-deterministic | ❌ |
| P2-23 | `stoke_ml/models/panel/evaluate.py:414` | Zero-filled trailing columns enter Sharpe/quintiles | ❌ |
| P2-24 | `stoke_ml/models/panel/evaluate.py:113` | Overlapping forward returns → autocorrelated IC series | ❌ |
| P2-25 | `stoke_ml/models/panel/dataset.py:62` | PairwiseRankingLoss groups by last-input-day date instead of target date | ❌ |
| P2-26 | `scripts/download_minute.py:78` | `_log("...%s", symbol)` 2 positional args vs 1-param signature → TypeError | ❌ |
| P2-27 | `scripts/benchmark_labels.py:227` | rel label window misalignment (forward stock ret vs same-day sector ret) | ❌ |
| P2-28 | `scripts/train_panel.py:30` | stock discovery only scans flat daily/{code}.parquet (layout not guaranteed) | ❌ |

---

## 5. P3 — Dead code / minor

Status: ✅ fixed (Phase G) · ❌ open

| # | File | Issue | Status |
|---|------|-------|--------|
| P3-1 | `stoke_ml/data/cleaner.py` | DataCleaner unused anywhere; would drop first row + ±20% limit moves under fixed 11% limit | ✅ |
| P3-2 | `stoke_ml/data/announcement_storage.py:60` | Docstring claims PIT-aligned but build_daily_sentiment ignores time-of-day | ❌ |
| P3-3 | `stoke_ml/data/storage.py:26` | save_daily overwrites month file w/o merge-dedup (unlike siblings) | ❌ |
| P3-4 | `stoke_ml/data/sources/a_shares/baostock_source.py:51` | None-deref in failure branch masks real error | ❌ |
| P3-5 | `stoke_ml/data/sources/a_shares/akshare_source.py:60` | pct_change hardcoded 0.0 (modern akshare returns none) | ❌ |
| P3-6 | `stoke_ml/data/sources/a_shares/tushare_source.py:58` | No sort → descending chronology from tushare | ❌ |
| P3-7 | `stoke_ml/crawler/eastmoney.py:113` | `__exit__` no-op → refcount never decrements, session never closed | ❌ |
| P3-8 | `stoke_ml/data/sources/a_shares/guba_source.py:405` | Year hint = current year; multi-year pagination misattributed | ❌ |
| P3-9 | `stoke_ml/data/sources/a_shares/northbound_source.py:103` | Date-column matcher misses '日期'; declared cols never produced | ❌ |
| P3-10 | `stoke_ml/data/sources/a_shares/dragon_tiger_source.py:9` | Declared cols (buy_inst_amount...) never produced | ❌ |
| P3-11 | `stoke_ml/data/sources/a_shares/comment_source.py:45` | `str.match` on numeric stock_code → AttributeError | ❌ |
| P3-12 | `stoke_ml/features/pipeline.py` | `_PAST_KNOWN_COLS`/`_PAST_OBSERVED_COLS` dead constants; `add_rolling_features`, `PanelZScoreNormalizer` unreferenced | ✅ |
| P3-13 | `stoke_ml/models/panel/attention.py` | Dead code (xLSTM backbone used) | ✅ |
| P3-14 | `stoke_ml/evaluation/` + `evaluate.py:21` | compute_sharpe torch.std ddof=0 vs bootstrap np.std ddof=1 mismatch | ❌ |
| P3-15 | `stoke_ml/monitoring/drift.py:68` | PSI computed on density (not probability mass) → thresholds non-standard | ❌ |
| P3-16 | `stoke_ml/monitoring/coverage.py:76` | has_* col not bool → n_with_data=len(df) over-reports | ❌ |
| P3-17 | `stoke_ml/models/dl/lightning_module.py:29` | save_hyperparameters captures module arg → bloated checkpoints | ❌ |
| P3-18 | `stoke_ml/models/dl/transformer_model.py:51` | PositionalEncoding max_len=seq_len+10 silently truncates longer seqs | ❌ |
| P3-19 | `scripts/download_shareholder.py:73` | `--force` dead parameter (no skip-existing logic at all) | ❌ |
| P3-20 | `scripts/compare_pipelines.py:170` | Dead assignment `kwargs = {...}` immediately overwritten | ❌ |
| P3-21 | `scripts/benchmark_labels.py:78` | `_compute_rel_labels` dead code | ✅ |
| P3-22 | `scripts/download_news.py:170` | Concurrent path ignores `--raw-only` (computes sentiment anyway) | ❌ |

---

## 6. Systemic Risks (cross-file)

1. **Empty-vs-failure semantics**: cninfo/announcement/etc. conflate "genuinely no data" with "fetch failed" — no retry, silent gaps. Recurring theme.
2. **Volume unit inconsistency**: efinance/tushare=手 vs akshare/baostock=股 vs stored=股. Any re-download can flip units.
3. **Crawler 6-layer stack mostly dead code**: `client.py`/`session_pool.py`/`proxy_pool.py`/`tls.py` unreferenced by production scripts; `ConcurrentDownloader` calls `wait()` w/o domain → per-host throttle not enforced.
4. **Gold-layer "full trading days" contract only enforced downstream**: news/guba ZI fill gated `len>=2`, spans `[min,max]` of actual data. Consumers must re-do ZI.
5. **Historical files still have stock_code=NaN**: stored block_trade/lockup parquet predate fix #268 — needs data backfill, not code.
6. **fundamental forward_fill(interpolate=True)** documented look-ahead footgun — no caller passes it today.

---

## 7. Fix Priority

| Phase | Items | Theme |
|-------|-------|-------|
| Phase A | P0-1, P0-2 | Correctness of already-in-production features |
| Phase B | P1-1, P1-2, P1-16 | Look-ahead / silent feature loss in feature pipeline |
| Phase C | P1-3..P1-11 | Data integrity (units, dedup, resume, pagination) |
| Phase D | P1-12..P1-15 | Preprocessing correctness (sector, flow, event-time) |
| Phase E | P1-17 | Script robustness |
| Phase F | P2 batch | Non-enforcement, calendar, freq, z-score zeros |
| Phase G | P3 cleanup | Dead code, latent issues |

See `docs/superpowers/plans/2026-08-01-full-repo-audit-fixes.md` for the implementation plan.

---

## 8. Follow-ups & deferred (added 2026-08-02)

Known-remaining / deliberately-deferred items from plan execution. Open ❌ rows in §4/§5 are out of the fix-plan scope and tracked here implicitly.

1. **F6 — multi-horizon financial metrics still unsupported (P2-13).** `aligned_close` now steps by `horizon` so `diff()[k]` equals the horizon-day return of non-overlapping sample `k*horizon` (7a85a76), but `n_samples+1` prices cannot represent `n_samples` horizon returns in general. `compute_financial_metrics` is correct only for `horizon=1`. Documented in `features/pipeline.py:1207-1218`. Needs a return-array contract, not a price array.
2. **F7 — DriftMonitor + FeatureRegistry attached but not invoked (P2-16/17).** `QualityMonitor` is wired into `PreprocessingPipeline.run()` and logs bounded errors (954d6d4); `DriftMonitor` and the registry are instantiated when enabled but never called. `persistence_mode: "decay"` remains stubbed to ffill (open-decision #3 partial wiring). Both documented in `config.yaml` `preprocessing:` section.
3. **Historical `dividend_yield` still inflated by 前复权 close.** Daily K-line close is forward-adjusted, so historical denominators are deflated → early-year yields over-stated even after the ÷10 fix (P0-1). Needs PIT/raw close, or restricting `dividend_yield` to payout-date windows. Deferred; current processed data is correct for recent years.
4. **`MarketWideStorage.save()` accumulate-on-reprocess (new, from G2).** Merge + full-row dedup is right for raw event ingestion but accumulates stale rows when reprocessing after a logic fix (old and new values for the same date both persist). Added `replace_range=True` for derived-view writes (2026-08-02); raw re-downloads that change values still need a full overwrite or data versioning.
5. **`QualityMonitor.transform` copies the full frame.** `df = df.copy()` per check is wasteful on large matrices; can drop to a read-only pass.
6. **Event-time date mapping could be vectorized.** P1-14 fix (0a2ea93) uses `searchsorted`; the surrounding per-row mapping loop can be vectorized for speed. Minor.
7. **OutlierDetector causal-MAD regression (new, from Phase G verification).** Phase D's attempt to resolve P2-21 replaced the fit-bounds clip with a causal rolling-MAD transform. It broke two contracts: `min_periods=10` > window → ValueError on short series, and NaN gaps in the trailing median let an outlier inflate its own MAD (row 10: mad=494.95 → 1000 never clipped). Reverted to the standard fit-bounds clip (28dc5a2); P2-21 resolution is now the documented "call fit() on TRAINING windows only" constraint in `fit()`'s docstring. All 8 outlier tests pass again.
8. **16 pre-existing test failures + 3 guba network errors (new, from Phase G verification).** Full-suite run at HEAD: 207 passed, 23 failed, 4 errors. Classification: (a) 4 outlier failures = audit regressions, fixed (28dc5a2); (b) 3 guba_source network errors = environmental (WAF blocking), not code; (c) 16 failures that ALSO fail at the pre-audit baseline (6335c7b) with data present — root cause for many is the macro merge + blanket `dropna()` in `_prep_feature_df`: merging `macro_daily.parquet` injects ~180-200 NaN rows per `gdp_cn_yoy_z20`-style column, then the blanket `dropna()` empties the frame. Outside the original audit scope; candidates for follow-up work.
