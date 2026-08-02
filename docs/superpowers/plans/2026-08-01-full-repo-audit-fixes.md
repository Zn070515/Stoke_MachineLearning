# 全仓库审计修复计划 (2026-08-01)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 2 P0 + 15 P1 + prioritized P2/P3 findings from the 2026-08-01 full-repo audit, restoring correctness to the feature pipeline and eliminating look-ahead leakage.

**Architecture:** Fixes are grouped by subsystem and applied in dependency order: P0 correctness → feature-pipeline leakage → data-integrity → preprocessing → eval/scripts. Each task is independently verifiable. No API changes that break existing callers.

**Tech Stack:** Python 3.10+, pandas, numpy, config.yaml (OmegaConf), pytest (where tests exist).

**Source of findings:** `docs/research-findings/2026-08-01-full-repo-code-audit.md`

---

## Phase A — P0: Correctness of in-production features

### Task A1: Fix dividend yield 10× inflation

**Files:**
- Modify: `stoke_ml/data/sources/a_shares/datacenter_sources.py:440`

**Background:** EastMoney `PRETAX_BONUS_RMB` is per-10-shares (每10股派息, 元). It is stored verbatim as `bonus_rmb` then `dividend_yield = bonus_rmb / close` in `aggregator.py:419` treats it as per-share → 10× inflation.

- [ ] **Step 1: Change the source to store per-share value**

```python
# datacenter_sources.py:440  (inside DividendSource.fetch_batch row loop)
"bonus_rmb": float(r.get("PRETAX_BONUS_RMB") or 0) / 10.0,
```

- [ ] **Step 2: Verify with a python one-liner**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.data.sources.a_shares.datacenter_sources import DividendSource
import inspect
src = inspect.getsource(DividendSource.fetch_batch)
assert '/ 10.0' in src, 'bonus_rmb not divided by 10'
print('OK: bonus_rmb now per-share')
"
```
Expected: `OK: bonus_rmb now per-share`

- [ ] **Step 3: Add a guard comment in aggregator so it's not re-divided**

**File:** `stoke_ml/preprocessing/event_sparse/aggregator.py:419`

```python
            if "close" in df.columns:
                # bonus_rmb is per-share (divided by 10 at the source) —
                # dividend_yield is a proper per-share yield.
                df["dividend_yield"] = (
                    df["bonus_rmb"] / df["close"].replace(0, np.nan)
                ).astype(np.float32)
```

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/data/sources/a_shares/datacenter_sources.py stoke_ml/preprocessing/event_sparse/aggregator.py
git commit -m "fix: dividend bonus_rmb per-10-share → per-share (10x yield inflation)"
```

> **Note:** Existing stored `dividend_processed` parquet files are still 10× wrong. After code fix, re-run dividend preprocessing (Phase G Task G2) to regenerate. See `docs/research-findings/2026-08-01-full-repo-code-audit.md` P0-1.

---

### Task A2: Remove Kalman post-gap future blending

**Files:**
- Modify: `stoke_ml/preprocessing/numeric/missing.py:126-129`

**Background:** `_kalman_fill` blends the causal forecast toward `post[0]` (first value AFTER the gap) with weight up to 0.5 → future data injected into imputed past values.

- [ ] **Step 1: Remove the post-gap blending**

Replace lines 125-130 with:

```python
            # Causal forecast only — do NOT blend toward the post-gap
            # anchor: that would leak future information into imputed rows.
            return forecast
```

- [ ] **Step 2: Verify no `post`-based blending remains**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/preprocessing/numeric/missing.py').read_text(encoding='utf-8')
assert 'anchor * blend' not in src, 'future-blend still present'
assert 'post[0]' not in src, 'post-gap anchor still used'
print('OK: kalman fill is causal-only')
"
```
Expected: `OK: kalman fill is causal-only`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/preprocessing/numeric/missing.py
git commit -m "fix: kalman gap fill no longer blends toward post-gap (future) data"
```

---

## Phase B — Feature pipeline leakage / silent feature loss

### Task B1: Move InteractionFeatures after aux merges

**Files:**
- Modify: `stoke_ml/features/pipeline.py:546-547`

**Background:** `_interaction.compute_all(df)` runs at line 546-547 BEFORE the aux merges (lines 550+). Sentiment/guba/ann columns don't exist yet → interaction features always empty.

- [ ] **Step 1: Remove the call from its current position (lines 546-547)**

```python
        if self.use_interaction:
            df = self._interaction.compute_all(df)
```

- [ ] **Step 2: Re-insert it AFTER the merge block (after line 569, before defrag at 571)**

```python
        df = self._merge_industry(df, industry_df)

        # Interaction features require merged sentiment columns — must run
        # after the aux merges (was previously a silent no-op at line 546).
        if self.use_interaction:
            df = self._interaction.compute_all(df)

        # Defragment after merge calls
        df = df.copy()
```

- [ ] **Step 3: Verify interaction columns now appear**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
from stoke_ml.features.pipeline import FeaturePipeline
p = FeaturePipeline(seq_len=5, horizon=1, flat_mode=True)
kl = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=50, freq='B'),
                   'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5,
                   'volume': 1_000_000, 'amount': 10_000_000.0})
sd = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=50, freq='B'),
                   'sentiment_mean': 0.3, 'sentiment_std': 0.1,
                   'news_count': 5, 'positive_ratio': 0.6, 'negative_ratio': 0.2,
                   'has_news': True})
X, y, ac = p.build_features(kl, sentiment_df=sd)
sent_x = [c for c in X.columns if 'sent_x_roc' in c or 'sent_bull_vol' in c]
assert sent_x, 'no interaction columns produced'
print('OK: interaction cols:', len(sent_x))
"
```
Expected: `OK: interaction cols: N` (N ≥ 4)

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "fix: interaction features now computed after aux merge (was silent no-op)"
```

---

### Task B2: Restrict static features `daily_ret_vol`/`daily_ret_mean` to first 20 days

**Files:**
- Modify: `stoke_ml/features/pipeline.py:1685-1695`

**Background:** `daily_ret_vol`/`daily_ret_mean` use the ENTIRE close history (incl. test period) → look-ahead in static features. `price_level_quantile` already correctly restricts to first 20 days.

- [ ] **Step 1: Restrict to first 20 days**

Replace lines 1685-1695 with:

```python
    if "daily_ret_vol" in needed or "daily_ret_mean" in needed:
        for i, df in enumerate(all_feat_dfs):
            if len(df) >= 3 and "close" in df.columns:
                n_days = min(first_n, len(df))   # first_n = 20, no look-ahead
                c = df["close"].iloc[:n_days].values.astype(np.float64)
                ret = np.diff(c) / (c[:-1] + 1e-8)
                ret = ret[np.isfinite(ret)]
                df["daily_ret_vol"] = float(np.std(ret)) if len(ret) > 1 else 0.0
                df["daily_ret_mean"] = float(np.mean(ret)) if len(ret) > 0 else 0.0
            else:
                df["daily_ret_vol"] = 0.0
                df["daily_ret_mean"] = 0.0
```

- [ ] **Step 2: Verify first-20-days restriction**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import inspect, re
from stoke_ml.features.pipeline import _compute_static_quantiles
src = inspect.getsource(_compute_static_quantiles)
assert 'first_n' in src and 'iloc[:n_days]' in src
assert '.values.astype(np.float64)' in src
print('OK: daily_ret_* restricted to first 20 days')
"
```
Expected: `OK: daily_ret_* restricted to first 20 days`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "fix: static daily_ret_vol/mean no longer use full history (look-ahead)"
```

---

### Task B3: WalkForwardSplitter purge gap for multi-day labels

**Files:**
- Modify: `stoke_ml/evaluation/splitter.py:18-28`

**Background:** `purge_days` defaults to 0; docstring says a purge is required for multi-day forward-return labels. All training scripts construct the splitter without passing `purge_days` (train_baseline.py:110, train_lstm.py:103, etc.).

- [ ] **Step 1: Make purge_days default derived from a horizon param**

Change constructor to:

```python
    def __init__(
        self,
        train_years: int = 2,
        val_months: int = 3,
        step_months: int = 3,
        purge_days: int | None = None,
        horizon: int = 1,
    ):
        self.train_days = train_years * 252
        self.val_days = val_months * 21
        self.step_days = step_months * 21
        # Purge (horizon - 1) days so multi-day labels at the fold boundary
        # don't read val-window closes.  Explicit purge_days wins.
        self.purge_days = purge_days if purge_days is not None else max(horizon - 1, 0)
```

- [ ] **Step 2: Update the 3 training scripts to pass horizon**

**Files:** `scripts/train_baseline.py:110`, `scripts/train_lstm.py:103`, `scripts/train_transformer.py:102`, `scripts/run_ablation.py:157`, `scripts/run_all_lstm.py:93`, `scripts/benchmark_preprocessing.py:157`, `scripts/benchmark_labels.py:160`, `scripts/benchmark_feature_selection.py:201`

In each, change:
```python
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    )
```
to:
```python
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
        horizon=cfg.features.target_horizon,
    )
```

- [ ] **Step 3: Verify default purge for horizon>1**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.evaluation.splitter import WalkForwardSplitter
import pandas as pd
dates = pd.date_range('2020-01-01', periods=1000, freq='B')
s = WalkForwardSplitter(train_years=2, val_months=3, horizon=5)
folds = list(s.split(dates))
tr, va = folds[0]
assert va[0] - tr[-1] == 5, (va[0], tr[-1])
print('OK: purge gap = 5 (horizon-1)')
"
```
Expected: `OK: purge gap = 5 (horizon-1)`

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/evaluation/splitter.py scripts/train_baseline.py scripts/train_lstm.py scripts/train_transformer.py scripts/run_ablation.py scripts/run_all_lstm.py scripts/benchmark_preprocessing.py scripts/benchmark_labels.py scripts/benchmark_feature_selection.py
git commit -m "fix: walk-forward purge gap defaults to horizon-1 to stop boundary leakage"
```

---

## Phase C — Data integrity

### Task C1: Fix failover backfill crash + volume unit mismatch

**Files:**
- Modify: `stoke_ml/data/sources/a_shares/failover.py:72, 79-85`

**Background:** Two defects in the date-range stitch: (a) `df["date"].max()[:10]` slices a `datetime.date` → TypeError at line 72 (outside try); (b) Baostock volume is 股 while efinance is 手 (100× mismatch) concatenated together.

- [ ] **Step 1: Fix the date-slice crash and normalize volume to 股**

Replace the logger line (72) with a safe string conversion, and normalize bs_df volume after concat:

```python
            gap_end = got_start - pd.Timedelta(days=1)
            gap_end_str = gap_end.strftime("%Y-%m-%d")
            got_start_str = str(got_start)
            got_end_str = str(pd.to_datetime(df["date"]).max().date() if len(df) else "?")
            logger.info(
                "  %s: %s returned %s→%s, backfilling %s→%s via Baostock",
                stock_code, source_used, got_start_str, got_end_str,
                start_date, gap_end_str,
            )
```

And inside the `try` block after concat (lines 79-81), add a volume-unit normalization. The stored convention is 股; efinance/tushare return 手, so convert the *stitched* series to 股 uniformly:

```python
                if len(bs_df) > 0:
                    df = pd.concat([bs_df, df], ignore_index=True)
                    df = df.sort_values("date").reset_index(drop=True)
                    df = df.drop_duplicates(subset="date", keep="last")
                    # Unit normalization: stored convention is 股.
                    # efinance/tushare volume is 手 (×100) — scale to 股.
                    if "volume" in df.columns:
                        vol = pd.to_numeric(df["volume"], errors="coerce")
                        df["volume"] = (vol * 100.0).astype(np.float32)
```

- [ ] **Step 2: Verify no `[:10]` on date and unit scale present**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/sources/a_shares/failover.py').read_text(encoding='utf-8')
assert 'df[\"date\"].max()[:10]' not in src
assert 'str(pd.to_datetime(df[\"date\"]).max().date()' in src
assert 'vol * 100.0' in src
print('OK: failover crash + volume unit fixed')
"
```
Expected: `OK: failover crash + volume unit fixed`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/sources/a_shares/failover.py
git commit -m "fix: failover backfill date-slice crash + volume 手→股 unit normalization"
```

---

### Task C2: MarketWideStorage.save dedup on full row, not date only

**Files:**
- Modify: `stoke_ml/data/market_wide_storage.py:69`

**Background:** `drop_duplicates(subset=["date"])` collapses multi-row-per-day event data (block_trade: multiple trades same day) to a single row.

- [ ] **Step 1: Dedup on full row**

```python
            # Dedup identical rows (not by date only — block_trade has
            # multiple trades per day that must all be preserved).
            new_rows = new_rows.drop_duplicates(keep="last")
```

- [ ] **Step 2: Verify dedup change**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/market_wide_storage.py').read_text(encoding='utf-8')
assert 'drop_duplicates(subset=[\"date\"]' not in src
print('OK: dedup is full-row')
"
```
Expected: `OK: dedup is full-row`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/market_wide_storage.py
git commit -m "fix: MarketWideStorage dedup preserves multi-row-per-day events (block_trade)"
```

---

### Task C3: Fundamental forward-fill must respect max_gap_days

**Files:**
- Modify: `stoke_ml/data/fundamental_storage.py:156`

**Background:** The final `.ffill()` carries stale values (beyond max_gap_days) to end_date indefinitely.

- [ ] **Step 1: Replace unbounded ffill with limit-aware ffill**

```python
            else:
                # ffill with the expiry window so stale values don't persist
                # past max_gap_days (which the per-row loop already NaN'd).
                result[col] = result[col].ffill(
                    limit=max_gap_days if max_gap_days > 0 else None
                )
```

- [ ] **Step 2: Verify limit applied**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/fundamental_storage.py').read_text(encoding='utf-8')
assert 'ffill(' in src and 'limit=max_gap_days' in src
print('OK: ffill respects max_gap_days')
"
```
Expected: `OK: ffill respects max_gap_days`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/fundamental_storage.py
git commit -m "fix: fundamental ffill no longer defeats max_gap_days expiry"
```

---

### Task C4: download_resume — stop destructive re-download + skip_completed_years `<=`

**Files:**
- Modify: `stoke_ml/data/download_resume.py:66-78, 133-137`

**Background:** Two bugs: (a) `skip_completed_stocks` DELETES the file when the oldest date doesn't reach start_date — for bounded-pagination sources this destroys data every resume run; (b) `skip_completed_years` flat-layout uses `year <= max_date.year` so the current year is always skipped.

- [ ] **Step 1: Stop deleting when coverage is partial (lines 66-78)**

```python
            if oldest <= start_ts:
                skipped += 1
                continue
            # Data exists but doesn't reach start_date (e.g. bounded
            # pagination on news).  Do NOT delete — just skip this run so
            # we don't destroy existing data and re-fetch it next time.
            skipped += 1
            logger.debug(
                "  %s: oldest %s is after %s — treating as complete (bounded source)",
                code, str(oldest.date()), str(start_ts.date()),
            )
            continue
```

Remove the `_safe_unlink(path)` call and the trailing `pending.append(code)` fallthrough for this branch.

- [ ] **Step 2: Fix skip_completed_years `<=` → `<` (line 134)**

```python
    for year in years:
        if year < max_date.year:
            skipped += 1
        else:
            pending.append(year)
```

- [ ] **Step 3: Verify both fixes**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/download_resume.py').read_text(encoding='utf-8')
assert 'year < max_date.year' in src, 'skip_completed_years not fixed'
assert src.count('_safe_unlink(path)') <= 1, 'destructive re-download still present'
print('OK: resume no longer destroys partial coverage; current year fetched')
"
```
Expected: `OK: resume no longer destroys partial coverage; current year fetched`

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/data/download_resume.py
git commit -m "fix: download_resume stops deleting partial-coverage files; skip_completed_years fetches current year"
```

---

### Task C5: news_pipeline fetch_bodies parameter honored

**Files:**
- Modify: `stoke_ml/data/sources/a_shares/news_pipeline.py:60-61`

**Background:** `kwargs.get("fetch_bodies", True)` reads a key never set — the caller's `fetch_bodies` arg is ignored.

- [ ] **Step 1: Thread the parameter through**

```python
                if source_name in _BODY_SOURCES:
                    kwargs["fetch_bodies"] = fetch_bodies
```

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/sources/a_shares/news_pipeline.py').read_text(encoding='utf-8')
assert 'kwargs[\"fetch_bodies\"] = fetch_bodies' in src
assert 'kwargs.get(\"fetch_bodies\", True)' not in src
print('OK: fetch_bodies honored')
"
```
Expected: `OK: fetch_bodies honored`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/sources/a_shares/news_pipeline.py
git commit -m "fix: news_pipeline honors fetch_bodies (--no-bodies was ignored)"
```

---

### Task C6: Tencent minute source column count mismatch

**Files:**
- Modify: `stoke_ml/data/sources/a_shares/minute_source_tencent.py:101-103`

**Background:** DataFrame declared 8 columns (`time_str, open, close, high, low, volume, n1, n2`) but Tencent mkline returns 6 fields/bar → ValueError on every non-empty fetch; `--source tencent` is dead.

- [ ] **Step 1: Confirm actual field count from the fetch path**

Read the top of the file to see what `_fetch_bars` returns. Tencent mkline returns 6 fields: `[datetime, open, close, high, low, volume]`.

```python
        columns = ["time_str", "open", "close", "high", "low", "volume"]
        df = pd.DataFrame(raw, columns=columns)
```

(Remove `n1`, `n2` from `columns` and delete the `drop(columns=["n1", "n2"])` line.)

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/sources/a_shares/minute_source_tencent.py').read_text(encoding='utf-8')
assert '\"n1\"' not in src, 'n1 still in columns'
print('OK: tencent columns match 6-field schema')
"
```
Expected: `OK: tencent columns match 6-field schema`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/sources/a_shares/minute_source_tencent.py
git commit -m "fix: tencent minute source 8-col vs 6-field mismatch (source was dead)"
```

---

### Task C7: cninfo pagination — retry transient failures instead of silent stop

**Files:**
- Modify: `stoke_ml/data/sources/a_shares/cninfo_source.py:261-288`

**Background:** `_query_page` returns `([], False)` on ANY exception → pagination terminates permanently on one flaky response; indistinguishable from "no data".

- [ ] **Step 1: Add retry with backoff, distinguish failure from end-of-data**

Replace the request/except block with:

```python
        for attempt in range(3):
            try:
                resp = self._session.post(
                    CNINFO_QUERY, data=body, headers=HEADERS,
                    timeout=30, impersonate="chrome120",
                )
                if resp.status_code != 200:
                    continue  # retry
                data = resp.json()
                items = []
                for ann in data.get("announcements", []):
                    title = (ann.get("announcementTitle") or "").strip()
                    title = re.sub(r"<[^>]+>", "", title)
                    ts = ann.get("announcementTime", 0)
                    date_str = (
                        time.strftime("%Y-%m-%d", time.localtime(ts / 1000)) if ts
                        else ""
                    )
                    adjunct = ann.get("adjunctUrl") or ""
                    items.append({
                        "date": date_str,
                        "title": title,
                        "notice_type": ann.get("announcementTypeName") or "",
                        "url": adjunct,
                    })
                has_more = bool(data.get("hasMore"))
                return items, has_more
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                # 3 failed attempts — genuinely degraded; surface as failure
                raise
        return [], False  # unreachable, for type-checker
```

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/data/sources/a_shares/cninfo_source.py').read_text(encoding='utf-8')
assert 'for attempt in range(3)' in src
assert 'raise' in src, 'no re-raise on persistent failure'
print('OK: cninfo pagination retries transient failures')
"
```
Expected: `OK: cninfo pagination retries transient failures`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/data/sources/a_shares/cninfo_source.py
git commit -m "fix: cninfo pagination retries transient failures instead of silent truncation"
```

---

## Phase D — Preprocessing correctness

### Task D1: SectorBroadcaster — compute sector stats on panel, not per-stock

**Files:**
- Modify: `stoke_ml/preprocessing/cross_sectional/sector.py:99-102`

**Background:** `SectorBroadcaster` is invoked per-stock, so `sector_relative_strength = change_pct - groupby("date")["change_pct"].transform("mean")` = x - x = 0. Same for breadth_z/turnover_z/alpha.

**Approach (minimal, correct):** The sector stats that collapse are those computed from per-stock groupbys on a single-stock df. The `industry_ranking`-derived features (momentum, RRG, rank_change, is_top5) are correct because they use the cross-sector `ir` frame. Fix: compute the collapse-prone features from `ir` (which has all sectors) and broadcast by sector_code, instead of the per-stock groupby.

- [ ] **Step 1: Add a broadcast helper and rewrite the three collapsing blocks**

Add to `transform()` after `df = self._add_sector_momentum(df, ir)`:

```python
        # Cross-sectional features must be computed on the PANEL (all
        # sectors from industry_ranking), then broadcast to each stock by
        # sector_code.  Per-stock groupbys collapse to x - x = 0.
        if not ir.empty and "sector_code" in ir.columns:
            panel = ir.groupby(["date", "sector_code"], as_index=False)["change_pct"].mean()
            panel = panel.rename(columns={"change_pct": "_sector_mean"})
            if "sector_code" in df.columns:
                df = df.merge(panel, on=["date", "sector_code"], how="left")
                if "change_pct" in df.columns and "_sector_mean" in df.columns:
                    df["sector_relative_strength"] = (
                        df["change_pct"] - df["_sector_mean"]
                    ).fillna(0.0).astype(np.float32)
                    df.drop(columns=["_sector_mean"], inplace=True)
```

- [ ] **Step 2: Remove the old per-stock block (lines 99-102)**

```python
        if "change_pct" in df.columns:
            df["sector_relative_strength"] = (
                df["change_pct"] - df.groupby("date")["change_pct"].transform("mean")
            ).astype(np.float32)
```

- [ ] **Step 3: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/preprocessing/cross_sectional/sector.py').read_text(encoding='utf-8')
assert '_sector_mean' in src and 'sector_relative_strength' in src
assert 'df.groupby(\"date\")[\"change_pct\"].transform(\"mean\")' not in src
print('OK: sector_relative_strength uses panel mean')
"
```
Expected: `OK: sector_relative_strength uses panel mean`

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/preprocessing/cross_sectional/sector.py
git commit -m "fix: sector broadcaster computes cross-sectional stats on panel not per-stock"
```

---

### Task D2: FlowDecomposer — wire close into residual/divergence

**Files:**
- Modify: `stoke_ml/preprocessing/daily_continuous/flow.py:78`

**Background:** `flow_alpha_residual`/`flow_price_divergence` gated on `"close" in df.columns`, but the flow df has no close column → features never produced.

**Approach (minimal):** The residualization needs close. `preprocess_new_data.py` passes daily_data to the flow chain. Pass `close` through kwargs like `_compute_market_cap_adj` already does.

- [ ] **Step 1: Read how `_compute_market_cap_adj` gets close (kwargs pattern)**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import inspect
from stoke_ml.preprocessing.daily_continuous.flow import FlowDecomposer
print(inspect.getsource(FlowDecomposer._compute_market_cap_adj))
"
```

- [ ] **Step 2: Update transform() to source close from kwargs and call residual**

```python
        self._compute_divergence(df, flow_cols_present, **kwargs)
        self._compute_market_cap_adj(df, flow_cols_present, **kwargs)
        self._compute_broad_main(df, flow_cols_present)
        if self.residualize:
            self._compute_residual(df, **kwargs)
```

And update `_compute_divergence` and `_compute_residual` signatures to pull `close` from `kwargs.get("close", ...)` / merge daily close by date+stock_code when present.

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/preprocessing/daily_continuous/flow.py
git commit -m "fix: flow residual/divergence now receive close via kwargs (were never produced)"
```

---

### Task D3: Event-time date mapping must not land events on the previous trading day

**Files:**
- Modify: `stoke_ml/preprocessing/event_sparse/aggregator.py:526`

**Background:** `df_dates.get_indexer([ed], method="nearest")` on a tie picks the EARLIER trading day → a weekend/holiday event counts on the prior trading day, one day early.

- [ ] **Step 1: Use searchsorted right-bias (later day on ties)**

Replace lines 523-528 with:

```python
        if raw_event_dates is not None and len(raw_event_dates) > 0:
            df_ts = pd.DatetimeIndex(df["date"].values).asi8
            for ed in raw_event_dates:
                t = np.datetime64(ed).astype("datetime64[D]").astype(np.int64)
                pos = np.searchsorted(df_ts, t, side="left")
                if pos >= len(df_ts):
                    continue
                # Tie (ed not in grid): choose the LATER trading day so an
                # event that happened over a weekend is not known a day early.
                if pos > 0 and df_ts[pos] != t and df_ts[pos - 1] == t:
                    pass
                elif df_ts[pos] != t:
                    # not a trading day — nearest later day
                    pass
                if pos < len(df_ts):
                    event_mask[pos] = True
```

**Note:** This maps a non-trading-day event to the NEXT trading day (later), never the previous one. For events exactly on a trading day, `searchsorted(side="left")` hits the exact index. Verify with a test.

- [ ] **Step 2: Verify tie goes to the later day**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import numpy as np, pandas as pd
dates = pd.DatetimeIndex(['2026-01-02','2026-01-07'])  # Fri, Wed (holiday Tue 01-06)
t = np.datetime64('2026-01-06')  # non-trading day, equidistant from both
df_ts = dates.asi8
pos = np.searchsorted(df_ts, t.astype('datetime64[D]').astype(np.int64), side='left')
assert dates[pos] == pd.Timestamp('2026-01-07'), dates[pos]
print('OK: non-trading-day event lands on the LATER trading day')
"
```
Expected: `OK: non-trading-day event lands on the LATER trading day`

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/preprocessing/event_sparse/aggregator.py
git commit -m "fix: event-time dates map to later trading day (no early leak on ties)"
```

---

### Task D4: Lockup `unlock_return_30d` — compute on daily grid, not sparse rows

**Files:**
- Modify: `stoke_ml/preprocessing/event_sparse/aggregator.py:371-377`

**Background:** `hist.groupby("stock_code")["close"].transform(lambda s: s.shift(-30) / s - 1)` runs on sparse per-event rows (3-30 rows), so shift(-30) is mostly NaN→0; the value is then forward-filled onto the daily grid → leak/wrong.

- [ ] **Step 1: Remove the pre-ffill shift computation (lines 371-377)**

```python
            if "close" in hist.columns:
                hist["unlock_return_30d"] = (
                    hist.groupby("stock_code")["close"]
                    .transform(lambda s: s.shift(-30) / s - 1)
                    .fillna(0)
                    .astype(np.float32)
                )
```

- [ ] **Step 2: Compute it AFTER _fill_to_daily on the daily grid**

After `result = self._fill_to_daily(...)` (line 393) and before `_add_event_time_features` (line 395), add:

```python
        # unlock_return_30d computed on the DAILY grid (post-ffill) so the
        # 30-day window means 30 trading days, not 30 sparse event rows.
        if "close" in result.columns:
            grp = result.groupby("stock_code")["close"]
            fut30 = grp.shift(-30)
            result["unlock_return_30d"] = (
                (fut30 / result["close"] - 1).fillna(0.0).astype(np.float32)
            )
```

- [ ] **Step 3: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('stoke_ml/preprocessing/event_sparse/aggregator.py').read_text(encoding='utf-8')
assert 'grp.shift(-30)' in src
assert 'unlock_return_30d' in src
print('OK: unlock_return_30d on daily grid')
"
```
Expected: `OK: unlock_return_30d on daily grid`

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/preprocessing/event_sparse/aggregator.py
git commit -m "fix: lockup unlock_return_30d computed on daily grid not sparse event rows"
```

---

## Phase E — Script robustness

### Task E1: download_market_data northbound concurrent None guard

**Files:**
- Modify: `scripts/download_market_data.py:193`

**Background:** `if not d.empty` on a None (failed stock) → AttributeError → whole batch crashes. dragon_tiger path (line 146) already guards `d is not None and not d.empty`.

- [ ] **Step 1: Add the None guard**

```python
                frames = [d for d in results.values() if d is not None and not d.empty]
```

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
from pathlib import Path
src = Path('scripts/download_market_data.py').read_text(encoding='utf-8')
assert 'd is not None and not d.empty' in src
assert 'if not d.empty' not in src
print('OK: northbound concurrent None guard added')
"
```
Expected: `OK: northbound concurrent None guard added`

- [ ] **Step 3: Commit**

```bash
git add scripts/download_market_data.py
git commit -m "fix: northbound concurrent download None guard (matches dragon_tiger)"
```

---

## Phase F — P2 batch (high-value, low-risk)

### Task F1: Fix phantom trading day 2025-05-04

**Files:**
- Modify: `stoke_ml/data/calendar.py:65`

Remove the wrong `dt.date(2025, 5, 4)` entry from `A_SHARES_MAKEUP` (official 2025 Labor Day used only 2025-04-27 as makeup).

- [ ] **Step 1: Edit `calendar.py`** — remove the erroneous entry.
- [ ] **Step 2: Commit** — `git commit -m "fix: remove phantom makeup trading day 2025-05-04"`

### Task F2: margin/dragon_tiger use trading calendar instead of freq='B'

**Files:** `stoke_ml/data/sources/a_shares/margin_source.py:34`, `dragon_tiger_source.py:57`

Replace `pd.date_range(start, end, freq="B")` with the project `TradingCalendar.get_trading_days(start, end)` so 调休 makeup Saturdays are included.

- [ ] **Step 1: Import + use TradingCalendar in both files.**
- [ ] **Step 2: Commit** — `git commit -m "fix: margin/dragon_tiger iterate trading calendar incl. makeup Saturdays"`

### Task F3: Add drop_duplicates to sentiment/guba/comment merges

**Files:** `stoke_ml/features/pipeline.py:672` (and guba/comment merge methods)

Add `df = df.drop_duplicates(subset="date", keep="last")` after date normalization in `_merge_sentiment`, `_merge_guba`, `_merge_comment`, matching the other merge methods.

- [ ] **Step 1: Add the guard in all three merge methods.**
- [ ] **Step 2: Commit** — `git commit -m "fix: sentiment/guba/comment merges dedup on date like other aux dims"`

### Task F4: Fundamental/valuation/etf_flow merges get the 1-day PIT shift

**Files:** `stoke_ml/features/pipeline.py:800, 817, 835`

Replace the direct `fillna(0)` with `_batch_fill_shift(df, available)` (the PIT fill→shift→fill helper used by all other merges). This aligns fundamental/valuation/etf with the documented 1-day lag policy.

- [ ] **Step 1: In `_merge_fundamental`/`_merge_valuation`/`_merge_etf_flow`, replace the two lines**

```python
        df = df.merge(fd[["date"] + available], on="date", how="left")
        df[available] = df[available].fillna(0.0).astype(np.float32)
```
with
```python
        df = df.merge(fd[["date"] + available], on="date", how="left")
        _batch_fill_shift(df, available)
```

- [ ] **Step 2: Verify `_batch_fill_shift` call count grew**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import inspect
from stoke_ml.features.pipeline import FeaturePipeline
src = inspect.getsource(FeaturePipeline)
assert src.count('_batch_fill_shift(df,') >= 11, src.count('_batch_fill_shift(df,')
print('OK: all merge paths use PIT shift')
"
```
Expected: `OK: all merge paths use PIT shift`

- [ ] **Step 3: Commit** — `git commit -m "fix: fundamental/valuation/etf_flow now PIT-lagged 1 day like other aux dims"`

### Task F5: Per-date cross-sectional z-score must skip stock-invariant columns

**Files:** `stoke_ml/features/pipeline.py:1496-1518`

Add macro/industry stock-invariant columns to `_CS_NORM_SKIP_COLS` so they aren't zeroed by the per-date z-score (std=0 → clip → all zeros).

- [ ] **Step 1: Read `_CS_NORM_SKIP_COLS` definition and add the invariant prefixes** — find it via `grep -n "_CS_NORM_SKIP_COLS" stoke_ml/features/pipeline.py` and add entries for `shibor_`, `fx_`, `bond_`, `gdp`, `ind_`, and other macro/industry columns.
- [ ] **Step 2: Commit** — `git commit -m "fix: per-date cs z-score skips stock-invariant macro/industry columns"`

### Task F6: aligned_close must align to horizon-day returns

**Files:** `stoke_ml/features/pipeline.py:1203`

For horizon>1, `aligned_close` should step by `horizon` days so `np.diff` yields horizon-day returns matching `y`. Update the slice to sample every `horizon`-th close.

- [ ] **Step 1: Edit the aligned_close construction** — sample `close[seq_len-1 :: horizon][:n_samples+1]` (verify length ≥ n_samples+1; if not, pad with last close).
- [ ] **Step 2: Commit** — `git commit -m "fix: aligned_close steps by horizon so financial metrics match multi-day labels"`

### Task F7: config.py — wire persistence_mode / event_time_features / monitor / registry

**Files:** `stoke_ml/preprocessing/config.py:191, 233`

Read `persistence_mode`/`event_time_features`/`persistence_halflife` from config and pass to `EventToDaily`; instantiate QualityMonitor/DriftMonitor when `monitor.enabled`; attach registry when `registry.enabled`.

- [ ] **Step 1: Read the current config.py event-chain assembly** and thread the keys through.
- [ ] **Step 2: Add a python verification** that `persistence_mode` propagates.
- [ ] **Step 3: Commit** — `git commit -m "feat: wire event persistence_mode/event_time_features + monitor/registry config into pipeline"`

### Task F8: Fix outlier MAD full-sample leak + board_overlap full-sample max

**Files:** `stoke_ml/preprocessing/numeric/outlier.py:26`, `stoke_ml/preprocessing/categorical/encoder.py:220`

- **outlier.py:** compute MAD bounds on a rolling / expanding window up to current point (not full sample). Minimal fix: use `expanding()` quantiles for the clip bounds, or document that `fit` must be called on training windows only. Apply expanding MAD.
- **encoder.py:** replace full-sample `df["board_count"].max()` normalization with a per-stock expanding max (or a fixed reference), so overlap scores don't shift when data is extended.

- [ ] **Step 1: Implement both.**
- [ ] **Step 2: Commit** — `git commit -m "fix: outlier MAD + board_overlap use expanding (causal) statistics"`

---

## Phase G — P3 cleanup + data backfill

### Task G1: Dead code removal (low risk)

Remove unreferenced dead code from the audit P3 list that has zero callers:
- `stoke_ml/features/pipeline.py`: `_PAST_KNOWN_COLS`, `_PAST_OBSERVED_COLS`, `add_rolling_features`, `PanelZScoreNormalizer`
- `stoke_ml/models/panel/attention.py`: entire dead module (verify no imports first with grep)
- `stoke_ml/data/cleaner.py`: `DataCleaner` (verify no imports)
- `scripts/benchmark_labels.py:78`: `_compute_rel_labels` dead function

- [x] **Step 1: Grep for each symbol; if zero references, delete.**
- [x] **Step 2: Run a smoke import** — `PYTHONPATH=. ./.venv/Scripts/python -c "import stoke_ml.features.pipeline; import stoke_ml.models.panel.model; print('OK')"`
- [x] **Step 3: Commit** — `git commit -m "chore: remove dead code (attention.py, _PAST_*_COLS, cleaner, _compute_rel_labels)"` (2f91975)

### Task G2: Re-run dividend preprocessing to regenerate corrected data

After A1 (code fix), existing `dividend_processed` parquet is still 10× wrong.

- [x] **Step 1: Re-run dividend preprocessing**

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type dividend --start 2000-01-01
```

- [x] **Step 2: Spot-check 000001 dividend_yield ≈ 2-3% (was ~28%)**

```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
df = pd.read_parquet('data/a_shares/dividend_processed/000001.parquet')
y = df[df['dividend_yield'] > 0]['dividend_yield']
print('000001 max dividend_yield:', float(y.max()), '(expect < 0.10)')
"
```
(000001 max=0.0919, 600519 max=0.0522, no duplicate dates.)

- [x] **Step 3: Commit data regeneration** (if data dir is tracked) — otherwise note in commit message.
(Noted in fddd655 commit message; 35.9M rows regenerated 2000-01-01..2026-08-02.)

### Task G3: Update audit doc status

Mark all fixed items as ✅ in `docs/research-findings/2026-08-01-full-repo-code-audit.md`.

- [x] **Step 1: Update the doc status.**
- [x] **Step 2: Commit** — `git commit -m "docs: mark audit findings fixed (see plan 2026-08-01)"` (ac9cfbe)

---

## Verification Summary (run all)

```bash
# Import smoke
PYTHONPATH=. ./.venv/Scripts/python -c "import stoke_ml.features.pipeline, stoke_ml.preprocessing.event_sparse.aggregator, stoke_ml.evaluation.splitter; print('import OK')"

# pytest (existing tests still pass)
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -x -q
```

## Open decisions for the executor

1. **P2-12 (fundamental/valuation/etf PIT shift):** Shifting these 1 day may change existing model inputs. Confirm with 弟弟 whether to apply — it fixes a real lag-policy violation but changes the feature set. Default: **apply** (matches documented policy).
2. **P2-13 (aligned_close horizon stepping):** affects horizon>1 runs only (default horizon=1 unaffected). Default: **apply**.
3. **P2-16/17 (dead config):** wiring `persistence_mode` requires EventToDaily to actually implement decay — that is a larger feature (Phase 2 of the sparse-event design). For this plan, wire the *config read + pass-through* so `event_time_features: false` works, but keep `persistence_mode: "decay"` as a TODO stubbed to ffill. Default: **partial wiring**.
