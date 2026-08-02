# Feature Engineering Build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 3 new feature families (pledge risk, market+macro env, index membership) into the existing 4-layer feature factory, add the L4 IC/leakage evaluation gate, parallelize the full 5530-stock build, and verify with tests.

> **SCOPE ADJUSTMENT (2026-08-02, user decision):** The limit-up ecology family (limit-up pools zt/zb/dt/yzt + market temperature) is **DEFERRED**. Verified empirically: EastMoney and AKShare limit-up pool APIs do NOT support historical backfill (dates before 2026-07 return 0 rows; the existing 1-month corpus is daily-incremental). With the 2021+ training window, limit-up features would be all-zero/constant — no signal. The preprocess script `_preprocess_limit_up.py` (Task A1) IS kept (its outputs are ready and will grow via daily increments), but Tasks A3/B1/B2/B5 are scoped to NOT wire limit-up columns (`market_zt_ratio`, `market_heat_z`, `has_market_sent`, `has_zt` etc.) into the pipeline. Dragon-tiger seat classification (B3) is UNAFFECTED (dragon_tiger spans 2015-2026).

**Architecture:** Follows `docs/superpowers/specs/2026-08-02-feature-engineering-deep-dive-design.md`. New raw sources are preprocessed into per-stock daily parquet (`*_processed/{code}.parquet`, mirroring `board_processed`) plus global daily files (`limit_up_market_daily.parquet`, `market_env_daily.parquet`). `FeaturePipeline` gains 4 `_merge_*` methods following the existing `_merge_macro` (global, disk-cached) and `_merge_daily_aux` (per-stock daily) patterns. A new L3 `MarketEnvRefiner` compresses raw macro cols into `menv_*` factors. L4 adds `feature_ic_report.py` + `feature_leakage_report.py`.

**Tech Stack:** pandas, numpy, pyarrow/lz4, scipy.stats.spearmanr, concurrent.futures (ProcessPoolExecutor), pytest.

---

## Key reference facts (verified 2026-08-02)

- **Per-stock file layout = stock_code from filename.** `limit_up_zt/zb/yzt/{code}.parquet` have NO `stock_code` column; only `limit_up_dt` has one. Because files are per-stock, we inject `code` from the filename — the spec §6 name→code mapping is NOT needed.
- **Schemas (probed):**
  - `limit_up_zt`: date, stock_name, price, pct, amount, float_cap, turnover, limit_days, first_seal("HH:MM:SS"), last_seal, seal_fund, break_times, industry, zt_stat
  - `limit_up_zb`: date, stock_name, price, limit_price, pct, turnover, first_seal, break_times, amplitude, speed, industry, zt_stat
  - `limit_up_dt`: date, stock_name, price, pct, turnover, pe, seal_fund, last_seal, board_amount, dt_days, open_times, industry, **stock_code**
  - `limit_up_yzt`: date, stock_name, price, pct, turnover, amplitude, speed, y_first_seal, y_limit_days, industry, zt_stat
  - `pledge/{code}.parquet`: 序号, 股票代码, 股票简称, 股东名称, 质押股份数量, 占所持股份比例, 占总股本比例, 质押机构, 最新价, 质押日收盘价, 预估平仓线, 质押开始日期, 质押结束日期, 状态("未解押"/"已解押"), 公告日期, stock_code
  - `market_breadth/account_stats.parquet` (monthly): 数据日期("2015-04"), 新增投资者-数量, 期末投资者-总量, 沪深总市值, 沪深户均市值, 上证指数-收盘
  - `market_breadth/highs_lows.parquet` (daily): date(str), close, high20, low20, high60, low60, high120, low120 (per-day counts of stocks at 20/60/120-day highs/lows)
  - `index_constituents/constituents.parquet`: 日期(str), index_code, index_name, stock_code, stock_name, weight, snapshot_date — monthly index snapshots (CSI300/CSI500/CSI1000/…) **⚠ single 2026-06-30 snapshot only → superseded by A4a Baostock backfill into `index_constituents_hist/membership.parquet` (3 indices, no weight)**
  - `limit_up_sentiment/sentiment.parquet` (global, **only 25 rows**): date, zt_count, zb_count, dt_count, yzt_count, break_rate, max_height, advance_rate, ladder_2..ladder_6plus — **too sparse alone; full history comes from per-stock aggregation**
  - `dragon_tiger/{code}.parquet`: date, stock_code, stock_name, lhb_reason, buy_amount, sell_amount, net_amount — **no seat/broker column**; `lhb_reason` strings like "日涨幅偏离值达7%", "日换手率达20%", "连续三个交易日内涨幅偏离值累计达20%"
- **Pipeline integration points** (`stoke_ml/features/pipeline.py`):
  - `_engineer_features` merges at lines 547-566; refiners at 577-586; temporal registration at 589-651
  - `build_features`/`engineer_features`/`save_features` all call `_engineer_features` positionally; new aux kwargs go at the END as keywords
  - `_merge_macro` (963-994) = global disk-load + cache pattern to copy for `_merge_market_env`
  - `_merge_daily_aux` (1833-1864) = per-stock daily merge (drops stock_code/collisions, ZI + shift-1) — used for limit_up/pledge/index_membership
  - `_batch_fill_shift` (1790-1830): partitions into float/int(`_count`/`_streak`/`_quadrant`)/bool(`has_*`), fills → shift(1) → fills
  - `TemporalTransformer` (transform.py): any col not in `_PK_PREFIXES` and not `_SKIP_COLS` is PO → gets `_ma5/10/20`, `_std20`, `_accel`, `_z20` (sparse skip z). Bool cols = `is_*`/`has_*` prefix → `_ma*` only. New cols use `zt_/zb_/dt_/yzt_/pledge_/lhb_/market_/idx_/index_/menv_/has_` prefixes → all PO, no `_PK_PREFIXES` edits needed.
- **New global files are loaded internally** (like macro/industry) so `build_features.py` does NOT load them.

---

## Phase A — Preprocess new sources

### Task A1: `scripts/_preprocess_limit_up.py`

**Files:**
- Create: `scripts/_preprocess_limit_up.py`
- Output: `data/a_shares/limit_up_processed/{code}.parquet` (per-stock daily)
- Output: `data/a_shares/limit_up_market_daily.parquet` (global daily)

Produces per-stock daily limit-up features (0/False on non-event days) and a global market-level limit-up temperature series aggregated from ALL per-stock files (full history, vs the 25-row `sentiment.parquet`).

- [ ] **Step 1: Write the script**

```python
"""Preprocess limit-up board data (zt/zb/dt/yzt) into per-stock daily features.

Outputs:
  data/a_shares/limit_up_processed/{code}.parquet  per-stock daily limit-up ecology
  data/a_shares/limit_up_market_daily.parquet      global daily zt/zb/dt/yzt counts + heat
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from stoke_ml.config import load_config

ZT_DIR = "limit_up_zt"
ZB_DIR = "limit_up_zb"
DT_DIR = "limit_up_dt"
YZT_DIR = "limit_up_yzt"
OUT_DIR = "limit_up_processed"
SENT_PATH = os.path.join("limit_up_sentiment", "sentiment.parquet")
MARKET_OUT = "limit_up_market_daily.parquet"
# Approx number of listed stocks (constant divisor for market_zt_ratio).
N_LISTED = 5530.0

ZT_COLS = [
    "date", "has_zt", "zt_first_seal_hour", "zt_last_seal_hour",
    "zt_seal_fund_ratio", "zt_break_times", "zt_limit_days", "zt_pct",
]
ZB_COLS = [
    "date", "has_zb", "zb_first_seal_hour", "zb_break_times",
    "zb_amplitude", "zb_speed",
]
DT_COLS = [
    "date", "has_dt", "dt_seal_fund_ratio", "dt_open_times",
    "dt_days", "dt_pe",
]
YZT_COLS = [
    "date", "has_yzt", "yzt_first_seal_hour", "yzt_limit_days",
]


def parse_hour(t):
    """Parse 'HH:MM:SS' to float hour (09:25 -> 9.42). NaN/'' -> 0.0."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return 0.0
    s = str(t).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0
    parts = s.split(":")
    try:
        return int(parts[0]) + int(parts[1]) / 60.0 + int(parts[2]) / 3600.0
    except (ValueError, IndexError):
        return 0.0


def _prepare_events(df, date_col="date"):
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    d = df.copy()
    d["date"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return d


def build_stock(zt, zb, dt, yzt) -> pd.DataFrame:
    """Merge zt/zb/dt/yzt event frames into one daily frame for a stock."""
    parts = [zt, zb, dt, yzt]
    date_idx = pd.DatetimeIndex([])
    for p in parts:
        if p is not None and not p.empty:
            date_idx = date_idx.union(p["date"])
    out = pd.DataFrame({"date": sorted(date_idx)})
    if out.empty:
        return out

    if zt is not None and not zt.empty:
        z = _prepare_events(zt)
        z["has_zt"] = True
        z["zt_first_seal_hour"] = z["first_seal"].map(parse_hour)
        z["zt_last_seal_hour"] = z["last_seal"].map(parse_hour)
        z["zt_seal_fund_ratio"] = (z["seal_fund"] / z["float_cap"].replace(0, np.nan)).astype(np.float32)
        z["zt_break_times"] = z["break_times"].astype(np.float32)
        z["zt_limit_days"] = z["limit_days"].astype(np.float32)
        z["zt_pct"] = z["pct"].astype(np.float32)
        out = out.merge(z[ZT_COLS], on="date", how="left")

    if zb is not None and not zb.empty:
        b = _prepare_events(zb)
        b["has_zb"] = True
        b["zb_first_seal_hour"] = b["first_seal"].map(parse_hour)
        b["zb_break_times"] = b["break_times"].astype(np.float32)
        b["zb_amplitude"] = b["amplitude"].astype(np.float32)
        b["zb_speed"] = b["speed"].astype(np.float32)
        out = out.merge(b[ZB_COLS], on="date", how="left")

    if dt is not None and not dt.empty:
        d = _prepare_events(dt)
        d["has_dt"] = True
        d["dt_seal_fund_ratio"] = (d["seal_fund"] / d["board_amount"].replace(0, np.nan)).astype(np.float32)
        d["dt_open_times"] = d["open_times"].astype(np.float32)
        d["dt_days"] = d["dt_days"].astype(np.float32)
        d["dt_pe"] = d["pe"].astype(np.float32)
        out = out.merge(d[DT_COLS], on="date", how="left")

    if yzt is not None and not yzt.empty:
        y = _prepare_events(yzt)
        y["has_yzt"] = True
        y["yzt_first_seal_hour"] = y["y_first_seal"].map(parse_hour)
        y["yzt_limit_days"] = y["y_limit_days"].astype(np.float32)
        out = out.merge(y[YZT_COLS], on="date", how="left")

    # ZI-fill: booleans False, numerics 0.0
    for c in ("has_zt", "has_zb", "has_dt", "has_yzt"):
        out[c] = out[c].fillna(False).astype(bool)
    for c in out.columns:
        if c != "date" and not c.startswith("has_"):
            out[c] = out[c].fillna(0.0).astype(np.float32)
    return out.sort_values("date").reset_index(drop=True)


def _date_counts(directory: str) -> pd.Series:
    """Count limit-up events per date across every per-stock file."""
    frames = []
    for f in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        try:
            d = pd.read_parquet(f, columns=["date"])
            frames.append(pd.to_datetime(d["date"], errors="coerce").dt.normalize().dropna())
        except Exception:
            continue
    if not frames:
        return pd.Series(dtype="int64")
    return pd.concat(frames).value_counts().sort_index()


def build_market_daily(base: str) -> pd.DataFrame:
    zt = _date_counts(os.path.join(base, ZT_DIR)).rename("zt_count")
    zb = _date_counts(os.path.join(base, ZB_DIR)).rename("zb_count")
    dt = _date_counts(os.path.join(base, DT_DIR)).rename("dt_count")
    yzt = _date_counts(os.path.join(base, YZT_DIR)).rename("yzt_count")
    df = pd.concat([zt, zb, dt, yzt], axis=1)  # Series already uniquely named — no keys= (avoids MultiIndex column mangling on parquet round-trip)
    df = df.fillna(0).reset_index(names="date")
    df["market_zt_ratio"] = (df["zt_count"] / N_LISTED).astype(np.float32)
    z20 = df["zt_count"].rolling(20)
    df["market_heat_z"] = ((df["zt_count"] - z20.mean()) / z20.std()).fillna(0.0).astype(np.float32)
    # Merge the sparse market sentiment snapshot (ladders/break_rate/...) if present.
    sent = os.path.join(base, SENT_PATH)
    if os.path.exists(sent):
        s = pd.read_parquet(sent)
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        df = df.merge(s, on="date", how="left")
        df["has_market_sent"] = df["break_rate"].notna()
    else:
        df["has_market_sent"] = False
    return df


def main():
    ap = argparse.ArgumentParser(description="Preprocess limit-up board data")
    ap.add_argument("--shard", type=str, default=None, help="k/n shard over stock codes")
    ap.add_argument("--stocks", type=str, default=None, help="comma-separated codes (test)")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares")

    # ---- per-stock daily ----
    zt_files = {os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(base, ZT_DIR, "*.parquet"))}
    zb_files = {os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(base, ZB_DIR, "*.parquet"))}
    dt_files = {os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(base, DT_DIR, "*.parquet"))}
    yzt_files = {os.path.splitext(os.path.basename(f))[0]: f
                 for f in glob.glob(os.path.join(base, YZT_DIR, "*.parquet"))}
    all_codes = sorted(set(zt_files) | set(zb_files) | set(dt_files) | set(yzt_files))
    if args.stocks:
        all_codes = [c for c in all_codes if c in set(args.stocks.split(","))]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        all_codes = [c for i, c in enumerate(all_codes) if i % n == k]

    out_dir = os.path.join(base, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    def _read(d, c):
        return pd.read_parquet(d[c]) if c in d else pd.DataFrame()

    for i, code in enumerate(all_codes):
        df = build_stock(
            _read(zt_files, code), _read(zb_files, code),
            _read(dt_files, code), _read(yzt_files, code),
        )
        if not df.empty:
            df["stock_code"] = code
            df.to_parquet(os.path.join(out_dir, f"{code}.parquet"), index=False, compression="lz4")
        if (i + 1) % 500 == 0:
            print(f"  limit_up processed {i+1}/{len(all_codes)}")

    # ---- global market daily ----
    market = build_market_daily(base)
    market.to_parquet(os.path.join(base, MARKET_OUT), index=False, compression="lz4")
    print(f"market daily: {len(market)} dates, zt span "
          f"{market['date'].min().date()} ~ {market['date'].max().date()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a few stocks**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_limit_up.py --stocks 000004,000021,000002`
Expected: prints market-daily span; `data/a_shares/limit_up_processed/000004.parquet` etc. created.

- [ ] **Step 3: Verify a single stock frame**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "import pandas as pd; df=pd.read_parquet('data/a_shares/limit_up_processed/000004.parquet'); print(df.shape); print(df.head(3).to_string()); print('has_zt sum', df['has_zt'].sum())"`
Expected: `has_zt` True on zt dates only; numeric 0 on non-event dates; columns = ZT_COLS+ZB_COLS+DT_COLS+YZT_COLS+stock_code.

- [ ] **Step 4: Run full per-stock build**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_limit_up.py`
Expected: all ~1600 codes processed; `limit_up_market_daily.parquet` has full zt/zb/dt/yzt history (2000+ rows).

- [ ] **Step 5: Commit**

```bash
git add scripts/_preprocess_limit_up.py
git commit -m "feat: preprocess limit-up board data into per-stock daily + market daily"
```

---

### Task A2: `scripts/_preprocess_pledge.py`

**Files:**
- Create: `scripts/_preprocess_pledge.py`
- Output: `data/a_shares/pledge_processed/{code}.parquet` (per-stock daily)

Derives PIT-safe daily pledge features keyed on `公告日期` (announcement date), using the stock's own K-line close for the margin-distance computation.

- [ ] **Step 1: Write the script**

```python
"""Preprocess pledge events into per-stock daily equity-risk features.

PIT rule: every feature is computed strictly from information available at or
before that trading day. Announcements are keyed on 公告日期; the margin-line
distance uses the stock's K-line close ON that date (never 最新价, which is a
scrape-time snapshot and would leak).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage

OUT_DIR = "pledge_processed"
CN = {
    "ratio": "占总股本比例",
    "held": "占所持股份比例",
    "margin_line": "预估平仓线",
    "status": "状态",
    "ann": "公告日期",
}


def build_stock(raw: pd.DataFrame, kline: pd.DataFrame) -> pd.DataFrame:
    """Pledge events + K-line -> daily pledge feature frame."""
    empty = pd.DataFrame(columns=["date", "pledge_ratio", "pledge_margin_dist",
                                  "pledge_risk", "pledge_count_20d", "has_pledge"])
    if raw.empty or kline.empty:
        return empty
    r = raw.copy()
    r["ann_dt"] = pd.to_datetime(r[CN["ann"]], errors="coerce").dt.normalize()
    r = r.dropna(subset=["ann_dt"])
    if r.empty:
        return empty

    # Net pledged-share fraction: +ratio for active, -ratio for released announcements.
    r["_delta"] = np.where(r[CN["status"]] == "未解押", r[CN["ratio"]], -r[CN["ratio"]])
    # pandas 3.0.3: Series.reset_index(names=...) unsupported -> rename instead.
    delta = r.groupby("ann_dt")["_delta"].sum().rename("_delta").reset_index().rename(columns={"ann_dt": "date"})

    # As-of margin line: last announced ACTIVE pledge's 预估平仓线, forward-filled.
    active = r[r[CN["status"]] == "未解押"]
    line = (active.groupby("ann_dt")[CN["margin_line"]]
            .last().rename("_margin_line").reset_index().rename(columns={"ann_dt": "date"}))

    k = kline[["date", "close"]].copy()
    k["date"] = pd.to_datetime(k["date"]).dt.normalize()
    k = k.drop_duplicates("date").sort_values("date")

    out = k.merge(delta, on="date", how="left").merge(line, on="date", how="left")
    out["_delta"] = out["_delta"].fillna(0.0)
    out["pledge_ratio"] = out["_delta"].cumsum().clip(lower=0.0).astype(np.float32)
    out["_margin_line"] = out["_margin_line"].ffill()
    out["pledge_margin_dist"] = (out["close"] / out["_margin_line"] - 1.0)
    out["pledge_risk"] = out["pledge_margin_dist"].notna() & (out["pledge_margin_dist"] < 0.20)
    out["pledge_count_20d"] = out["_delta"].ne(0).astype(int).rolling(20, min_periods=1).sum().astype("int16")
    out["has_pledge"] = out["pledge_ratio"].gt(0).cummax()
    out = out.drop(columns=["close", "_delta", "_margin_line"])
    out["pledge_margin_dist"] = out["pledge_margin_dist"].fillna(0.0).astype(np.float32)
    out["pledge_ratio"] = out["pledge_ratio"].fillna(0.0)
    return out.sort_values("date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Preprocess pledge events")
    ap.add_argument("--shard", type=str, default=None, help="k/n shard")
    ap.add_argument("--stocks", type=str, default=None, help="comma-separated codes")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares")
    storage = DataStorage(data_dir)

    files = sorted(glob.glob(os.path.join(base, "pledge", "*.parquet")))
    codes = [os.path.splitext(os.path.basename(f))[0] for f in files]
    # pledge/ dir also holds aggregate tables (market_pledge_stats, pledge_ratios) — keep per-stock only.
    codes = [c for c in codes if len(c) == 6 and c.isdigit()]
    if args.stocks:
        codes = [c for c in codes if c in set(args.stocks.split(","))]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        codes = [c for i, c in enumerate(codes) if i % n == k]

    out_dir = os.path.join(base, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    for i, code in enumerate(codes):
        try:
            raw = pd.read_parquet(os.path.join(base, "pledge", f"{code}.parquet"))
            # load_daily requires start/end; wide range never excludes dates.
            kline = storage.load_daily(code, "1990-12-19", "2030-12-31")
            df = build_stock(raw, kline)
            if not df.empty:
                df["stock_code"] = code
                df.to_parquet(os.path.join(out_dir, f"{code}.parquet"), index=False, compression="lz4")
        except Exception as e:
            print(f"  {code}: SKIP {e}")
        if (i + 1) % 500 == 0:
            print(f"  pledge processed {i+1}/{len(codes)}")
    print(f"pledge done: {len(codes)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_pledge.py --stocks 000002,000006`
Expected: `pledge_processed/000002.parquet` created with a few hundred daily rows (万科A has 72 pledge events spanning 2021+). `pledge_ratio` grows stepwise at announcement dates, 0 before the first.

- [ ] **Step 3: Verify PIT (no look-ahead)**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "import pandas as pd; df=pd.read_parquet('data/a_shares/pledge_processed/000002.parquet'); d=df[df['pledge_count_20d']>0]; print('first event date', d['date'].min()); print(d.head(3).to_string())"`
Expected: first nonzero `pledge_count_20d` equals the stock's earliest 公告日期 (not earlier), and `pledge_margin_dist` at that date uses that date's close.

- [ ] **Step 4: Run full**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_pledge.py`
Expected: ~3259 pledge files processed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_preprocess_pledge.py
git commit -m "feat: preprocess pledge events into PIT-safe daily equity-risk features"
```

---

### Task A3: `scripts/_preprocess_market_env.py`

**Files:**
- Create: `scripts/_preprocess_market_env.py`
- Output: `data/a_shares/market_breadth/market_env_daily.parquet` (global daily)

Combines investor stats (monthly), high/low breadth (daily), limit-up market temperature (from A1), industry advance ratio, and market-turnover z-score into one global daily file consumed by `_merge_market_env`.

- [ ] **Step 1: Write the script**

```python
"""Build a global daily market-environment panel.

Merges: account_stats (monthly investor/mkt-cap), highs_lows (daily high/low
breadth), limit_up_market_daily (daily zt/zb/dt/yzt temperature from A1),
industry advance ratio, and a market-turnover z-score computed by scanning
daily K-line flat files.

Output: data/a_shares/market_breadth/market_env_daily.parquet (date + cols)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config


def _z(s: pd.Series, win: int = 20) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0)


def build_turnover_daily(base: str) -> pd.Series:
    """Sum 'amount' across all daily flat files per date -> turnover series."""
    amounts = []
    for f in glob.glob(os.path.join(base, "daily", "*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["date", "amount"])
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            amounts.append(d.groupby("date")["amount"].sum())
        except Exception:
            continue
    if not amounts:
        return pd.Series(dtype="float64")
    tot = pd.concat(amounts).groupby(level=0).sum()
    return _z(tot)


def build_industry_advance(base: str) -> pd.Series:
    """Fraction of industries with positive return per date (breadth proxy)."""
    path = os.path.join(base, "industry", "industry_ranking_computed.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype="float64")
    raw = pd.read_parquet(path)
    if isinstance(raw.index, pd.DatetimeIndex):
        idx = raw.index
        vals = raw
    elif "date" in raw.columns:
        idx = pd.to_datetime(raw["date"])
        vals = raw.drop(columns=["date"])
    else:
        return pd.Series(dtype="float64")
    cols = [c for c in vals.columns if pd.api.types.is_numeric_dtype(vals[c])]
    if not cols:
        return pd.Series(dtype="float64")
    m = vals[cols].astype(float)
    adv = (m > 0).sum(axis=1) / m.notna().sum(axis=1).replace(0, np.nan)
    return adv.fillna(0.0).rename("market_adv_ratio").set_axis(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-turnover", action="store_true",
                    help="skip the slow daily-file turnover scan")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares")
    br = os.path.join(base, "market_breadth")

    acc = pd.read_parquet(os.path.join(br, "account_stats.parquet"))
    acc["date"] = pd.to_datetime(acc["数据日期"].astype(str) + "-28", errors="coerce")
    acc = acc.sort_values("date").set_index("date")
    acc = acc.rename(columns={
        "新增投资者-数量": "investor_new_num",
        "沪深总市值": "mkt_cap_total",
        "沪深户均市值": "avg_account_cap",
    })
    cols = ["investor_new_num", "mkt_cap_total", "avg_account_cap"]
    acc_daily = acc[cols].resample("D").ffill()

    hl = pd.read_parquet(os.path.join(br, "highs_lows.parquet"))
    hl["date"] = pd.to_datetime(hl["date"])
    hl = hl.set_index("date").sort_index()
    hl["high_low_ratio"] = hl["high20"] / (hl["high20"] + hl["low20"]).replace(0, np.nan)

    # market daily (from Task A1)
    md = pd.read_parquet(os.path.join(base, "limit_up_market_daily.parquet"))
    md["date"] = pd.to_datetime(md["date"]).dt.normalize()
    md = md.set_index("date").sort_index()

    out = md[["market_zt_ratio", "market_heat_z", "has_market_sent"]].copy()
    # sparse ladder snapshot cols ride along (mostly 0, has_market_sent guards)
    for c in ("break_rate", "max_height", "advance_rate", "ladder_2", "ladder_3",
              "ladder_4", "ladder_5", "ladder_6plus"):
        if c in md.columns:
            out[c] = md[c]

    out["high_low_ratio"] = hl["high_low_ratio"]
    out["mkt_cap_total_z"] = _z(acc_daily["mkt_cap_total"])
    out["avg_account_cap_z"] = _z(acc_daily["avg_account_cap"])
    out["investor_new_num"] = acc_daily["investor_new_num"]
    out["investor_new_z"] = _z(acc_daily["investor_new_num"])

    adv = build_industry_advance(base)
    if not adv.empty:
        out["market_adv_ratio"] = adv
    if not args.skip_turnover:
        out["market_turnover_z"] = build_turnover_daily(base)

    out = out.fillna(0.0)
    for c in ("break_rate", "max_height", "advance_rate", "ladder_2", "ladder_3",
              "ladder_4", "ladder_5", "ladder_6plus"):
        if c in out.columns:
            out[c] = out[c].astype(np.float32)
    out["has_market_sent"] = out["has_market_sent"].astype(bool)
    out.index.name = "date"
    out.to_parquet(os.path.join(br, "market_env_daily.parquet"), compression="lz4")
    print(f"market_env_daily: {len(out)} dates "
          f"({out.index.min().date()} ~ {out.index.max().date()}), "
          f"{len(out.columns)} cols")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (skip turnover scan first for speed)**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_market_env.py --skip-turnover`
Expected: `market_env_daily.parquet` with ~4000+ dates (from limit_up market daily), cols include `market_zt_ratio`, `market_heat_z`, `high_low_ratio`, `mkt_cap_total_z`, `investor_new_num`, `market_adv_ratio`.

- [ ] **Step 3: Run full incl. turnover scan**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_market_env.py`
Expected: `market_turnover_z` column added; run may take 2-5 min scanning daily flats.

- [ ] **Step 4: Commit**

```bash
git add scripts/_preprocess_market_env.py
git commit -m "feat: build global market-environment daily panel (breadth + sentiment temperature)"
```

---

### Task A4a: `scripts/download_index_hist.py`

> **SCOPE ADJUSTMENT (2026-08-02, user decision):** Index constituents was a single 2026-06-30 snapshot (1900 rows) — unusable for the 2021+ window. AKShare (no historical API) and JRJ (404) are both dead. Pivot to **Baostock**, which has official free historical constituent queries for exactly 3 indices: **000300 (沪深300), 000905 (中证500), 000016 (上证50)**. **000852 (中证1000) and 000688 (科创50) are coverage gaps** (no Baostock API). Consequently `index_weight` is DROPPED (Baostock provides no historical weights) — A4 outputs `is_index_member` / `n_indexes` / `idx_change_30d` only.

**Files:**
- Create: `scripts/download_index_hist.py`
- Output: `data/a_shares/index_constituents_hist/membership.parquet` (intervals) + `snapshots/{index}/{date}.parquet` (raw audit trail)

Queries Baostock constituent sets on a monthly grid; reconstructs per-stock membership intervals.

- [ ] **Step 1: Write the script**

```python
"""Download historical index-constituent membership from Baostock.

Queries constituent sets on a monthly grid for the 3 major indices Baostock
covers (000300 / 000905 / 000016) and reconstructs per-stock membership
intervals. Baostock has NO historical data for 000852 (CSI1000) or 000688
(科创50) — documented coverage gaps.

Baostock row semantics: a query at date D returns every constituent as of D,
each row carrying `updateDate` = the exact date that membership spell became
effective (an index adjustment date). A stock present across many queries
keeps the SAME updateDate until it is dropped and re-added (new updateDate).

Reconstruction: for each (index, stock), each distinct updateDate starts a
membership spell. The spell's out_date = the last query date that still
reported that spell (the last confirmation before removal); NaT means the
stock was still a member at the final grid query (open-ended).

Outputs:
  data/a_shares/index_constituents_hist/snapshots/{index}/{YYYY-MM-DD}.parquet
      raw query result per index-date (audit trail)
  data/a_shares/index_constituents_hist/membership.parquet
      long-form intervals: stock_code, index_code, in_date, out_date
"""
import argparse
import os
import time

import pandas as pd

from stoke_ml.config import load_config

INDICES = {
    "000300": ("query_hs300_stocks", "沪深300"),
    "000905": ("query_zz500_stocks", "中证500"),
    "000016": ("query_sz50_stocks", "上证50"),
}
GRID_START = "2015-01-01"
GRID_END = "2026-12-31"
RETRIES = 3


def query_flat(bs, fn_name, date, retries=RETRIES):
    """One constituent query -> DataFrame, with retry."""
    fn = getattr(bs, fn_name)
    last_err = None
    for attempt in range(retries):
        try:
            rs = fn(date=date)
            if rs.error_code != "0":
                raise RuntimeError(f"{fn_name}@{date}: {rs.error_code} {rs.error_msg}")
            cols = rs.get_fields()
            rows = rs.get_row_data()
            df = pd.DataFrame(rows, columns=cols)
            df["query_date"] = pd.Timestamp(date)
            return df
        except Exception as e:
            last_err = e
            print(f"  retry {fn_name}@{date} ({attempt+1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{fn_name}@{date}: failed after {retries} attempts: {last_err}")


def rebuild_membership(snap_dir):
    """Concatenate cached snapshots -> membership interval DataFrame."""
    frames = []
    for idx in INDICES:
        sd = os.path.join(snap_dir, idx)
        if not os.path.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(sd, f))
            if df.empty:
                continue
            df["index_code"] = idx
            frames.append(df)
    if not frames:
        raise SystemExit("no snapshots collected — run without --resume first")
    allq = pd.concat(frames, ignore_index=True)
    allq["stock_code"] = allq["code"].astype(str).str.rsplit(".", n=1).str[-1]
    allq["updateDate"] = pd.to_datetime(allq["updateDate"], errors="coerce").dt.normalize()
    allq["query_date"] = pd.to_datetime(allq["query_date"]).dt.normalize()
    allq = allq.dropna(subset=["stock_code", "updateDate"])

    last_grid = allq["query_date"].max()
    rows = []
    for (idx, code), g in allq.groupby(["index_code", "stock_code"]):
        last_seen = g.groupby("updateDate")["query_date"].max()
        for u, ls in last_seen.items():
            still_active = ls >= last_grid
            rows.append({"stock_code": code, "index_code": idx,
                         "in_date": u, "out_date": pd.NaT if still_active else ls})
    return pd.DataFrame(rows, columns=["stock_code", "index_code", "in_date", "out_date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=GRID_START)
    ap.add_argument("--end", default=GRID_END)
    ap.add_argument("--resume", action="store_true",
                    help="skip (index, date) snapshots already cached")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares", "index_constituents_hist")
    snap_dir = os.path.join(base, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    try:
        import baostock as bs
    except ImportError:
        raise SystemExit("baostock not installed: pip install baostock")

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"bs.login failed: {lg.error_code} {lg.error_msg}")

    grid = pd.date_range(args.start, args.end, freq="MS")
    try:
        for i, d in enumerate(grid):
            day = d.strftime("%Y-%m-%d")
            for idx, (fn, _name) in INDICES.items():
                out = os.path.join(snap_dir, idx, f"{day}.parquet")
                if args.resume and os.path.exists(out):
                    continue
                df = query_flat(bs, fn, day)
                df["index_code"] = idx
                df.to_parquet(out, index=False, compression="lz4")
            if (i + 1) % 12 == 0:
                print(f"  {i+1}/{len(grid)} grid months done")
    finally:
        bs.logout()

    mem = rebuild_membership(snap_dir)
    mem.to_parquet(os.path.join(base, "membership.parquet"), index=False, compression="lz4")
    print(f"membership.parquet: {len(mem)} spells, {mem['stock_code'].nunique()} stocks")
    print(mem.groupby("index_code")["stock_code"].nunique().to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (small grid)**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/download_index_hist.py --start 2015-01-01 --end 2015-06-01`
Expected: 18 snapshots (3 indices × 6 months); `membership.parquet` built; `600519` has both `000016` and `000300` spells; `000001` has an `000300` spell.

- [ ] **Step 3: Run full grid**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/download_index_hist.py`
Expected: ~414 snapshots; membership covers every stock ever in the 3 indices over 2015-2026 (a few hundred unique).

- [ ] **Step 4: Commit**

```bash
git add scripts/download_index_hist.py
git commit -m "feat: backfill historical index-constituent membership from Baostock"
```

---

### Task A4: `scripts/_preprocess_index_membership.py`

**Files:**
- Create: `scripts/_preprocess_index_membership.py`
- Output: `data/a_shares/index_membership_processed/{code}.parquet` (per-stock daily)

Converts historical membership intervals into per-stock daily membership with 30-day add/drop events. NO `index_weight` (see A4a scope note).

- [ ] **Step 1: Write the script**

```python
"""Preprocess historical index-membership intervals into per-stock daily features.

PIT rule: membership is judged purely from interval data produced by
scripts/download_index_hist.py (Baostock monthly-grid reconstruction).
in_date is the exact adjustment effective date; a stock is a member on
trading day d iff some interval has in_date <= d < out_date (out_date NaT
means still active). Expanded on the stock's own K-line trading calendar.
"""
import argparse
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage

OUT_DIR = "index_membership_processed"


def build_stock(membership: pd.DataFrame, kline: pd.DataFrame) -> pd.DataFrame:
    """Membership intervals + K-line -> daily membership feature frame."""
    empty = pd.DataFrame(columns=["date", "is_index_member", "n_indexes", "idx_change_30d"])
    if membership.empty or kline.empty:
        return empty
    k = kline[["date"]].copy()
    k["date"] = pd.to_datetime(k["date"]).dt.normalize()
    k = k.drop_duplicates("date").sort_values("date")
    dates = k["date"].to_numpy()  # datetime64[ns]
    d0, d1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])

    m = membership.copy()
    m["in_date"] = pd.to_datetime(m["in_date"]).dt.normalize()
    m["out_date"] = pd.to_datetime(m["out_date"], errors="coerce").dt.normalize()
    # Half-open interval [in, out); NaT out_date = still active -> cap past data end.
    m["out_date"] = m["out_date"].fillna(d1 + pd.Timedelta(days=1))
    m = m[(m["in_date"] <= d1) & (m["out_date"] > d0)]

    n = len(dates)
    is_mem = np.zeros(n, dtype=bool)
    n_idx = np.zeros(n, dtype="int16")
    for _, row in m.iterrows():
        lo = int(np.searchsorted(dates, row["in_date"].to_datetime64(), side="left"))
        hi = int(np.searchsorted(dates, row["out_date"].to_datetime64(), side="left"))
        if hi > lo:
            is_mem[lo:hi] = True
            n_idx[lo:hi] += 1

    out = pd.DataFrame({"date": pd.to_datetime(dates), "is_index_member": is_mem,
                        "n_indexes": n_idx})
    # Net membership change within trailing 30 trading days (+add, -drop).
    out["idx_change_30d"] = (out["is_index_member"].astype(int).diff()
                             .rolling(30).sum().fillna(0).astype("int16"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--stocks", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares")
    storage = DataStorage(data_dir)

    membership = pd.read_parquet(os.path.join(base, "index_constituents_hist", "membership.parquet"))

    codes = sorted(membership["stock_code"].astype(str).unique())
    if args.stocks:
        codes = [c for c in codes if c in set(args.stocks.split(","))]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        codes = [c for i, c in enumerate(codes) if i % n == k]

    out_dir = os.path.join(base, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for i, code in enumerate(codes):
        try:
            m = membership[membership["stock_code"].astype(str) == code]
            kline = storage.load_daily(code, "1990-12-19", "2030-12-31")
            df = build_stock(m, kline)
            if not df.empty:
                df["stock_code"] = code
                df.to_parquet(os.path.join(out_dir, f"{code}.parquet"), index=False, compression="lz4")
                written += 1
        except Exception as e:
            print(f"  {code}: SKIP {e}")
        if (i + 1) % 500 == 0:
            print(f"  index membership processed {i+1}/{len(codes)}")
    print(f"index membership done: {written}/{len(codes)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_index_membership.py --stocks 000001,600519`
Expected: files written; `000001` (平安银行, 沪深300) has `is_index_member=True` and `n_indexes==1` on recent dates; `600519` (贵州茅台, 沪深300+上证50) has `n_indexes==2`.

- [ ] **Step 3: Run full**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/_preprocess_index_membership.py`
Expected: membership files for every stock appearing in `membership.parquet` (all 3-index members 2015-2026).

- [ ] **Step 4: Commit**

```bash
git add scripts/_preprocess_index_membership.py
git commit -m "feat: preprocess index-membership intervals into daily features"
```

---

### Task A5: Verify all preprocessed outputs

- [ ] **Step 1: Run the completeness check**

Run:
```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -c "
import glob, os, pandas as pd
base = 'data/a_shares'
for d in ['pledge_processed', 'index_membership_processed']:
    fs = glob.glob(os.path.join(base, d, '*.parquet'))
    print(d, len(fs), 'files')
me = pd.read_parquet(os.path.join(base, 'market_breadth', 'market_env_daily.parquet'))
print('market_env_daily dates:', len(me), me.index.min(), me.index.max())
"
```
Expected: `pledge_processed` ≈3259, `index_membership_processed` few-hundred (all 3-index members 2015-2026), `market_env_daily` has full 2021+ history. (Limit-up preprocess outputs are excluded here — that family is deferred per the top scope note, though A1 keeps running incrementally.)

---

## Phase B — Pipeline integration

### Task B1: New constants + constructor flags

**Files:**
- Modify: `stoke_ml/features/pipeline.py` (constants block ~line 126; constructor ~line 145-231)

- [ ] **Step 1: Add constant lists** (after `INDUSTRY_COLS` block, ~line 135)

```python
LIMIT_UP_COLS = [
    "zt_first_seal_hour", "zt_last_seal_hour", "zt_seal_fund_ratio",
    "zt_break_times", "zt_limit_days", "zt_pct",
    "zb_first_seal_hour", "zb_break_times", "zb_amplitude", "zb_speed",
    "dt_seal_fund_ratio", "dt_open_times", "dt_days", "dt_pe",
    "yzt_first_seal_hour", "yzt_limit_days",
    "has_zt", "has_zb", "has_dt", "has_yzt",
]  # DEFERRED (limit-up ecology family, top scope note) — defined for future re-enable, NOT wired

PLEDGE_COLS = [
    "pledge_ratio", "pledge_margin_dist", "pledge_risk",
    "pledge_count_20d", "has_pledge",
]

INDEX_MEMBER_COLS = [
    "is_index_member", "n_indexes", "idx_change_30d",
]  # no index_weight — Baostock has no historical weights (A4a scope note)

# Must match scripts/_preprocess_market_env.py output exactly (7 cols, no
# limit-up temperature cols — that family is deferred).
MARKET_ENV_COLS = [
    "high_low_ratio", "mkt_cap_total_z", "avg_account_cap_z",
    "investor_new_num", "investor_new_z", "market_adv_ratio", "market_turnover_z",
]

DRAGON_TIGER_SEAT_COLS = [
    "lhb_is_wave", "lhb_is_sustained", "lhb_is_drop", "lhb_count_5d",
]
```

- [ ] **Step 2: Add constructor params + state** (after `use_industry`, before `minute_mode`)

```python
        use_limit_up: bool = False,  # DEFERRED (limit-up ecology, top scope note)
        use_pledge: bool = True,
        use_market_env: bool = True,
        use_index_membership: bool = True,
        use_market_env_refine: bool = True,
```

In the body (mirroring `self.use_macro = use_macro`):

```python
        self.use_limit_up = use_limit_up  # inert while deferred (not wired in _engineer_features)
        self.use_pledge = use_pledge
        self.use_market_env = use_market_env
        self.use_index_membership = use_index_membership
        self.use_market_env_refine = use_market_env_refine
        self._market_env_cache: pd.DataFrame | None = None
```

And after the existing refiner construction (line ~231):

```python
        from stoke_ml.features.market_env import MarketEnvRefiner
        self._market_env_refiner = MarketEnvRefiner() if use_market_env_refine else None
```

> Note: `MarketEnvRefiner` is created in Task B4; until then the import is a no-op guard because B4 wires the module. Implement B4 before running tests.

- [ ] **Step 3: Verify import**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "from stoke_ml.features.pipeline import FeaturePipeline; print('ok')"`
Expected: ok (after Task B4 creates `market_env.py`).

---

### Task B2: Merge methods + engine wiring

**Files:**
- Modify: `stoke_ml/features/pipeline.py`

- [ ] **Step 1: Add `_merge_limit_up` / `_merge_pledge` / `_merge_index_membership`** (per-stock daily, delegate to `_merge_daily_aux`) — add right after `_merge_concept` (~line 961)

```python
    def _merge_limit_up(self, df: pd.DataFrame,
                        limit_up_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_limit_up:
            return df
        if limit_up_df is None or limit_up_df.empty:
            self._warn_if_missing("limit_up")
            return df
        return _merge_daily_aux(df, limit_up_df)

    def _merge_pledge(self, df: pd.DataFrame,
                      pledge_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_pledge:
            return df
        if pledge_df is None or pledge_df.empty:
            self._warn_if_missing("pledge")
            return df
        return _merge_daily_aux(df, pledge_df)

    def _merge_index_membership(self, df: pd.DataFrame,
                                im_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_index_membership:
            return df
        if im_df is None or im_df.empty:
            self._warn_if_missing("index_membership")
            return df
        return _merge_daily_aux(df, im_df)
```

- [ ] **Step 2: Add `_merge_market_env`** (global, disk-loaded + cached — mirror `_merge_macro` at line 963) — add after `_merge_macro`

```python
    def _merge_market_env(self, df: pd.DataFrame,
                          market_env_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.use_market_env:
            return df
        if market_env_df is None:
            market_env_df = self._market_env_cache
            if market_env_df is None:
                import os
                from stoke_ml.config import load_config
                cfg = load_config()
                path = os.path.join(cfg.project.data_dir, "a_shares", "market_breadth",
                                    "market_env_daily.parquet")
                if not os.path.exists(path):
                    self._warn_if_missing("market_env")
                    return df
                market_env_df = pd.read_parquet(path)
                self._market_env_cache = market_env_df
        if market_env_df is None or market_env_df.empty:
            return df
        me = market_env_df.copy()
        if me.index.name == "date":
            me = me.reset_index()
        me["date"] = pd.to_datetime(me["date"]).dt.normalize()
        me = me.drop_duplicates(subset="date", keep="last")
        available = [c for c in MARKET_ENV_COLS if c in me.columns]
        if not available:
            return df
        df = df.merge(me[["date"] + available], on="date", how="left")
        _batch_fill_shift(df, available)
        return df
```

- [ ] **Step 3: Wire into `_engineer_features` merge sequence** — after `df = self._merge_industry(...)` (line 566)

```python
        # _merge_limit_up is DEFERRED (limit-up ecology family, top scope note):
        # the method exists (Step 1) but is intentionally NOT wired here.
        df = self._merge_pledge(df, pledge_df)
        df = self._merge_market_env(df, market_env_df)
        df = self._merge_index_membership(df, index_membership_df)
```

- [ ] **Step 4: Add `MarketEnvRefiner` call** — after `self._fundamental_refiner.refine(df)` (line 582)

```python
        # 4b. Market-environment factors (macro composite + regime score)
        if self._market_env_refiner is not None:
            df = self._market_env_refiner.refine(df)
```

- [ ] **Step 5: Add new params to `_engineer_features` signature** (after `industry_df`, line 523)

```python
        limit_up_df: pd.DataFrame | None = None,  # unused hook while deferred (top scope note)
        pledge_df: pd.DataFrame | None = None,
        market_env_df: pd.DataFrame | None = None,
        index_membership_df: pd.DataFrame | None = None,
```

- [ ] **Step 6: Thread new kwargs through `build_features` / `engineer_features` / `save_features`**

Each of the three public methods (lines 277, 329, 368) gains the same 4 kwargs in its signature, and passes them into its `_engineer_features(...)` call as keywords:

```python
        macro_df=macro_df, industry_df=industry_df,
        limit_up_df=limit_up_df, pledge_df=pledge_df,
        market_env_df=market_env_df, index_membership_df=index_membership_df,
```

- [ ] **Step 7: Register new columns in temporal lag block** — inside `if self.use_temporal and not skip_temporal:` (after the `INDUSTRY_COLS` line, ~616)

```python
            temporal_cols += _active_cols(df, LIMIT_UP_COLS)
            temporal_cols += _active_cols(df, PLEDGE_COLS)
            temporal_cols += _active_cols(df, INDEX_MEMBER_COLS)
            temporal_cols += _active_cols(df, MARKET_ENV_COLS)
            temporal_cols += _active_cols(df, DRAGON_TIGER_SEAT_COLS)
            # MarketEnvRefiner outputs
            temporal_cols += _active_cols(df, [
                c for c in df.columns if c.startswith("menv_")
            ])
```

- [ ] **Step 8: Import check**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "from stoke_ml.features.pipeline import FeaturePipeline, LIMIT_UP_COLS, PLEDGE_COLS, MARKET_ENV_COLS, INDEX_MEMBER_COLS, DRAGON_TIGER_SEAT_COLS; print('ok')"`
Expected: ok.

- [ ] **Step 9: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "feat: wire limit-up/pledge/market-env/index-membership merges into pipeline"
```

---

### Task B3: Extend `_merge_dragon_tiger` with seat classification

**Files:**
- Modify: `stoke_ml/features/pipeline.py` (`_merge_dragon_tiger`, line 748)

**Constraint (spec §3.2):** The dragon_tiger data has NO seat/broker column, so institution/northbound seat labels are **not derivable**. We classify `lhb_reason` into wave/sustained/drop flags and add `lhb_count_5d` (rolling frequency). True seat-level `lhb_is_institution`/`lhb_is_north` requires a future `lhb_detail` download (documented, deferred).

- [ ] **Step 1: Rewrite `_merge_dragon_tiger`**

```python
    def _merge_dragon_tiger(self, df: pd.DataFrame,
                            dt_df: pd.DataFrame | None) -> pd.DataFrame:
        if not self.use_dragon_tiger:
            return df
        if dt_df is None or dt_df.empty:
            self._warn_if_missing("dragon_tiger")
            return df
        dt = dt_df.copy()
        dt["date"] = pd.to_datetime(dt["date"])
        reason = dt.get("lhb_reason", pd.Series(index=dt.index, dtype=str)).fillna("").astype(str)
        dt["lhb_is_wave"] = reason.str.contains("振幅|换手", regex=True)
        dt["lhb_is_sustained"] = reason.str.contains("连续", regex=False)
        dt["lhb_is_drop"] = reason.str.contains("跌幅|跌停|下跌", regex=True)
        dt = dt.drop(columns=["stock_code", "stock_name", "lhb_reason"], errors="ignore")
        agg = dt.groupby("date").agg(
            lhb_net_amount=("net_amount", "sum"),
            lhb_buy_ratio=(
                "buy_amount",
                lambda x: x.sum() / (x.sum()
                                     + dt.loc[x.index, "sell_amount"].sum()
                                     + 1),
            ),
            lhb_present=("net_amount", "count"),
            lhb_is_wave=("lhb_is_wave", "any"),
            lhb_is_sustained=("lhb_is_sustained", "any"),
            lhb_is_drop=("lhb_is_drop", "any"),
        ).reset_index()
        agg["lhb_present"] = (agg["lhb_present"] > 0).astype(np.float32)
        agg["lhb_buy_ratio"] = agg["lhb_buy_ratio"].fillna(0.5).astype(np.float32)
        agg["lhb_net_amount"] = agg["lhb_net_amount"].fillna(0.0).astype(np.float32)
        for c in ("lhb_is_wave", "lhb_is_sustained", "lhb_is_drop"):
            agg[c] = agg[c].fillna(False).astype(np.float32)
        df = df.merge(agg, on="date", how="left")
        _batch_fill_shift(df, [c for c in DRAGON_TIGER_COLS if c in df.columns])
        # Past-5-trading-day LHB frequency (computed AFTER the PIT shift, so it
        # never looks ahead; must NOT be shifted again).
        if "lhb_present" in df.columns:
            df["lhb_count_5d"] = df["lhb_present"].rolling(5, min_periods=1).sum().astype("int16")
        return df
```

- [ ] **Step 2: Verify on a real stock**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.config import load_config
import pandas as pd
cfg = load_config(); s = MarketWideStorage(cfg.project.data_dir, 'dragon_tiger')
df = pd.read_parquet('data/a_shares/daily/000001.parquet')
from stoke_ml.features.pipeline import FeaturePipeline
p = FeaturePipeline(use_technical=False, use_scoring=False, use_temporal=False,
                    use_sentiment=False, use_announcements=False, use_guba=False,
                    use_comment=False, use_margin=False, use_northbound=False,
                    use_fundamental=False, use_valuation=False, use_etf_flow=False,
                    use_interaction=False, use_capital_flow=False, use_block_trade=False,
                    use_shareholder=False, use_lockup=False, use_dividend=False,
                    use_board=False, use_sector=False, use_concept=False,
                    use_macro=False, use_industry=False, use_emotion_refine=False,
                    use_fundamental_refine=False, use_temporal_stats=False,
                    use_limit_up=False, use_pledge=False, use_market_env=False,
                    use_index_membership=False)
dt = s.load('000001', '2020-01-01', '2024-01-01')
out = p.engineer_features(df, dragon_tiger_df=dt)
print('lhb cols:', [c for c in out.columns if c.startswith('lhb_')])
print(out[out['lhb_present']>0][['date','lhb_present','lhb_is_wave','lhb_is_sustained','lhb_count_5d']].head())
"`
Expected: `lhb_is_wave`/`lhb_is_sustained` flags set on LHB days; `lhb_count_5d` bounded 0-5.

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "feat: classify dragon-tiger lhb_reason into wave/sustained/drop + 5d frequency"
```

---

### Task B4: `stoke_ml/features/market_env.py` — MarketEnvRefiner

**Files:**
- Create: `stoke_ml/features/market_env.py`

Compresses the raw `MACRO_COLS` already merged by `_merge_macro` into 6 factors plus a composite regime score. All outputs use the `menv_` prefix (PO → gets rolling/lag variants automatically).

- [ ] **Step 1: Write the module**

```python
"""L3 Deepen: market-environment factors from raw macro + breadth columns.

MarketEnvRefiner compresses the ~28 raw macro columns (already merged as PO by
_merge_macro) into six regime factors, then assembles a composite regime score.
All outputs use the ``menv_`` prefix so TemporalTransformer treats them as
past-observed. Graceful when any input column is absent (sparse configs).
"""
import numpy as np
import pandas as pd

# factor name -> (source col, z-window); window=None means use the raw value.
_FACTOR_Z = {
    "menv_shibor_1m_z": ("shibor_1M", 60),
    "menv_fx_usd_cny_z": ("fx_usd_cny", 60),
    "menv_cpi_z": ("cpi_yoy", 60),
}
_FACTOR_RAW = {
    "menv_bond_10y2y_spread": "bond_cn_10y2y_spread",
    "menv_us_cn_10y_spread": ("bond_us_10y", "bond_cn_10y"),
    "menv_m1_m2_spread": ("m1_yoy", "m2_yoy"),
}
_COMPOSITE = [
    "menv_shibor_1m_z", "menv_us_cn_10y_spread", "menv_fx_usd_cny_z",
    "menv_m1_m2_spread", "menv_cpi_z", "menv_bond_10y2y_spread",
]


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0).astype(np.float32)


class MarketEnvRefiner:
    """Compress raw macro cols into menv_* market-environment factors."""

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" not in out.columns:
            return out
        for col, (src, win) in _FACTOR_Z.items():
            if src in out.columns and col not in out.columns:
                out[col] = _rolling_z(out[src], win)
        for col, src in _FACTOR_RAW.items():
            if col in out.columns:
                continue
            if isinstance(src, tuple):
                a, b = src
                if a in out.columns and b in out.columns:
                    out[col] = (out[a] - out[b]).astype(np.float32)
            elif src in out.columns:
                out[col] = out[src].astype(np.float32)
        present = [c for c in _COMPOSITE if c in out.columns]
        if present:
            out["menv_regime_z"] = out[present].mean(axis=1).fillna(0.0).astype(np.float32)
        return out
```

- [ ] **Step 2: Unit smoke**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.market_env import MarketEnvRefiner
import pandas as pd, numpy as np
d = pd.date_range('2021-01-01', periods=200, freq='D')
df = pd.DataFrame({'date': d, 'shibor_1M': 2.0 + np.sin(np.arange(200)/10),
                   'fx_usd_cny': 7.0, 'cpi_yoy': 1.0, 'm1_yoy': 5.0, 'm2_yoy': 8.0,
                   'bond_cn_10y2y_spread': 0.5, 'bond_us_10y': 4.0, 'bond_cn_10y': 3.0})
out = MarketEnvRefiner().refine(df)
print([c for c in out.columns if c.startswith('menv_')])
print(out[['menv_m1_m2_spread','menv_regime_z']].head(3))
"`
Expected: all 7 `menv_*` cols present; `menv_m1_m2_spread` = -3.0; `menv_regime_z` non-null.

- [ ] **Step 3: Commit**

```bash
git add stoke_ml/features/market_env.py
git commit -m "feat: add MarketEnvRefiner compressing macro cols into menv_* factors"
```

---

### Task B5: Wire new sources + ablation flags into `build_features.py`

**Files:**
- Modify: `scripts/build_features.py`

- [ ] **Step 1: Add directories + ablation flags**

In `main()`, after the existing `_a` dir block (line ~100-109):

```python
    limit_up_dir = os.path.join(_a, "limit_up_processed")
    pledge_dir = os.path.join(_a, "pledge_processed")
    index_membership_dir = os.path.join(_a, "index_membership_processed")
    # market_env_daily is global -> auto-loaded by _merge_market_env internally
```

After `--no-comment` (line 78):

```python
    # limit-up ecology family is DEFERRED (top scope note) — no --no-limit-up flag
    parser.add_argument("--no-pledge", action="store_true", help="Exclude pledge risk")
    parser.add_argument("--no-market-env", action="store_true", help="Exclude market env")
    parser.add_argument("--no-index-membership", action="store_true",
                        help="Exclude index membership")
```

- [ ] **Step 2: Pass flags to `FeaturePipeline(...)`** (after `use_comment=use_cm,` line 135)

```python
        use_limit_up=False,  # limit-up family deferred (top scope note)
        use_pledge=not args.no_pledge,
        use_market_env=not args.no_market_env,
        use_index_membership=not args.no_index_membership,
```

- [ ] **Step 3: Load new per-stock files** (after `concept_df = ...` line 195)

```python
            # limit_up_df intentionally NOT loaded (family deferred, top scope note)
            pledge_df = _load_stock_parquet(pledge_dir, code)
            index_membership_df = _load_stock_parquet(index_membership_dir, code)
```

Add to the `loaded_parts` list (line ~204):

```python
                ("Pledge", pledge_df), ("IdxM", index_membership_df),
```

- [ ] **Step 4: Pass to `save_features(...)`** (after `concept_df=...` line 231)

```python
                limit_up_df=None,  # deferred (top scope note)
                pledge_df=pledge_df if not pledge_df.empty else None,
                index_membership_df=index_membership_df if not index_membership_df.empty else None,
```

- [ ] **Step 5: Verify a single-stock build with new sources**

Run:
```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/build_features.py --stock 000001 --force --output-dir data/features_dev
```
Then inspect columns:
```bash
PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
df = pd.read_parquet('data/features_dev/000001.parquet')
new = [c for c in df.columns if c.startswith(('pledge_','lhb_','market_','menv_','is_index','idx_'))]
print('new cols:', len(new)); print(sorted(new)[:60])
print('pledge_risk nonzero dates:', int((df['pledge_risk']>0).sum()))
"
```
Expected: new cols present (with `_ma5/_ma10/_ma20/_std20/_accel/_z20` variants from TemporalTransformer), `pledge_risk`>0 on some dates. (No `zt_/zb_/dt_/yzt_/has_zt` cols — limit-up family deferred per top scope note.)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_features.py
git commit -m "feat: wire new sources + ablation flags into build_features.py"
```

---

## Phase C — L4 evaluation gate

### Task C1: `scripts/feature_ic_report.py`

**Files:**
- Create: `scripts/feature_ic_report.py`
- Output: `reports/feature_ic_report.csv`

Per-feature Spearman RankIC vs forward return, dual-window (full + 2021+), both cross-sectional (per date) and time-series (per stock) variants.

- [ ] **Step 1: Write the script**

```python
"""L4 gate: per-feature IC report over pre-built feature panels.

Computes, per feature column:
  - ic_cross   : mean cross-sectional Spearman RankIC vs forward return (per date)
  - icir_cross : ic_cross / std(IC across dates)
  - ic_pos_ratio: fraction of dates with positive IC
  - coverage   : fraction of dates with >= MIN_STOCKS observations
  - ic_ts      : mean per-stock time-series Spearman IC (handles global/regime features)
Dual window: full history + 2021+ primary window.

Output: reports/feature_ic_report.csv
"""
import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stoke_ml.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HORIZON = 1
MIN_STOCKS = 20
PRIMARY = pd.Timestamp("2021-01-01")

# Columns to skip (metadata / target / non-features).
SKIP = {"date", "stock_code", "sector", "sector_code", "size_proxy",
        "open", "high", "low", "close", "volume", "amount"}


def forward_return(df: pd.DataFrame, h: int = HORIZON) -> pd.Series:
    close = df["close"]
    return close.shift(-h) / close - 1.0


def ic_row(feature: str, panel: pd.DataFrame) -> dict:
    """One row: cross-sectional + time-series IC for a feature across the panel."""
    rows = panel[["date", "stock_code", feature]].copy()
    rows["fwd"] = None  # filled in caller loop? -> use precomputed returns
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--max-stocks", type=int, default=None,
                    help="cap stock count (dev runs); None = all")
    ap.add_argument("--feature-subset", default=None,
                    help="comma-separated features; None = all (minus SKIP)")
    args = ap.parse_args()

    cfg = load_config()
    feat_dir = args.features_dir or os.path.join(cfg.project.data_dir, "features")
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if args.max_stocks:
        files = files[: args.max_stocks]
    if not files:
        log.error("no feature files under %s", feat_dir)
        return

    # Discover feature columns from the first file.
    first = pd.read_parquet(files[0])
    cols = [c for c in first.columns
            if c not in SKIP and not c.startswith("fwd_")]
    if args.feature_subset:
        cols = [c for c in cols if c in set(args.feature_subset.split(","))]
    log.info("%d stocks, %d candidate features", len(files), len(cols))

    # Per-window: collect (date, stock, feature, fwd_ret) long frames.
    windows = {"full": [], "primary": []}
    n_loaded = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date", "stock_code", "close"] + cols)
        except Exception as e:
            log.warning("skip %s: %s", os.path.basename(f), e)
            continue
        if len(df) < 30 or "close" not in df:
            continue
        df["fwd_ret"] = forward_return(df)
        df["date"] = pd.to_datetime(df["date"])
        n_loaded += 1
        for win, mask in (("full", df["date"] >= pd.Timestamp("2010-01-01")),
                          ("primary", df["date"] >= PRIMARY)):
            sub = df.loc[mask, ["date", "stock_code", "fwd_ret"] + cols]
            sub = sub.dropna(subset=["fwd_ret"])
            windows[win].append(sub)
    log.info("loaded %d stocks", n_loaded)

    results = []
    for win, parts in windows.items():
        if not parts:
            continue
        panel = pd.concat(parts, ignore_index=True)
        dates = panel["date"].unique()
        # ---- cross-sectional IC (long frame -> pivot per feature) ----
        # For each date, rank feature and fwd across stocks; Spearman.
        for col in cols:
            ics = []
            for d in dates:
                sub = panel[panel["date"] == d][[col, "fwd_ret"]].dropna()
                if len(sub) < MIN_STOCKS or sub[col].nunique() < 2:
                    continue
                rho, _ = spearmanr(sub[col], sub["fwd_ret"])
                if np.isfinite(rho):
                    ics.append(rho)
            if not ics:
                continue
            ics = np.asarray(ics)
            ic_cross = float(ics.mean())
            icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
            pos = float((ics > 0).mean())
            coverage = float(len(ics) / len(dates))
            # ---- time-series IC (per stock) ----
            ts_ics = []
            for code, g in panel.groupby("stock_code"):
                gg = g[[col, "fwd_ret"]].dropna()
                if len(gg) >= 30 and gg[col].nunique() >= 2:
                    rho, _ = spearmanr(gg[col], gg["fwd_ret"])
                    if np.isfinite(rho):
                        ts_ics.append(rho)
            ic_ts = float(np.mean(ts_ics)) if ts_ics else np.nan
            results.append({
                "window": win, "feature": col,
                "ic_cross": ic_cross, "icir_cross": icir,
                "ic_pos_ratio": pos, "coverage": coverage, "ic_ts": ic_ts,
            })
            log.info("[%s] %s ic=%.4f icir=%.2f ts=%.4f", win, col, ic_cross, icir, ic_ts)

    rep = pd.DataFrame(results)
    if rep.empty:
        log.error("no results produced")
        return

    def _grade(r):
        ic = max(abs(r["ic_cross"]), abs(r["ic_ts"]) if np.isfinite(r["ic_ts"]) else 0)
        return "high" if ic >= 0.02 else ("medium" if ic >= 0.01 else "low")

    rep["grade"] = rep.apply(_grade, axis=1)
    rep = rep.sort_values(["window", "grade", "ic_cross"], ascending=[True, True, False])
    os.makedirs("reports", exist_ok=True)
    rep.to_csv("reports/feature_ic_report.csv", index=False)
    log.info("wrote reports/feature_ic_report.csv (%d rows)", len(rep))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a small subset**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/feature_ic_report.py --features-dir data/features_dev --max-stocks 15 --feature-subset pledge_ratio,is_index_member,high_low_ratio,market_turnover_z,menv_regime_z,lhb_count_5d`
Expected: `reports/feature_ic_report.csv` with rows for the 6 features; `menv_regime_z`/`high_low_ratio` show `ic_cross≈0` (global — expected) but nonzero `ic_ts`.

- [ ] **Step 3: Commit**

```bash
git add scripts/feature_ic_report.py
git commit -m "feat: per-feature dual-window IC report (cross-sectional + time-series)"
```

---

### Task C2: `scripts/feature_leakage_report.py`

**Files:**
- Create: `scripts/feature_leakage_report.py`
- Output: `reports/feature_leakage_report.csv`

Audits the PIT shift for every new source by re-deriving it from raw parquet vs feature parquet, plus a high-IC heuristic scan.

- [ ] **Step 1: Write the script**

```python
"""L4 gate: leakage audit for pre-built features.

For each new source, on a sample of stocks, verify the PIT invariant:
  feature[source_col] at trading day t == raw source value at day t-1.
Also flags any feature whose cross-sectional |IC| is implausibly high (> 0.15)
for manual review (the classic look-ahead signature).

Output: reports/feature_leakage_report.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config

SAMPLES = 30
# source_dir -> (feature_source_col, feature_col_in_panel, shift)
# limit-up family excluded (deferred per top scope note).
CHECKS = [
    ("pledge_processed", "has_pledge", "has_pledge", 1),
    ("index_membership_processed", "is_index_member", "is_index_member", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--stocks", default=None, help="sample stocks (comma-sep); default random 30")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares")
    feat_dir = args.features_dir or os.path.join(cfg.project.data_dir, "features")
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if not feat_files:
        print("no feature files")
        return
    codes = args.stocks.split(",") if args.stocks else None
    if codes is None:
        rng = np.random.default_rng(0)
        codes = [os.path.basename(f).replace(".parquet", "")
                 for f in rng.choice(feat_files, size=min(SAMPLES, len(feat_files)), replace=False)]

    rows = []
    for src_dir, raw_col, feat_col, lag in CHECKS:
        ok = skip = bad = 0
        for code in codes:
            raw_path = os.path.join(base, src_dir, f"{code}.parquet")
            feat_path = os.path.join(feat_dir, f"{code}.parquet")
            if not (os.path.exists(raw_path) and os.path.exists(feat_path)):
                skip += 1
                continue
            raw = pd.read_parquet(raw_path, columns=["date", raw_col])
            feat = pd.read_parquet(feat_path, columns=["date", feat_col])
            if raw.empty or feat.empty or raw_col not in raw or feat_col not in feat:
                skip += 1
                continue
            raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
            feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
            raw_val = raw.set_index("date")[raw_col]
            # Expected: feature at t == raw at t-1.
            merged = feat.merge(raw.rename(columns={raw_col: "raw"}), on="date", how="left")
            merged["raw_lag"] = merged["raw"].shift(lag)
            merged = merged.dropna(subset=["raw_lag"])
            if len(merged) < 10:
                skip += 1
                continue
            match = np.isclose(merged[feat_col].astype(float),
                               merged["raw_lag"].astype(float)).mean()
            if match >= 0.95:
                ok += 1
            else:
                bad += 1
                rows.append({"source": src_dir, "code": code,
                             "check": "pit_lag", "pass": False,
                             "match_rate": float(match), "detail": f"expected lag {lag}"})
        rows.append({"source": src_dir, "code": "AGG", "check": "pit_lag",
                     "pass": ok > 0 and bad == 0, "match_rate": None,
                     "detail": f"{ok} ok, {bad} bad, {skip} skipped"})

    # High-|IC| heuristic scan (uses feature_ic_report.csv if present).
    ic_path = os.path.join("reports", "feature_ic_report.csv")
    if os.path.exists(ic_path):
        ic = pd.read_csv(ic_path)
        flagged = ic[(ic["window"] == "primary") & (ic["ic_cross"].abs() > 0.15)]
        for _, r in flagged.iterrows():
            rows.append({"source": "IC", "code": "AGG", "check": "high_ic",
                         "pass": False, "match_rate": None,
                         "detail": f"{r['feature']} ic_cross={r['ic_cross']:.3f} — review for leakage"})

    rep = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    rep.to_csv("reports/feature_leakage_report.csv", index=False)
    fails = rep[rep["pass"] == False]  # noqa: E712
    print(f"leakage report written: {len(rep)} rows, {len(fails)} failures/suspicious")
    for _, r in fails.iterrows():
        print("  -", r["source"], r["code"], r["check"], r["detail"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/feature_leakage_report.py --features-dir data/features_dev`
Expected: `pit_lag` AGG rows `pass=True` for all 3 sources (match_rate ≥ 0.95 on the 15-stock dev panel), plus any high-IC flags.

- [ ] **Step 3: Commit**

```bash
git add scripts/feature_leakage_report.py
git commit -m "feat: leakage audit — verifies PIT lag for new sources + high-IC scan"
```

---

## Phase D — Parallel build

### Task D1: Multiprocessing + shards in `build_features.py`

**Files:**
- Modify: `scripts/build_features.py`

- [ ] **Step 1: Extract per-stock work into a module-level function** (workers can't pickle `main()` closures on Windows spawn). Add after `_load_stock_parquet`:

```python
def build_one(args: dict) -> tuple[str, str]:
    """Build features for one stock. args carries ALL inputs; returns (code, status)."""
    code = args["code"]
    try:
        pipeline = FeaturePipeline(
            seq_len=args["seq_len"], horizon=args["horizon"],
            use_sentiment=args["use_sentiment"], use_guba=args["use_guba"],
            use_comment=args["use_comment"], use_limit_up=args["use_limit_up"],
            use_pledge=args["use_pledge"], use_market_env=args["use_market_env"],
            use_index_membership=args["use_index_membership"],
        )
        df = args["storage"].load_daily(code, start_date=args["start"], end_date=args["end"])
        if df.empty:
            return code, "empty"
        output_path = os.path.join(args["output_dir"], f"{code}.parquet")
        if os.path.exists(output_path) and not args["force"]:
            return code, "exists"
        loaders = {
            # limit_up_df intentionally absent (family deferred, top scope note)
            "pledge_df": ("pledge_dir", "pledge_processed"),
            "index_membership_df": ("index_dir", "index_membership_processed"),
        }
        aux = {}
        for kw, (dirkey, sub) in loaders.items():
            aux[kw] = _load_stock_parquet(os.path.join(args["data_dir"], "a_shares", sub), code)
        pipeline.save_features(
            output_path, df,
            sentiment_df=_load_opt(args, "news_storage", "load_daily_sentiment", code),
            margin_df=_load_opt(args, "margin_storage", "load", code),
            northbound_df=_load_opt(args, "nb_storage", "load", code),
            dragon_tiger_df=_load_opt(args, "dt_storage", "load", code),
            fundamental_df=_load_opt(args, "fund_storage", "forward_fill_to_daily", code),
            valuation_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "valuation"), code),
            capital_flow_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "capital_flow_processed"), code),
            board_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "board_processed"), code),
            sector_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "industry_ranking_processed"), code),
            block_trade_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "block_trade_processed"), code),
            dividend_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "dividend_processed"), code),
            lockup_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "lockup_processed"), code),
            shareholder_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "shareholder_processed"), code),
            concept_df=_load_stock_parquet(os.path.join(args["data_dir"], "a_shares", "concept_blocks_processed"), code),
            guba_df=_load_opt(args, "guba_storage", "load_daily_sentiment", code),
            comment_df=_load_opt(args, "comment_storage", "build_features", code),
            announcement_df=_load_opt(args, "ann_storage", "load_daily_sentiment", code),
            etf_flow_df=_load_etf(args, code),
            limit_up_df=None,  # deferred (top scope note)
            pledge_df=aux["pledge_df"] or None,
            index_membership_df=aux["index_membership_df"] or None,
        )
        return code, "built"
    except Exception:
        logging.getLogger(__name__).exception("[%s] failed", code)
        return code, "failed"
```

Plus the module-level loaders:

```python
def _load_opt(args, storage_key, method, code):
    obj = args.get(storage_key)
    if obj is None:
        return None
    try:
        out = getattr(obj, method)(code, args["start"], args["end"])
        return out if not out.empty else None
    except Exception:
        return None


def _load_etf(args, code):
    try:
        sector = args["sector_mapper"].get_sector(code)
        if not sector:
            return None
        out = args["etf_storage"].load_sector_flow(sector, args["start"], args["end"])
        return out if not out.empty else None
    except Exception:
        return None
```

- [ ] **Step 2: Add `--shard` and `--jobs` CLI args** (in `main()`)

```python
    parser.add_argument("--shard", type=str, default=None,
                        help="k/n shard over codes, e.g. 0/4")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes (0 = cpu_count)")
    # limit-up family deferred (top scope note): use_limit_up hard-False, no flag
    parser.add_argument("--pledge", dest="use_pledge", action="store_true", default=True)
    parser.add_argument("--no-pledge", dest="use_pledge", action="store_false")
    parser.add_argument("--market-env", dest="use_market_env", action="store_true", default=True)
    parser.add_argument("--no-market-env", dest="use_market_env", action="store_false")
    parser.add_argument("--index-membership", dest="use_index_membership", action="store_true", default=True)
    parser.add_argument("--no-index-membership", dest="use_index_membership", action="store_false")
```

(Replace the earlier `--no-*` flags from Task B5 with these dual-store flags — same behavior, one source of truth.)

- [ ] **Step 3: Shard the code list** (after `codes = ...` line 114)

```python
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        codes = [c for i, c in enumerate(codes) if i % n == k]
```

- [ ] **Step 4: Dispatch via ProcessPoolExecutor when `--jobs > 1`** (replace the serial `for code in codes:` loop)

```python
    worker_args = {
        "code": None, "seq_len": cfg.features.seq_len,
        "horizon": cfg.features.target_horizon,
        "use_sentiment": cfg.features.get("use_sentiment", True),
        "use_guba": use_gb, "use_comment": use_cm,
        "use_limit_up": False,  # limit-up deferred (top scope note)
        "use_pledge": args.use_pledge,
        "use_market_env": args.use_market_env,
        "use_index_membership": args.use_index_membership,
        "storage": storage, "news_storage": news_storage,
        "margin_storage": margin_storage, "nb_storage": nb_storage,
        "dt_storage": dt_storage, "fund_storage": fund_storage,
        "etf_storage": etf_storage, "guba_storage": guba_storage,
        "comment_storage": comment_storage, "ann_storage": ann_storage,
        "sector_mapper": sector_mapper,
        "data_dir": data_dir, "output_dir": output_dir,
        "start": date_start, "end": date_end, "force": args.force,
    }

    import concurrent.futures as cf
    jobs = args.jobs
    if jobs <= 1:
        results = [build_one({**worker_args, "code": c}) for c in codes]
    else:
        workers = jobs if jobs > 0 else min(32, (os.cpu_count() or 1) + 2)
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(
                lambda c: build_one({**worker_args, "code": c}), codes))
    counts = Counter(s for _, s in results)
    logger.info("Done: %s (out of %d stocks)", dict(counts), len(codes))
```

Add `from collections import Counter` at the top.

- [ ] **Step 5: Determinism smoke (single vs parallel, one shard)**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --stock 000001 --force --output-dir data/features_det_single
PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --stock 000001 --force --output-dir data/features_det_multi --jobs 2
PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
a = pd.read_parquet('data/features_det_single/000001.parquet')
b = pd.read_parquet('data/features_det_multi/000001.parquet')
print('identical:', a.equals(b))
"
```
Expected: `identical: True` (bit-identical determinism between serial and parallel).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_features.py
git commit -m "feat: parallel multiprocessing build + shards in build_features.py"
```

---

## Phase E — Tests

### Task E1: `tests/features/test_new_sources.py`

**Files:**
- Create: `tests/features/test_new_sources.py`

- [ ] **Step 1: Write the tests**

```python
"""Tests for the 4 new feature families (spec §7.1-7.5)."""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.market_env import MarketEnvRefiner
from stoke_ml.features.pipeline import (
    FeaturePipeline, _batch_fill_shift, _merge_daily_aux,
)


def _kline(n=120, start="2021-01-04"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "date": idx, "open": close - 0.5, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1_000_000, 2_000_000, n),
        "amount": rng.uniform(1e8, 5e8, n), "pct_change": rng.normal(0, 1, n),
    })


def test_merge_daily_aux_pit_lag():
    df = _kline()
    aux = pd.DataFrame({"date": df["date"], "pledge_ratio": 3.0, "has_pledge": True})
    merged = _merge_daily_aux(df.copy(), aux)
    # feature at t == source at t-1 (PIT lag 1)
    assert np.allclose(merged["pledge_ratio"].iloc[1:].values,
                       aux["pledge_ratio"].iloc[:-1].values)
    assert merged["has_pledge"].iloc[1]  # first aux row shifted to second K-line day
    assert not merged["has_pledge"].iloc[0]


def test_merge_market_env_global(tmp_path):
    me = pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=10),
                       "high_low_ratio": np.linspace(-1, 1, 10)})
    from stoke_ml.features import pipeline as P
    me_path = tmp_path / "market_env_daily.parquet"
    me.to_parquet(me_path)
    df = _kline(n=10)
    p = FeaturePipeline(use_technical=False, use_scoring=False, use_temporal=False,
                        use_sentiment=False, use_announcements=False, use_guba=False,
                        use_comment=False, use_margin=False, use_northbound=False,
                        use_dragon_tiger=False, use_fundamental=False, use_valuation=False,
                        use_etf_flow=False, use_interaction=False, use_capital_flow=False,
                        use_block_trade=False, use_shareholder=False, use_lockup=False,
                        use_dividend=False, use_board=False, use_sector=False,
                        use_concept=False, use_macro=False, use_industry=False,
                        use_emotion_refine=False, use_fundamental_refine=False,
                        use_temporal_stats=False, use_pledge=False, use_limit_up=False,
                        use_index_membership=False)
    # _merge_market_env reads from cfg data_dir; bypass by injecting cache.
    p._market_env_cache = me
    out = p._merge_market_env(df)
    assert "high_low_ratio" in out.columns
    # lagged: out[t] == me[t-1]
    assert np.allclose(out["high_low_ratio"].iloc[1:].values, me["high_low_ratio"].iloc[:-1].values)


def test_market_env_refiner_factors():
    d = pd.date_range("2021-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": d, "shibor_1M": 2.0, "fx_usd_cny": 7.0, "cpi_yoy": 1.0,
        "m1_yoy": 5.0, "m2_yoy": 8.0, "bond_cn_10y2y_spread": 0.5,
        "bond_us_10y": 4.0, "bond_cn_10y": 3.0,
    })
    out = MarketEnvRefiner().refine(df)
    for c in ("menv_shibor_1m_z", "menv_us_cn_10y_spread", "menv_m1_m2_spread",
              "menv_regime_z", "menv_bond_10y2y_spread"):
        assert c in out.columns
    assert np.allclose(out["menv_m1_m2_spread"], -3.0)
    assert np.allclose(out["menv_us_cn_10y_spread"], 1.0)


def test_market_env_refiner_graceful_missing():
    df = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=10)})
    out = MarketEnvRefiner().refine(df)
    assert "menv_regime_z" not in out.columns  # no inputs -> no composite


def test_batch_fill_shift_int16_count():
    df = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=5)})
    df["pledge_count_20d"] = [1, 2, 3, 4, 5]
    df["has_pledge"] = [True, True, True, True, True]
    _batch_fill_shift(df, ["pledge_count_20d", "has_pledge"])
    assert df["pledge_count_20d"].dtype == np.int16
    assert df["has_pledge"].dtype == bool
    assert df["pledge_count_20d"].iloc[1] == 1  # shifted


def test_ic_correctness_known_signal():
    """Inject a feature == forward return + noise; cross-sectional IC ~ high."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=30)
    ics = []
    for d in dates:
        n = 80
        noise = rng.normal(0, 0.1, n)
        sig = rng.normal(0, 1, n)
        fwd = sig + noise
        feat = fwd + rng.normal(0, 0.05, n)
        rho, _ = spearmanr(feat, fwd)
        ics.append(rho)
    mean_ic = float(np.mean(ics))
    assert mean_ic > 0.7
```

- [ ] **Step 2: Run the tests**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m pytest tests/features/test_new_sources.py -q`
Expected: all 6 tests pass (merge PIT lag, market-env global merge, refiner factors, graceful missing, int16 shift, IC sanity).

- [ ] **Step 3: Run the full feature test suite**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python -m pytest tests/features/ -q`
Expected: existing feature tests still pass (no regression from B2/B3/B4 changes).

- [ ] **Step 4: Commit**

```bash
git add tests/features/test_new_sources.py
git commit -m "test: cover 4 new feature families (PIT lag, market env, refiner, dtypes)"
```

---

## Phase F — Full build + evaluation gate

Runs the complete 5530-stock feature build (all 4 new sources enabled), then applies the L4 evaluation gate (IC report + leakage audit) at scale and records results.

### Task F1: Full parallel build

**Files:**
- None new (uses `scripts/build_features.py` from D1)

- [ ] **Step 1: Confirm preprocess outputs are complete**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "import glob,os; [print(d, len(glob.glob(os.path.join('data/a_shares', d, '*.parquet')))) for d in ('pledge_processed','index_membership_processed')]; print('market_env_daily.parquet', os.path.exists('data/a_shares/market_breadth/market_env_daily.parquet'))"`
Expected: `pledge_processed` ≈3257, `index_membership_processed` few-hundred (all 3-index members 2015-2026), market_env_daily exists. (limit-up family excluded — deferred per top scope note.)

- [ ] **Step 2: Launch full build over 8 shards × 4 workers**

Run (8 shards, each `--jobs 4`, output to `data/features`):
```bash
for k in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python \
    scripts/build_features.py --shard "$k/8" --jobs 4 > "logs/build_shard_$k.log" 2>&1 &
done
wait
```
Expected: `data/features/*.parquet` for every stock with a K-line flat; each shard log ends with `Done:` and `fail=0`. (Run `grep -h "Done:" logs/build_shard_*.log` to aggregate.)

- [ ] **Step 3: Verify completeness**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "import glob,os; n=len(glob.glob('data/features/*.parquet')); m=len(glob.glob('data/a_shares/daily/*.parquet')); print(f'features={n} daily_flats={m} coverage={n/m:.1%}')"`
Expected: coverage ≥ 99% of daily flats (stocks whose K-line is too short for `seq_len=60` are legitimately skipped).

- [ ] **Step 4: Determinism re-check on one shard**

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --shard 0/8 --output-dir data/features_det2 --jobs 4
PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --shard 0/8 --output-dir data/features_det1 --jobs 1
PYTHONPATH=. ./.venv/Scripts/python -c "
import glob, pandas as pd
for p in sorted(glob.glob('data/features_det1/*.parquet')):
    b = pd.read_parquet(p.replace('det1','det2'))
    a = pd.read_parquet(p)
    assert a.equals(b), p
print('all identical')
"
```
Expected: `all identical` (parallel build is bit-deterministic).

- [ ] **Step 5: Commit**

Nothing new to commit (build is output data; scripts already committed). Skip commit if no script changed.

---

### Task F2: IC report at scale

**Files:**
- None new (uses `scripts/feature_ic_report.py` from C1)

- [ ] **Step 1: Run IC report over the full feature set**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/feature_ic_report.py --features-dir data/features --max-stocks 300`
Expected: `reports/feature_ic_report.csv` with one row per feature (all features, not just the 6 smoke-tested). Confirm `pledge_risk`, `is_index_member` have plausible `ic_cross` values; global features (`market_*`, `menv_*`) show `ic_cross≈0` with nonzero `ic_ts`. (No `has_zt`/`zt_*` rows — limit-up family deferred per top scope note.)

- [ ] **Step 2: Commit**

```bash
git add reports/feature_ic_report.csv
git commit -m "report: full feature IC report (300-stock panel)"
```

---

### Task F3: Leakage audit at scale

**Files:**
- None new (uses `scripts/feature_leakage_report.py` from C2)

- [ ] **Step 1: Run leakage audit**

Run: `PYTHONPATH=. PYTHONIOENCODING=utf-8 ./.venv/Scripts/python scripts/feature_leakage_report.py --features-dir data/features`
Expected: `reports/feature_leakage_report.csv`; `pit_lag` rows `pass=True` (match_rate ≥ 0.95) for all new sources; high-IC flags (if any) are global features already explained by `ic_cross≈0`.

- [ ] **Step 2: Commit**

```bash
git add reports/feature_leakage_report.csv
git commit -m "report: PIT leakage audit at scale for new sources"
```

---

### Task F4: Record results + wrap up

**Files:**
- Modify: `docs/research-findings.md` (append a section)

- [ ] **Step 1: Record build + IC results**

Append to `docs/research-findings.md` a section `## 2026-08-02 FE v2: 4 new feature families` containing: stock count, feature count (columns per stock), the top-10 |IC| features from `feature_ic_report.csv`, and any leakage flags. Keep it 15 lines or fewer.

- [ ] **Step 2: Commit**

```bash
git add docs/research-findings.md
git commit -m "docs: record FE v2 build results (4 new families, IC, leakage audit)"
```

---

## Self-review (writing-plans)

**Spec coverage:**
- §3.1 limit-up ecology → A1 (preprocess) + B2 (`_merge_limit_up` implemented but NOT wired — deferred per top scope note) + B3 (`_merge_dragon_tiger` lhb_reason) ✓
- §3.2 equity/capital risk → A2 (pledge) + B2 (`_merge_pledge`) ✓
- §3.3 market+macro → A3 (`_preprocess_market_env`) + A4 (index membership) + B4 (`MarketEnvRefiner`) ✓
- §3.4 feature evaluation → C1 (IC) + C2 (leakage) + F2/F3 ✓
- §5 architecture (4-layer factory) → B2 (fuse layer) + B4 (deepen layer) ✓
- §7 parallel build → D1 (shards/jobs) + F1 (full run) ✓

**Placeholder scan:** Every step has concrete code or a run command; no TBD/TODO found. ✓

**Type consistency:**
- MARKET_ENV_COLS (B1) match `_preprocess_market_env.py` output exactly (7 cols, no limit-up temperature) ✓
- `market_adv_ratio` (derived in A3) vs `advance_rate` (raw sentiment col kept as-is) — both in B1 `MARKET_ENV_COLS`; intentional (raw rides along, guarded by `has_market_sent`). ✓
- `zt_first_seal_hour`, `pledge_ratio`, `is_index_member`, `idx_change_30d` match between A1/A2/A4 outputs and B1 constants and C1 `--feature-subset`. ✓
- B5 build_features flags (`--pledge/--market-env/--index-membership`) match D1 dual-store flags; limit-up hard-False (deferred). ✓
- New PO prefixes (`zt_/zb_/dt_/yzt_/pledge_/lhb_/market_/idx_/index_/menv_/has_`) require no `_PK_PREFIXES` edits per TemporalTransformer scan. ✓
