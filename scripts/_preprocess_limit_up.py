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
    """Parse 'HH:MM[:SS]' to float hour (09:25 -> 9.42, 09:25:00 -> 9.42). NaN/'' -> 0.0."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return 0.0
    s = str(t).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0.0
    parts = s.split(":")
    try:
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) / 3600.0 if len(parts) > 2 else 0.0
        return h + m / 60.0 + sec
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

    # ZI-fill: booleans False, numerics 0.0.
    # Guarantee a stable schema (all event-type columns) even when a stock has
    # no events of a given type -- e.g. 000004 has no dt file, so has_dt and the
    # dt_* numerics would otherwise be missing entirely.
    for c in ("has_zt", "has_zb", "has_dt", "has_yzt"):
        if c not in out.columns:
            out[c] = False
    for c in (ZT_COLS + ZB_COLS + DT_COLS + YZT_COLS):
        if c != "date" and c not in out.columns:
            out[c] = 0.0
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
        except Exception as e:
            print(f"  WARN {os.path.basename(f)}: {e!r}", file=sys.stderr)
            continue
    if not frames:
        return pd.Series(dtype="int64")
    return pd.concat(frames).value_counts().sort_index()


def build_market_daily(base: str) -> pd.DataFrame:
    zt = _date_counts(os.path.join(base, ZT_DIR)).rename("zt_count")
    zb = _date_counts(os.path.join(base, ZB_DIR)).rename("zb_count")
    dt = _date_counts(os.path.join(base, DT_DIR)).rename("dt_count")
    yzt = _date_counts(os.path.join(base, YZT_DIR)).rename("yzt_count")
    # Series are already uniquely named by .rename(); no keys= (which would wrap
    # a MultiIndex on some pandas versions and mangle column names on write).
    df = pd.concat([zt, zb, dt, yzt], axis=1)
    df = df.fillna(0).reset_index(names="date")
    df["market_zt_ratio"] = (df["zt_count"] / N_LISTED).astype(np.float32)
    z20 = df["zt_count"].rolling(20)
    df["market_heat_z"] = ((df["zt_count"] - z20.mean()) / z20.std()).fillna(0.0).astype(np.float32)
    # Merge the sparse market sentiment snapshot (ladders/break_rate/...) if present.
    sent = os.path.join(base, SENT_PATH)
    if os.path.exists(sent):
        s = pd.read_parquet(sent)
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        # sentiment.parquet carries its own zt/zb/dt/yzt counts; drop them so the
        # merge doesn't collide with the full-history aggregation (which would
        # otherwise rename both sides to *_x/*_y).
        s = s.drop(columns=["zt_count", "zb_count", "dt_count", "yzt_count"], errors="ignore")
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
