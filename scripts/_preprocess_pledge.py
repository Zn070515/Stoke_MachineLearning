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
    delta = (r.groupby("ann_dt")["_delta"].sum().rename("_delta")
             .reset_index().rename(columns={"ann_dt": "date"}))

    # As-of margin line: last announced ACTIVE pledge's 预估平仓线, forward-filled.
    active = r[r[CN["status"]] == "未解押"]
    line = (active.groupby("ann_dt")[CN["margin_line"]]
            .last().rename("_margin_line").reset_index()
            .rename(columns={"ann_dt": "date"}))

    k = kline[["date", "close"]].copy()
    k["date"] = pd.to_datetime(k["date"]).dt.normalize()
    k = k.drop_duplicates("date").sort_values("date")

    out = k.merge(delta, on="date", how="left").merge(line, on="date", how="left")
    out["_delta"] = out["_delta"].fillna(0.0)
    out["pledge_ratio"] = out["_delta"].cumsum().clip(lower=0.0).astype(np.float32)
    out["_margin_line"] = out["_margin_line"].ffill()
    # No active pledge exposure -> no margin line (clears stale line after full release).
    out.loc[out["pledge_ratio"] < 1e-7, "_margin_line"] = np.nan
    out["pledge_margin_dist"] = (out["close"] / out["_margin_line"] - 1.0)
    out["pledge_risk"] = out["pledge_margin_dist"].notna() & (out["pledge_margin_dist"] < 0.20)
    out["pledge_count_20d"] = out["_delta"].ne(0).astype(int).rolling(20, min_periods=1).sum().astype("int16")
    out["has_pledge"] = out["pledge_ratio"].gt(0).cummax()
    out = out.drop(columns=["close", "_delta", "_margin_line"])
    out["pledge_margin_dist"] = out["pledge_margin_dist"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
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
    # Keep only per-stock files: the pledge dir also holds aggregate tables
    # (market_pledge_stats.parquet, pledge_ratios.parquet) that are not stocks.
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
            # Full-history K-line via the storage layer (flat daily/{code}.parquet
            # preferred, partitioned fallback). Wide range never excludes dates.
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
