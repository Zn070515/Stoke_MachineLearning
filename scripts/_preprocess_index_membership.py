"""Preprocess historical index-membership intervals into per-stock daily features.

PIT rule: membership is judged purely from interval data produced by
scripts/download_index_hist.py (Baostock monthly-grid reconstruction).
in_date is the run's earliest Baostock monthly-refresh date (a proxy for
when membership became effective, per A4a's corrected semantics — NOT an
exact adjustment date); a stock is a member on trading day d iff some
interval has in_date <= d < out_date (out_date NaT means still active).
Expanded on the stock's own K-line trading calendar.
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
