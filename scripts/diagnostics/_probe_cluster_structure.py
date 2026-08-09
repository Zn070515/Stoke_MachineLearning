"""Inspect the ratio distribution structure of a few daily files.

For each stock, prints per-year ratio clusters to reveal whether the 股/手
split is clean, and whether the high cluster median is ~100x the low cluster
median (the 100x shares/lots relationship).  Also prints rows near the global
outlier (600795 gu_max ~50) to understand what qfq factor means there.

Read-only.  Writes nothing.
"""
import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"


def ratios(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    vol = df["volume"].astype("float64")
    amt = df["amount"].astype("float64")
    close = df["close"].astype("float64")
    ok = (vol > 0) & (amt > 0) & (close > 0) & np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    df = df[ok].copy()
    df["ratio"] = amt[ok] / vol[ok] / close[ok]
    df["year"] = df["date"].dt.year
    return df


def report(stock: str) -> None:
    path = f"{DAILY}/{stock}.parquet"
    df = ratios(path)
    print(f"\n=== {stock} (n={len(df)}) ===", flush=True)
    # per-year median + min/max of ratio
    g = df.groupby("year")["ratio"].agg(["median", "min", "max", "count"])
    for year, row in g.iterrows():
        flag = ""
        if row["max"] > 50:
            flag = "  <-- has ratio>50"
        print(f"  {year}: med={row['median']:10.4f} min={row['min']:10.4f} max={row['max']:10.4f} n={int(row['count']):6d}{flag}", flush=True)
    # global cluster split check: largest gap in sorted ratio
    r = np.sort(df["ratio"].to_numpy())
    gaps = np.diff(r)
    k = int(np.argmax(gaps))
    split = (r[k] + r[k + 1]) / 2
    lo_cluster = r[r <= split]
    hi_cluster = r[r > split]
    if len(lo_cluster) and len(hi_cluster):
        lo_med = float(np.median(lo_cluster))
        hi_med = float(np.median(hi_cluster))
        print(f"  largest-gap split at {split:.4f}: lo_med={lo_med:.4f} hi_med={hi_med:.4f} "
              f"hi/lo={hi_med / lo_med:.1f}x  lo_n={len(lo_cluster)} hi_n={len(hi_cluster)}", flush=True)
    else:
        print(f"  unimodal (all one cluster): max={r[-1]:.4f} min={r[0]:.4f}", flush=True)


def outlier_rows(stock: str, lo: float, hi: float) -> None:
    path = f"{DAILY}/{stock}.parquet"
    df = ratios(path)
    band = df[(df["ratio"] >= lo) & (df["ratio"] <= hi)].head(10)
    if len(band) == 0:
        print(f"\n({stock} no rows in ratio [{lo},{hi}])", flush=True)
        return
    print(f"\n=== {stock} rows with ratio in [{lo},{hi}] ===", flush=True)
    cols = ["date", "open", "high", "low", "close", "volume", "amount", "ratio"]
    print(band[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    for stock in ["600795", "000001", "000725", "600519"]:
        report(stock)
    outlier_rows("600795", 30, 60)
    outlier_rows("000001", 50, 200)
