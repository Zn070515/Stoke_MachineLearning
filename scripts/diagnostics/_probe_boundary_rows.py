"""Inspect rows in the ambiguous ratio band [75, 105] across all files.

Also inspect the two anomaly rows found by the histogram scan:
  000858 ratio~293777 (junk?), 000510 ratio~0.895 (hand-mislabel?).

Prints per-file counts in the band plus sample rows, so we can judge whether
the boundary rows are gu (qfq~84, would break a global threshold) or hand
(qfq~0.84 reverse-split stocks).

Read-only.  Writes nothing.
"""
import glob
import os

import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"
BAND = (74.0, 95.0)


def main() -> None:
    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    band_rows = []  # (file, ratio, volume, amount, close, date)
    n_band = 0
    n_files_with_band = 0
    for path in files:
        df = pd.read_parquet(path)
        if not {"volume", "amount", "close", "date", "high", "low"}.issubset(df.columns):
            continue
        vol = df["volume"].astype("float64").to_numpy()
        amt = df["amount"].astype("float64").to_numpy()
        close = df["close"].astype("float64").to_numpy()
        high = df["high"].astype("float64").to_numpy()
        low = df["low"].astype("float64").to_numpy()
        ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
        ok &= (vol > 0) & (amt > 0) & (close > 0) & (high > 0) & (low > 0)
        if not ok.any():
            continue
        ratio = amt[ok] / vol[ok] / close[ok]
        in_band = (ratio >= BAND[0]) & (ratio <= BAND[1])
        n = int(in_band.sum())
        if n:
            n_band += n
            n_files_with_band += 1
            fname = os.path.basename(path)
            band_pos = np.where(in_band)[0]
            orig_idx = np.where(ok)[0][band_pos]
            for m in range(min(5, len(band_pos))):
                j = orig_idx[m]
                band_rows.append((fname, ratio[band_pos[m]], vol[j], amt[j], close[j], str(df["date"].iloc[j]), high[j], low[j]))
            if len(band_pos) > 5:
                band_rows.append((fname, "...", len(band_pos), 0, 0, f"({len(band_pos)} more in band)", 0, 0))

    print(f"files with rows in [{BAND[0]},{BAND[1]}]: {n_files_with_band}, total band rows: {n_band}", flush=True)
    print("\nband rows (file, ratio, volume, amount, close, date, high, low):", flush=True)
    for r in band_rows:
        if r[1] == "...":
            print(f"  {r[0]} {r[1]} {r[2]}", flush=True)
        else:
            print(f"  {r[0]} ratio={r[1]:.3f} vol={r[2]:.0f} amt={r[3]:.0f} close={r[4]:.4f} date={r[5]} high={r[6]:.4f} low={r[7]:.4f}", flush=True)

    # specific anomalies
    for stock, lo, hi, label in [("000858", 100000, np.inf, "000858 junk high-ratio"), ("000510", 0, 3, "000510 low-ratio")]:
        path = f"{DAILY}/{stock}.parquet"
        df = pd.read_parquet(path)
        vol = df["volume"].astype("float64").to_numpy()
        amt = df["amount"].astype("float64").to_numpy()
        close = df["close"].astype("float64").to_numpy()
        ok = (vol > 0) & (amt > 0) & (close > 0)
        ratio = amt[ok] / vol[ok] / close[ok]
        sel = (ratio >= lo) & (ratio <= hi)
        idx = np.where(ok)[0][sel]
        print(f"\n{label}: {len(idx)} rows", flush=True)
        if len(idx):
            cols = ["date", "open", "high", "low", "close", "volume", "amount"]
            print(df.iloc[idx[:10]][cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
