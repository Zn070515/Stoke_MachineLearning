"""Per-year ratio medians + out-of-band row inspection for the anomalous
files (000408, 000858) to decide whether the classifier needs an iterative
robust year-median.

Read-only.
"""
import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"


def main() -> None:
    codes = ["000408", "000858"]
    import sys
    if len(sys.argv) > 1:
        codes = sys.argv[1:]
    for code in codes:
        df = pd.read_parquet(f"{DAILY}/{code}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        vol = df["volume"].astype("float64").to_numpy()
        amt = df["amount"].astype("float64").to_numpy()
        close = df["close"].astype("float64").to_numpy()
        ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
        ok &= (vol > 0) & (amt > 0) & (close > 0)
        ratio = amt[ok] / vol[ok] / close[ok]
        ok_idx = np.where(ok)[0]
        years = df.loc[ok, "date"].dt.year.to_numpy()
        s = pd.Series(ratio, index=years)
        print(f"\n=== {code} (n_valid={ok.sum()}/{len(df)}) ===", flush=True)
        med = s.groupby(level=0).median()
        cnt = s.groupby(level=0).count()
        for y, m, c in zip(med.index, med.values, cnt.values):
            yr = s[s.index == y]
            hi = float(yr.quantile(0.99))
            print(f"  {y}: med={m:9.2f} n={c:5d} p99={hi:12.1f}", flush=True)
        # rows with ratio > 100 or < 0.01 (out of formal band)
        oob_rows = ratio[(ratio > 100) | (ratio < 0.01)]
        print(f"  out-of-band rows: {int(oob_rows.size)}", flush=True)
        idx = np.where((ratio > 100) | (ratio < 0.01))[0]
        for k in idx[:15]:
            j = ok_idx[k]
            r = ratio[k]
            y = years[k]
            print(f"    {df.loc[j,'date'].date()} y={y} ratio={r:.1f} "
                  f"vol={vol[j]:.0f} amt={amt[j]:.0f} close={close[j]:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
