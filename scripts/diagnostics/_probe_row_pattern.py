"""Show 000001's recent ratio-by-date pattern to reveal the hand-row structure.

Read-only.
"""
import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"


def main() -> None:
    for stock, lo_year in [("000001", 2015), ("000725", 2015)]:
        df = pd.read_parquet(f"{DAILY}/{stock}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        vol = df["volume"].astype("float64")
        amt = df["amount"].astype("float64")
        close = df["close"].astype("float64")
        ok = (vol > 0) & (amt > 0) & (close > 0)
        df = df[ok].copy()
        df["ratio"] = amt[ok] / vol[ok] / close[ok]
        df = df[df["date"].dt.year >= lo_year]
        print(f"\n=== {stock} rows >= {lo_year}: n={len(df)} ===", flush=True)
        # print ratio per row as a compact time series, marking the hand rows
        prev_unit = None
        run_start = None
        for _, r in df.iterrows():
            hu = r["ratio"] > 50
            unit = "H" if hu else "g"
            if unit != prev_unit:
                if prev_unit is not None:
                    print(f"  [{prev_unit}] {run_start} .. {prev_date}  ({run_n} rows)", flush=True)
                run_start = str(r["date"].date())
                run_n = 1
            else:
                run_n += 1
            prev_unit = unit
            prev_date = str(r["date"].date())
        print(f"  [{prev_unit}] {run_start} .. {prev_date}  ({run_n} rows)", flush=True)


if __name__ == "__main__":
    main()
