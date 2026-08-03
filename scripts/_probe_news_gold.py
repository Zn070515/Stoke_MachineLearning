"""Compare sentiment flat vs partitioned history for sample stocks."""
import glob
from pathlib import Path

import pandas as pd

R = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")

for code in ["000001", "600519", "000725"]:
    for sub in ["news_silver", "news_raw", "sentiment"]:
        p = R / sub / f"{code}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            dc = "date" if "date" in df.columns else "aligned_date"
            s = pd.to_datetime(df[dc])
            print(f"{code} {sub}: {len(df)} rows {s.min().date()}~{s.max().date()}")
        else:
            print(f"{code} {sub}: MISSING")
    parts = sorted(glob.glob(str(R / "sentiment" / "*" / "*" / f"{code}.parquet")))
    print(f"{code} sentiment partitions: {len(parts)}")
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        s = pd.to_datetime(df["date"])
        print(f"   partition total {len(df)} rows {s.min().date()}~{s.max().date()}")
        print(f"   cols: {list(df.columns)}")
