# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Compare news_raw vs news_silver start/end for sample stocks; inspect a raw file."""
import pandas as pd
from pathlib import Path

R = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")

for code in ["000001", "600519", "000725", "601318"]:
    raw_p = R / "news_raw" / f"{code}.parquet"
    sil_p = R / "news_silver" / f"{code}.parquet"
    raw = pd.read_parquet(raw_p) if raw_p.exists() else None
    sil = pd.read_parquet(sil_p) if sil_p.exists() else None
    r = f"rows={len(raw)} {pd.to_datetime(raw['date']).min().date()}~{pd.to_datetime(raw['date']).max().date()}" if raw is not None else "MISSING"
    s = f"rows={len(sil)} {pd.to_datetime(sil['aligned_date']).min().date()}~{pd.to_datetime(sil['aligned_date']).max().date()}" if sil is not None else "MISSING"
    print(f"{code}: raw {r}")
    print(f"       sil {s}")
    if raw is not None:
        print(f"       raw cols={list(raw.columns)}")
        yr = raw["date"].astype(str).str[:4].value_counts().to_dict()
        print(f"       raw year dist: {dict(sorted(yr.items()))}")
