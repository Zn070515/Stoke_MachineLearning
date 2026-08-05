# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Probe all datasets under data/a_shares: file counts, sample columns, shape."""
import glob, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")

def probe_dir(name, pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        print(f"\n== {name} ==  NO FILES ({pattern})")
        return
    sample = files[0] if len(files) < 6 else files[len(files)//2]
    try:
        df = pd.read_parquet(sample)
        cols = list(df.columns)
        n = len(df)
        # try to find date column
        datecol = None
        for c in ("date","datetime","report_date","trade_date","time"):
            if c in cols:
                datecol = c
                break
        tspan = ""
        if datecol:
            try:
                s = pd.to_datetime(df[datecol])
                tspan = f" {s.min().date()}~{s.max().date()}"
            except Exception:
                pass
        print(f"\n== {name} ==  {len(files)} files  |  sample: {Path(sample).name}  rows={n}{tspan}")
        print(f"   cols({len(cols)}): {cols}")
    except Exception as e:
        print(f"\n== {name} ==  {len(files)} files  |  UNREADABLE {Path(sample).name}: {e}")

# flat per-stock datasets
for d in sorted(os.listdir(ROOT)):
    p = ROOT / d
    if not p.is_dir():
        continue
    files = sorted(glob.glob(str(p / "*.parquet")))
    if files:
        probe_dir(d, f"{d}/*.parquet")
    else:
        # maybe partitioned
        parts = sorted(glob.glob(str(p / "**/*.parquet"), recursive=True))
        if parts:
            sample = parts[len(parts)//2]
            try:
                df = pd.read_parquet(sample)
                cols = list(df.columns)
                print(f"\n== {d} ==  {len(parts)} partitioned files  |  sample: {Path(sample).relative_to(ROOT)}  rows={len(df)}")
                print(f"   cols({len(cols)}): {cols}")
            except Exception as e:
                print(f"\n== {d} ==  {len(parts)} partitioned  |  UNREADABLE: {e}")
        else:
            # maybe csv/txt
            others = list(p.iterdir())[:5]
            print(f"\n== {d} ==  no parquet; entries: {[o.name for o in others]}")
