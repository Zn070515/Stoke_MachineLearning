# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Quantify per-stock time coverage by reading only the date column."""
import glob, sys, time
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

ROOT = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")

def coverage(globpat, datecol, label):
    files = sorted(glob.glob(str(ROOT / globpat)))
    starts, ends = Counter(), Counter()
    n_ok = 0
    t0 = time.time()
    for f in files:
        try:
            t = pq.read_table(f, columns=[datecol], use_threads=True)
            col = t.column(0).to_pylist()
            if not col:
                continue
            s = min(col); e = max(col)
            starts[str(s)[:4]] += 1
            ends[str(e)[:4]] += 1
            n_ok += 1
        except Exception:
            pass
    print(f"### {label}  ({len(files)} files, {n_ok} ok)  {time.time()-t0:.0f}s")
    print("  start-year:", dict(sorted(starts.items())))
    print("  end-year:  ", dict(sorted(ends.items())), flush=True)

coverage("daily/*.parquet", "date", "daily K-line")
coverage("sentiment/*.parquet", "date", "news sentiment (Gold)")
coverage("guba_sentiment/*.parquet", "date", "guba sentiment (Gold)")
coverage("fundamentals_daily/*.parquet", "date", "fundamentals_daily")
coverage("valuation/*.parquet", "date", "valuation")
