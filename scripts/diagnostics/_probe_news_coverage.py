# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Full-market coverage stats for news_raw and sentiment partitions vs flat."""
import glob
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq

R = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")


def date_scan(globpat, datecol):
    starts, ends = Counter(), Counter()
    n_ok = 0
    for f in sorted(glob.glob(globpat)):
        try:
            t = pq.read_table(f, columns=[datecol], use_threads=True)
            col = t.column(0).to_pylist()
            if not col:
                continue
            s, e = min(col), max(col)
            starts[str(s)[:4]] += 1
            ends[str(e)[:4]] += 1
            n_ok += 1
        except Exception:
            pass
    return starts, ends, n_ok


print("### news_raw (per-stock start years)")
st, en, n = date_scan(str(R / "news_raw" / "*.parquet"), "date")
print(f"  {n} files start-year:", dict(sorted(st.items())))

print("### news_silver (per-stock start years)")
st, en, n = date_scan(str(R / "news_silver" / "*.parquet"), "aligned_date")
print(f"  {n} files start-year:", dict(sorted(st.items())))

print("### sentiment flat (per-stock start years)")
st, en, n = date_scan(str(R / "sentiment" / "*.parquet"), "date")
print(f"  {n} files start-year:", dict(sorted(st.items())))

# how many stocks have partitioned sentiment
part_codes = set()
for p in glob.glob(str(R / "sentiment" / "*" / "*" / "*.parquet")):
    part_codes.add(Path(p).stem)
print(f"\n### sentiment partitions cover {len(part_codes)} stocks")

# industry_ranking_computed structure
print("\n### industry/industry_ranking_computed.parquet")
df = __import__("pandas").read_parquet(R / "industry" / "industry_ranking_computed.parquet")
print(f"  rows={len(df)} cols={list(df.columns)}")
print(f"  industries: {df['industry'].nunique()}  date range: {df['date'].min()}~{df['date'].max()}")
print(f"  sample industries: {sorted(df['industry'].unique())[:10]}")
