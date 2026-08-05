# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Check existing capital_flow tier data + margin 2012 depth."""
from pathlib import Path

import akshare as ak
import pandas as pd

R = Path(r"D:\Projects\Stoke_MachineLearning\data\a_shares")

print("=== existing capital_flow raw tiers (000001) ===")
df = pd.read_parquet(R / "capital_flow" / "000001.parquet")
print(f"rows={len(df)} range={df['date'].min()}~{df['date'].max()}")
for c in ["main_net", "super_net", "large_net", "mid_net", "small_net"]:
    nnz = (df[c].abs() > 1).sum()
    print(f"  {c}: nonzero={nnz} ({nnz/len(df):.1%})  sample={df[c].iloc[-1]:.0f}")

print("\n=== margin SSE 2012 ===")
try:
    sse = ak.stock_margin_detail_sse(date="20120104")
    print(f"SSE 20120104: {len(sse)} rows")
except Exception as e:
    print(f"SSE 20120104: ERR {e}")

print("\n=== margin SZSE 2018 ===")
try:
    sz = ak.stock_margin_detail_szse(date="20180102")
    print(f"SZSE 20180102: {len(sz)} rows cols={list(sz.columns)[:4]}")
except Exception as e:
    print(f"SZSE 20180102: ERR {e}")
