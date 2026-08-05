# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Probe: does rebuilt feature file pct_change match the clean flat?

Rebuilt features for 000001 at 22:31 (--force). Flat repaired at 22:02-22:12.
If feature pct_change still zero where flat is non-zero, feature build is
reading dirty data somewhere.
"""
import os
from datetime import datetime

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(PROJECT, "data", "features", "000001.parquet")
FLAT = os.path.join(PROJECT, "data", "a_shares", "daily", "000001.parquet")

print("feature file mtime:", datetime.fromtimestamp(os.path.getmtime(FEAT)))
print("flat file mtime:", datetime.fromtimestamp(os.path.getmtime(FLAT)))

feat = pd.read_parquet(FEAT)
flat = pd.read_parquet(FLAT)
feat["date"] = pd.to_datetime(feat["date"])
flat["date"] = pd.to_datetime(flat["date"])

for name, df in [("feature", feat), ("flat", flat)]:
    m = df["date"] >= "2026-06-18"
    print(f"\n{name} pct_change >= 2026-06-18 ({m.sum()} rows):")
    print(df.loc[m, ["date", "pct_change"]].tail(12).to_string())
    print(f"{name} n_zero_in_window: {(df.loc[m, 'pct_change'] == 0).sum()}")

# date-aligned comparison, last 40 rows from 2026-06-01
merged = feat[["date", "pct_change"]].rename(columns={"pct_change": "feat_pc"})
merged = merged.merge(flat[["date", "pct_change"]].rename(columns={"pct_change": "flat_pc"}),
                      on="date", how="inner")
merged = merged[merged["date"] >= "2026-06-01"].tail(40)
merged["diff"] = (merged["feat_pc"] - merged["flat_pc"]).abs()
print("\naligned feat vs flat (>= 2026-06-01, tail 40):")
print(merged.to_string())

# global alignment stats
full = feat[["date", "pct_change"]].rename(columns={"pct_change": "feat_pc"})
full = full.merge(flat[["date", "pct_change"]].rename(columns={"pct_change": "flat_pc"}),
                  on="date", how="inner")
full["diff"] = (full["feat_pc"] - full["flat_pc"]).abs()
print(f"\nglobal: n_aligned={len(full)}, exact_equal={(full['diff'] < 1e-6).mean():.4f}, "
      f"max_diff={full['diff'].max():.4f}")
# where does feature have 0 but flat non-zero?
bad = full[(full["feat_pc"] == 0) & (full["flat_pc"].abs() > 1e-6)]
print(f"feature-zero/flat-nonzero rows: {len(bad)}")
if len(bad):
    print(bad.head(10).to_string())
