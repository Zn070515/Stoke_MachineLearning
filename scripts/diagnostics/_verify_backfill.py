# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Verify margin + capital_flow backfill coverage.

Scans every per-stock flat parquet and reports:
  - file count, earliest-date distribution (floor attainment)
  - files below floor / no data
  - rows with NaN stock_code
  - duplicate (date) rows per file
"""
import argparse
import os
import sys

import pandas as pd

from stoke_ml.config import load_config

FLOORS = {
    "margin": pd.Timestamp("2012-01-01"),
    "capital_flow": pd.Timestamp("2010-03-01"),
}


def scan(data_type, floor):
    base = os.path.join(cfg.project.data_dir, "a_shares", data_type)
    if not os.path.isdir(base):
        print(f"{data_type}: dir missing")
        return
    files = sorted(p for p in os.listdir(base) if p.endswith(".parquet"))
    n = len(files)
    n_floor = n_nodata = n_nan = 0
    n_dup = []
    earliest = []
    for i, f in enumerate(files):
        path = os.path.join(base, f)
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  {data_type} {f}: READ ERR {e}")
            continue
        if df.empty:
            n_nodata += 1
            continue
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if d.empty:
            n_nodata += 1
            continue
        e = d.min()
        earliest.append((e, f))
        if e <= floor:
            n_floor += 1
        if "stock_code" in df.columns and df["stock_code"].isna().any():
            n_nan += 1
        dups = d.duplicated().sum()
        if dups:
            n_dup.append((f, dups))
        if (i + 1) % 1000 == 0:
            print(f"  {data_type}: scanned {i+1}/{n}")
    print(f"\n=== {data_type}: {n} files, {n_floor} reach floor ({floor.date()}), "
          f"{n_nodata} empty, {n_nan} NaN stock_code, {len(n_dup)} dup-date files ===")
    if earliest:
        earliest.sort()
        # bucket earliest dates by year
        import collections
        cnt = collections.Counter(d.year for d, _ in earliest)
        print(f"  earliest-date by year: {dict(sorted(cnt.items()))}")
        print(f"  top-10 earliest: {[(f, d.date()) for d, f in earliest[:10]]}")
        print(f"  bottom-5 latest: {[(f, d.date()) for d, f in earliest[-5:]]}")
    if n_dup:
        print(f"  dup-date files (up to 10): {n_dup[:10]}")


if __name__ == "__main__":
    global cfg
    cfg = load_config()
    for dt, floor in FLOORS.items():
        scan(dt, floor)
