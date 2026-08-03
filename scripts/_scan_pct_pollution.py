"""Scan all a_shares parquet files for pct_change pollution since 2026-06-18.

A row is polluted if: date >= 2026-06-18, close present and non-null, and
close != prev close (real trading) but pct_change == 0. This distinguishes a
genuine zero move from suspended/halted days.

Cheap pass: read only date/close/pct_change columns (or schema first).
"""
import os
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

BASE = r"D:\Projects\Stoke_MachineLearning\data\a_shares"
CUTOFF = pd.Timestamp("2026-06-18")
# dirs we know are text/event raw or already repaired — skip heavy dirs
SKIP_DIRS = {"news_raw", "guba_raw", "comment_sentiment", "news_silver",
             "guba_silver", "sentiment", "minute", "minute_flat", "analyst",
             "guba_raw", "index_constituents", "index_constituents_hist",
             "universe"}


def scan_one(path, max_rows=None):
    try:
        cols = pq.read_schema(path).names
    except Exception:
        return None
    if "pct_change" not in cols:
        return None
    read_cols = [c for c in ("date", "close", "pct_change") if c in cols]
    try:
        df = pd.read_parquet(path, columns=read_cols)
    except Exception:
        return None
    if df.empty:
        return None
    if "date" not in df.columns:
        # some files use a DatetimeIndex
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df["date"].max() < CUTOFF:
        return None  # no data in polluted window
    recent = df[df["date"] >= CUTOFF].copy()
    if recent.empty:
        return None
    has_close = "close" in recent.columns
    polluted = 0
    n = 0
    prev = None
    for _, r in recent.iterrows():
        if has_close and pd.isna(r.get("close")):
            continue
        pc = r.get("pct_change")
        if pd.isna(pc):
            continue
        n += 1
        # real move: close differs from previous close (by >0.001%)
        if has_close and prev is not None and prev != 0:
            moved = abs(r["close"] / prev - 1) > 1e-5
        else:
            moved = True  # no close column — count pct_change==0 as polluted
        if pc == 0 and moved:
            polluted += 1
        prev = r["close"] if has_close else None
    if n == 0:
        return None
    return (n, polluted)


def main():
    by_dir = defaultdict(lambda: [0, 0, 0])  # files_scanned, files_with_pc, files_polluted
    detail = []
    for root, _dirs, files in os.walk(BASE):
        rel = os.path.relpath(root, BASE)
        top = rel.split(os.sep)[0]
        if top in SKIP_DIRS:
            continue
        for f in files:
            if not f.endswith(".parquet"):
                continue
            path = os.path.join(root, f)
            by_dir[top][0] += 1
            res = scan_one(path)
            if res is None:
                continue
            n, polluted = res
            by_dir[top][1] += 1
            if polluted > 0:
                by_dir[top][2] += 1
                if len(detail) < 15:
                    detail.append((path, n, polluted))
    print(f"{'dir':<32}{'scanned':>8}{'has_pc':>8}{'polluted':>9}")
    for d in sorted(by_dir, key=lambda x: -by_dir[x][2]):
        s, h, p = by_dir[d]
        print(f"{d:<32}{s:>8}{h:>8}{p:>9}")
    if detail:
        print("\nfirst polluted files:")
        for path, n, polluted in detail:
            print(f"  {os.path.relpath(path, BASE)}  (recent_rows={n}, polluted={polluted})")


if __name__ == "__main__":
    main()
