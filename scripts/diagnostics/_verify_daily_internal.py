# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Verify daily flat pct_change == close.pct_change()*100 (internal consistency).

The feature rebuild restores pct_change from daily flat, so any residual
stale-zero pollution left in daily flat (e.g. a redownload that re-introduced
the fill-0 bug) would silently re-pollute rebuilt features. This checks every
daily flat file for the pollution signature: close changed (recomputed non-zero)
but pct_change == 0.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_daily_internal.py --sample 100
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_daily_internal.py  # all
"""
import argparse
import glob
import os
import sys

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(PROJECT, "data", "a_shares", "daily")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="check only first N daily files (0 = all)")
    parser.add_argument("--max-diff", type=float, default=0.01,
                        help="tolerated |pct_change - recomputed| (default 0.01)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    if args.sample:
        files = files[: args.sample]
    print(f"checking {len(files)} daily files (max_diff tol {args.max_diff})")

    bad = []
    max_diff_global = 0.0
    poll_total = 0
    for i, fp in enumerate(files):
        code = os.path.splitext(os.path.basename(fp))[0]
        try:
            d = pd.read_parquet(fp, columns=["date", "close", "pct_change"])
        except Exception as e:
            bad.append((code, f"read_err:{e}", 0, 0))
            continue
        if "pct_change" not in d.columns or "close" not in d.columns:
            bad.append((code, "missing_col", 0, 0))
            continue
        d["date"] = pd.to_datetime(d["date"])
        d = d.drop_duplicates("date", keep="last")
        close = d["close"].astype("float64")
        pc = d["pct_change"].astype("float64")
        recomputed = close.pct_change() * 100.0
        diff = (pc - recomputed).abs().dropna()
        md = float(diff.max()) if len(diff) else 0.0
        max_diff_global = max(max_diff_global, md)
        # Pollution signature: close moved but stored pct_change == 0.
        poll = int(((pc == 0.0) & (recomputed.abs() > 1e-4)).sum())
        poll_total += poll
        if md > args.max_diff or poll:
            bad.append((code, f"md={md:.4f}", int(len(diff)), poll))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} (max_diff {max_diff_global:.4f}, poll_total {poll_total})")

    print(f"\nglobal max |pct_change - close.pct_change()*100|: {max_diff_global:.6f}")
    print(f"total pollution rows (close moved, pct_change==0): {poll_total}")
    print(f"problem files: {len(bad)}")
    for code, why, n, poll in bad[:40]:
        print(f"  {code}: {why} (n={n}, poll={poll})")
    if not bad:
        print("ALL CLEAN")


if __name__ == "__main__":
    sys.exit(main())
