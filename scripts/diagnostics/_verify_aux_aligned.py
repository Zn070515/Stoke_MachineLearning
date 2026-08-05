# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Verify board_processed / industry_ranking_processed pct_change aligns to daily.

The repair (#353) rewrote aux pct_change to the canonical daily flat values,
date-aligned. The scanner's "polluted" hits on these files are false positives:
the aux close column is on an old adjustment basis while pct_change now matches
daily, so a self-close-based move check misfires. This verifier uses daily as the
authority: every aux pct_change row must equal the daily-aligned pct_change.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_aux_aligned.py --sample 100
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_aux_aligned.py  # all
"""
import argparse
import glob
import os
import sys

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_SHARES = os.path.join(PROJECT, "data", "a_shares")
AUX_DIRS = ["board_processed", "industry_ranking_processed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="check only first N files (0 = all)")
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    files = []
    for d in AUX_DIRS:
        files.extend(glob.glob(os.path.join(A_SHARES, d, "*.parquet")))
    files.sort()
    if args.sample:
        files = files[: args.sample]
    print(f"checking {len(files)} aux files (max_diff tol {args.max_diff})")

    bad = []
    max_diff_global = 0.0
    n_rows = 0
    for i, fp in enumerate(files):
        d = os.path.basename(os.path.dirname(fp))
        code = os.path.splitext(os.path.basename(fp))[0]
        daily_p = os.path.join(A_SHARES, "daily", f"{code}.parquet")
        if not os.path.exists(daily_p):
            bad.append((d, code, "no_daily", 0, 0))
            continue
        try:
            a = pd.read_parquet(fp, columns=["date", "pct_change"])
            dly = pd.read_parquet(daily_p, columns=["date", "pct_change"])
        except Exception as e:
            bad.append((d, code, f"read_err:{e}", 0, 0))
            continue
        if "pct_change" not in a.columns:
            continue
        a["date"] = pd.to_datetime(a["date"])
        dly["date"] = pd.to_datetime(dly["date"])
        dly = dly.drop_duplicates("date", keep="last")
        dly["pct_change"] = dly["pct_change"].fillna(0.0).astype("float64")
        aligned = a["date"].map(dly.set_index("date")["pct_change"])
        diff = (a["pct_change"].astype("float64") - aligned).abs().dropna()
        md = float(diff.max()) if len(diff) else 0.0
        n_rows += int(len(diff))
        max_diff_global = max(max_diff_global, md)
        if md > args.max_diff:
            bad.append((d, code, f"md={md:.6f}", int(len(diff)), int((diff > args.max_diff).sum())))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} ({n_rows} rows, max_diff {max_diff_global:.8f})")

    print(f"\nchecked {len(files)} files, {n_rows} rows")
    print(f"global max |aux_pc - daily_pc|: {max_diff_global:.8f}")
    print(f"mismatch files: {len(bad)}")
    for d, code, why, n, m in bad[:40]:
        print(f"  {d}/{code}: {why} (rows={n}, mismatched={m})")
    if not bad:
        print("ALL CLEAN")


if __name__ == "__main__":
    sys.exit(main())
