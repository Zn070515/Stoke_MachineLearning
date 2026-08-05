# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Verify feature pct_change == daily flat pct_change (no pollution).

After the pipeline fix (pct_change restored from K-line, aux pct_change
excluded) every rebuilt feature should carry the same-day daily return.
This checks: for each feature file, date-aligned |feat_pc - daily_pc| and the
pollution signature (feature 0 where daily non-zero) in the 06-18+ window.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_feature_pct.py --sample 100
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_feature_pct.py  # all
"""
import argparse
import glob
import os
import sys

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(PROJECT, "data", "features")
DAILY = os.path.join(PROJECT, "data", "a_shares", "daily")
CUTOFF = pd.Timestamp("2026-06-18")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="check only first N feature files (0 = all)")
    args = parser.parse_args()

    feats = sorted(glob.glob(os.path.join(FEAT, "*.parquet")))
    if args.sample:
        feats = feats[: args.sample]
    print(f"checking {len(feats)} feature files")

    bad = []
    max_diff = 0.0
    for i, fp in enumerate(feats):
        code = os.path.splitext(os.path.basename(fp))[0]
        dp = os.path.join(DAILY, f"{code}.parquet")
        if not os.path.exists(dp):
            bad.append((code, "no_daily", 0, 0))
            continue
        try:
            f = pd.read_parquet(fp, columns=["date", "pct_change"])
            d = pd.read_parquet(dp, columns=["date", "pct_change"])
        except Exception as e:
            bad.append((code, f"read_err:{e}", 0, 0))
            continue
        f["date"] = pd.to_datetime(f["date"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.drop_duplicates("date").set_index("date")["pct_change"].fillna(0.0)
        if "pct_change" not in f.columns:
            bad.append((code, "no_feat_pc", 0, 0))
            continue
        fp_v = f.set_index("date")["pct_change"].astype("float64")
        m = pd.concat([fp_v, d], axis=1, keys=["feat", "daily"]).dropna()
        if m.empty:
            continue
        diff = (m["feat"] - m["daily"]).abs()
        md = diff.max()
        max_diff = max(max_diff, md)
        post = m.index >= CUTOFF
        if post.any():
            poll = int(((m["feat"] == 0) & (m["daily"].abs() > 1e-4))[post].sum())
            if poll or md > 0.5:
                bad.append((code, f"md={md:.4f}", int(post.sum()), poll))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(feats)} (max_diff so far {max_diff:.4f})")

    print(f"\nglobal max |feat-daily| diff: {max_diff:.6f}")
    print(f"problem files: {len(bad)}")
    for code, why, post, poll in bad[:30]:
        print(f"  {code}: {why} (post={post}, poll_zero={poll})")
    if not bad:
        print("ALL CLEAN")


if __name__ == "__main__":
    main()
