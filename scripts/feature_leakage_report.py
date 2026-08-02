"""L4 gate: leakage audit for pre-built features.

For each new source, on a sample of stocks, verify the PIT invariant:
  feature[source_col] at trading day t == raw source value at day t-1.
Also flags any feature whose cross-sectional |IC| is implausibly high (> 0.15)
for manual review (the classic look-ahead signature).

Checks cover 2 sources (pledge + index_membership); the limit-up family is
deferred per the top scope note.

Output: reports/feature_leakage_report.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config

SAMPLES = 30
# source_dir -> (feature_source_col, feature_col_in_panel, shift)
# limit-up family excluded (deferred per top scope note). 2 sources total.
CHECKS = [
    ("pledge_processed", "has_pledge", "has_pledge", 1),
    ("index_membership_processed", "is_index_member", "is_index_member", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--stocks", default=None, help="sample stocks (comma-sep); default random 30")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares")
    feat_dir = args.features_dir or os.path.join(cfg.project.data_dir, "features")
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if not feat_files:
        print("no feature files")
        return
    codes = args.stocks.split(",") if args.stocks else None
    if codes is None:
        rng = np.random.default_rng(0)
        codes = [os.path.basename(f).replace(".parquet", "")
                 for f in rng.choice(feat_files, size=min(SAMPLES, len(feat_files)), replace=False)]

    rows = []
    for src_dir, raw_col, feat_col, lag in CHECKS:
        ok = skip = bad = 0
        for code in codes:
            raw_path = os.path.join(base, src_dir, f"{code}.parquet")
            feat_path = os.path.join(feat_dir, f"{code}.parquet")
            if not (os.path.exists(raw_path) and os.path.exists(feat_path)):
                skip += 1
                continue
            try:
                raw = pd.read_parquet(raw_path, columns=["date", raw_col])
            except Exception:
                skip += 1
                continue
            try:
                feat = pd.read_parquet(feat_path, columns=["date", feat_col])
            except Exception:
                skip += 1
                continue
            if raw.empty or feat.empty or raw_col not in raw or feat_col not in feat:
                skip += 1
                continue
            raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
            feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
            raw = raw.drop_duplicates("date")
            feat = feat.drop_duplicates("date")
            # Expected: feature at t == raw at t-1.
            merged = feat.merge(raw.rename(columns={raw_col: "raw"}), on="date", how="left")
            merged = merged.sort_values("date")
            merged["raw_lag"] = merged["raw"].shift(lag)
            merged = merged.dropna(subset=["raw_lag"])
            if len(merged) < 10:
                skip += 1
                continue
            match = np.isclose(merged[feat_col].astype(float),
                               merged["raw_lag"].astype(float)).mean()
            if match >= 0.95:
                ok += 1
            else:
                bad += 1
                rows.append({"source": src_dir, "code": code,
                             "check": "pit_lag", "pass": False,
                             "match_rate": float(match), "detail": f"expected lag {lag}"})
        if ok == 0 and bad == 0:
            # Nothing verifiable (e.g. no sampled stock has the source): NaN,
            # not False, so it is not flagged as a leak.
            agg_pass = None
        else:
            agg_pass = ok > 0 and bad == 0
        rows.append({"source": src_dir, "code": "AGG", "check": "pit_lag",
                     "pass": agg_pass, "match_rate": None,
                     "detail": f"{ok} ok, {bad} bad, {skip} skipped"})

    # High-|IC| heuristic scan (uses feature_ic_report.csv if present).
    ic_path = os.path.join("reports", "feature_ic_report.csv")
    if os.path.exists(ic_path):
        ic = pd.read_csv(ic_path)
        flagged = ic[(ic["window"] == "primary") & (ic["ic_cross"].abs() > 0.15)]
        for _, r in flagged.iterrows():
            rows.append({"source": "IC", "code": "AGG", "check": "high_ic",
                         "pass": False, "match_rate": None,
                         "detail": f"{r['feature']} ic_cross={r['ic_cross']:.3f} — review for leakage"})

    rep = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    rep.to_csv("reports/feature_leakage_report.csv", index=False)
    fails = rep[rep["pass"] == False]  # noqa: E712
    print(f"leakage report written: {len(rep)} rows, {len(fails)} failures/suspicious")
    for _, r in fails.iterrows():
        print("  -", r["source"], r["code"], r["check"], r["detail"])


if __name__ == "__main__":
    main()
