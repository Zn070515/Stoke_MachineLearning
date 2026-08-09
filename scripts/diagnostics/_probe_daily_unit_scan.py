"""Full-universe scan of the daily store's implied unit ratio.

ratio = amount / volume / close for every row with volume>0, amount>0,
close>0, all finite.  A CORRECT row (volume in shares) has ratio ≈ the qfq
factor (≈1 for recent dates, growing for early dates of dividend payers).
A 手 row (volume in lots) has ratio ≈ 100× that.  This probe reports, per
file, the two clusters' bounds and the global gap, so the migration's
classification threshold is chosen from evidence, not a guess.

Read-only.  Writes nothing.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"
CLUSTER_HAND_MIN = 50.0  # a ratio above this is a 手 row, whatever the qfq factor


def scan_file(path: str) -> dict:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"file": os.path.basename(path), "error": str(exc)}
    if not {"volume", "amount", "close"}.issubset(df.columns):
        return {"file": os.path.basename(path), "error": "missing cols"}
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    if not ok.any():
        return {"file": os.path.basename(path), "n_valid": 0}
    ratio = amt[ok] / vol[ok] / close[ok]
    gu = ratio < CLUSTER_HAND_MIN   # 股 candidate
    hu = ratio >= CLUSTER_HAND_MIN  # 手 candidate
    out = {
        "file": os.path.basename(path),
        "n_valid": int(ok.sum()),
        "n_hand_cand": int(hu.sum()),
        "gu_med": float(np.median(ratio[gu])) if gu.any() else None,
        "gu_max": float(ratio[gu].max()) if gu.any() else None,
        "hu_min": float(ratio[hu].min()) if hu.any() else None,
        "hu_max": float(ratio[hu].max()) if hu.any() else None,
    }
    return out


def main() -> None:
    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    print(f"scanning {len(files)} daily files", flush=True)
    t0 = time.time()
    rows = []
    gu_max_global = 0.0
    hu_min_global = np.inf
    gu_max_file = None
    for i, path in enumerate(files):
        r = scan_file(path)
        rows.append(r)
        if r.get("gu_max") is not None:
            if r["gu_max"] > gu_max_global:
                gu_max_global = r["gu_max"]
                gu_max_file = r["file"]
        if r.get("hu_min") is not None and r["hu_min"] < hu_min_global:
            hu_min_global = r["hu_min"]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "n_files": len(files),
        "gu_max_global": gu_max_global,
        "gu_max_file": gu_max_file,
        "hu_min_global": None if np.isinf(hu_min_global) else float(hu_min_global),
        "gap": None,
    }
    if not np.isinf(hu_min_global) and gu_max_global:
        out["gap"] = float(hu_min_global / gu_max_global)
    with open("reports/daily_unit_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # files with BOTH clusters present (the ones needing a per-file split)
    both = [r for r in rows if r.get("gu_max") and r.get("hu_min")]
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"files with BOTH clusters: {len(both)}", flush=True)
    if both:
        print("sample both-cluster files:", flush=True)
        for r in both[:15]:
            print(f"  {r}", flush=True)


if __name__ == "__main__":
    main()
