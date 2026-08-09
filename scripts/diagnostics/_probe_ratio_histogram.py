"""Unbiased global scan of the daily-store unit structure.

No fixed threshold.  For each file:
  1. compute ratio = amount/volume/close per row (vol/amt/close>0, finite)
  2. group by year, median ratio per year (>=5 rows)
  3. detect unit switches via year-median jumps >= SWITCH (15x): a switch
     changes the ratio by ~100x (hand segment vs gu segment at adjacent
     dates where qfq is smooth)
  4. classify each year as gu/hand, update global stats:
       gu_max_global  = max ratio of any year classified gu
       hu_min_global  = min ratio of any year classified hand
     plus a log-space histogram of ALL ratios (for gap visualization)
  5. uniform files (no switch): classify by whether the median ratio is
     closer to the gu median (~1-2) or the hand median (~100+); record
     both their gu_max (if classified gu) and hu_min (if classified hand).

Read-only.  Writes nothing.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"
SWITCH = 15.0          # year-median ratio jump (either direction) = unit switch
GU_LABEL = 1           # volume in shares
HU_LABEL = 100         # volume in lots


def classify_year_medians(med: np.ndarray) -> np.ndarray:
    """med: array of year-median ratios, sorted by year. Returns unit per year."""
    n = len(med)
    unit = np.full(n, -1)
    # forward pass: propagate label, flip at jumps >= SWITCH
    cur = GU_LABEL if med[0] < 50 else HU_LABEL
    unit[0] = cur
    for i in range(1, n):
        r = med[i] / med[i - 1] if med[i - 1] > 0 else np.inf
        if r >= SWITCH:
            cur = HU_LABEL
        elif r <= 1.0 / SWITCH:
            cur = GU_LABEL
        unit[i] = cur
    return unit


def scan_file(path: str) -> dict:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"file": os.path.basename(path), "error": str(exc)}
    if not {"volume", "amount", "close", "date"}.issubset(df.columns):
        return {"file": os.path.basename(path), "error": "missing cols"}
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    if not ok.any():
        return {"file": os.path.basename(path), "n_valid": 0}
    ratio = amt[ok] / vol[ok] / close[ok]
    dates = pd.to_datetime(df.loc[ok, "date"])
    years = dates.dt.year.to_numpy()
    # per-year medians
    s = pd.Series(ratio, index=years)
    med = s.groupby(level=0).median()
    cnt = s.groupby(level=0).count()
    years_all = med.index.to_numpy()
    med_vals = med.to_numpy()
    use = cnt.to_numpy() >= 5
    years_u = years_all[use]
    med_u = med_vals[use]
    if len(med_u) == 0:
        # no year has >=5 rows; classify the whole file by its overall median
        all_unit = GU_LABEL if np.median(ratio) < 50 else HU_LABEL
        row_unit = np.full(len(ratio), all_unit, dtype=float)
        return {
            "file": os.path.basename(path),
            "n_valid": int(ok.sum()),
            "gu_max": float(ratio[row_unit == GU_LABEL].max()) if (row_unit == GU_LABEL).any() else None,
            "hu_min": float(ratio[row_unit == HU_LABEL].min()) if (row_unit == HU_LABEL).any() else None,
            "n_gu": int((row_unit == GU_LABEL).sum()),
            "n_hu": int((row_unit == HU_LABEL).sum()),
            "n_unlabeled": int((row_unit < 0).sum()),
            "ratio_list": ratio.tolist(),
        }
    unit = classify_year_medians(med_u)
    year_unit = dict(zip(years_u, unit))
    # row-level label
    row_unit = np.array([year_unit.get(y, -1) for y in years], dtype=float)
    gu = row_unit == GU_LABEL
    hu = row_unit == HU_LABEL
    return {
        "file": os.path.basename(path),
        "n_valid": int(ok.sum()),
        "gu_max": float(ratio[gu].max()) if gu.any() else None,
        "hu_min": float(ratio[hu].min()) if hu.any() else None,
        "n_gu": int(gu.sum()),
        "n_hu": int(hu.sum()),
        "n_unlabeled": int((row_unit < 0).sum()),
        "ratio_list": ratio.tolist(),
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    print(f"scanning {len(files)} daily files", flush=True)
    t0 = time.time()
    rows = []
    gu_max_global = 0.0
    hu_min_global = np.inf
    gu_max_file = None
    hu_min_file = None
    n_gu_rows = 0
    n_hu_rows = 0
    n_unlabeled = 0
    # log10 histogram of all ratios
    bins = np.arange(-1.0, 4.2, 0.05)  # ratio in [0.1, 15850]
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    for i, path in enumerate(files):
        r = scan_file(path)
        rows.append(r)
        if r.get("gu_max") is not None:
            if r["gu_max"] > gu_max_global:
                gu_max_global = r["gu_max"]
                gu_max_file = r["file"]
            n_gu_rows += r["n_gu"]
        if r.get("hu_min") is not None and r["hu_min"] < hu_min_global:
            hu_min_global = r["hu_min"]
            hu_min_file = r["file"]
            n_hu_rows += r["n_hu"]
        n_unlabeled += r.get("n_unlabeled", 0)
        vals = np.array(r["ratio_list"])
        hist += np.histogram(np.log10(vals), bins=bins)[0]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "n_files": len(files),
        "gu_max_global": gu_max_global,
        "gu_max_file": gu_max_file,
        "hu_min_global": None if np.isinf(hu_min_global) else float(hu_min_global),
        "hu_min_file": hu_min_file,
        "n_gu_rows": n_gu_rows,
        "n_hu_rows": n_hu_rows,
        "n_unlabeled": n_unlabeled,
    }
    if not np.isinf(hu_min_global) and gu_max_global:
        out["gap"] = float(hu_min_global / gu_max_global)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)

    # print histogram (log10 ratio bins), showing the gap location
    centers = 10 ** ((bins[:-1] + bins[1:]) / 2)
    print("\nratio histogram (log10 bins):", flush=True)
    for c, h in zip(centers, hist):
        if h > 0:
            bar = "#" * min(60, int(h / max(1, hist.max() / 60)))
            print(f"  {c:12.3f}  {h:10d}  {bar}", flush=True)


if __name__ == "__main__":
    main()
