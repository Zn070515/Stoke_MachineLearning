"""Definitive per-file unit classifier + global gap scan.

Per file:
  - ratio = amount/volume/close per row (vol/amt/close>0, finite)
  - per-year median ratio (>=5 rows/year)
  - unit switches detected via year-median jumps >= SWITCH (15x):
      ratio[y]/ratio[y-1] >= 15  -> gu->hand
      ratio[y]/ratio[y-1] <= 1/15 -> hand->gu
  - base unit for uniform files (no switch): median ratio of the LAST
    year with >=5 rows.  All-gu files have it ~1; all-hand files ~100.
    A value in the ambiguous band [40, 60] is flagged for review.
  - every row inherits its year's unit.

Outputs to reports/daily_unit_definitive.json:
  gu_max_global / gu_max_file   (max ratio of rows in gu-classified years,
                                 within sanity band [0.2, 200])
  hu_min_global / hu_min_file   (min ratio of rows in hand-classified years,
                                 within sanity band [0.2, 20000])
  gap, counts, ambiguous files list.

Read-only.  Writes only the report json.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

DAILY = "data/a_shares/daily"
SWITCH = 15.0
GU, HU = 1, 100
STATS_BAND = (0.3, 3000.0)   # ratios outside = corrupt rows, excluded from stats
STRAY = 30.0                 # row vs year-median beyond 30x = the other unit
AMBIGUOUS_BAND = (40.0, 60.0)


def classify_file(df: pd.DataFrame) -> dict:
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    if not ok.any():
        return {"n_valid": 0, "row_unit": np.array([], dtype=float), "ratio": np.array([]),
                "flags": [], "years": [], "meds": []}
    ratio = amt[ok] / vol[ok] / close[ok]
    years = pd.to_datetime(df.loc[ok, "date"]).dt.year.to_numpy()
    s = pd.Series(ratio, index=years)
    med = s.groupby(level=0).median()
    cnt = s.groupby(level=0).count()
    years_all = med.index.to_numpy()
    use = cnt.to_numpy() >= 5
    years_u = years_all[use]
    med_u = med.to_numpy()[use]
    flags = []
    if len(med_u) == 0:
        return {"n_valid": int(ok.sum()), "row_unit": np.array([], dtype=float),
                "ratio": ratio, "flags": ["no_year_with_5rows"], "years": years_u.tolist(),
                "meds": med_u.tolist()}

    # backward pass: base unit from last-year median (qfq~1 at latest date:
    # all-gu files have last-year med ~1, all-hand files ~100), then walk
    # backward flipping on jumps.  r = newer/older:
    #   r >= SWITCH   => newer is hand, older is gu  => older=GU
    #   r <= 1/SWITCH => newer is gu,    older is hand => older=HU
    n = len(med_u)
    unit = np.full(n, -1)
    last_med = med_u[-1]
    if last_med < AMBIGUOUS_BAND[0]:
        base = GU
    elif last_med > AMBIGUOUS_BAND[1]:
        base = HU
    else:
        base = GU
        flags.append(f"last_year_median_in_ambiguous_band:{last_med:.2f}")
    unit[-1] = base
    for i in range(n - 2, -1, -1):
        r = med_u[i + 1] / med_u[i] if med_u[i] > 0 else np.inf
        if r >= SWITCH:          # older is gu
            unit[i] = GU
        elif r <= 1.0 / SWITCH:  # older is hand
            unit[i] = HU
        else:
            unit[i] = unit[i + 1]
    # sanity: any 3+ consecutive flips is suspicious
    flips = sum(1 for i in range(n - 1) if unit[i] != unit[i + 1])
    if flips >= 3:
        flags.append(f"many_unit_flips:{flips}")

    year_unit = dict(zip(years_u.tolist(), unit.tolist()))
    row_unit = np.array([year_unit.get(y, -1) for y in years], dtype=float)
    # row-level stray rule: within each year, a row deviating >30x from the
    # year median is the other unit (gu<->hand).  A hand stray in a gu year is
    # ~100x the median; a gu stray in a hand year is ~1/100x.
    year_med_map = dict(zip(years_u.tolist(), med_u.tolist()))
    n_flip = 0
    for i in range(len(ratio)):
        ym = year_med_map.get(years[i], None)
        if ym is None or row_unit[i] < 0:
            continue
        if row_unit[i] == GU and ratio[i] > STRAY * ym:
            row_unit[i] = HU
            n_flip += 1
        elif row_unit[i] == HU and ratio[i] < ym / STRAY:
            row_unit[i] = GU
            n_flip += 1
    if n_flip:
        flags.append(f"row_stray_flips:{n_flip}")
    return {"n_valid": int(ok.sum()), "row_unit": row_unit, "ratio": ratio,
            "flags": flags, "years": years_u.tolist(), "meds": med_u.tolist()}


def main() -> None:
    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    print(f"scanning {len(files)} daily files", flush=True)
    t0 = time.time()
    gu_max_global = 0.0
    hu_min_global = np.inf
    gu_max_file = None
    hu_min_file = None
    n_gu = 0
    n_hu = 0
    n_unlabeled = 0
    ambiguous = []
    for i, path in enumerate(files):
        df = pd.read_parquet(path)
        res = classify_file(df)
        fname = os.path.basename(path)
        row_unit = res["row_unit"]
        ratio = res["ratio"]
        if len(row_unit) and len(ratio) and len(row_unit) == len(ratio):
            ok_u = (row_unit == GU) & (ratio > STATS_BAND[0]) & (ratio < STATS_BAND[1])
            ok_h = (row_unit == HU) & (ratio > STATS_BAND[0]) & (ratio < STATS_BAND[1])
            if ok_u.any():
                m = float(ratio[ok_u].max())
                if m > gu_max_global:
                    gu_max_global = m
                    gu_max_file = fname
                n_gu += int(ok_u.sum())
            if ok_h.any():
                m = float(ratio[ok_h].min())
                if m < hu_min_global:
                    hu_min_global = m
                    hu_min_file = fname
                n_hu += int(ok_h.sum())
            n_unlabeled += int((row_unit < 0).sum())
        if res["flags"]:
            ambiguous.append({"file": fname, "flags": res["flags"],
                              "last_year_med": res["meds"][-1] if res["meds"] else None,
                              "n_gu": int((row_unit == GU).sum()) if len(row_unit) else 0,
                              "n_hu": int((row_unit == HU).sum()) if len(row_unit) else 0})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "n_files": len(files),
        "gu_max_global": gu_max_global,
        "gu_max_file": gu_max_file,
        "hu_min_global": None if np.isinf(hu_min_global) else float(hu_min_global),
        "hu_min_file": hu_min_file,
        "gap": None if np.isinf(hu_min_global) or not gu_max_global else float(hu_min_global / gu_max_global),
        "n_gu_rows": n_gu,
        "n_hu_rows": n_hu,
        "n_unlabeled": n_unlabeled,
        "n_ambiguous_files": len(ambiguous),
        "ambiguous": ambiguous,
    }
    with open("reports/daily_unit_definitive.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "ambiguous"}, indent=2, ensure_ascii=False), flush=True)
    print(f"ambiguous files: {len(ambiguous)}", flush=True)
    for a in ambiguous:
        print(f"  {a}", flush=True)


if __name__ == "__main__":
    main()
