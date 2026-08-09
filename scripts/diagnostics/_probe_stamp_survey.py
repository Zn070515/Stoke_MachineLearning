"""Survey the daily store for the Phase-2 qfq provenance stamp (#85).

Read-only.  For every unknown-adjust file, run the stamp safety gate:
  1. classify_file -> flips == 0 AND junk == 0   (unit-clean, no corruption)
  2. qfq latest-year band: median(amount/volume/close) in [0.3, 3] for the
     most recent year (a per-share current price must have VWAP ~ close)
  3. qfq monotonicity: no OLDER year whose median ratio is below the NEXT
     newer year's median by more than 2x (qfq factors only push old ratios
     UP; an older year lower than a newer one contradicts qfq)

Files failing any check are SKIPPED (keep adjust=unknown) + reported.
Files already qfq are skipped as already-declared.
Writes reports/stamp_survey.json only.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from scripts.maintenance.current.migrate_daily_units import classify_file

DAILY = os.path.join("data", "a_shares", "daily")
LATEST_BAND = (0.3, 3.0)
MONO_TOL = 2.0  # older-year med must be >= newer-year med / MONO_TOL


def qfq_signature(df: pd.DataFrame) -> dict:
    """Per-year median ratio + latest-year band + monotonicity violation."""
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    if not ok.any():
        return {"n_valid": 0, "latest_med": None, "mono_violations": 0}
    ratio = amt[ok] / vol[ok] / close[ok]
    years = pd.to_datetime(df.loc[ok, "date"]).dt.year.to_numpy()
    s = pd.Series(ratio, index=years)
    med = s.groupby(level=0).median()
    cnt = s.groupby(level=0).count()
    use = cnt.to_numpy() >= 5
    years_u = med.index[use].to_numpy()
    med_u = med.to_numpy()[use]
    if len(med_u) == 0:
        latest_med = float(np.median(ratio))
        return {"n_valid": int(ok.sum()), "latest_med": latest_med,
                "mono_violations": 0}
    latest_med = float(med_u[-1])
    mono = 0
    for i in range(len(med_u) - 1):
        if med_u[i] < med_u[i + 1] / MONO_TOL:
            mono += 1
    return {"n_valid": int(ok.sum()), "latest_med": latest_med,
            "mono_violations": mono}


def main() -> None:
    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    t0 = time.time()
    n_qfq = 0
    n_unknown = 0
    n_pass = 0
    n_skip = 0
    skips: dict[str, list[str]] = {}  # code -> reasons
    per_file: dict[str, dict] = {}
    for i, path in enumerate(files):
        code = os.path.basename(path)[: -len(".parquet")]
        mf = os.path.join(DAILY, f"{code}.manifest.json")
        m = json.load(open(mf, encoding="utf-8")) if os.path.isfile(mf) else {}
        adjust = m.get("adjust", "<missing>")
        if adjust == "qfq":
            n_qfq += 1
            per_file[code] = {"adjust": "qfq", "gate": "skip", "reason": "already_qfq"}
            continue
        n_unknown += 1
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        res = classify_file(df)
        flips = int(res["flip"].sum())
        junks = int(res["junk"].sum())
        reasons = []
        if flips or junks:
            reasons.append(f"unit_flips={flips}_junk={junks}")
        sig = qfq_signature(df)
        if reasons and (sig["latest_med"] is not None
                        and not (LATEST_BAND[0] <= sig["latest_med"] <= LATEST_BAND[1])):
            reasons.append(f"latest_med={sig['latest_med']:.2f}")
        if sig["mono_violations"]:
            reasons.append(f"mono_violations={sig['mono_violations']}")
        if not reasons and sig["latest_med"] is not None \
                and not (LATEST_BAND[0] <= sig["latest_med"] <= LATEST_BAND[1]):
            reasons.append(f"latest_med_out_of_band={sig['latest_med']:.2f}")
        if reasons:
            n_skip += 1
            skips.setdefault(code, reasons)
            per_file[code] = {"adjust": "unknown", "gate": "skip",
                              "reasons": reasons, "flips": flips, "junk": junks,
                              "latest_med": sig["latest_med"],
                              "mono_violations": sig["mono_violations"]}
        else:
            n_pass += 1
            per_file[code] = {"adjust": "unknown", "gate": "pass",
                              "flips": flips, "junk": junks,
                              "latest_med": sig["latest_med"],
                              "mono_violations": sig["mono_violations"]}
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} pass={n_pass} skip={n_skip} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "n_files": len(files),
        "n_qfq_already": n_qfq,
        "n_unknown": n_unknown,
        "n_pass_gate": n_pass,
        "n_skip": n_skip,
        "skips": skips,
        "per_file": per_file,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/stamp_survey.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nn_files={len(files)} qfq={n_qfq} unknown={n_unknown} "
          f"pass={n_pass} skip={n_skip}  ({time.time()-t0:.0f}s)", flush=True)
    print("skip reasons:", flush=True)
    reason_cnt: dict[str, int] = {}
    for code, rs in skips.items():
        for r in rs:
            reason_cnt[r] = reason_cnt.get(r, 0) + 1
    for r, c in sorted(reason_cnt.items()):
        print(f"  {r}: {c}", flush=True)
    print("first 30 skipped:", flush=True)
    for code, rs in list(skips.items())[:30]:
        print(f"  {code}: {rs}", flush=True)


if __name__ == "__main__":
    main()
