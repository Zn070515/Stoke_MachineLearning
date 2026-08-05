# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Scan flat-only stocks for qfq seam corruption (close self-consistency).

flat-only stocks (no partition files) get their close entirely from the old
flat, which may have been repeatedly consolidated across sources with different
qfq (前复权) bases — the "frankenstein" close. Real qfq adjustment seams occur
only on dividend/split ex-dates and are a fixed ratio; a *sawtooth* (over-limit
jump immediately reversed next day) or seams concentrated in the 2026-06-18+
AKShare-fallback window are the corruption signature.

Output: reports/flatonly_seam_scan.csv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.config import get_project_root

PROJECT = get_project_root()
DAILY_DIR = PROJECT / "data" / "a_shares" / "daily"

FALLBACK_TS = pd.Timestamp("2026-06-18")


def limit_for(code: str) -> float:
    if code.startswith(("30", "68")):
        return 20.5
    if code.startswith(("4", "8", "92")):
        return 30.5
    return 10.5


def has_partitions(code: str) -> bool:
    return any(
        Path(f).parent != DAILY_DIR
        for f in glob.glob(str(DAILY_DIR / "**" / f"{code}.parquet"), recursive=True)
    )


def scan():
    codes = sorted(
        Path(f).stem for f in glob.glob(str(DAILY_DIR / "*.parquet"))
        if not has_partitions(Path(f).stem)
    )
    print(f"scanning {len(codes)} flat-only stocks", flush=True)

    rows = []
    for i, code in enumerate(codes):
        try:
            df = pd.read_parquet(DAILY_DIR / f"{code}.parquet")
        except Exception as e:
            rows.append({"code": code, "error": str(e)})
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        pct = close.pct_change() * 100.0
        if "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            vol = pd.Series(np.nan, index=df.index)
        lim = limit_for(code)
        traded_prev = vol.shift(1).fillna(1) > 0  # exclude halt->resume jump
        over = (pct.abs() > lim) & traded_prev
        n_over = int(over.sum())
        n_recent = int((over & (df["date"] > FALLBACK_TS)).sum())
        # Sawtooth: consecutive over-limit days with opposite sign — the
        # signature of stitching two different qfq bases, not a real ex-date.
        sign = np.sign(pct)
        saw = (over & over.shift(-1).fillna(False)
               & (sign != sign.shift(-1)) & sign.notna() & sign.shift(-1).notna())
        n_saw = int(saw.sum())
        rows.append({
            "code": code,
            "rows": len(df),
            "seams": n_over,
            "recent_seams": n_recent,
            "sawtooth": n_saw,
            "last_seam": str(df.loc[over, "date"].max().date()) if n_over else "",
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(codes)}", flush=True)

    rep = pd.DataFrame(rows)
    if "error" not in rep.columns:
        rep["error"] = ""
    out = PROJECT / "reports" / "flatonly_seam_scan.csv"
    out.parent.mkdir(exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"saved {out}")
    ok = rep[rep["error"] == ""]
    print(f"  rows: {len(ok)}  seams distribution:\n"
          f"    seams=0: {int((ok['seams']==0).sum())}  "
          f"seams=1-2: {int(((ok['seams']>=1)&(ok['seams']<=2)).sum())}  "
          f"seams>=3: {int((ok['seams']>=3).sum())}")
    print(f"  with sawtooth>0: {int((ok['sawtooth']>0).sum())}  "
          f"with recent_seams>0: {int((ok['recent_seams']>0).sum())}")
    susp = ok[(ok["sawtooth"] > 0) | (ok["recent_seams"] > 0)]
    if len(susp):
        print(f"  suspicious candidates ({len(susp)}):")
        print(susp.sort_values(["sawtooth", "recent_seams", "seams"], ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    sys.exit(scan())
