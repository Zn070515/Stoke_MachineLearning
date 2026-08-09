"""Post re-download verification for the 7 residual files (#85 Phase 2b-2).

Read-only.  After download_data.py --stocks <7codes>:
  - every code: file exists?  manifest present + validate_manifest ok
  - classify_file flips/junks (unit cleanliness on the FRESH download)
  - qfq_signature problems (per-year ratio sanity on the fresh qfq data)
  - formal load_daily(require_valid_manifest=True) — the Phase 3 gate
  - NaN-volume bar count (if the source still emits no-trade bars)
Writes reports/relog_7_verify.json.
"""
import json
import os

import pandas as pd

from stoke_ml.data.storage import DataStorage
from scripts.maintenance.current.migrate_daily_units import classify_file
from scripts.maintenance.current.stamp_qfq_provenance import qfq_signature

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DAILY = os.path.join(ROOT, "data", "a_shares", "daily")
CODES = ["000338", "000596", "302132",
         "000520", "001872", "001914", "002506"]


def main() -> None:
    storage = DataStorage(os.path.join(ROOT, "data"))
    out: dict[str, dict] = {}
    for code in CODES:
        path = os.path.join(DAILY, f"{code}.parquet")
        rec: dict = {"file_exists": os.path.isfile(path)}
        if not rec["file_exists"]:
            out[code] = rec
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        rec["rows"] = int(len(df))
        res = classify_file(df)
        rec["flips"] = int(res["flip"].sum())
        rec["junks"] = int(res["junk"].sum())
        sig = qfq_signature(df)
        rec["sig_problems"] = sig["problems"]
        rec["sig_flags"] = sig["flags"]
        vol = df["volume"].astype("float64")
        amt = df["amount"].astype("float64")
        rec["nan_vol"] = int(vol.isna().sum())
        rec["nan_amt"] = int(amt.isna().sum())
        rep = storage.validate_manifest(code)
        rec["validate_ok"] = bool(rep.get("ok"))
        if not rep.get("ok"):
            rec["validate_reason"] = rep.get("reason") or rep.get("mismatches")
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
            rec["formal_ok"] = True
            rec["formal_rows"] = int(len(d))
        except ValueError as exc:
            rec["formal_ok"] = False
            rec["formal_reason"] = str(exc)[:140]
        out[code] = rec
        print(f"{code}: rows={rec['rows']} flips={rec['flips']} "
              f"junks={rec['junks']} sig_problems={sig['problems']} "
              f"nan_vol={rec['nan_vol']} validate={rec['validate_ok']} "
              f"formal={rec['formal_ok']}", flush=True)

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "relog_7_verify.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("report: reports/relog_7_verify.json", flush=True)


if __name__ == "__main__":
    main()
