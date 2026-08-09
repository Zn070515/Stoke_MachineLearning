"""0-fill NaN volume/amount bars on a restored canonical daily file.

Used for 001872 (#85 Phase 2b-2): the fresh source download has MORE NaN
no-trade bars (5.6%) than the stored version (1.4%), so the stored version was
restored and only its NaN flat bars are 0-filled — volume/amount NaN -> 0 on
bars that are already flat OHLC with pct=0 and close == prev close (verified
contract-consistent).  Writes via save_daily_repair so the qfq provenance is
carried forward unchanged.

Run:
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_repair_nan_volume.py
"""
import os

import pandas as pd

from stoke_ml.data.storage import DataStorage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CODE = "001872"


def main() -> None:
    storage = DataStorage(os.path.join(ROOT, "data"))
    df = storage.load_daily(CODE, "1970-01-01", "2099-12-31")
    assert len(df), f"{CODE} not on disk — restore from backup first"
    df["date"] = pd.to_datetime(df["date"])
    nanv = df["volume"].isna()
    nana = df["amount"].isna()
    print(f"{CODE}: rows={len(df)} nan_vol={int(nanv.sum())} "
          f"nan_amt={int(nana.sum())}", flush=True)
    if nanv.sum() or nana.sum():
        # 0-fill only the bars that are flat no-trade placeholders
        flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) \
            & (df["low"] == df["close"])
        nan_rows = (nanv | nana)
        if not bool(flat[nan_rows].all()):
            print(f"WARNING: some of {int(nan_rows.sum())} NaN rows are not "
                  f"flat OHLC — refusing to blanket 0-fill", file=os.sys.stderr)
            raise SystemExit(1)
        df.loc[nanv, "volume"] = 0.0
        df.loc[nana, "amount"] = 0.0
        print(f"  0-filled {int(nan_rows.sum())} flat no-trade bars",
              flush=True)
    df["stock_code"] = CODE
    storage.save_daily_repair(df)
    # verify
    from scripts.maintenance.current.migrate_daily_units import classify_file
    from scripts.maintenance.current.stamp_qfq_provenance import qfq_signature
    back = storage.load_daily(CODE, "1970-01-01", "2099-12-31")
    res = classify_file(back)
    sig = qfq_signature(back)
    rep = storage.validate_manifest(CODE)
    print(f"  post-repair: flips={int(res['flip'].sum())} "
          f"junks={int(res['junk'].sum())} sig_problems={sig['problems']} "
          f"nan_vol={int(back['volume'].isna().sum())} "
          f"validate={rep.get('ok')}", flush=True)
    formal = storage.load_daily(CODE, "1970-01-01", "2099-12-31",
                                require_valid_manifest=True)
    print(f"  formal read OK: rows={len(formal)}", flush=True)


if __name__ == "__main__":
    main()
