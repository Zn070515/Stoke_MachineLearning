# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Re-download flat-only stocks whose close series is corrupted (frankenstein).

Identified by scripts/diagnostics/_probe_flatonly_seams.py (sawtooth > 0 = opposite-sign
over-limit jumps on consecutive days, the signature of stitching two different
qfq bases). The flat file is replaced with a clean single-source series from the
failover chain (efinance -> AKShare -> ... -> Baostock backfill), so close and
pct_change are self-consistent. Writes route through ``DataStorage.save_daily``
so the contract manifest + source segments stay in sync (§八-1).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_redownload_flatonly.py --limit 5
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_redownload_flatonly.py
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.storage import DataStorage

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data" / "a_shares" / "daily"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=str, default=None,
                        help="comma-separated stock codes")
    parser.add_argument("--limit", type=int, default=0,
                        help="max stocks to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        rep = pd.read_csv(PROJECT / "reports" / "flatonly_seam_scan.csv")
        codes = rep[rep["sawtooth"] > 0]["code"].astype(str).tolist()
    if args.limit:
        codes = codes[:args.limit]
    print(f"re-downloading {len(codes)} stocks", flush=True)
    if args.dry_run:
        return 0

    dl = AShareDownloader()
    storage = DataStorage(str(PROJECT / "data"))
    t0 = time.time()
    ok = skip = fail = 0
    for i, code in enumerate(codes):
        flat_path = DAILY_DIR / f"{code}.parquet"
        try:
            old = pd.read_parquet(flat_path, columns=["date"])
            start = pd.to_datetime(old["date"]).min().strftime("%Y-%m-%d")
            end = pd.to_datetime(old["date"]).max().strftime("%Y-%m-%d")
        except Exception:
            print(f"  {code}: cannot read old flat, skip")
            skip += 1
            continue
        try:
            df = dl.fetch_daily(code, start, end)
        except Exception as e:
            print(f"  {code}: download error: {e}")
            fail += 1
            continue
        if df.empty or "close" not in df.columns:
            print(f"  {code}: empty result, skip")
            skip += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        if "stock_code" in df.columns:
            df["stock_code"] = code
        # save_daily performs the non-destructive merge (new clean rows win on
        # date), keeps the lock/manifest/source segments in sync (§八-1), and
        # carries the fetch layer's source/adjustment attrs.
        storage.save_daily(df, market="a_shares")
        ok += 1
        if (ok + skip + fail) <= 10 or ok % 50 == 0:
            print(f"  {code}: {len(df)} rows [{start}..{end}]  ({ok}/{len(codes)})", flush=True)
    print(f"done: {ok} ok, {skip} skip, {fail} fail ({time.time() - t0:.0f}s)")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
