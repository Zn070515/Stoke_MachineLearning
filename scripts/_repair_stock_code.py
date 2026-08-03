"""Repair stock_code=NaN in MarketWide flat files (legacy write-path bug).

Some per-stock flat files under data/a_shares/{type}/{code}.parquet were
written before the stock_code fix and carry stock_code=NaN rows. NaN breaks
both feature-pipeline joins (merge on date+stock_code never matches) and
full-row dedup during backfills (same date appears twice: NaN + correct code).

This fills stock_code=NaN -> the file's code, drops now-identical duplicate
rows, sorts by date, and writes back atomically. Non-destructive for
genuinely-distinct rows (block_trade multi-row-per-day preserved).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_stock_code.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_stock_code.py --types margin,capital_flow
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_stock_code.py --dry-run
"""
import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
BASE = PROJECT / "data" / "a_shares"

DEFAULT_TYPES = [
    "margin", "capital_flow", "dragon_tiger", "northbound", "block_trade",
    "shareholder", "lockup", "dividend",
]


def repair_one(type_dir: Path, code: str, dry: bool) -> tuple[bool, int]:
    """Return (had_nan_or_missing_col, n_rows_dropped)."""
    path = type_dir / f"{code}.parquet"
    df = pd.read_parquet(path)
    if "stock_code" not in df.columns:
        df["stock_code"] = code
        changed = True
    else:
        s = df["stock_code"].astype("string")
        changed = bool(s.isna().any())
        df["stock_code"] = s.fillna(code)
    before = len(df)
    df = df.drop_duplicates(keep="last")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    changed = changed or before != len(df)
    if changed and not dry:
        tmp = str(path) + ".tmp"
        df.to_parquet(tmp, index=False, compression="lz4")
        os.replace(tmp, path)
    return changed, before - len(df)


def main():
    ap = argparse.ArgumentParser(description="Repair stock_code=NaN in flat files")
    ap.add_argument("--types", type=str, default=",".join(DEFAULT_TYPES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    t0 = time.time()
    for t in types:
        tdir = BASE / t
        if not tdir.is_dir():
            logger.warning("skip missing type dir: %s", t)
            continue
        files = sorted(glob.glob(str(tdir / "*.parquet")))
        changed = ok = dropped = 0
        for f in files:
            code = Path(f).stem
            try:
                was_changed, n_dropped = repair_one(tdir, code, args.dry_run)
            except Exception as e:
                logger.error("%s/%s: %s", t, code, e)
                continue
            ok += 1
            changed += int(was_changed)
            dropped += n_dropped
        logger.info("%s: %d files ok, %d changed, %d dup rows dropped (%.1fs)",
                    t, ok, changed, dropped, time.time() - t0)
    logger.info("done (%.1fs)", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
