"""Consolidate quarterly fundamentals into flat fundamentals/{code}.parquet.

Merges flat fundamentals/{code}.parquet with partitioned fundamentals/{year}/Q{n}/{code}.parquet,
de-duping on report_date keeping the most recent disclosure (max disclose_date), and writes
back to flat. Keeps ALL columns (union). Does NOT delete partitions (safe/reversible).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_consolidate_fund_flat.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/_consolidate_fund_flat.py --shard 0/8
"""
import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
FUND_DIR = PROJECT / "data" / "a_shares" / "fundamentals"


def collect_codes():
    codes = set()
    for f in glob.glob(str(FUND_DIR / "**" / "*.parquet"), recursive=True):
        codes.add(Path(f).stem)
    return sorted(codes)


def consolidate_one(code):
    flat_path = FUND_DIR / f"{code}.parquet"
    part_files = sorted(
        Path(f) for f in glob.glob(str(FUND_DIR / "**" / f"{code}.parquet"), recursive=True)
        if Path(f).parent != FUND_DIR
    )
    chunks = []
    if flat_path.exists():
        chunks.append(pd.read_parquet(flat_path))
    for f in part_files:
        chunks.append(pd.read_parquet(f))
    if not chunks:
        return "rows=0"
    df = pd.concat(chunks, ignore_index=True)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
        if "disclose_date" in df.columns:
            df["disclose_date"] = pd.to_datetime(df["disclose_date"])
            df = df.sort_values(["report_date", "disclose_date"], kind="mergesort")
        else:
            df = df.sort_values("report_date")
        df = df.drop_duplicates(subset="report_date", keep="last").reset_index(drop=True)
    if "stock_code" in df.columns:
        df["stock_code"] = str(code)
    tmp_path = str(flat_path) + ".tmp"
    df.to_parquet(tmp_path, index=False, compression="lz4")
    os.replace(tmp_path, flat_path)
    return f"rows={len(df)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default="0/1")
    args = parser.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    codes = collect_codes()
    mine = [c for i, c in enumerate(codes) if i % n == k]
    logger.info("shard %d/%d: %d stocks (total %d)", k, n, len(mine), len(codes))
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(mine):
        try:
            consolidate_one(code)
            ok += 1
        except Exception as e:
            logger.error("%s: %s", code, e)
            fail += 1
        if (ok + fail) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d done (%.1f stk/s)", ok + fail, len(mine),
                        (ok + fail) / max(elapsed, 1e-9))
    logger.info("shard %d/%d done: %d ok, %d fail (%.1fs)", k, n, ok, fail,
                time.time() - t0)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
