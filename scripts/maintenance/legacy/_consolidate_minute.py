# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Consolidate minute K-line from partitioned {freq}/{year}/{month}/{stock}.parquet
into flat minute_flat/{freq}/{stock}.parquet.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_minute.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_minute.py --freq 60
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_minute.py --dry-run
"""
import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT / "data" / "a_shares" / "minute"
OUT_DIR = PROJECT / "data" / "a_shares" / "minute_flat"

SORT_COL = "datetime"
KEEP_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount",
             "stock_code", "bar_period"]


def consolidate_freq(freq: str, dry_run: bool = False):
    src = SRC_DIR / freq
    if not src.exists():
        logger.warning("%s: source directory not found, skipping", src)
        return 0, 0

    # Group files by stock_code
    stock_files: dict[str, list[Path]] = defaultdict(list)
    for pq in src.glob("**/*.parquet"):
        stock_files[pq.stem].append(pq)

    n_stocks = len(stock_files)
    n_files = sum(len(v) for v in stock_files.values())
    logger.info("%smin: %d files across %d stocks", freq, n_files, n_stocks)

    if dry_run:
        sizes = [len(v) for v in stock_files.values()]
        logger.info("  Files/stock: min=%d, max=%d, mean=%.1f",
                    min(sizes), max(sizes), sum(sizes) / len(sizes))
        return n_stocks, n_files

    out = OUT_DIR / freq
    out.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    t0 = time.time()

    for code, files in sorted(stock_files.items()):
        out_path = out / f"{code}.parquet"
        try:
            # Read all monthly chunks
            chunks = []
            for f in files:
                df = pd.read_parquet(f)
                # Keep only needed columns to save memory
                cols = [c for c in KEEP_COLS if c in df.columns]
                chunks.append(df[cols])

            merged = pd.concat(chunks, ignore_index=True)
            merged = merged.sort_values(SORT_COL).reset_index(drop=True)
            merged = merged.drop_duplicates(subset=[SORT_COL, "stock_code"], keep="last")

            merged.to_parquet(out_path, index=False, compression="lz4")
            ok += 1

        except Exception as e:
            logger.error("%s: %s", code, e)
            fail += 1

        if (ok + fail) % 500 == 0:
            elapsed = time.time() - t0
            logger.info("  %smin: %d/%d done (%.1f stk/s)",
                        freq, ok + fail, n_stocks, (ok + fail) / elapsed)

    elapsed = time.time() - t0
    logger.info("%smin: %d ok, %d fail (%.1fs, %.1f stk/s)",
                freq, ok, fail, elapsed, ok / elapsed)
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Consolidate minute K-line to flat per-stock")
    parser.add_argument("--freq", type=str, default=None,
                        help="Single frequency (5/15/30/60) or omit for all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan only, don't consolidate")
    args = parser.parse_args()

    freqs = [args.freq] if args.freq else ["5", "15", "30", "60"]
    total_ok, total_fail = 0, 0

    for freq in freqs:
        ok, fail = consolidate_freq(freq, dry_run=args.dry_run)
        total_ok += ok
        total_fail += fail

    logger.info("Done: %d ok, %d fail", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
