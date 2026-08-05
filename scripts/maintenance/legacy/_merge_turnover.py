# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Merge turnover_raw data into daily K-line parquet files.

Reads date + turnover from data/a_shares/turnover_raw/{code}.parquet,
left-joins into data/a_shares/daily/{code}.parquet on date,
removes the raw file after successful merge.

Resume-safe: skips stocks that already have a non-null turnover column
in their daily file.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_merge_turnover.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_merge_turnover.py --stocks 000001,600519
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Merge turnover_raw into daily K-line parquet files"
    )
    parser.add_argument(
        "--stocks", type=str, default=None,
        help="Comma-separated stock codes (default: all with raw files)",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw files after merge (default: delete on success)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.storage import DataStorage

    cfg = load_config()
    data_dir = Path(cfg.project.data_dir) / "a_shares"
    daily_dir = data_dir / "daily"
    raw_dir = data_dir / "turnover_raw"
    storage = DataStorage(cfg.project.data_dir)

    if not raw_dir.exists():
        logger.error("turnover_raw directory not found: %s", raw_dir)
        return 1

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([f.stem for f in raw_dir.glob("*.parquet")])

    logger.info("Scanning %d stocks with turnover raw data...", len(codes))

    # Discover stocks that actually need merging
    todo = []
    already_ok = 0
    for code in codes:
        daily_path = daily_dir / f"{code}.parquet"
        if not daily_path.exists():
            logger.warning("%s: raw exists but no daily file, skipping", code)
            continue
        try:
            # Check if daily already has turnover with actual values
            df = pd.read_parquet(daily_path, columns=["turnover"])
            if "turnover" in df.columns and df["turnover"].notna().sum() > 0:
                already_ok += 1
                continue
        except Exception:
            pass  # turnover column doesn't exist, needs merge
        todo.append(code)

    if not todo:
        logger.info("All %d stocks already have turnover data (or no raw). Done.", len(codes))
        return 0

    logger.info("%d to merge, %d already ok, %d total raw files",
                len(todo), already_ok, len(codes))

    success, fail = 0, 0
    t_start = time.time()

    for i, code in enumerate(todo):
        raw_path = raw_dir / f"{code}.parquet"
        daily_path = daily_dir / f"{code}.parquet"

        try:
            raw = pd.read_parquet(raw_path)
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"])
            if raw.empty:
                fail += 1
                logger.warning("%s: raw file empty", code)
                continue

            daily = pd.read_parquet(daily_path)
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

            # Left-join: keep all daily rows, add turnover where available
            daily = daily.merge(
                raw[["date", "turnover"]], on="date", how="left",
                suffixes=("", "_raw"),
            )
            # If daily already had a turnover column, prefer new data but keep old as fallback
            if "turnover_raw" in daily.columns:
                daily["turnover"] = daily["turnover_raw"].combine_first(
                    daily.get("turnover", pd.NA)
                ) if "turnover" in daily.columns else daily["turnover_raw"]
                daily = daily.drop(columns=["turnover_raw"])

            # Write through the storage API so the manifest + source segments
            # stay in sync; provenance is preserved from the existing file
            # (§八-1).
            storage.save_daily_repair(daily, market="a_shares")

            if not args.keep_raw:
                os.remove(raw_path)

            success += 1
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                logger.info("  %d/%d done, %.1f stk/min",
                            i + 1, len(todo), (i + 1) / elapsed * 60)

        except Exception as e:
            logger.error("%s: merge failed: %s", code, e)
            fail += 1

    elapsed = time.time() - t_start
    logger.info("Done: %d ok, %d fail, %d skipped (%.1f min)",
                success, fail, already_ok, elapsed / 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
