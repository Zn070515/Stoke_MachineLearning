# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Backfill turnover + amplitude columns into existing K-line parquet files.

EastMoney API returns these fields but the old source normalization dropped them.
This script re-fetches daily data for stocks missing the columns, using the
failover chain (efinance → akshare → tushare → baostock), and merges
only the new columns into existing files — no re-download of OHLCV needed.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_kline_columns.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_kline_columns.py --stocks 000001,600519
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent

NEW_COLS = ["turnover", "amplitude"]


def main():
    parser = argparse.ArgumentParser(description="Backfill K-line columns")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Delay between stocks (default: 0.3s)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.sources.a_shares.failover import AShareDownloader
    from stoke_ml.data.storage import DataStorage

    cfg = load_config()
    data_dir = Path(cfg.project.data_dir) / "a_shares" / "daily"
    storage = DataStorage(cfg.project.data_dir)

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([f.stem for f in data_dir.glob("*.parquet")])

    # ── Discover stocks missing new columns ──
    todo = []
    for code in codes:
        path = data_dir / f"{code}.parquet"
        try:
            existing = pd.read_parquet(path, columns=["date"])  # only read column list
            existing_cols = set(pd.read_parquet(path).columns)
            missing = [c for c in NEW_COLS if c not in existing_cols]
            if missing:
                todo.append((code, missing))
        except Exception:
            todo.append((code, NEW_COLS))

    if not todo:
        logger.info("All %d files already have %s — nothing to do.",
                    len(codes), NEW_COLS)
        return 0

    logger.info("%d/%d stocks missing columns, downloading...",
                len(todo), len(codes))

    downloader = AShareDownloader()
    success, fail = 0, 0

    for i, (code, missing) in enumerate(todo):
        if i > 0:
            time.sleep(args.sleep)

        try:
            fresh = downloader.fetch_daily(code, "2015-01-01", "2026-12-31")
            if fresh.empty:
                logger.warning("[%d/%d] %s: fetch returned empty", i + 1, len(todo), code)
                fail += 1
                continue

            # Merge only the new columns into existing file
            existing = pd.read_parquet(data_dir / f"{code}.parquet")
            fresh["date"] = pd.to_datetime(fresh["date"]).dt.date
            existing["date"] = pd.to_datetime(existing["date"]).dt.date

            merge_cols = ["date"] + [c for c in NEW_COLS if c in fresh.columns]
            merged = existing.merge(
                fresh[merge_cols], on="date", how="left", suffixes=("", "_fresh"),
            )

            # If any NEW_COLS came through as _fresh suffix, replace with those
            for c in NEW_COLS:
                fresh_c = f"{c}_fresh"
                if fresh_c in merged.columns:
                    merged[c] = merged[c].fillna(merged[fresh_c])
                    merged = merged.drop(columns=[fresh_c])

            # Write through the storage API so the manifest + source segments
            # stay in sync; provenance is preserved from the existing file
            # (§八-1).
            if "stock_code" not in merged.columns:
                merged["stock_code"] = code
            storage.save_daily_repair(merged, market="a_shares")

            got = [c for c in NEW_COLS if c in fresh.columns]
            logger.info("[%d/%d] %s: added %s", i + 1, len(todo), code, got)
            success += 1
        except Exception as e:
            logger.error("[%d/%d] %s: %s", i + 1, len(todo), code, e)
            fail += 1

    logger.info("Done: %d success, %d fail", success, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
