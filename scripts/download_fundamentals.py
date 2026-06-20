"""Download quarterly fundamental data for all A-share stocks.

Usage:
  python scripts/download_fundamentals.py
  python scripts/download_fundamentals.py --stocks 000001,600519
  python scripts/download_fundamentals.py --start 2020-01-01 --end 2024-12-31 --sleep 0.5
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.sources.a_shares.fundamental_source import FundamentalSource
from stoke_ml.data.fundamental_storage import FundamentalStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stocks_from_disk(data_dir: str) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    codes = set()
    for root, _dirs, files in os.walk(daily_dir):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description="Download fundamental data")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date filter")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter (default: today)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between stocks (default: 0.5)")
    args = parser.parse_args()

    if args.end is None:
        from datetime import datetime
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found.")
        sys.exit(1)

    calendar = TradingCalendar("a_shares")
    source = FundamentalSource()
    storage = FundamentalStorage(data_dir, calendar)

    logger.info("Downloading fundamentals for %d stocks", len(codes))

    total_rows = 0
    success, fail, empty = 0, 0, 0

    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

        try:
            df = source.fetch_indicators(code)
        except Exception as e:
            logger.error("  %s: fetch failed: %s", code, e)
            fail += 1
            continue

        if df.empty:
            empty += 1
            continue

        # Filter by date
        if "report_date" in df.columns:
            df = df[df["report_date"] >= pd.Timestamp(args.start)]

        storage.save(df)
        logger.info("  %s: %d quarters saved", code, len(df))
        total_rows += len(df)
        success += 1

    logger.info("Done: %d success, %d fail, %d empty, %d total quarters",
                success, fail, empty, total_rows)


if __name__ == "__main__":
    main()
