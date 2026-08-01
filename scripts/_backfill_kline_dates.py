"""Backfill K-line date range: prepend 2000-2014 data to existing files.

Existing files start at 2015-01-01. This fetches data from 2000-01-01
to each stock's current earliest date, prepends, and deduplicates.
Stocks listed after 2000 will naturally return fewer rows (the failover
chain handles IPO date gaps).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_backfill_kline_dates.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/_backfill_kline_dates.py --stocks 000001,600519
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

BACKFILL_START = "2000-01-01"


def main():
    parser = argparse.ArgumentParser(description="Backfill K-line date range")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Delay between stocks (default: 0.3s)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.sources.a_shares.failover import AShareDownloader

    cfg = load_config()
    data_dir = Path(cfg.project.data_dir) / "a_shares" / "daily"

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([f.stem for f in data_dir.glob("*.parquet")])

    # ── Discover stocks needing backfill ──
    todo = []
    for code in codes:
        path = data_dir / f"{code}.parquet"
        try:
            existing = pd.read_parquet(path, columns=["date"])
            existing["date"] = pd.to_datetime(existing["date"])
            min_date = existing["date"].min().date()
            if min_date > pd.Timestamp(BACKFILL_START).date():
                todo.append((code, min_date))
        except Exception as e:
            logger.warning("%s: read failed: %s", code, e)

    if not todo:
        logger.info("All %d files already start at or before %s.",
                    len(codes), BACKFILL_START)
        return 0

    logger.info("%d/%d stocks need date backfill", len(todo), len(codes))

    downloader = AShareDownloader()
    success, fail = 0, 0

    for i, (code, min_date) in enumerate(todo):
        if i > 0:
            time.sleep(args.sleep)

        end_date = (min_date - pd.Timedelta(days=1)).isoformat()
        logger.info("[%d/%d] %s: fetching %s → %s",
                    i + 1, len(todo), code, BACKFILL_START, end_date)

        try:
            backfill = downloader.fetch_daily(code, BACKFILL_START, end_date)
            if backfill.empty:
                logger.info("  %s: no data before %s (IPO after 2000)", code, min_date)
                success += 1
                continue

            backfill["date"] = pd.to_datetime(backfill["date"]).dt.date

            existing = pd.read_parquet(data_dir / f"{code}.parquet")
            existing["date"] = pd.to_datetime(existing["date"]).dt.date

            # Concatenate: older data first, keep first on duplicate dates
            combined = pd.concat([backfill, existing], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="first")
            combined = combined.sort_values("date", ascending=True)
            combined.to_parquet(data_dir / f"{code}.parquet", index=False, compression='lz4')

            old_rows = len(backfill)
            logger.info("  %s: +%d rows, now %d total [%s → %s]",
                        code, old_rows, len(combined),
                        combined["date"].min().date(),
                        combined["date"].max().date())
            success += 1
        except Exception as e:
            logger.error("  %s: %s", code, e)
            fail += 1

    logger.info("Done: %d success, %d fail", success, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
