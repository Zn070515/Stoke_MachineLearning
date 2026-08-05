"""Download historical quarterly shareholder count for all A-shares.

Uses EastMoney RPT_HOLDERNUM_DET — one paginated API call per quarter-end
date returns ALL stocks' holder counts (~5000 stocks × ~100 quarters).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_shareholder.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_shareholder.py --start 2015-01-01
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_shareholder.py --date 2023-09-30
"""
import argparse
import logging
import sys
import time
from datetime import date, timedelta

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.datacenter_sources import ShareholderSource
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.download_manifest import write_run_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_shareholder.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# Months that mark quarter-end report dates
QUARTER_MONTHS = [3, 6, 9, 12]
QUARTER_DAYS = {3: 31, 6: 30, 9: 30, 12: 31}


def generate_quarters(start_date: str, end_date: str) -> list[str]:
    """Generate all quarter-end dates between start and end."""
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    quarters = []
    y, m = start.year, start.month
    # Find first quarter month >= start
    qm = next((qm for qm in QUARTER_MONTHS if qm >= m), None)
    if qm is None:
        y += 1
        qm = 3
    current = date(y, qm, QUARTER_DAYS[qm])
    while current <= end:
        if current >= start:
            quarters.append(current.strftime("%Y-%m-%d"))
        # Next quarter
        idx = QUARTER_MONTHS.index(current.month)
        if idx == 3:  # December → next year March
            current = date(current.year + 1, 3, 31)
        else:
            next_m = QUARTER_MONTHS[idx + 1]
            current = date(current.year, next_m, QUARTER_DAYS[next_m])
    return quarters


def main():
    parser = argparse.ArgumentParser(
        description="Download historical quarterly shareholder count data"
    )
    parser.add_argument("--start", type=str, default="2000-01-01",
                        help="Earliest quarter to fetch (default: 2000-01-01)")
    parser.add_argument("--end", type=str, default=None,
                        help="Latest quarter (default: today)")
    parser.add_argument("--date", type=str, default=None,
                        help="Fetch a single quarter-end date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if data already exists")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.date:
        dates = [args.date]
    else:
        end = args.end or date.today().strftime("%Y-%m-%d")
        dates = generate_quarters(args.start, end)
        logger.info("Quarter range: %s → %s (%d quarters)", dates[0], dates[-1], len(dates))

    source = ShareholderSource(min_interval=0.6)
    storage = MarketWideStorage(data_dir, "shareholder")

    t0 = time.time()
    total_stocks = 0
    success = 0
    done_dates: set[str] = set()
    failed_dates: list[str] = []

    for i, dt in enumerate(dates):
        try:
            df = source.fetch_quarter(dt)
            n = len(df)
            if n == 0:
                logger.info("[%d/%d] %s: 0 stocks (empty)", i + 1, len(dates), dt)
                continue

            # Save per-stock files (merge with existing)
            storage.save(df)
            total_stocks += n
            success += 1
            done_dates.add(dt)

            if (i + 1) % 10 == 0 or (i + 1) == len(dates):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60
                eta = (len(dates) - i - 1) / max(rate, 0.01)
                logger.info(
                    "[%d/%d] %.0f qtrs/min | %.0fK total stocks | ETA %.0f min",
                    i + 1, len(dates), rate, total_stocks / 1000, eta,
                )

        except Exception as e:
            logger.error("[%d/%d] %s: error: %s", i + 1, len(dates), dt, str(e)[:100])
            failed_dates.append(dt)

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/shareholder",
            start_date=args.start, end_date=args.end,
            requested=dates, failed=failed_dates, complete=done_dates,
            success_count=success,
        )
    except Exception as exc:
        logger.error("run manifest write failed: %s", exc)

    elapsed = time.time() - t0
    stored = [f.replace(".parquet", "")
              for f in __import__("os").listdir(storage._base_dir())
              if f.endswith(".parquet")]
    logger.info(
        "Done: %d/%d quarters, %.0fK total rows, %d stocks in %.1f min",
        success, len(dates), total_stocks / 1000, len(stored), elapsed / 60,
    )


if __name__ == "__main__":
    main()
