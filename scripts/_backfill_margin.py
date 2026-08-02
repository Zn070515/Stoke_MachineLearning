"""Backfill margin trading data 2012-2024-08 (existing coverage is heterogeneous).

Existing per-stock flat files start in scattered years (2015..2026), so many
stocks lack early history. This fetches daily SSE+SZSE margin snapshots for the
full range 2012-01-01 ~ 2024-08-31 and merges into the existing flat files
(MarketWideStorage.save dedups identical rows; stocks not yet margin-eligible
simply don't appear in that day's snapshot).

Resumable per year via marker files under data/a_shares/margin/_backfill_done/.
Pre-2015 trading days come from AKShare's official calendar (TradingCalendar
only has holidays from 2015 onward).
"""
import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.margin_source import MarginTradingSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def trading_days_for_year(year: int) -> list[dt.date]:
    if year >= 2015:
        return TradingCalendar("a_shares").get_trading_days(
            dt.date(year, 1, 1), dt.date(year, 12, 31)
        )
    import akshare as ak

    cal = ak.tool_trade_date_hist_sina()
    days = pd.to_datetime(cal["trade_date"])
    return [d.date() for d in days if d.year == year]


def main():
    ap = argparse.ArgumentParser(description="Backfill margin 2012-2024-08")
    ap.add_argument("--start", type=int, default=2012)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit to first N days of start year (validation)")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    storage = MarketWideStorage(data_dir, "margin")
    source = MarginTradingSource()
    marker_dir = Path(data_dir) / "a_shares" / "margin" / "_backfill_done"
    marker_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        marker = marker_dir / f"{year}.marker"
        if marker.exists() and not args.force:
            logger.info("year %d already backfilled, skip", year)
            continue

        days = trading_days_for_year(year)
        if year == 2024:
            days = [d for d in days if d <= dt.date(2024, 8, 31)]
        if args.smoke and year == args.start:
            days = days[: args.smoke]
        if not days:
            logger.warning("year %d: no trading days", year)
            continue

        logger.info("margin backfill %s ~ %s (%d days)", days[0], days[-1], len(days))
        try:
            df = source.fetch_daily(
                days[0].isoformat(), days[-1].isoformat(),
                sleep=args.sleep, dates=days,
            )
        except Exception as e:
            logger.error("year %d FAILED: %s", year, e)
            continue

        if df is not None and not df.empty:
            storage.save(df)
            logger.info("year %d saved %d rows", year, len(df))
        else:
            logger.warning("year %d: no data fetched", year)

        if not (args.smoke and year == args.start):
            marker.touch()

    logger.info("margin backfill done")


if __name__ == "__main__":
    main()
