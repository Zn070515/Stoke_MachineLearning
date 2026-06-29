"""Download market-wide data: dragon-tiger, margin, northbound.

Usage:
  python scripts/download_market_data.py --type all
  python scripts/download_market_data.py --type margin --start 2020-01-01
  python scripts/download_market_data.py --type northbound --stocks 600519
  python scripts/download_market_data.py --type all --concurrent
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.margin_source import MarginTradingSource
from stoke_ml.data.sources.a_shares.dragon_tiger_source import DragonTigerSource
from stoke_ml.data.sources.a_shares.northbound_source import NorthboundSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stocks_from_disk(data_dir: str) -> list[str]:
    """Discover stock codes from existing K-line data on disk."""
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
    parser = argparse.ArgumentParser(description="Download A-share market data")
    parser.add_argument("--type", type=str, default="all",
                        choices=["margin", "dragon_tiger", "northbound", "all"],
                        help="Data type to download (default: all)")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Seconds between API calls (default: from config)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    args = parser.parse_args()

    if args.end is None:
        from datetime import datetime
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.sleep is None:
        args.sleep = float(cfg.crawler.rate_limit.base_delay_sec)

    to_download = []
    if args.type == "all":
        to_download = [
            ("margin", MarginTradingSource, "margin"),
            ("dragon_tiger", DragonTigerSource, "dragon_tiger"),
            ("northbound", NorthboundSource, "northbound"),
        ]
    elif args.type == "margin":
        to_download = [("margin", MarginTradingSource, "margin")]
    elif args.type == "dragon_tiger":
        to_download = [("dragon_tiger", DragonTigerSource, "dragon_tiger")]
    elif args.type == "northbound":
        to_download = [("northbound", NorthboundSource, "northbound")]

    for label, source_cls, storage_type in to_download:
        logger.info("=== Downloading %s (%s to %s) ===", label, args.start, args.end)
        t0 = time.time()

        try:
            source = source_cls()
        except Exception as e:
            logger.error("Failed to init %s source: %s", label, e)
            continue

        storage = MarketWideStorage(data_dir, storage_type)

        if label == "margin":
            # Fetch year-by-year with incremental saves to survive interruptions
            start_dt = pd.Timestamp(args.start).date()
            end_dt = pd.Timestamp(args.end).date()
            all_frames = []
            for year in range(start_dt.year, end_dt.year + 1):
                y_start = f"{year}-01-01"
                y_end = f"{min(year, end_dt.year)}-12-31"
                if year == start_dt.year:
                    y_start = args.start
                if year == end_dt.year:
                    y_end = args.end
                logger.info("  margin: downloading %s to %s", y_start, y_end)
                y_df = source.fetch_daily(y_start, y_end)
                if y_df is not None and not y_df.empty:
                    storage.save(y_df)
                    all_frames.append(y_df)
                    logger.info("  margin saved %d rows for year %d", len(y_df), year)
                else:
                    logger.warning("  margin: no data for year %d", year)
            df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        elif label == "dragon_tiger":
            if args.stocks:
                stock_list = [c.strip() for c in args.stocks.split(",")]
                if args.concurrent:
                    from stoke_ml.crawler.rate_limiter import RateLimiter
                    from stoke_ml.crawler.concurrent import ConcurrentDownloader

                    rate_limiter = RateLimiter(
                        base_delay_sec=args.sleep,
                        daily_quota=cfg.crawler.rate_limit.daily_quota_per_domain,
                    )
                    downloader = ConcurrentDownloader(
                        rate_limiter=rate_limiter, max_workers=args.workers,
                    )
                    results = downloader.download_all(
                        stock_list,
                        lambda c: source.fetch_by_stock(c, args.start, args.end),
                    )
                    frames = [d for d in results.values() if d is not None and not d.empty]
                else:
                    frames = []
                    for code in stock_list:
                        sdf = source.fetch_by_stock(code, args.start, args.end)
                        if not sdf.empty:
                            frames.append(sdf)
                        time.sleep(args.sleep)
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df = source.fetch_daily(args.start, args.end)
        elif label == "northbound":
            stock_codes = None
            if args.stocks:
                stock_codes = [c.strip() for c in args.stocks.split(",")]
            else:
                stock_codes = get_stocks_from_disk(data_dir)
                if not stock_codes:
                    logger.error("No stock codes found. Run download_data.py first.")
                    sys.exit(1)
                logger.info("  northbound: loaded %d stock codes from disk", len(stock_codes))
            if stock_codes and args.concurrent:
                from stoke_ml.crawler.rate_limiter import RateLimiter
                from stoke_ml.crawler.concurrent import ConcurrentDownloader

                rate_limiter = RateLimiter(
                    base_delay_sec=args.sleep,
                    daily_quota=cfg.crawler.rate_limit.daily_quota_per_domain,
                )
                downloader = ConcurrentDownloader(
                    rate_limiter=rate_limiter, max_workers=args.workers,
                )
                results = downloader.download_all(
                    stock_codes,
                    lambda c: source.fetch_individual(
                        args.start, args.end, stock_codes=[c],
                    ),
                )
                frames = [d for d in results.values() if not d.empty]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            elif stock_codes:
                # Save incrementally in batches of 50 to avoid timeout
                batch_size = 50
                all_frames = []
                for i in range(0, len(stock_codes), batch_size):
                    batch = stock_codes[i:i + batch_size]
                    batch_df = source.fetch_individual(
                        args.start, args.end, stock_codes=batch,
                    )
                    if batch_df is not None and not batch_df.empty:
                        all_frames.append(batch_df)
                        storage.save(batch_df)
                        logger.info("  saved batch %d/%d: %d rows",
                                    i // batch_size + 1,
                                    (len(stock_codes) + batch_size - 1) // batch_size,
                                    len(batch_df))
                df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
                logger.info("  northbound: %d rows total (%d batches)",
                            len(df), len(all_frames))
            else:
                df = source.fetch_individual(args.start, args.end, stock_codes=stock_codes)
        else:
            df = pd.DataFrame()

        if df is not None and not df.empty:
            storage.save(df)
            logger.info(
                "  %s: %d rows saved (%.1fs)",
                label, len(df), time.time() - t0,
            )
        else:
            logger.warning("  %s: no data returned", label)

    logger.info("Done.")


if __name__ == "__main__":
    main()
