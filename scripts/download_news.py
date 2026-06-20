"""Download news for A-share stocks from multiple sources, compute sentiment.

Usage:
  python scripts/download_news.py                              # all stocks, all sources
  python scripts/download_news.py --stocks 000001,600519       # specific stocks
  python scripts/download_news.py --source sina                # single source
  python scripts/download_news.py --source xueqiu,ths,sina     # selected sources
  python scripts/download_news.py --max-pages 5 --sleep 1      # deeper, faster
  python scripts/download_news.py --concurrent                 # parallel download
  python scripts/download_news.py --skip-sentiment             # raw only
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.sources.a_shares.news_pipeline import NewsPipeline
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

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
    parser = argparse.ArgumentParser(description="Download A-share news")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--source", type=str, default="all",
                        help="News source(s): sina, xueqiu, ths, all (default: all)")
    parser.add_argument("--max-pages", type=int, default=3,
                        help="Pages per stock per source (default: 3)")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Seconds between stocks (default: from config)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation (raw only)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.sleep is None:
        args.sleep = float(cfg.crawler.rate_limit.base_delay_sec)

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found. Run download_data.py first.")
        sys.exit(1)

    # Select sources
    if args.source == "all":
        active_sources = None  # pipeline uses all available
    else:
        active_sources = [s.strip() for s in args.source.split(",")]

    calendar = TradingCalendar("a_shares")
    news_storage = NewsStorage(data_dir, calendar)
    news_pipeline = NewsPipeline(active_sources=active_sources)
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer()

    source_label = args.source if args.source != "all" else "sina+xueqiu+ths"
    mode_label = "concurrent" if args.concurrent else "sequential"
    logger.info(
        "Downloading news for %d stocks (sources=%s, max_pages=%d, sleep=%.1fs, %s)",
        len(codes), source_label, args.max_pages, args.sleep, mode_label,
    )

    total_articles = 0
    success, fail, empty = 0, 0, 0

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

        def _fetch_one(code: str):
            df = news_pipeline.fetch_all_news(
                code,
                start_date=args.start,
                end_date=args.end,
                max_pages=args.max_pages,
            )
            if not args.skip_sentiment and not df.empty:
                df = compute_raw_sentiment(df, analyzer)
            return df

        results = downloader.download_all(codes, _fetch_one)

        for i, code in enumerate(codes):
            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)
            df = results.get(code)
            if df is None:
                logger.error("  %s: fetch failed (exception in worker)", code)
                fail += 1
                continue

            if df.empty:
                logger.info("  %s: no news found", code)
                empty += 1
                continue

            # Save raw (Bronze)
            news_storage.save_raw_news(code, df)
            logger.info("  %s: %d articles saved (raw)", code, len(df))
            total_articles += len(df)

            # PIT-align -> Silver
            silver = news_storage.bronze_to_silver(code)
            if not silver.empty:
                news_storage.save_silver_news(code, silver)

            # Daily aggregation -> Gold
            if not args.skip_sentiment:
                gold = news_storage.silver_to_gold(code, analyzer)
                if not gold.empty:
                    news_storage.save_daily_sentiment(gold)
                    news_days = gold["has_news"].sum()
                    logger.info("  %s: %d sentiment days (%d with news)",
                                code, len(gold), news_days)

            success += 1
    else:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.sleep)

            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

            try:
                df = news_pipeline.fetch_all_news(
                    code,
                    start_date=args.start,
                    end_date=args.end,
                    max_pages=args.max_pages,
                )
            except Exception as e:
                logger.error("  %s: fetch failed: %s", code, e)
                fail += 1
                continue

            if df.empty:
                logger.info("  %s: no news found", code)
                empty += 1
                continue

            # Compute sentiment on titles
            if not args.skip_sentiment:
                df = compute_raw_sentiment(df, analyzer)

            # Save raw (Bronze)
            news_storage.save_raw_news(code, df)
            logger.info("  %s: %d articles saved (raw)", code, len(df))
            total_articles += len(df)

            # PIT-align -> Silver
            silver = news_storage.bronze_to_silver(code)
            if not silver.empty:
                news_storage.save_silver_news(code, silver)

            # Daily aggregation -> Gold
            if not args.skip_sentiment:
                gold = news_storage.silver_to_gold(code, analyzer)
                if not gold.empty:
                    news_storage.save_daily_sentiment(gold)
                    news_days = gold["has_news"].sum()
                    logger.info("  %s: %d sentiment days (%d with news)",
                                code, len(gold), news_days)

            success += 1

    logger.info("Done: %d success, %d fail, %d empty, %d total articles",
                success, fail, empty, total_articles)


if __name__ == "__main__":
    main()
