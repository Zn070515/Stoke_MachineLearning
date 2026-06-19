"""Download news for A-share stocks, compute sentiment, store in 3-layer format.

Usage:
  python scripts/download_news.py                          # all stocks on disk
  python scripts/download_news.py --stocks 000001,600519   # specific stocks
  python scripts/download_news.py --max-pages 5 --sleep 2  # deeper, slower
  python scripts/download_news.py --skip-sentiment          # raw only
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
from stoke_ml.data.sources.a_shares.news_source import SinaNewsSource
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
    parser.add_argument("--max-pages", type=int, default=3,
                        help="Pages per stock (default: 3)")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between stocks (default: 2.0)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation (raw only)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found. Run download_data.py first.")
        sys.exit(1)

    calendar = TradingCalendar("a_shares")
    news_storage = NewsStorage(data_dir, calendar)
    news_source = SinaNewsSource()
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer()

    logger.info("Downloading news for %d stocks (max_pages=%d, sleep=%.1fs)",
                len(codes), args.max_pages, args.sleep)

    total_articles = 0
    success, fail, empty = 0, 0, 0

    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

        try:
            df = news_source.fetch_news(
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

        # PIT-align → Silver
        silver = news_storage.bronze_to_silver(code)
        if not silver.empty:
            news_storage.save_silver_news(code, silver)

        # Daily aggregation → Gold
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
