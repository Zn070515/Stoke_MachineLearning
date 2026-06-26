"""Download company announcements for all stocks and compute daily sentiment."""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.announcement_storage import AnnouncementStorage
from stoke_ml.data.sources.a_shares.announcement_source import AnnouncementSource
from stoke_ml.features.news_nlp import compute_raw_sentiment, NewsSentimentAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def available_stocks(data_dir: str) -> list[str]:
    base = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description="Download company announcements")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between API calls")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    storage = AnnouncementStorage(cfg.project.data_dir)
    source = AnnouncementSource()

    codes = (args.stocks.split(",") if args.stocks
             else available_stocks(cfg.project.data_dir))

    if not codes:
        logger.error("No stocks found")
        sys.exit(1)

    end_date = args.end or time.strftime("%Y-%m-%d")
    logger.info("Downloading announcements for %d stocks (%s to %s)",
                len(codes), args.start, end_date)

    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer()

    success = 0
    for i, code in enumerate(codes):
        try:
            logger.info("[%d/%d] %s", i + 1, len(codes), code)
            df = source.fetch_announcements(code, args.start, end_date)
            if df.empty:
                logger.info("  %s: 0 announcements, skipping", code)
                continue

            df["stock_code"] = code

            if analyzer is not None and len(df) > 0:
                df = compute_raw_sentiment(df, analyzer)

            storage.save_raw(code, df)
            storage.build_daily_sentiment(code)
            success += 1
            logger.info("  %s: %d announcements saved", code, len(df))

        except Exception as e:
            logger.error("  %s: ERROR %s", code, e)

        time.sleep(args.sleep)

    logger.info("Done: %d/%d stocks with announcements", success, len(codes))


if __name__ == "__main__":
    main()
