"""Build Gold (daily sentiment) from existing Silver news data.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_news_gold.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_news_gold.py --stocks 100
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build news Gold from Silver")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=None,
                        help="Limit to first N stocks (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    silver_dir = os.path.join(data_dir, "a_shares", "news_silver")

    if not os.path.isdir(silver_dir):
        logger.error("Silver directory not found: %s", silver_dir)
        sys.exit(1)

    silver_files = sorted(
        f for f in os.listdir(silver_dir) if f.endswith(".parquet")
    )
    if args.stocks:
        silver_files = silver_files[:args.stocks]

    logger.info("Building Gold for %d stocks from Silver", len(silver_files))

    calendar = TradingCalendar("a_shares")
    storage = NewsStorage(data_dir, calendar)
    analyzer = NewsSentimentAnalyzer(force_lexicon=True)

    total_days = 0
    total_news_days = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    for i, fname in enumerate(silver_files):
        code = fname.replace(".parquet", "")
        try:
            silver = storage.load_silver_news(code)
            if silver.empty:
                skipped += 1
                continue

            gold = storage.silver_to_gold(code, analyzer)
            if gold.empty:
                skipped += 1
                continue

            storage.save_daily_sentiment(gold)
            news_days = int(gold["has_news"].sum()) if "has_news" in gold.columns else 0
            total_days += len(gold)
            total_news_days += news_days

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(silver_files) - i - 1) / rate
                logger.info(
                    "[%d/%d] %s: %d gold days (%d news), %.1f stock/s, ETA %.0f min",
                    i + 1, len(silver_files), code, len(gold), news_days,
                    rate, eta / 60,
                )

        except Exception as e:
            errors += 1
            logger.error("[%d/%d] %s: %s", i + 1, len(silver_files), code, e)
            if errors > 20:
                logger.error("Too many errors, stopping")
                break

    elapsed = time.time() - start_time
    logger.info(
        "Done: %d stocks, %d gold days (%d with news), %d errors, %d skipped, %.1f min",
        len(silver_files) - errors - skipped, total_days, total_news_days,
        errors, skipped, elapsed / 60,
    )


if __name__ == "__main__":
    main()
