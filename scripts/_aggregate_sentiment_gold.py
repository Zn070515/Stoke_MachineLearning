"""One-shot: aggregate news silver → gold for a list of stocks.
Usage: PYTHONPATH=. ./.venv/Scripts/python scripts/_aggregate_sentiment_gold.py <shard_file>
shard_file: comma-separated stock codes
"""
import logging
import os
import sys

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.news_storage import NewsStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    shard_file = sys.argv[1]
    with open(shard_file) as f:
        codes = [c.strip() for c in f.read().split(",") if c.strip()]

    cfg = load_config()
    data_dir = cfg.project.data_dir
    calendar = TradingCalendar("a_shares")
    storage = NewsStorage(data_dir, calendar)

    logger.info("Aggregating silver→gold for %d stocks", len(codes))
    success, empty, fail = 0, 0, 0
    for i, code in enumerate(codes):
        try:
            gold = storage.silver_to_gold(code)
            if gold.empty:
                empty += 1
            else:
                storage.save_daily_sentiment(gold)
                success += 1
        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)
            fail += 1

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (ok=%d, empty=%d, fail=%d)",
                        i + 1, len(codes), success, empty, fail)

    logger.info("Done: %d success, %d empty, %d fail", success, empty, fail)


if __name__ == "__main__":
    main()
