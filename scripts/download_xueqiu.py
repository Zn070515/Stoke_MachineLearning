"""Download Xueqiu forum posts for A-share stocks, compute FinBERT sentiment.

Xueqiu is a social investing platform with user discussions similar to Guba.
Each stock returns up to 1000 posts (20 items/page × 50 pages) via the
internal status API.

Uses Playwright for WAF bypass (aliyun_waf_oo JS challenge).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --max-pages 50
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --workers 2
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.xueqiu_storage import XueqiuStorage
from stoke_ml.data.sources.a_shares.xueqiu_source import XueqiuNewsSource
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

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


def process_stock(code, start_date, end_date, max_pages, storage, analyzer):
    """Fetch Xueqiu posts for one stock, compute sentiment, save bronze/silver/gold.

    Returns (code, post_count).
    """
    source = XueqiuNewsSource()
    try:
        df = source.fetch_news(code, start_date, end_date, max_pages)
    except Exception as e:
        logger.error("%s: fetch failed — %s", code, e)
        return code, 0

    if df.empty:
        return code, 0

    df = compute_raw_sentiment(df, analyzer)
    storage.save_raw(code, df)

    silver = storage.bronze_to_silver(code)
    if not silver.empty:
        storage.save_silver(code, silver)

    gold = storage.silver_to_gold(code, analyzer)
    if not gold.empty:
        storage.save_daily_sentiment(gold)
        post_days = gold["has_xueqiu_post"].sum()
        logger.info(
            "%s: %d posts, %d sentiment days (%d with posts)",
            code, len(df), len(gold), post_days,
        )

    return code, len(df)


def main():
    parser = argparse.ArgumentParser(description="Download A-share Xueqiu forum posts")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Pages per stock, 20 posts/page, max 50 (default: 20)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel stocks (default: 1, Playwright is thread-bound)")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between stocks (default: 2.0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    end_date = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    max_pages = min(args.max_pages, 50)

    storage = XueqiuStorage(data_dir)
    analyzer = NewsSentimentAnalyzer()

    logger.info(
        "Downloading Xueqiu posts for %d stocks (%s to %s, %d pages/stock)",
        len(codes), args.start, end_date, max_pages,
    )

    total_posts = 0
    for i, code in enumerate(codes):
        logger.info("[%d/%d] Processing %s", i + 1, len(codes), code)
        try:
            _, count = process_stock(
                code, args.start, end_date, max_pages, storage, analyzer,
            )
            total_posts += count
        except Exception as e:
            logger.error("%s: failed — %s", code, e)

        if args.sleep > 0 and i < len(codes) - 1:
            time.sleep(args.sleep)

    logger.info("Done: %d total posts across %d stocks", total_posts, len(codes))


if __name__ == "__main__":
    main()
