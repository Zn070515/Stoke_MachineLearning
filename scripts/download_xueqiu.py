"""Download Xueqiu forum posts for A-share stocks with FinBERT sentiment.

Uses the existing XueqiuNewsSource (Playwright browser per stock, handles WAF).
Auto-skips already-downloaded stocks so interrupted runs can resume.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --max-pages 50
"""
import argparse
import logging
import os
import signal
import sys
import time
import traceback

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.xueqiu_storage import XueqiuStorage
from stoke_ml.data.sources.a_shares.xueqiu_source import XueqiuNewsSource
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logging.getLogger().handlers[0].setLevel(logging.INFO)
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


def main():
    parser = argparse.ArgumentParser(description="Download Xueqiu forum posts")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages per stock, 20 posts/page (default: 10)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between stocks (default: 1.0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    # Skip already-downloaded stocks
    xueqiu_dir = os.path.join(data_dir, "a_shares", "xueqiu_raw")
    if os.path.exists(xueqiu_dir):
        done = {f.replace(".parquet", "") for f in os.listdir(xueqiu_dir)
                if f.endswith(".parquet")}
        remaining = [c for c in codes if c not in done]
        skipped = len(codes) - len(remaining)
        if skipped:
            logger.info("Skip %d already-downloaded, %d remaining", skipped, len(remaining))
    else:
        remaining = codes

    if not remaining:
        logger.info("All %d stocks already downloaded", len(codes))
        return

    end_date = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    max_pages = min(args.max_pages, 50)

    storage = XueqiuStorage(data_dir)
    analyzer = NewsSentimentAnalyzer()
    source = XueqiuNewsSource()

    total_posts = 0
    total_errors = 0
    start_time = time.time()

    for i, code in enumerate(remaining):
        sys.stdout.flush()
        t0 = time.time()

        try:
            df = source.fetch_news(code, args.start, end_date, max_pages)
        except Exception as e:
            total_errors += 1
            logger.error("[%d/%d] %s: fetch crashed — %s",
                         i + 1, len(remaining), code, e)
            if total_errors > 5:
                logger.error("Too many errors (%d), stopping", total_errors)
                break
            time.sleep(args.sleep * 2)
            source = XueqiuNewsSource()  # fresh source after error
            continue

        elapsed = time.time() - t0

        if df.empty:
            if (i + 1) % 20 == 0:
                logger.info("[%d/%d] %s: 0 posts (%.1fs)",
                            i + 1, len(remaining), code, elapsed)
            continue

        try:
            df = compute_raw_sentiment(df, analyzer)
            storage.save_raw(code, df)

            silver = storage.bronze_to_silver(code)
            if not silver.empty:
                storage.save_silver(code, silver)

            gold = storage.silver_to_gold(code, analyzer)
            post_days = gold["has_xueqiu_post"].sum() if not gold.empty else 0
            if not gold.empty:
                storage.save_daily_sentiment(gold)
        except Exception as e:
            total_errors += 1
            logger.error("[%d/%d] %s: save failed — %s",
                         i + 1, len(remaining), code, e)
            continue

        total_posts += len(df)

        if (i + 1) % 10 == 0 or elapsed > 30:
            eta = (time.time() - start_time) / (i + 1) * (len(remaining) - i - 1)
            logger.info(
                "[%d/%d] %s: %d posts, %d sent days (%.1fs, ETA %.0f min)",
                i + 1, len(remaining), code, len(df), post_days,
                elapsed, eta / 60,
            )

        if args.sleep > 0 and i < len(remaining) - 1:
            time.sleep(args.sleep)

    elapsed = time.time() - start_time
    logger.info(
        "Done: %d posts across %d stocks in %.1f min (%.1f sec/stock, %d errors)",
        total_posts, len(remaining), elapsed / 60,
        elapsed / len(remaining) if remaining else 0, total_errors,
    )


if __name__ == "__main__":
    main()
