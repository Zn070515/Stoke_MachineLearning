"""Download Xueqiu forum posts for A-share stocks with FinBERT sentiment.

Per-stock hard timeout via ProcessPoolExecutor — each stock runs in its own
process to avoid Playwright greenlet conflicts, with a 120s kill switch.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_xueqiu.py --max-pages 50
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd

from stoke_ml.config import load_config, get_project_root
from stoke_ml.data.xueqiu_storage import XueqiuStorage
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

PER_STOCK_TIMEOUT = 120  # seconds — kill worker if stock takes longer


def _fetch_one_stock(args_tuple: tuple) -> dict:
    """Run in a child process — has its own main thread, Playwright-safe."""
    code, start, end, max_pages = args_tuple
    from stoke_ml.data.sources.a_shares.xueqiu_source import XueqiuNewsSource
    source = XueqiuNewsSource()
    df = source.fetch_news(code, start, end, max_pages)
    return {"code": code, "df": df}


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
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip FinBERT sentiment (raw + PIT only, no Gold)")
    parser.add_argument("--per-stock-timeout", type=int, default=PER_STOCK_TIMEOUT,
                        help=f"Seconds before killing a stuck stock (default: {PER_STOCK_TIMEOUT})")
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
    per_stock_timeout = args.per_stock_timeout

    storage = XueqiuStorage(data_dir)
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer(force_lexicon=True)

    total_posts = 0
    total_errors = 0
    total_timeouts = 0
    start_time = time.time()

    for i, code in enumerate(remaining):
        sys.stdout.flush()
        t0 = time.time()

        # Run fetch in a child process to get hard timeout without Playwright greenlet issues
        df = pd.DataFrame()
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _fetch_one_stock, (code, args.start, end_date, max_pages)
                )
                result = future.result(timeout=per_stock_timeout)
                df = result["df"]
        except FutureTimeoutError:
            total_timeouts += 1
            logger.warning("[%d/%d] %s: TIMEOUT after %ds, skipping",
                           i + 1, len(remaining), code, per_stock_timeout)
            continue
        except Exception as e:
            total_errors += 1
            logger.error("[%d/%d] %s: fetch crashed — %s",
                         i + 1, len(remaining), code, e)
            if total_errors > 5:
                logger.error("Too many errors (%d), stopping", total_errors)
                break
            time.sleep(args.sleep * 2)
            continue

        elapsed = time.time() - t0

        if df.empty:
            if (i + 1) % 20 == 0:
                logger.info("[%d/%d] %s: 0 posts (%.1fs)",
                            i + 1, len(remaining), code, elapsed)
            continue

        try:
            if not args.skip_sentiment and analyzer is not None:
                df = compute_raw_sentiment(df, analyzer)
            storage.save_raw(code, df)

            if not args.skip_sentiment:
                silver = storage.bronze_to_silver(code)
                if not silver.empty:
                    storage.save_silver(code, silver)

                gold = storage.silver_to_gold(code, analyzer)
                post_days = gold["has_xueqiu_post"].sum() if not gold.empty else 0
                if not gold.empty:
                    storage.save_daily_sentiment(gold)
            else:
                post_days = 0
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
        "Done: %d posts across %d stocks in %.1f min (%.1f sec/stock, %d errors, %d timeouts)",
        total_posts, len(remaining), elapsed / 60,
        elapsed / len(remaining) if remaining else 0, total_errors, total_timeouts,
    )


if __name__ == "__main__":
    main()
