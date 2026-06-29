"""Download Guba forum posts for A-share stocks, compute FinBERT sentiment.

Usage:
  python scripts/download_guba.py                              # all stocks
  python scripts/download_guba.py --stocks 000001,600519       # specific stocks
  python scripts/download_guba.py --max-pages 5 --sleep 1      # deeper, faster
  python scripts/download_guba.py --concurrent                 # parallel download
  python scripts/download_guba.py --skip-sentiment             # raw + PIT only
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.sources.a_shares.guba_source import GubaSource
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
    parser = argparse.ArgumentParser(description="Download A-share Guba forum posts")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: auto)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date filter YYYY-MM-DD (default: 2015-01-01)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter YYYY-MM-DD (default: today)")
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages per stock, ~80 posts/page (default: 10)")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Seconds between stocks (default: from config)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip FinBERT sentiment computation")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.sleep is None:
        args.sleep = float(cfg.crawler.rate_limit.base_delay_sec)

    if args.end is None:
        args.end = time.strftime("%Y-%m-%d")

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found. Run download_data.py first.")
        sys.exit(1)

    calendar = TradingCalendar("a_shares")
    guba_storage = GubaStorage(data_dir, calendar)
    guba_source = GubaSource()
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer()

    mode_label = "concurrent" if args.concurrent else "sequential"
    logger.info(
        "Downloading Guba posts for %d stocks (%s to %s, max_pages=%d, sleep=%.1fs, %s)",
        len(codes), args.start, args.end, args.max_pages, args.sleep, mode_label,
    )

    total_posts = 0
    success, fail, empty, skipped = 0, 0, 0, 0

    # Resume: skip stocks already on disk
    pending = []
    for code in codes:
        raw_path = os.path.join(data_dir, "a_shares", "guba_raw", f"{code}.parquet")
        if os.path.exists(raw_path):
            skipped += 1
        else:
            pending.append(code)
    if skipped:
        logger.info("Skipping %d stocks already on disk, %d remaining", skipped, len(pending))
    codes = pending

    # Shared save function used by both paths
    def _save_stock(code: str, df: pd.DataFrame) -> int:
        """Save one stock through the full medallion pipeline. Returns post count."""
        guba_storage.save_raw(code, df)
        post_count = len(df)
        silver = guba_storage.bronze_to_silver(code)
        if not silver.empty:
            guba_storage.save_silver(code, silver)
        if not args.skip_sentiment:
            gold = guba_storage.silver_to_gold(code, analyzer)
            if not gold.empty:
                guba_storage.save_daily_sentiment(gold)
                post_days = gold["has_guba_post"].sum()
                logger.info("  %s: %d posts saved, %d sentiment days (%d with posts)",
                            code, post_count, len(gold), post_days)
        else:
            logger.info("  %s: %d posts saved (raw)", code, post_count)
        return post_count

    if args.concurrent:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from stoke_ml.crawler.rate_limiter import RateLimiter

        rate_limiter = RateLimiter(
            base_delay_sec=0,
            daily_quota=cfg.crawler.rate_limit.daily_quota_per_domain,
        )
        lock = threading.Lock()
        completed = 0

        def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
            try:
                df = guba_source.fetch_posts(
                    code,
                    start_date=args.start,
                    end_date=args.end,
                    max_pages=args.max_pages,
                    fetch_bodies=True,
                )
                if not args.skip_sentiment and not df.empty:
                    df = compute_raw_sentiment(df, analyzer)
                return code, df, None
            except Exception as e:
                return code, None, str(e)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in codes}
            for future in as_completed(futures):
                code, df, err = future.result()
                with lock:
                    completed += 1
                    logger.info("[%d/%d] %s ...", completed, len(codes) + skipped, code)

                if err:
                    logger.error("  %s: fetch failed: %s", code, err)
                    fail += 1
                    continue

                if df is None or df.empty:
                    logger.info("  %s: no posts found", code)
                    empty += 1
                    continue

                total_posts += _save_stock(code, df)
                success += 1
    else:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.sleep)

            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

            try:
                df = guba_source.fetch_posts(
                    code,
                    start_date=args.start,
                    end_date=args.end,
                    max_pages=args.max_pages,
                    fetch_bodies=True,
                )
            except Exception as e:
                logger.error("  %s: fetch failed: %s", code, e)
                fail += 1
                continue

            if df.empty:
                logger.info("  %s: no posts found", code)
                empty += 1
                continue

            # Compute sentiment on titles + bodies
            if not args.skip_sentiment:
                df = compute_raw_sentiment(df, analyzer)

            # Save raw (Bronze)
            guba_storage.save_raw(code, df)
            logger.info("  %s: %d posts saved (raw)", code, len(df))
            total_posts += len(df)

            # PIT-align -> Silver
            silver = guba_storage.bronze_to_silver(code)
            if not silver.empty:
                guba_storage.save_silver(code, silver)

            # Daily aggregation -> Gold
            if not args.skip_sentiment:
                gold = guba_storage.silver_to_gold(code, analyzer)
                if not gold.empty:
                    guba_storage.save_daily_sentiment(gold)
                    post_days = gold["has_guba_post"].sum()
                    logger.info("  %s: %d sentiment days (%d with posts)",
                                code, len(gold), post_days)

            success += 1

    logger.info("Done: %d success, %d fail, %d empty, %d skipped, %d total posts",
                success, fail, empty, skipped, total_posts)


if __name__ == "__main__":
    main()
