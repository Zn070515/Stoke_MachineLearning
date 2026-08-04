"""Download news for A-share stocks from multiple sources, compute sentiment.

Usage:
  python scripts/download_news.py                              # all stocks, all sources
  python scripts/download_news.py --stocks 000001,600519       # specific stocks
  python scripts/download_news.py --source sina                # single source
  python scripts/download_news.py --source ths,sina       # selected sources
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
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.download_resume import (
    evidence_says_complete,
    mark_stock_result,
    skip_completed_stocks,
)
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.sources.a_shares.news_pipeline import NewsPipeline
from stoke_ml.data.storage import DataStorage
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
                        help="News source(s): sina, ths, all (default: all)")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Pages per stock per source (default: 20)")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Seconds between stocks (default: from config)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation (raw only)")
    parser.add_argument("--raw-only", action="store_true",
                        help="Only download and save raw Bronze, skip Silver/Gold")
    parser.add_argument("--no-bodies", action="store_true",
                        help="Skip article body fetching (defer to post-processing)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter YYYY-MM-DD")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-download all stocks (ignore existing files)")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based) for parallel download")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (required with --shard)")
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

    if (args.shard is not None) != (args.num_shards is not None):
        parser.error("--shard and --num-shards must be used together")
    if args.shard is not None and not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards})")

    shard_tag = f"_shard{args.shard}" if args.shard is not None else ""

    if args.shard is not None:
        total = len(codes)
        size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * size
        end = min(start + size, total)
        codes = codes[start:end]
        if not codes:
            logger.error("Shard %d: no stocks in range [%d:%d]", args.shard, start, end)
            sys.exit(1)
        logger.info("Shard %d/%d: %d stocks [%d:%d]", args.shard, args.num_shards, len(codes), start, end)

    # Resume: skip stocks whose raw data already covers start_date
    raw_dir = os.path.join(data_dir, "a_shares", "news_raw")
    if not args.no_resume:
        codes, _n_skipped = skip_completed_stocks(
            raw_dir, codes, start_date=args.start,
        )
    elif not os.path.isdir(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    if not codes:
        logger.info("All stocks already downloaded. Nothing to do.")
        sys.exit(0)

    # Select sources
    if args.source == "all":
        active_sources = None  # pipeline uses all available
    else:
        active_sources = [s.strip() for s in args.source.split(",")]

    calendar = TradingCalendar("a_shares")
    news_storage = NewsStorage(data_dir, calendar)
    news_pipeline = NewsPipeline(active_sources=active_sources)
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer(force_lexicon=True)

    source_label = args.source if args.source != "all" else "sina+ths"
    mode_label = "concurrent" if args.concurrent else "sequential"
    logger.info(
        "Downloading news for %d stocks (sources=%s, max_pages=%d, sleep=%.1fs, %s)",
        len(codes), source_label, args.max_pages, args.sleep, mode_label,
    )

    total_articles = 0
    success, fail, empty, partial = 0, 0, 0, 0

    def _mark(code: str, df: pd.DataFrame, meta: dict) -> None:
        """Write the per-stock manifest with pagination evidence."""
        mark_stock_result(
            raw_dir, code, df, dataset="news_raw",
            requested_start=args.start, requested_end=args.end,
            source=source_label,
            pages_requested=meta.get("pages_requested"),
            pages_fetched=meta.get("pages_fetched"),
            pagination_exhausted=meta.get("pagination_exhausted"),
        )

    def _classify(df: pd.DataFrame, meta: dict) -> str:
        """complete / partial / degraded — mirrors mark_stock_result's decision."""
        if df is None or df.empty:
            return "degraded"
        if evidence_says_complete(
            df, pagination_exhausted=meta.get("pagination_exhausted"),
        ):
            return "complete"
        return "partial"

    if args.concurrent:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from stoke_ml.crawler.rate_limiter import RateLimiter

        rate_limiter = RateLimiter(
            base_delay_sec=0,
            daily_quota=cfg.crawler.rate_limit.daily_quota_per_domain,
        )

        def _fetch_one(code: str) -> tuple[pd.DataFrame, dict]:
            meta: dict = {}
            df = news_pipeline.fetch_all_news(
                code,
                start_date=args.start,
                end_date=args.end,
                max_pages=args.max_pages,
                fetch_bodies=not args.no_bodies,
                meta=meta,
            )
            if not args.skip_sentiment and not df.empty:
                df = compute_raw_sentiment(df, analyzer)
            return df, meta

        def _worker(code: str):
            rate_limiter.wait()
            return code, _fetch_one(code)

        # Stream: save each stock as its fetch completes, so a mid-run
        # crash keeps everything already downloaded (no all-or-nothing buffer).
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, code): code for code in codes}
            done = 0
            for future in as_completed(futures):
                code = futures[future]
                done += 1
                logger.info("[%d/%d] %s ...", done, len(codes), code)
                try:
                    _code, (df, meta) = future.result()
                except Exception as e:
                    logger.error("  %s: fetch failed: %s", code, e)
                    fail += 1
                    continue

                if df is None:
                    logger.error("  %s: fetch failed (no data)", code)
                    fail += 1
                    continue

                outcome = _classify(df, meta)
                if outcome == "degraded":
                    logger.info("  %s: no news found", code)
                    _mark(code, df, meta)
                    empty += 1
                    continue

                # Save raw (Bronze)
                news_storage.save_raw_news(code, df)
                _mark(code, df, meta)
                logger.info("  %s: %d articles saved (raw)", code, len(df))
                total_articles += len(df)

                if not args.raw_only:
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

                if outcome == "complete":
                    success += 1
                else:
                    partial += 1
    else:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.sleep)

            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

            meta: dict = {}
            try:
                df = news_pipeline.fetch_all_news(
                    code,
                    start_date=args.start,
                    end_date=args.end,
                    max_pages=args.max_pages,
                    fetch_bodies=not args.no_bodies,
                    meta=meta,
                )
            except Exception as e:
                logger.error("  %s: fetch failed: %s", code, e)
                fail += 1
                continue

            outcome = _classify(df, meta)
            if outcome == "degraded":
                logger.info("  %s: no news found", code)
                _mark(code, df, meta)
                empty += 1
                continue

            # Compute sentiment on titles
            if not args.skip_sentiment and not args.raw_only:
                df = compute_raw_sentiment(df, analyzer)

            # Save raw (Bronze)
            news_storage.save_raw_news(code, df)
            _mark(code, df, meta)
            logger.info("  %s: %d articles saved (raw)", code, len(df))
            total_articles += len(df)

            if not args.raw_only:
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

            if outcome == "complete":
                success += 1
            else:
                partial += 1

    logger.info(
        "Done: %d complete, %d partial, %d empty, %d fail, %d total articles",
        success, partial, empty, fail, total_articles,
    )

    # Exit code reflects the outcome — 0 when every result is complete
    # or skipped, 1 on fetch failures, 2 when any result is partial/degraded.
    if fail:
        sys.exit(1)
    if partial or empty:
        sys.exit(2)


if __name__ == "__main__":
    main()
