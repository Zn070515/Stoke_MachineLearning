"""Download company announcements for all stocks and compute daily sentiment.

Usage:
  python scripts/production/download_announcements.py                              # all stocks
  python scripts/production/download_announcements.py --shard 0/4                  # process 1/4 of stocks
  python scripts/production/download_announcements.py --stocks 600519              # specific stocks
  python scripts/production/download_announcements.py --skip-sentiment             # raw only
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.announcement_storage import AnnouncementStorage
from stoke_ml.data.download_resume import (
    evidence_says_complete,
    mark_stock_result,
    skip_completed_stocks,
)
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
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-download all stocks (ignore existing files)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec: k/N (e.g. 0/4 processes first quarter of stocks)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    storage = AnnouncementStorage(cfg.project.data_dir)
    source = AnnouncementSource()

    codes = (args.stocks.split(",") if args.stocks
             else available_stocks(cfg.project.data_dir))

    if not codes:
        logger.error("No stocks found")
        sys.exit(1)

    # Apply shard filter
    shard_label = ""
    if args.shard:
        k, n = args.shard.split("/")
        k, n = int(k), int(n)
        codes = [c for i, c in enumerate(codes) if i % n == k]
        shard_label = f" [shard {k}/{n}]"

    end_date = args.end or time.strftime("%Y-%m-%d")

    # Resume: skip stocks whose raw data already covers start_date
    raw_dir = os.path.join(cfg.project.data_dir, "a_shares", "announcements")
    if not args.no_resume:
        codes, _n_skipped = skip_completed_stocks(
            raw_dir, codes, start_date=args.start,
        )
    elif not os.path.isdir(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    if not codes:
        logger.info("All stocks already downloaded. Nothing to do.")
        sys.exit(0)

    logger.info("Downloading announcements for %d stocks (%s to %s)%s",
                len(codes), args.start, end_date, shard_label)

    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer(force_lexicon=True)

    success = 0
    partial = 0
    empty = 0
    fail = 0

    def _mark(code: str, df: pd.DataFrame, meta: dict) -> None:
        """Write the per-stock manifest with pagination evidence."""
        mark_stock_result(
            raw_dir, code, df, dataset="announcements",
            requested_start=args.start, requested_end=end_date,
            expected_rows=meta.get("expected_rows"),
            pages_fetched=meta.get("pages_fetched"),
            pagination_exhausted=meta.get("pagination_exhausted"),
        )

    for i, code in enumerate(codes):
        meta: dict = {}
        try:
            df = source.fetch_announcements(code, args.start, end_date, meta=meta)
            if df.empty:
                _mark(code, df, meta)
                empty += 1
                if (i + 1) % 250 == 0:
                    logger.info("[%d/%d] %s: 0 announcements", i + 1, len(codes), code)
                continue

            df["stock_code"] = code

            if analyzer is not None and len(df) > 0:
                df = compute_raw_sentiment(df, analyzer)

            storage.save_raw(code, df)
            _mark(code, df, meta)
            storage.build_daily_sentiment(code)
            if evidence_says_complete(
                df,
                expected_rows=meta.get("expected_rows"),
                pagination_exhausted=meta.get("pagination_exhausted"),
            ):
                success += 1
            else:
                partial += 1

            if success % 250 == 0:
                logger.info("[%d/%d] %s: %d announcements saved",
                            i + 1, len(codes), code, len(df))

        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)
            fail += 1

    logger.info("Done%s: %d complete, %d partial, %d empty, %d fail / %d stocks",
                shard_label, success, partial, empty, fail, len(codes))

    # Exit code reflects the outcome — 0 all good, 1 fetch failures,
    # 2 any partial/degraded (truncated or empty) result.
    if fail:
        sys.exit(1)
    if partial or empty:
        sys.exit(2)


if __name__ == "__main__":
    main()
