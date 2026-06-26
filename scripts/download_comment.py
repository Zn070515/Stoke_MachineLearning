"""Download AKShare market comment sentiment for all A-share stocks.

Two modes:
  snapshot — single call gets all 5184 stocks (current day only)
  history — per-stock 30-day detail via stock_comment_detail_zhpj_lspf_em

Usage:
  python scripts/download_comment.py                     # snapshot only
  python scripts/download_comment.py --history           # snapshot + per-stock 30d
  python scripts/download_comment.py --stocks 600519     # specific stocks
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.comment_storage import CommentStorage
from stoke_ml.data.sources.a_shares.comment_source import CommentSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stocks_from_disk(data_dir: str) -> list[str]:
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
    parser = argparse.ArgumentParser(
        description="Download market comment sentiment from AKShare"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--history", action="store_true",
                        help="Also download per-stock 30-day history")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between per-stock API calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    storage = CommentStorage(data_dir)
    source = CommentSource()

    # Step 1: Full-market snapshot (always)
    logger.info("Downloading full-market comment snapshot...")
    try:
        snapshot = source.fetch_all_snapshot()
        if not snapshot.empty:
            storage.save_snapshot(snapshot)
            logger.info(
                "Snapshot saved: %d stocks, score range [%.1f, %.1f]",
                len(snapshot),
                snapshot["comment_score"].min(),
                snapshot["comment_score"].max(),
            )
    except Exception as e:
        logger.error("Snapshot failed: %s", e)
        if not args.history:
            sys.exit(1)

    # Step 2: Per-stock 30-day history (optional)
    if not args.history:
        logger.info("Done (snapshot only). Use --history for per-stock daily data.")
        return

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found.")
        sys.exit(1)

    logger.info("Downloading 30-day comment history for %d stocks...", len(codes))
    success = 0
    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        try:
            df = source.fetch_stock_history(code)
            if df.empty:
                logger.debug("[%d/%d] %s: no data", i + 1, len(codes), code)
                continue
            storage.save_daily(df)
            success += 1
            if success % 100 == 0:
                logger.info("[%d/%d] %s: %d days", i + 1, len(codes), code, len(df))
        except Exception as e:
            logger.debug("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)

    logger.info("Done: %d/%d stocks with comment history", success, len(codes))


if __name__ == "__main__":
    main()
