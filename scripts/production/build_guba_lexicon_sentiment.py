"""One-shot: compute lexicon sentiment on Guba Silver posts → Gold daily aggregation.

Reads all saved Silver parquet files, applies financial lexicon sentiment
on titles (and bodies when available), then aggregates to daily Gold.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_guba_lexicon_sentiment.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_guba_lexicon_sentiment.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_guba_lexicon_sentiment.py --shard 0/8
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.features.news_nlp import NewsSentimentAnalyzer, compute_raw_sentiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def available_stocks(data_dir: str) -> list[str]:
    silver_dir = os.path.join(data_dir, "a_shares", "guba_silver")
    if not os.path.isdir(silver_dir):
        return []
    return sorted(
        f.replace(".parquet", "")
        for f in os.listdir(silver_dir)
        if f.endswith(".parquet")
    )


def main():
    parser = argparse.ArgumentParser(description="Build Guba lexicon sentiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None)
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if Gold already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    gs = GubaStorage(data_dir)

    codes = (args.stocks.split(",") if args.stocks
             else available_stocks(data_dir))

    if not codes:
        logger.error("No Guba silver files found")
        sys.exit(1)

    if args.shard:
        k, n = args.shard.split("/")
        k, n = int(k), int(n)
        codes = [c for i, c in enumerate(codes) if i % n == k]
        logger.info("Shard %s/%s: %d stocks", k, n, len(codes))

    # Skip stocks that already have Gold (unless --force)
    if not args.force:
        gold_base = os.path.join(data_dir, "a_shares", "guba_sentiment")
        remaining = []
        for code in codes:
            gold_path = os.path.join(gold_base, f"{code}.parquet")
            if not os.path.isfile(gold_path):
                remaining.append(code)
        if len(remaining) < len(codes):
            logger.info("Skipping %d stocks with existing Gold, %d remaining",
                       len(codes) - len(remaining), len(remaining))
        codes = remaining

    if not codes:
        logger.info("All stocks already have Gold sentiment. Nothing to do.")
        sys.exit(0)

    logger.info("Computing lexicon sentiment for %d stocks", len(codes))
    analyzer = NewsSentimentAnalyzer(force_lexicon=True)

    success = 0
    total_posts = 0
    for i, code in enumerate(codes):
        try:
            silver = gs.load_silver(code)
            if silver.empty:
                continue

            n_posts = len(silver)
            total_posts += n_posts

            # Compute lexicon sentiment on titles (and bodies if available)
            if "sentiment_title" not in silver.columns:
                silver = compute_raw_sentiment(silver, analyzer)

            gold = gs.silver_to_gold(code, analyzer)
            if not gold.empty:
                gs.save_daily_sentiment(gold)
                post_days = gold["has_guba_post"].sum()
                success += 1
                if success % 100 == 0:
                    logger.info("[%d/%d] %s: %d posts → %d sentiment days (%d with posts)",
                               i + 1, len(codes), code, n_posts, len(gold), post_days)
        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)

    logger.info("Done: %d/%d stocks, %d total posts processed", success, len(codes), total_posts)


if __name__ == "__main__":
    main()
