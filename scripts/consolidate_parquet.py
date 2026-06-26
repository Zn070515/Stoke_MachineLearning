"""Consolidate fragmented parquet files into one file per stock per category.

Current layout:
  {data_type}/{year}/{month}/{stock_code}.parquet  (500K+ tiny files)

After consolidation:
  {data_type}_merged/{stock_code}.parquet  (one file per stock, all history)

Skips news_raw, news_silver, sentiment (news download in progress).
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = ["daily", "dragon_tiger", "margin", "northbound", "fundamentals", "etf_flow"]


def collect_files(base_dir: str, category: str) -> dict[str, list[str]]:
    """Group all parquet files by stock_code.

    Returns: {stock_code: [file_path, ...]}
    """
    cat_dir = os.path.join(base_dir, category)
    if not os.path.exists(cat_dir):
        return {}
    stocks: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(cat_dir):
        for f in files:
            if f.endswith(".parquet"):
                code = f.replace(".parquet", "")
                stocks.setdefault(code, []).append(os.path.join(root, f))
    return stocks


def consolidate_category(base_dir: str, category: str, min_file_size_kb: float = 0) -> dict:
    """Merge all year/month/stock parquet into one file per stock.

    Args:
        base_dir: Path to a_shares data directory.
        category: Data type name (daily, margin, etc.).
        min_file_size_kb: Skip files smaller than this (for filtering empties).

    Returns:
        {stock_code: (old_count, new_size_bytes)}
    """
    cat_dir = os.path.join(base_dir, category)
    if not os.path.exists(cat_dir):
        logger.warning("Category %s not found at %s", category, cat_dir)
        return {}

    out_dir = os.path.join(base_dir, f"{category}_merged")
    os.makedirs(out_dir, exist_ok=True)

    stocks = collect_files(base_dir, category)
    if not stocks:
        logger.warning("No parquet files found for %s", category)
        return {}

    logger.info("%s: %d stocks, %d total files", category, len(stocks), sum(len(v) for v in stocks.values()))

    stats = {}
    failed = 0
    for code, files in tqdm(stocks.items(), desc=f"Consolidating {category}", unit="stock"):
        try:
            frames = []
            for fp in files:
                if min_file_size_kb > 0:
                    size_kb = os.path.getsize(fp) / 1024
                    if size_kb < min_file_size_kb:
                        continue
                df = pd.read_parquet(fp)
                frames.append(df)

            if not frames:
                continue

            merged = pd.concat(frames, ignore_index=True)
            if "date" in merged.columns:
                merged["date"] = pd.to_datetime(merged["date"])
                merged = merged.drop_duplicates()
                merged = merged.sort_values("date")

            out_path = os.path.join(out_dir, f"{code}.parquet")
            merged.to_parquet(out_path, index=False)
            stats[code] = (len(files), os.path.getsize(out_path))
        except Exception as e:
            logger.warning("Failed to consolidate %s/%s: %s", category, code, e)
            failed += 1

    logger.info("%s: %d stocks consolidated, %d failed", category, len(stats), failed)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Consolidate fragmented parquet files")
    parser.add_argument("--data-dir", type=str,
                        default="C:/Users/16275/Desktop/Stoke_MachineLearning/data/a_shares")
    parser.add_argument("--categories", type=str, nargs="*",
                        default=CATEGORIES,
                        help="Categories to consolidate (default: all except news)")
    parser.add_argument("--category", type=str, default=None,
                        help="Single category to consolidate")
    args = parser.parse_args()

    categories = [args.category] if args.category else args.categories

    total_old_files = 0
    total_stocks = 0
    for cat in categories:
        t0 = time.time()
        stats = consolidate_category(args.data_dir, cat)
        elapsed = time.time() - t0
        if stats:
            old_count = sum(v[0] for v in stats.values())
            new_size_mb = sum(v[1] for v in stats.values()) / 1024 / 1024
            logger.info("  %s: %d stocks -> %d files merged in %.1f min, %.1f MB",
                        cat, len(stats), old_count, elapsed / 60, new_size_mb)
            total_old_files += old_count
            total_stocks += len(stats)

    logger.info("Done: %d stocks consolidated, %d files merged", total_stocks, total_old_files)


if __name__ == "__main__":
    main()
