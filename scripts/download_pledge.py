"""Download share pledge data — pledge ratios and market-wide statistics.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_pledge.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_pledge.py --stocks 600519,000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_pledge.py --market-only
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.pledge_source import PledgeSource

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


def _existing_stocks(pledge_dir: str) -> set[str]:
    """Return set of stock codes that already have per-stock pledge files."""
    if not os.path.exists(pledge_dir):
        return set()
    return {
        f.replace(".parquet", "")
        for f in os.listdir(pledge_dir)
        if f.endswith(".parquet") and f not in ("pledge_ratios.parquet", "market_pledge_stats.parquet")
    }


def main():
    parser = argparse.ArgumentParser(description="Download share pledge data")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--market-only", action="store_true",
                        help="Only fetch market-wide pledge stats (fast)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between per-stock API calls (default: 0.3)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-download all stocks (ignore existing per-stock files)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    out_dir = os.path.join(data_dir, "a_shares", "pledge")
    os.makedirs(out_dir, exist_ok=True)

    src = PledgeSource()

    # 1. Market-wide pledge stats (single call)
    logger.info("=== Step 1/2: Market-wide pledge statistics ===")
    market_stats = src.fetch_market_pledge_stats()
    if not market_stats.empty:
        path = os.path.join(out_dir, "market_pledge_stats.parquet")
        market_stats.to_parquet(path, index=False, compression='lz4')
        logger.info("Saved %d rows -> %s", len(market_stats), path)

    if args.market_only:
        logger.info("Done (market only).")
        return

    # 2. Per-stock pledge ratios with resume support
    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stock codes found.")
        sys.exit(1)

    # Resume: skip stocks that already have per-stock files
    if not args.no_resume:
        existing = _existing_stocks(out_dir)
        if existing:
            remaining = [c for c in codes if c not in existing]
            logger.info("Skipping %d already-complete stocks, %d remaining",
                        len(existing), len(remaining))
            codes = remaining

    if not codes:
        logger.info("All stocks already downloaded. Nothing to do.")
        return

    logger.info("=== Step 2/2: Per-stock pledge ratios for %d stocks ===", len(codes))

    success = 0
    fail = 0
    start_time = time.time()

    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        try:
            df = src.fetch_pledge_ratio(code)
        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning("[%d/%d] %s: fetch failed — %s", i + 1, len(codes), code, e)
            continue

        if not df.empty:
            # Save per-stock immediately for resume safety
            stock_path = os.path.join(out_dir, f"{code}.parquet")
            df.to_parquet(stock_path, index=False, compression='lz4')
            success += 1

        if (i + 1) % 250 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(codes) - i - 1) / rate if rate > 0 else 0
            logger.info("[%d/%d] %s: %d success, %d fail (%.1f/s, ETA %.0f min)",
                        i + 1, len(codes), code, success, fail, rate, eta / 60)

    # Build combined pledge_ratios.parquet from per-stock files
    all_per_stock = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".parquet") and f not in ("pledge_ratios.parquet", "market_pledge_stats.parquet")
    ]
    if all_per_stock:
        frames = [pd.read_parquet(p) for p in all_per_stock]
        combined = pd.concat(frames, ignore_index=True)
        combined_path = os.path.join(out_dir, "pledge_ratios.parquet")
        combined.to_parquet(combined_path, index=False, compression='lz4')
        n_stocks = combined["stock_code"].nunique()
        logger.info("Combined %d pledge rows for %d stocks -> %s",
                      len(combined), n_stocks, combined_path)

    elapsed = time.time() - start_time
    logger.info("Done: %d success, %d fail in %.1f min",
                success, fail, elapsed / 60)


if __name__ == "__main__":
    main()
