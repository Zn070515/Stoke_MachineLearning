"""Download market breadth indicators — new highs/lows, account statistics.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_market_breadth.py
"""
import argparse
import logging
import os

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.market_breadth_source import MarketBreadthSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download market breadth data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: data/a_shares/market_breadth/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output or os.path.join(
        cfg.project.data_dir, "a_shares", "market_breadth"
    )
    os.makedirs(out_dir, exist_ok=True)

    src = MarketBreadthSource()
    data = src.fetch_all()

    for name, df in data.items():
        if df.empty:
            logger.warning("Skipping %s (empty)", name)
            continue
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False, compression='lz4')
        logger.info("Saved %s: %d rows → %s", name, len(df), path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
