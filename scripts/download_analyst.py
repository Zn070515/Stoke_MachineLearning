"""Download analyst profit forecasts + ratings for A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_analyst.py
"""
import argparse
import logging
import os

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.analyst_source import AnalystSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download analyst data")
    parser.add_argument("--year", type=str, default="2024",
                        help="Year for analyst ranking (default: 2024)")
    parser.add_argument("--ranking-only", action="store_true",
                        help="Only download analyst ranking")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    out_dir = os.path.join(data_dir, "a_shares", "analyst")
    os.makedirs(out_dir, exist_ok=True)

    src = AnalystSource()

    if not args.ranking_only:
        # 1. Profit forecasts — single call, market-wide
        logger.info("=== Step 1/2: Profit forecasts ===")
        forecasts = src.fetch_profit_forecast()
        if not forecasts.empty:
            path = os.path.join(out_dir, "profit_forecasts.parquet")
            forecasts.to_parquet(path, index=False, compression='lz4')
            n_stocks = forecasts["stock_code"].nunique()
            logger.info("Saved %d forecast rows for %d stocks → %s",
                          len(forecasts), n_stocks, path)

    # 2. Analyst ranking
    logger.info("=== Step 2/2: Analyst ranking ===")
    ranking = src.fetch_analyst_ranking(year=args.year)
    if not ranking.empty:
        path = os.path.join(out_dir, "analyst_ranking.parquet")
        ranking.to_parquet(path, index=False, compression='lz4')
        logger.info("Saved %d analyst rankings → %s", len(ranking), path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
