"""Download industry index returns and stock-to-industry mapping.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_industry.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_industry.py --no-mapping
"""
import argparse
import logging
import os
import sys

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.industry_source import IndustrySource
from stoke_ml.data.download_manifest import write_run_manifest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download industry index data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: data/a_shares/industry/)")
    parser.add_argument("--no-mapping", action="store_true",
                        help="Skip stock-to-industry mapping")
    parser.add_argument("--start", type=str, default="20150101")
    parser.add_argument("--end", type=str, default="20260716")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    out_dir = args.output or os.path.join(data_dir, "a_shares", "industry")
    os.makedirs(out_dir, exist_ok=True)

    src = IndustrySource()

    done: set[str] = set()
    failed: list[str] = []
    requested = ["industry_returns"] + ([] if args.no_mapping else ["stock_industry_map"])

    # Fetch all industry returns
    logger.info("Fetching industry index returns...")
    try:
        returns = src.fetch_all_returns(start_date=args.start, end_date=args.end)
        returns_path = os.path.join(out_dir, "industry_returns.parquet")
        returns.to_parquet(returns_path)
        logger.info("Saved %d days × %d industries to %s",
                    len(returns), len(returns.columns), returns_path)
        done.add("industry_returns")
    except Exception as e:
        failed.append("industry_returns")
        logger.error("industry_returns: %s", str(e)[:120])

    # Fetch stock-to-industry mapping
    if not args.no_mapping:
        logger.info("Fetching stock-to-industry mapping...")
        try:
            mapping = src.fetch_stock_industry_map()
            if not mapping.empty:
                map_path = os.path.join(out_dir, "stock_industry_map.parquet")
                mapping.to_parquet(map_path)
                logger.info("Saved %d stock mappings to %s", len(mapping), map_path)
                done.add("stock_industry_map")
            else:
                failed.append("stock_industry_map")
                logger.warning("Could not fetch stock-industry mapping")
        except Exception as e:
            failed.append("stock_industry_map")
            logger.error("stock_industry_map: %s", str(e)[:120])

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/industry",
            start_date=args.start, end_date=args.end,
            requested=requested, failed=failed, complete=done,
            success_count=len(done),
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)


if __name__ == "__main__":
    main()
