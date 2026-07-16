"""Download macro-economic indicators and save as daily features.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_macro.py
"""
import argparse
import logging
import os
import sys

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.macro_source import MacroSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download macro-economic indicators")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/a_shares/macro/macro_daily.parquet)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    output_path = args.output or os.path.join(data_dir, "a_shares", "macro", "macro_daily.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Fetching macro indicators...")
    ms = MacroSource()
    df = ms.fetch_all()

    logger.info("Saving %d rows × %d columns to %s", len(df), len(df.columns), output_path)
    df.to_parquet(output_path)

    logger.info("Done. Columns: %s", list(df.columns))
    logger.info("Date range: %s to %s", df.index.min().date(), df.index.max().date())


if __name__ == "__main__":
    main()
