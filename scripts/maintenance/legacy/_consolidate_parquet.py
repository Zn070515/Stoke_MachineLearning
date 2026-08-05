# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Consolidate partitioned parquet files into flat {code}.parquet per stock.

Handles: sentiment/, fundamentals/, guba_sentiment/

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_parquet.py sentiment
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_parquet.py fundamentals
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_consolidate_parquet.py all
"""
import logging
import os
import shutil
import sys

import pandas as pd

from stoke_ml.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CONSOLIDATION_TARGETS = {
    "sentiment": {
        "dir": "a_shares/sentiment",
        "skip_dirs": [],  # year dirs are the partitioned data to merge
    },
    "fundamentals": {
        "dir": "a_shares/fundamentals",
        "skip_dirs": [],
    },
    "guba_sentiment": {
        "dir": "a_shares/guba_sentiment",
        "skip_dirs": [],
    },
}


def consolidate_directory(data_dir: str, target_dir: str, dry_run: bool = False):
    """Merge all partitioned parquet files into flat {code}.parquet."""
    base = os.path.join(data_dir, target_dir)
    if not os.path.exists(base):
        logger.warning("%s does not exist, skipping", base)
        return

    # Find all partitioned files
    code_files: dict[str, list[str]] = {}
    flat_codes = set()

    for root, dirs, files in os.walk(base):
        if root == base:
            # Top level: track flat files already present
            for f in files:
                if f.endswith(".parquet"):
                    flat_codes.add(f.replace(".parquet", ""))
            continue
        for f in files:
            if f.endswith(".parquet"):
                code = f.replace(".parquet", "")
                code_files.setdefault(code, []).append(os.path.join(root, f))

    if not code_files:
        logger.info("%s: no partitioned files to consolidate", target_dir)
        return

    logger.info("%s: %d stocks with partitioned data, %d already flat",
                target_dir, len(code_files), len(flat_codes))

    # For stocks that already have flat files, skip
    merge_count = 0
    for code, paths in code_files.items():
        if code in flat_codes:
            continue

        try:
            frames = []
            for p in paths:
                try:
                    df = pd.read_parquet(p)
                    if not df.empty:
                        frames.append(df)
                except Exception:
                    logger.warning("  Corrupt file: %s", p)

            if not frames:
                continue

            combined = pd.concat(frames, ignore_index=True)

            # Dedup by date
            date_col = "date" if "date" in combined.columns else (
                "report_date" if "report_date" in combined.columns else None
            )
            if date_col:
                combined[date_col] = pd.to_datetime(combined[date_col])
                if "stock_code" in combined.columns:
                    combined = combined.drop_duplicates(subset=[date_col, "stock_code"])
                else:
                    combined = combined.drop_duplicates(subset=[date_col])
                combined = combined.sort_values(date_col).reset_index(drop=True)

            out_path = os.path.join(base, f"{code}.parquet")
            if not dry_run:
                combined.to_parquet(out_path, index=False, compression='lz4')
            merge_count += 1

        except Exception as e:
            logger.error("  %s: ERROR %s", code, e)

    logger.info("%s: merged %d stocks into flat files", target_dir, merge_count)

    # Delete partitioned directories
    if not dry_run:
        for item in os.listdir(base):
            item_path = os.path.join(base, item)
            if os.path.isdir(item_path):
                logger.info("  Removing partitioned dir: %s", item)
                shutil.rmtree(item_path)

    logger.info("%s: consolidation complete", target_dir)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    dry_run = "--dry-run" in sys.argv

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if target == "all":
        targets = list(CONSOLIDATION_TARGETS.keys())
    else:
        targets = [target]

    for t in targets:
        if t not in CONSOLIDATION_TARGETS:
            logger.error("Unknown target: %s", t)
            continue
        consolidate_directory(data_dir, CONSOLIDATION_TARGETS[t]["dir"], dry_run)


if __name__ == "__main__":
    main()
