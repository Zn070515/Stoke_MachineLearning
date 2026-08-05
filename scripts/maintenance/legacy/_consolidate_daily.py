# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""LEGACY — not directly usable on current canonical data.

Historical one-shot migration (July 2026) that consolidated partitioned
``daily/{year}/{month}/{code}.parquet`` files into flat ``daily/{code}.parquet``.
Since v7-P0 the flat file is the only canonical layout and partitioned files are
ignored, so this script has no remaining purpose. Writes also bypassed
``DataStorage`` governance (§八-1). Keep for history; do not run on current data.
"""
import logging
import os
import sys
import time
from collections import defaultdict

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "a_shares", "daily")

if not os.path.exists(DATA_DIR):
    logger.error("Daily data dir not found: %s", DATA_DIR)
    sys.exit(1)


def scan_partitions():
    """Scan year/month subdirectories and group parquet files by stock code."""
    stock_files = defaultdict(list)
    year_dirs = []
    total_files = 0

    for item in sorted(os.listdir(DATA_DIR)):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
            year_dirs.append(item_path)
            for month_dir in sorted(os.listdir(item_path)):
                month_path = os.path.join(item_path, month_dir)
                if not os.path.isdir(month_path):
                    continue
                for f in os.listdir(month_path):
                    if f.endswith(".parquet"):
                        code = f.replace(".parquet", "")
                        stock_files[code].append(os.path.join(month_path, f))
                        total_files += 1

    return stock_files, year_dirs, total_files


def main():
    stock_files, year_dirs, total_files = scan_partitions()

    if not stock_files:
        logger.info("No partitioned files found. Nothing to consolidate.")
        return

    n_stocks = len(stock_files)
    avg_files = total_files / n_stocks
    logger.info("Found %d partition files across %d stocks (avg %.0f files/stock)",
                total_files, n_stocks, avg_files)

    # Check for existing flat files
    existing_flat = {f.replace(".parquet", "") for f in os.listdir(DATA_DIR)
                     if f.endswith(".parquet")}
    logger.info("Existing flat files: %d", len(existing_flat))

    consolidated = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, (code, paths) in enumerate(sorted(stock_files.items())):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_stocks - i - 1) / rate if rate > 0 else 0
            logger.info("[%d/%d] %.1f stocks/s, ETA %.0f sec",
                        i + 1, n_stocks, rate, eta)

        output_path = os.path.join(DATA_DIR, f"{code}.parquet")
        tmp_path = output_path + ".tmp"

        try:
            # Read all partition files for this stock
            dfs = []
            for p in paths:
                try:
                    chunk = pd.read_parquet(p)
                    if not chunk.empty:
                        dfs.append(chunk)
                except Exception as e:
                    logger.warning("  %s: failed to read %s — %s", code, p, e)

            if not dfs:
                skipped += 1
                continue

            merged = pd.concat(dfs, ignore_index=True)

            # If a flat file already exists, merge it in too
            if code in existing_flat and os.path.exists(output_path):
                try:
                    existing = pd.read_parquet(output_path)
                    if not existing.empty:
                        merged = pd.concat([existing, merged], ignore_index=True)
                except Exception:
                    pass

            # Deduplicate by date + stock_code
            merged = merged.drop_duplicates(subset=["date"], keep="last")

            # Standardize column order
            col_order = ["date", "open", "high", "low", "close", "volume", "amount",
                         "pct_change", "stock_code"]
            available = [c for c in col_order if c in merged.columns]
            merged = merged[available]

            # Ensure stock_code is consistent
            if "stock_code" in merged.columns:
                merged["stock_code"] = str(code)
            else:
                merged["stock_code"] = str(code)

            # Sort by date
            if "date" in merged.columns:
                merged["date"] = pd.to_datetime(merged["date"])
                merged = merged.sort_values("date").reset_index(drop=True)

            # Write to temp file then rename (atomic on same FS)
            merged.to_parquet(tmp_path, index=False, compression='lz4')
            os.replace(tmp_path, output_path)

            # Delete partition files
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

            consolidated += 1

        except Exception as e:
            errors += 1
            logger.error("  %s: consolidation failed — %s", code, e)
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # Remove empty year/month directories
    for yd in year_dirs:
        for month_dir in sorted(os.listdir(yd)):
            mp = os.path.join(yd, month_dir)
            if os.path.isdir(mp):
                try:
                    remaining = os.listdir(mp)
                    if not remaining:
                        os.rmdir(mp)
                except OSError:
                    pass
        try:
            remaining = os.listdir(yd)
            if not remaining:
                os.rmdir(yd)
        except OSError:
            pass

    elapsed = time.time() - start_time
    logger.info("Done: %d consolidated, %d skipped, %d errors in %.1f sec",
                consolidated, skipped, errors, elapsed)
    logger.info("Flat files on disk: %d",
                len([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]))


if __name__ == "__main__":
    main()
