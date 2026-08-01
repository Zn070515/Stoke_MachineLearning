"""One-shot: clip negative OHLCV values in daily K-line parquet files.
Backs up original to data/a_shares/daily_backup/ before modifying.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_fix_negative_ohlcv.py --dry-run
  PYTHONPATH=. ./.venv/Scripts/python scripts/_fix_negative_ohlcv.py
"""
import logging
import os
import shutil
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close"]
FLOOR = 0.01


def scan_and_fix(data_dir: str, dry_run: bool = True):
    daily_dir = os.path.join(data_dir, "daily")
    backup_dir = os.path.join(data_dir, "daily_backup")
    parquets = sorted(f for f in os.listdir(daily_dir) if f.endswith(".parquet"))

    affected = []
    for fname in parquets:
        path = os.path.join(daily_dir, fname)
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.warning("Skipping unreadable: %s", fname)
            continue

        fixed_cols = {}
        for c in OHLCV_COLS:
            if c in df.columns:
                neg_mask = df[c] < 0
                if neg_mask.any():
                    fixed_cols[c] = int(neg_mask.sum())

        if not fixed_cols:
            continue

        code = fname.replace(".parquet", "")
        date_range = ""
        neg_dates = df.loc[
            df[[c for c in fixed_cols if c in df.columns]].lt(0).any(axis=1), "date"
        ]
        if "date" in df.columns and len(neg_dates) > 0:
            date_range = f" [{neg_dates.min()} to {neg_dates.max()}]"

        logger.info(
            "%s: %s%s",
            code,
            ", ".join(f"{c}({n})" for c, n in fixed_cols.items()),
            date_range,
        )
        affected.append((fname, fixed_cols))

        if not dry_run:
            # Backup
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, fname)
            if not os.path.exists(backup_path):
                shutil.copy2(path, backup_path)

            # Clip
            for c in fixed_cols:
                df[c] = df[c].clip(lower=FLOOR)
            df.to_parquet(path, index=False, compression='lz4')

    logger.info(
        "Total: %d stocks affected. %s",
        len(affected),
        "DRY RUN — no changes made" if dry_run else "Fixed and backed up.",
    )


def main():
    dry_run = "--dry-run" in sys.argv
    data_dir = "data/a_shares"
    logger.info(
        "%s negative OHLCV in %s...",
        "Scanning" if dry_run else "Fixing",
        data_dir + "/daily",
    )
    scan_and_fix(data_dir, dry_run)


if __name__ == "__main__":
    main()
