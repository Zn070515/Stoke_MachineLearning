# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Verify consolidated flat files cover all partition data (post disk-move).

Checks per dataset (daily / fundamentals / minute_{5,15,30,60}):
1. Every stock present in partitions has a flat file (no missing stocks).
2. For a random sample, flat timestamps superset of partition timestamps.
3. Flat parquet files readable after the disk move; report row count + newest date.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/diagnostics/_verify_consolidation.py
"""
import glob
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd

from stoke_ml.config import get_project_root

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ROOT = get_project_root() / "data" / "a_shares"
SAMPLE = 30
MINUTE_SAMPLE = 15
SEED = 42


def collect_codes(pattern):
    return {Path(f).stem for f in glob.glob(str(pattern))}


def check_missing(name, flat_pattern, part_pattern):
    flat = collect_codes(flat_pattern)
    part = collect_codes(part_pattern)
    missing = sorted(part - flat)
    logger.info(
        "%s: flat_stocks=%d part_stocks=%d missing=%d",
        name, len(flat), len(part), len(missing),
    )
    if missing:
        logger.warning("  MISSING flat for: %s", missing[:20])
    return part


def verify_coverage(name, flat_path, part_glob, time_col):
    flat = pd.read_parquet(flat_path)
    part_files = sorted(
        Path(f) for f in glob.glob(str(part_glob)) if Path(f).parent != flat_path.parent
    )
    if not part_files:
        logger.info("  %s: no partition files, skip", name)
        return
    parts, bad = [], []
    for f in part_files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            bad.append(f"{f.parent.name}/{f.name}: {e}")
    if not parts:
        logger.info("  %s: all %d partition files unreadable [PROBLEM]",
                    name, len(part_files))
        for b in bad:
            logger.warning("    bad file: %s", b)
        return
    part = pd.concat(parts, ignore_index=True)
    fs = set(pd.to_datetime(flat[time_col]).dt.normalize())
    ps = set(pd.to_datetime(part[time_col]).dt.normalize())
    only_part = ps - fs
    part_unique = len(part.drop_duplicates(subset=[time_col]))
    ok = not only_part and len(flat) >= part_unique and not bad
    logger.info(
        "  %s: flat rows=%d part_unique=%d only_in_partition=%d newest=%s bad_files=%d [%s]",
        name, len(flat), part_unique, len(only_part),
        pd.to_datetime(flat[time_col]).max().date(), len(bad),
        "OK" if ok else "PROBLEM",
    )
    if only_part:
        logger.warning("    only_in_partition sample: %s", sorted(only_part)[:5])
    for b in bad:
        logger.warning("    bad file: %s", b)


def sample_codes(all_codes, n, seed=SEED):
    rng = random.Random(seed)
    return rng.sample(sorted(all_codes), min(n, len(all_codes)))


def main():
    t0 = time.time()
    datasets = []

    # ---- daily ----
    part_codes = check_missing(
        "daily", "data/a_shares/daily/*.parquet", "data/a_shares/daily/*/*/*.parquet"
    )
    for code in sample_codes(part_codes, SAMPLE):
        verify_coverage(code, ROOT / "daily" / f"{code}.parquet",
                        f"data/a_shares/daily/*/*/{code}.parquet", "date")

    # ---- fundamentals ----
    part_codes = check_missing(
        "fundamentals", "data/a_shares/fundamentals/*.parquet",
        "data/a_shares/fundamentals/*/*/*.parquet"
    )
    for code in sample_codes(part_codes, SAMPLE):
        verify_coverage(code, ROOT / "fundamentals" / f"{code}.parquet",
                        f"data/a_shares/fundamentals/*/*/{code}.parquet", "report_date")

    # ---- minute ----
    for freq in ["5", "15", "30", "60"]:
        part_codes = check_missing(
            f"minute_{freq}",
            f"data/a_shares/minute_flat/{freq}/*.parquet",
            f"data/a_shares/minute/{freq}/*/*/*.parquet",
        )
        codes = sample_codes(part_codes, MINUTE_SAMPLE)
        if freq == "30" and "600630" in part_codes:
            codes = ["600630"] + [c for c in codes if c != "600630"]
        for code in codes:
            verify_coverage(code, ROOT / "minute_flat" / freq / f"{code}.parquet",
                            f"data/a_shares/minute/{freq}/*/*/{code}.parquet", "datetime")

    logger.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
