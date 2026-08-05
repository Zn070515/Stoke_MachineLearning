# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""LEGACY — not directly usable on current canonical data.

Historical one-shot migration (Aug 2026) that merged partitioned
``daily/{year}/{month}/{code}.parquet`` into flat ``daily/{code}.parquet``
(non-destructive). Since v7-P0 the flat file is the only canonical layout and
partitioned files are ignored, so this script has no remaining purpose. Writes
also bypassed ``DataStorage`` governance (§八-1). Keep for history; do not run
on current data.
"""
import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data" / "a_shares" / "daily"


def collect_codes():
    codes = set()
    for f in glob.glob(str(DAILY_DIR / "**" / "*.parquet"), recursive=True):
        codes.add(Path(f).stem)
    return sorted(codes)


def consolidate_one(code):
    flat_path = DAILY_DIR / f"{code}.parquet"
    part_files = sorted(
        Path(f) for f in glob.glob(str(DAILY_DIR / "**" / f"{code}.parquet"), recursive=True)
        if Path(f).parent != DAILY_DIR
    )
    chunks = []
    if flat_path.exists():
        chunks.append(pd.read_parquet(flat_path))
    for f in part_files:
        chunks.append(pd.read_parquet(f))
    if not chunks:
        return "rows=0"
    df = pd.concat(chunks, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if "pct_change" in df.columns and "close" in df.columns:
        # Guard: a broken source (e.g. AKShare stock_zh_a_daily omits 涨跌幅)
        # can persist pct_change=0. Merging those into flat with keep="last"
        # would spread zeros to every consumer (load_daily prefers flat). Repair
        # rows where pct_change is 0/missing but close actually moved, and log —
        # never silently overwrite a legitimate value.
        close = pd.to_numeric(df["close"], errors="coerce")
        expected = close.pct_change() * 100.0
        actual = pd.to_numeric(df["pct_change"], errors="coerce")
        bad = (actual.isna() | (actual == 0)) & close.notna() & (expected.abs() > 0.05)
        if bad.any():
            n = int(bad.sum())
            logger.warning(
                "%s: %d pct_change=0 rows but close moved — repaired from close", code, n
            )
            df.loc[bad, "pct_change"] = expected[bad]
    if "stock_code" in df.columns:
        df["stock_code"] = str(code)
    tmp_path = str(flat_path) + ".tmp"
    df.to_parquet(tmp_path, index=False, compression="lz4")
    os.replace(tmp_path, flat_path)
    return f"rows={len(df)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default="0/1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    codes = collect_codes()
    mine = [c for i, c in enumerate(codes) if i % n == k]
    logger.info("shard %d/%d: %d stocks (total %d)", k, n, len(mine), len(codes))
    if args.dry_run:
        return 0
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(mine):
        try:
            consolidate_one(code)
            ok += 1
        except Exception as e:
            logger.error("%s: %s", code, e)
            fail += 1
        if (ok + fail) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d done (%.1f stk/s)", ok + fail, len(mine),
                        (ok + fail) / max(elapsed, 1e-9))
    logger.info("shard %d/%d done: %d ok, %d fail (%.1fs)", k, n, ok, fail,
                time.time() - t0)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
