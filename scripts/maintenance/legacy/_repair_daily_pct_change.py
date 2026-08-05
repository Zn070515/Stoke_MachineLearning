# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Repair daily pct_change by deriving it from close (task #342).

Root cause: akshare stock_zh_a_daily (Sina qfq) omits 涨跌幅 and akshare_source
filled pct_change=0.0. With efinance down, failover fell back to AKShare, so
partition files were persisted with pct_change=0 and _consolidate_daily_flat
spread those zeros into flat, also mixing adjustment bases per-date.

close is the single source of truth (verified self-consistent). This script:

- Partition stocks: stitch partitions, recompute pct_change across the full
  series (month boundaries included), write it back into every partition, then
  rebuild flat = old flat ∪ repaired partitions (partition wins on date). This
  also removes the frankenstein close seams for fully-partitioned stocks.
- Flat-only stocks: recompute zero/NaN pct_change rows from close in place.

Idempotent. All canonical-daily writes route through
``DataStorage.save_daily_repair`` (§八-1) so the flat keeps its
lock/merge/manifest/source-segment governance; legacy partition files are no
longer touched (the flat is the only canonical layout since v7-P0).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_daily_pct_change.py --dry-run
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_daily_pct_change.py --codes 000001,601212
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_daily_pct_change.py --shard 0/8
"""
import argparse
import glob
import logging
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

sys.path.insert(0, str(PROJECT))
from stoke_ml.data.storage import DataStorage


def collect_codes():
    codes = set()
    for f in glob.glob(str(DAILY_DIR / "**" / "*.parquet"), recursive=True):
        codes.add(Path(f).stem)
    return sorted(codes)


def _has_partitions(code: str) -> bool:
    return any(
        Path(f).parent != DAILY_DIR
        for f in glob.glob(str(DAILY_DIR / "**" / f"{code}.parquet"), recursive=True)
    )


def _recompute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Set pct_change = close.pct_change()*100 for every row (clean basis)."""
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0
    return df


def repair_one(code: str, storage: DataStorage) -> str:
    part_files = sorted(
        Path(f) for f in glob.glob(str(DAILY_DIR / "**" / f"{code}.parquet"), recursive=True)
        if Path(f).parent != DAILY_DIR
    )
    flat_path = DAILY_DIR / f"{code}.parquet"

    if part_files:
        # Stitch partitions into one series, then rebuild flat = old flat ∪
        # partitions (partition wins on date). The final pct_change is derived
        # from the merged flat's own close so it equals close.pct_change()*100
        # exactly — the same invariant good stocks already satisfy. Old-flat
        # rows (suspension days / dates partitions miss) keep their close; the
        # recompute keeps pct_change consistent with it.
        stitched = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
        stitched["date"] = pd.to_datetime(stitched["date"])
        stitched = stitched.drop_duplicates("date", keep="last")

        chunks = []
        if flat_path.exists():
            chunks.append(pd.read_parquet(flat_path))
        chunks.append(stitched)
        flat = pd.concat(chunks, ignore_index=True)
        flat["date"] = pd.to_datetime(flat["date"])
        flat = flat.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        mode = "partition rebuild"
    elif flat_path.exists():
        flat = pd.read_parquet(flat_path)
        flat["date"] = pd.to_datetime(flat["date"])
        mode = "flat-only"
    else:
        return "no data"

    before = int((flat["pct_change"].isna() | (flat["pct_change"] == 0)).sum())
    flat = _recompute_all(flat)
    after = int((flat["pct_change"].isna() | (flat["pct_change"] == 0)).sum())
    flat["stock_code"] = str(code)
    # Route through the storage API so the manifest + source segments stay in
    # sync and provenance is preserved from the existing file (§八-1).
    storage.save_daily_repair(flat, market="a_shares")
    return f"{mode}: recomputed all ({before} -> {after} zero rows, {len(flat)} total)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=str, default=None,
                        help="comma-separated stock codes (default: all)")
    parser.add_argument("--flat-only", action="store_true",
                        help="only repair stocks without partition files")
    parser.add_argument("--shard", type=str, default="0/1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.flat_only:
        all_codes = collect_codes()
        codes = [c for c in all_codes if not _has_partitions(c)]
        logger.info("flat-only mode: %d stocks", len(codes))
    else:
        k, n = (int(x) for x in args.shard.split("/"))
        all_codes = collect_codes()
        codes = [c for i, c in enumerate(all_codes) if i % n == k]
        logger.info("shard %d/%d: %d stocks (total %d)", k, n, len(codes), len(all_codes))

    if args.dry_run:
        logger.info("dry-run: %d stocks would be repaired", len(codes))
        return 0

    storage = DataStorage(str(PROJECT / "data"))
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(codes):
        try:
            detail = repair_one(code, storage)
            ok += 1
            if (ok + fail) <= 5 or (ok + fail) % 500 == 0:
                logger.info("  %s: %s", code, detail)
        except Exception as e:
            logger.error("%s: %s", code, e)
            fail += 1
        if (ok + fail) % 500 == 0:
            logger.info("  %d/%d done (%.1f stk/s)", ok + fail, len(codes),
                        (ok + fail) / max(time.time() - t0, 1e-9))
    logger.info("done: %d ok, %d fail (%.1fs)", ok, fail, time.time() - t0)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
