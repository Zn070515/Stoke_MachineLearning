# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Align aux pct_change to the canonical daily flat (task #353).

Root cause of the 2026-06-18+ pct_change=0 pollution: board_processed and
industry_ranking_processed carry a pct_change column that went stale-zero in
the efinance-down / AKShare-fill-0 window. _merge_daily_aux previously injected
it into features as if it were the stock's own daily return.

The feature layer is already fixed (pipeline preserves K-line pct_change and
excludes it from aux injection). This script repairs the aux files on disk so
the column reflects the true daily return: date-aligned to daily flat
pct_change, the canonical (already-repaired) K-line source.

- Aligns the FULL history (not just the pollution window), so the 2015
  close=0.01 placeholder rows also get their real returns back instead of
  zero, and adjustment-base differences are removed.
- Only touches the pct_change column; all other aux columns are preserved.
- Atomic per-file replace (tmp + os.replace). Idempotent.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_aux_pct_change.py --dry-run
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_aux_pct_change.py --codes 000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_repair_aux_pct_change.py --shard 0/4
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
A_SHARES = PROJECT / "data" / "a_shares"
DAILY_DIR = A_SHARES / "daily"
AUX_DIRS = ["board_processed", "industry_ranking_processed"]


def collect_codes() -> list[str]:
    codes = set()
    for d in AUX_DIRS:
        codes.update(Path(p).stem for p in glob.glob(str(A_SHARES / d / "*.parquet")))
    return sorted(codes)


def _write_atomic(path: Path, df: pd.DataFrame) -> None:
    tmp = str(path) + ".tmp"
    df.to_parquet(tmp, index=False, compression="lz4")
    os.replace(tmp, path)


def repair_one(code: str) -> str:
    daily_path = DAILY_DIR / f"{code}.parquet"
    if not daily_path.exists():
        return "no daily flat"

    daily = pd.read_parquet(daily_path, columns=["date", "pct_change"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.drop_duplicates("date", keep="last")
    # daily first row is NaN (no previous close) — treat as flat 0, same as the
    # feature layer's fillna(0) convention.
    daily["pct_change"] = daily["pct_change"].fillna(0.0)

    changed_rows = 0
    for d in AUX_DIRS:
        path = A_SHARES / d / f"{code}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "pct_change" not in df.columns:
            continue
        orig_dtype = df["pct_change"].dtype
        df["date"] = pd.to_datetime(df["date"])
        aligned = df["date"].map(daily.set_index("date")["pct_change"])
        # Guard: aux dates missing from daily (should be none) get 0, and keep
        # the column's original dtype so parquet schema is unchanged.
        new_pc = aligned.fillna(0.0).astype(orig_dtype)
        changed = int((df["pct_change"].astype("float64").ne(
            new_pc.astype("float64"))).sum())
        df["pct_change"] = new_pc
        changed_rows += changed
        _write_atomic(path, df)
    return f"board/industry aligned, rows changed={changed_rows}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=str, default=None,
                        help="comma-separated stock codes (default: all)")
    parser.add_argument("--shard", type=str, default="0/1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        k, n = (int(x) for x in args.shard.split("/"))
        all_codes = collect_codes()
        codes = [c for i, c in enumerate(all_codes) if i % n == k]
        logger.info("shard %d/%d: %d stocks (total %d)", k, n, len(codes), len(all_codes))

    if args.dry_run:
        logger.info("dry-run: %d stocks would be repaired", len(codes))
        return 0

    t0 = time.time()
    ok = fail = total_changed = 0
    for i, code in enumerate(codes):
        try:
            detail = repair_one(code)
            ok += 1
            if "changed=" in detail:
                total_changed += int(detail.split("changed=")[1])
            if (ok + fail) <= 5 or (ok + fail) % 500 == 0:
                logger.info("  %s: %s", code, detail)
        except Exception as e:
            logger.error("%s: %s", code, e)
            fail += 1
        if (ok + fail) % 500 == 0:
            logger.info("  %d/%d done (%.1f stk/s)", ok + fail, len(codes),
                        (ok + fail) / max(time.time() - t0, 1e-9))
    logger.info("done: %d ok, %d fail, %d rows changed (%.1fs)",
                ok, fail, total_changed, time.time() - t0)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
