"""Align aux OHLC to the canonical daily flat (调整基准统一).

Root cause: board_processed / industry_ranking_processed / block_trade_processed /
dividend_processed / lockup_processed / shareholder_processed were preprocessed
from a daily snapshot whose qfq anchor differs from the current canonical daily
flat. Their embedded open/high/low/close carry the OLD adjustment basis, so any
price-level or cross-date-ratio feature derived from them is inconsistent with
daily-based features. pct_change was already aligned (#353); this aligns OHLC.

The close/OHLC in these files is a pass-through of the daily close (merged via
EventToDaily close_prices / BoardBroadcaster base), so overwriting it with the
canonical daily value on matching dates is safe and correct — it does NOT touch
event-native columns (e.g. block_trade deal_price, dividend bonus_rmb).

- Aligns FULL history, not just a window.
- Only touches OHLC columns present in both aux and daily; all other columns
  (including pct_change) are preserved.
- Atomic per-file replace (tmp + os.replace). Idempotent.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_aux_close_basis.py --dry-run
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_aux_close_basis.py --codes 000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/_repair_aux_close_basis.py --shard 0/4
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
# All processed dirs that embed a daily-derived OHLC column on a stale basis.
AUX_DIRS = [
    "block_trade_processed",
    "board_processed",
    "dividend_processed",
    "industry_ranking_processed",
    "lockup_processed",
    "shareholder_processed",
]
OHLC = ["open", "high", "low", "close"]


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

    daily = pd.read_parquet(daily_path, columns=["date"] + OHLC)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.drop_duplicates("date", keep="last")

    total_changed = 0
    details = []
    for d in AUX_DIRS:
        path = A_SHARES / d / f"{code}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date" not in df.columns:
            continue
        present = [c for c in OHLC if c in df.columns]
        if not present:
            continue
        df["date"] = pd.to_datetime(df["date"])
        daily_idx = daily.set_index("date")
        changed = 0
        for c in present:
            # Keep the column's original dtype so the parquet schema is unchanged.
            orig_dtype = df[c].dtype
            aligned_c = df["date"].map(daily_idx[c])
            new = aligned_c.fillna(df[c]).astype(orig_dtype)
            changed += int(df[c].astype("float64").ne(new.astype("float64")).sum())
            df[c] = new
        if changed:
            _write_atomic(path, df)
        total_changed += changed
        details.append(f"{d}:{changed}")
    return f"rows changed={total_changed} ({', '.join(details)})"


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
                total_changed += int(detail.split("changed=")[1].split(" ")[0])
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
