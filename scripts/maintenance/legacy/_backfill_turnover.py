# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Download turnover (换手率) raw data via Baostock for stocks missing it.

Saves date + turnover columns to data/a_shares/turnover_raw/{code}.parquet.
Merge into daily K-line files will be done in a later unified step.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_turnover.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_turnover.py --stocks 000001,600519
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Download turnover raw data via Baostock")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Delay between stocks (default: 0s)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all missing)")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config

    cfg = load_config()
    daily_dir = Path(cfg.project.data_dir) / "a_shares" / "daily"
    raw_dir = Path(cfg.project.data_dir) / "a_shares" / "turnover_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([f.stem for f in daily_dir.glob("*.parquet")])

    # ── Discover stocks missing turnover ──
    todo = []
    for code in codes:
        path = daily_dir / f"{code}.parquet"
        raw_path = raw_dir / f"{code}.parquet"
        if raw_path.exists():
            continue  # already downloaded
        try:
            df = pd.read_parquet(path, columns=["date"])
            df["date"] = pd.to_datetime(df["date"])
            if "turnover" in pd.read_parquet(path).columns:
                continue  # already has turnover in daily file
            min_date = df["date"].min().strftime("%Y-%m-%d")
            max_date = df["date"].max().strftime("%Y-%m-%d")
            todo.append((code, min_date, max_date))
        except Exception as e:
            logger.warning("%s: scan failed: %s", code, e)

    if not todo:
        logger.info("All %d stocks already have turnover or raw data.", len(codes))
        return 0

    logger.info("%d/%d stocks need turnover download", len(todo), len(codes))

    # ── Download via Baostock ──
    import baostock as bs

    success, fail, skipped = 0, 0, 0
    total_rows = 0

    for i, (code, min_date, max_date) in enumerate(todo):
        if i > 0:
            time.sleep(args.sleep)

        logger.info("[%d/%d] %s: turnover %s → %s",
                    i + 1, len(todo), code, min_date, max_date)

        try:
            lg = bs.login()
            if lg is None or lg.error_code != "0":
                logger.error("  %s: baostock login failed", code)
                fail += 1
                continue

            if code.startswith("6") or code.startswith("9"):
                bs_code = f"sh.{code}"
            elif code.startswith("8") or code.startswith("4"):
                bs_code = f"bj.{code}"
            else:
                bs_code = f"sz.{code}"

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,turn",
                start_date=min_date,
                end_date=max_date,
                frequency="d",
                adjustflag="2",
            )
            if rs is None or rs.error_code != "0":
                err = "None" if rs is None else rs.error_msg
                logger.warning("  %s: query failed: %s", code, err)
                bs.logout()
                fail += 1
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            bs.logout()

            if not rows:
                logger.info("  %s: no data returned", code)
                skipped += 1
                continue

            df = pd.DataFrame(rows, columns=["date", "turnover"])
            df["date"] = pd.to_datetime(df["date"])
            df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
            df = df.dropna(subset=["turnover"])
            if df.empty:
                skipped += 1
                continue

            df.to_parquet(raw_dir / f"{code}.parquet", index=False, compression='lz4')
            total_rows += len(df)
            success += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                logger.info("  ... %d/%d done, %.1f stk/min",
                            i + 1, len(todo), (i + 1) / elapsed * 60)

        except Exception as e:
            logger.error("  %s: %s", code, e)
            try:
                bs.logout()
            except Exception:
                pass
            fail += 1

    logger.info("Done: %d ok, %d fail, %d skip — %d turnover rows saved to %s",
                success, fail, skipped, total_rows, raw_dir)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    t_start = time.time()
    sys.exit(main())
