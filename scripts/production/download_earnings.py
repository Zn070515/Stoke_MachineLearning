"""Download 业绩预告 (earnings forecasts) + 业绩快报 (express reports) market-wide.

These are the highest-timeliness fundamental signals — a forecast/express lands
weeks before the actual quarterly report, so it is the first market reaction to
a beat/miss.

IMPORTANT: the AKShare/EM endpoint must be called with an explicit report period
(date=YYYYMMDD); calling without a date returns a stale 2020 default batch. The
downloader auto-generates quarter-end report periods and accumulates each into
the snapshot parquet, deduping on the announcement key so history survives
across runs.

Usage:
  # default: recent ~2.5 years of report periods
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_earnings.py
  # full historical backfill
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_earnings.py --start-year 2020
  # explicit period list
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_earnings.py --report-dates 20260331,20251231
"""
import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.earnings_source import EarningsSource
from stoke_ml.data.download_manifest import write_run_manifest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Announcement identity per source. New-schema 业绩预告 has no report_date, so
# (stock, announce_date, metric) uniquely identifies one forecast row; express
# is keyed by (stock, announce_date).
FORECAST_DEDUP = ["stock_code", "announce_date", "forecast_metric"]
EXPRESS_DEDUP = ["stock_code", "announce_date"]

QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def quarter_ends(start_year: int, today: datetime) -> list[str]:
    """All quarter-end report periods from *start_year* up to the latest <= today."""
    periods = []
    for y in range(start_year, today.year + 1):
        for m, d in QUARTER_ENDS:
            p = f"{y}{m:02d}{d:02d}"
            if pd.Timestamp(p) <= pd.Timestamp(today.date()):
                periods.append(p)
    return periods


def _accumulate(out_dir: str, name: str, new: pd.DataFrame, subset: list[str]) -> str:
    """Append *new* rows to {name}.parquet, deduping on the announcement key."""
    path = os.path.join(out_dir, name)
    if os.path.isfile(path):
        try:
            old = pd.read_parquet(path)
        except Exception as e:
            logger.warning("existing %s unreadable (%s); starting fresh", name, e)
            old = pd.DataFrame()
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = merged.drop_duplicates(subset=[c for c in subset if c in merged.columns],
                                    keep="last")
    merged.to_parquet(path, index=False, compression="lz4")
    return path


def _dedup(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=[c for c in subset if c in df.columns], keep="last")


def main():
    ap = argparse.ArgumentParser(description="Download earnings forecast/express snapshots")
    ap.add_argument(
        "--report-dates", type=str, default=None,
        help="comma-separated report periods (e.g. 20260331,20251231); "
             "overrides --start-year",
    )
    ap.add_argument(
        "--start-year", type=int, default=2024,
        help="first report year to fetch (default 2024); use 2020 for full backfill",
    )
    ap.add_argument("--skip-express", action="store_true",
                    help="only download 业绩预告")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = os.path.join(cfg.project.data_dir, "a_shares", "earnings")
    os.makedirs(out_dir, exist_ok=True)

    periods = args.report_dates.split(",") if args.report_dates \
        else quarter_ends(args.start_year, datetime.now())
    logger.info("%d report periods: %s ... %s", len(periods), periods[0], periods[-1])

    src = EarningsSource()

    done: set[str] = set()
    failed: list[str] = []
    requested = ["forecasts"] + ([] if args.skip_express else ["express"])

    # 1. 业绩预告
    logger.info("=== 业绩预告 ===")
    fc_frames = []
    for p in periods:
        try:
            df = src.fetch_forecasts(date=p)
            if not df.empty:
                fc_frames.append(df)
                logger.info("  period %s: %d rows", p, len(df))
        except Exception as e:
            logger.warning("  period %s: forecast fetch failed: %s", p, str(e)[:100])
    if fc_frames:
        new = _dedup(pd.concat(fc_frames, ignore_index=True), FORECAST_DEDUP)
        p = _accumulate(out_dir, "forecasts.parquet", new, FORECAST_DEDUP)
        n_stocks = new["stock_code"].nunique()
        logger.info("  saved %d rows (%d stocks) → %s", len(new), n_stocks, p)
        done.add("forecasts")
    else:
        failed.append("forecasts")
        logger.warning("  no forecast data fetched")

    # 2. 业绩快报
    if not args.skip_express:
        logger.info("=== 业绩快报 ===")
        ex_frames = []
        for p in periods:
            try:
                df = src.fetch_express(date=p)
                if not df.empty:
                    ex_frames.append(df)
                    logger.info("  period %s: %d rows", p, len(df))
            except Exception as e:
                logger.warning("  period %s: express fetch failed: %s", p, str(e)[:100])
        if ex_frames:
            new = _dedup(pd.concat(ex_frames, ignore_index=True), EXPRESS_DEDUP)
            p = _accumulate(out_dir, "express.parquet", new, EXPRESS_DEDUP)
            n_stocks = new["stock_code"].nunique()
            logger.info("  saved %d rows (%d stocks) → %s", len(new), n_stocks, p)
            done.add("express")
        else:
            failed.append("express")
            logger.warning("  no express data fetched")

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/earnings",
            requested=requested, failed=failed, complete=done,
            success_count=len(done),
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
