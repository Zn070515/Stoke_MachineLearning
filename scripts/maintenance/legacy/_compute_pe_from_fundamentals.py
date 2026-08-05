# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Compute PE_TTM from quarterly fundamental EPS + daily K-line close prices.

Pure local computation — no API calls, no rate limits. Uses PIT-aligned
merge_asof to ensure only EPS data available on or before each trading day
is used (no look-ahead bias).

EPS in fundamentals is CUMULATIVE (累计):
  Q1=3mo, Q2=6mo(H1), Q3=9mo, Q4=12mo(FY)

TTM EPS formula:
  TTM = latest_FY + current_partial_year - prior_year_same_period

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_compute_pe_from_fundamentals.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_compute_pe_from_fundamentals.py --stocks 600519,000001

Output: data/a_shares/valuation/{code}.parquet
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent

# Months that mark quarter-end report_dates
QM = {3, 6, 9, 12}



def compute_ttm_eps(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Given cumulative EPS data sorted by report_date, compute TTM EPS.

    fund_df columns: report_date, eps (cumulative, sorted)
    Returns DataFrame with report_date and eps_ttm.
    """
    df = fund_df.sort_values("report_date").copy()
    # Drop rows with NaN EPS or NaT report_date
    df = df.dropna(subset=["eps", "report_date"])
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df = df.dropna(subset=["eps"])

    if df.empty:
        return pd.DataFrame(columns=["report_date", "eps_ttm"])

    rows = []

    for _, row in df.iterrows():
        rd = row["report_date"]
        this_eps = float(row["eps"])
        qtr = rd.month

        if qtr == 12:
            ttm = this_eps
        else:
            fy_rows = df[
                (df["report_date"].dt.month == 12)
                & (df["report_date"] < rd)
            ]
            if fy_rows.empty:
                ttm = this_eps
            else:
                last_fy_eps = float(fy_rows.iloc[-1]["eps"])
                prior_rd = pd.Timestamp(year=rd.year - 1, month=rd.month, day=rd.day)
                prior_rows = df[df["report_date"] == prior_rd]
                if not prior_rows.empty:
                    prior_ytd_eps = float(prior_rows.iloc[0]["eps"])
                else:
                    prior_ytd_eps = 0.0
                ttm = last_fy_eps + this_eps - prior_ytd_eps

        rows.append({"report_date": rd, "eps_ttm": max(ttm, 0.001)})

    return pd.DataFrame(rows)


def build_pit_pe(kline_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """PIT merge daily K-line with quarterly fundamentals.

    kline_df: daily data with date, close columns
    fund_df: fundamentals with report_date, disclose_date, eps columns

    Uses merge_asof on disclose_date to ensure no look-ahead.
    """
    daily = kline_df[["date", "close"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=False).dt.tz_localize(None)
    daily = daily.sort_values("date")

    # Compute TTM EPS from cumulative quarterly EPS (based on report_date)
    ttm_df = compute_ttm_eps(fund_df[["report_date", "eps"]].copy())

    # Merge TTM EPS back to fundamentals (keyed by report_date)
    fund = fund_df.merge(ttm_df, on="report_date", how="left")
    fund["disclose_date"] = pd.to_datetime(fund["disclose_date"], utc=False).dt.tz_localize(None)
    fund = fund.sort_values("disclose_date")

    # PIT-safe merge: for each trading day, use the latest disclosed EPS data
    # Normalize key column to datetime64[us] to avoid ms/us mismatch
    trade_dates = daily["date"].to_numpy().astype("datetime64[us]")
    eps_dates = fund["disclose_date"].to_numpy().astype("datetime64[us]")

    trade_col = pd.DataFrame({"trade_date": trade_dates})
    eps_col = pd.DataFrame({
        "trade_date": eps_dates,
        "report_date": fund["report_date"].values,
        "eps_ttm": fund["eps_ttm"].values,
    })

    daily_eps = pd.merge_asof(
        trade_col,
        eps_col,
        on="trade_date",
        direction="backward",
    )
    daily_eps["close"] = daily["close"].values

    # Record the report_date used for transparency
    daily_eps["eps_ttm"] = daily_eps["eps_ttm"].ffill()

    # Compute PE
    daily_eps["pe_ttm"] = daily_eps["close"] / daily_eps["eps_ttm"]
    # Guard: negative EPS → negative PE is nonsense, set to NaN
    daily_eps.loc[daily_eps["eps_ttm"] <= 0, "pe_ttm"] = np.nan
    # Guard: absurd PE values (>10,000 or <0) → cap
    daily_eps.loc[daily_eps["pe_ttm"] > 10000, "pe_ttm"] = np.nan

    result = daily_eps[["trade_date", "pe_ttm"]].rename(
        columns={"trade_date": "date"}
    )
    result["pb_mrq"] = np.nan
    result["ps_ttm"] = np.nan
    result["pcf_ttm"] = np.nan

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute PE_TTM from fundamentals + K-line data"
    )
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all missing)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing valuation files")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    cfg = load_config()
    data_dir = Path(cfg.project.data_dir) / "a_shares"

    daily_dir = data_dir / "daily"
    fund_dir = data_dir / "fundamentals"
    val_dir = data_dir / "valuation"
    val_dir.mkdir(exist_ok=True)

    all_stocks_set = {f.stem for f in daily_dir.glob("*.parquet")}
    fund_stocks = {f.stem for f in fund_dir.glob("*.parquet")}

    if args.force:
        existing = set()
    else:
        existing = {f.stem for f in val_dir.glob("*.parquet")}

    if args.stocks:
        stocks = [c.strip() for c in args.stocks.split(",")]
    else:
        stocks = sorted(all_stocks_set - existing)

    # Only process stocks that have fundamental data
    stocks = [s for s in stocks if s in fund_stocks]

    if not stocks:
        logger.info("Nothing to compute — all %d stocks have valuation data.", len(existing))
        return 0

    no_fund = len(all_stocks_set - existing - fund_stocks)
    logger.info(
        "PE_TTM: %d/%d stocks cached, %d to compute (%d without fundamentals, skipped)",
        len(existing), len(all_stocks_set), len(stocks), max(0, no_fund)
    )

    t0 = time.time()
    done = fail = skip_no_price = 0
    n = len(stocks)

    for i, code in enumerate(stocks):
        try:
            kline_df = pd.read_parquet(daily_dir / f"{code}.parquet")
            if "close" not in kline_df.columns:
                skip_no_price += 1
                continue

            fund_df = pd.read_parquet(fund_dir / f"{code}.parquet")
            if fund_df.empty or "eps" not in fund_df.columns:
                fail += 1
                continue

            result = build_pit_pe(kline_df, fund_df)
            if result.empty:
                fail += 1
                continue

            result["stock_code"] = code
            result.to_parquet(
                val_dir / f"{code}.parquet", index=False, compression="lz4"
            )
            done += 1
            if result["pe_ttm"].isna().all():
                logger.debug("  %s: EPS all negative, PE set to NaN", code)

        except Exception:
            fail += 1

        if (i + 1) % 200 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] done=%d fail=%d noclose=%d (%.1f stk/s, ETA %.0fs)",
                i + 1, n, done, fail, skip_no_price, rate, eta,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d ok, %d fail, %d no-close, %.0fs",
        done, fail, skip_no_price, elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
