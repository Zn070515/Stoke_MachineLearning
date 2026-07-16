"""Download daily valuation ratios (PE/PB/PS/PCF) from Baostock.

Baostock provides daily PE(TTM), PB(MRQ), PS(TTM), PCF(NCF TTM) since ~1990.
Free, no API key required. Incremental: skips stocks already on disk.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python -u scripts/download_valuation.py
  PYTHONPATH=. ./.venv/Scripts/python -u scripts/download_valuation.py --stocks 600519,000001
  PYTHONPATH=. ./.venv/Scripts/python -u scripts/download_valuation.py --start 2010-01-01
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

VALUATION_COLS = ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]


def _bs_code(stock_code: str) -> str:
    if stock_code.startswith("6"):
        return f"sh.{stock_code}"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        return f"sz.{stock_code}"
    elif stock_code.startswith("8") or stock_code.startswith("4"):
        return f"bj.{stock_code}"
    raise ValueError(f"Unknown exchange for {stock_code}")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} {msg}", flush=True)


def get_stocks_from_disk(data_dir: str) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    codes = {f.replace(".parquet", "") for f in os.listdir(daily_dir) if f.endswith(".parquet")}
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description="Download daily valuation ratios from Baostock")
    parser.add_argument("--stocks", type=str, default=None)
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    from stoke_ml.config import load_config
    from stoke_ml.data.market_wide_storage import MarketWideStorage

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.stocks:
        stock_list = [c.strip() for c in args.stocks.split(",")]
    else:
        stock_list = get_stocks_from_disk(data_dir)
        if not stock_list:
            _log("ERROR: No stocks found")
            sys.exit(1)

    # Skip already-downloaded stocks
    val_base = os.path.join(data_dir, "a_shares", "valuation")
    existing = set()
    if os.path.isdir(val_base):
        existing = {f.replace(".parquet", "") for f in os.listdir(val_base) if f.endswith(".parquet")}
    to_download = [c for c in stock_list if c not in existing]
    _log(f"Valuation: {len(stock_list)} total, {len(existing)} cached, {len(to_download)} to fetch")

    if not to_download:
        _log("All stocks already downloaded.")
        return

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        _log(f"ERROR: Baostock login failed: {lg.error_code} {lg.error_msg}")
        sys.exit(1)

    storage = MarketWideStorage(data_dir, "valuation")
    batch_size = 50
    errors = 0
    t0 = time.time()

    for i, code in enumerate(to_download):
        try:
            bsc = _bs_code(code)
            rs = bs.query_history_k_data_plus(
                bsc, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=args.start, end_date=args.end,
                frequency="d", adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=["date", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"])
                df["date"] = pd.to_datetime(df["date"])
                for col in VALUATION_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=VALUATION_COLS, how="all")
                if not df.empty:
                    df["stock_code"] = code
                    storage.save(df)
        except Exception:
            errors += 1
            if errors <= 3:
                _log(f"  ERROR {code}: check traceback")

        # Save progress every batch_size stocks
        if (i + 1) % batch_size == 0 or i == len(to_download) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_download) - i - 1) / rate if rate > 0 else 0
            _log(f"  {i+1}/{len(to_download)} stocks ({rate:.1f}/s, ETA {eta/60:.0f}m)")

        # Re-login every 200 stocks
        if (i + 1) % 200 == 0:
            bs.logout()
            time.sleep(0.5)
            lg = bs.login()
            if lg.error_code != "0":
                _log(f"ERROR: re-login failed at stock {i}")
                break

    bs.logout()
    _log(f"Done: {len(to_download)} stocks, {errors} errors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
