# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Download valuation for a shard of stocks, with resume support.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_download_valuation_shard.py 0 4
"""

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


def get_stocks_from_daily(data_dir: str) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    return sorted({f.replace(".parquet", "") for f in os.listdir(daily_dir) if f.endswith(".parquet")})


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <shard_index> <total_shards> [start_date] [end_date]")
        sys.exit(1)

    shard_idx = int(sys.argv[1])
    total_shards = int(sys.argv[2])
    start_date = sys.argv[3] if len(sys.argv) > 3 else "2015-01-01"
    end_date = sys.argv[4] if len(sys.argv) > 4 else datetime.now().strftime("%Y-%m-%d")

    from stoke_ml.config import load_config
    from stoke_ml.data.market_wide_storage import MarketWideStorage

    cfg = load_config()
    data_dir = cfg.project.data_dir
    all_stocks = get_stocks_from_daily(data_dir)
    if not all_stocks:
        _log("ERROR: No stocks found")
        sys.exit(1)

    # Shard + resume
    stocks = [s for i, s in enumerate(all_stocks) if i % total_shards == shard_idx]
    val_base = os.path.join(data_dir, "a_shares", "valuation")
    existing = set()
    if os.path.isdir(val_base):
        existing = {f.replace(".parquet", "") for f in os.listdir(val_base) if f.endswith(".parquet")}
    todo = [c for c in stocks if c not in existing]
    _log(f"Shard {shard_idx}/{total_shards}: {len(todo)} to fetch ({len(existing)} cached, {len(stocks)} total)")

    if not todo:
        _log("All stocks in shard already cached.")
        return

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        _log(f"ERROR: Baostock login failed: {lg.error_code} {lg.error_msg}")
        sys.exit(1)

    storage = MarketWideStorage(data_dir, "valuation")
    t0 = time.time()
    errors = 0

    for i, code in enumerate(todo):
        try:
            bsc = _bs_code(code)
            rs = bs.query_history_k_data_plus(
                bsc, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=start_date, end_date=end_date,
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
                _log(f"  ERROR {code}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            _log(f"  {i+1}/{len(todo)} ({rate:.1f}/s, ETA {eta/60:.0f}m)")

        if (i + 1) % 200 == 0:
            bs.logout()
            time.sleep(0.3)
            lg = bs.login()
            if lg.error_code != "0":
                _log(f"ERROR: re-login failed at stock {i}")
                break

    bs.logout()
    _log(f"Shard {shard_idx} done: {len(todo)} done, {errors} errors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
