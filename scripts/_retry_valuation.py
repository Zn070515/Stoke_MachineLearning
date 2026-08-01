"""Retry Baostock valuation for stocks missed in first pass. Sharded with resume.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_retry_valuation.py 0 4
"""
import os, sys, time
from datetime import datetime
import pandas as pd
import numpy as np

VALUATION_COLS = ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]


def _bs_code(stock_code: str) -> str:
    if stock_code.startswith("6"): return f"sh.{stock_code}"
    elif stock_code.startswith("0") or stock_code.startswith("3"): return f"sz.{stock_code}"
    elif stock_code.startswith("8") or stock_code.startswith("4"): return f"bj.{stock_code}"
    raise ValueError(f"Unknown exchange for {stock_code}")


def _log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <shard_index> <total_shards>")
        sys.exit(1)
    shard_idx = int(sys.argv[1])
    total_shards = int(sys.argv[2])

    from stoke_ml.config import load_config
    from stoke_ml.data.market_wide_storage import MarketWideStorage

    cfg = load_config()
    data_dir = cfg.project.data_dir

    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    all_stocks = sorted({f.replace(".parquet", "") for f in os.listdir(daily_dir) if f.endswith(".parquet")})
    val_dir = os.path.join(data_dir, "a_shares", "valuation")
    existing = {f.replace(".parquet", "") for f in os.listdir(val_dir) if f.endswith(".parquet")} if os.path.isdir(val_dir) else set()
    missing = [c for c in all_stocks if c not in existing]
    stocks = [s for i, s in enumerate(missing) if i % total_shards == shard_idx]

    _log(f"Shard {shard_idx}/{total_shards}: {len(stocks)} missing stocks to retry")
    if not stocks:
        return

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        _log(f"ERROR: login failed: {lg.error_code} {lg.error_msg}")
        sys.exit(1)

    storage = MarketWideStorage(data_dir, "valuation")
    t0 = time.time()
    done = errors = 0

    for i, code in enumerate(stocks):
        ok = False
        for attempt in range(3):
            try:
                bsc = _bs_code(code)
                rs = bs.query_history_k_data_plus(
                    bsc, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                    start_date="2015-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
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
                        ok = True
                break
            except Exception:
                time.sleep(1.0)
        if ok:
            done += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(stocks) - i - 1) / rate if rate > 0 else 0
            _log(f"  {i+1}/{len(stocks)} ({rate:.1f}/s, ETA {eta/60:.0f}m) done={done} err={errors}")

        if (i + 1) % 200 == 0:
            bs.logout(); time.sleep(0.3)
            lg = bs.login()
            if lg.error_code != "0":
                _log(f"ERROR: re-login failed at {i}"); break

    bs.logout()
    _log(f"Shard {shard_idx} done: {done} ok, {errors} err, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
