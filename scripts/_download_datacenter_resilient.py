"""Resilient batch wrapper for download_datacenter — small batches, incremental save, resume.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_datacenter_resilient.py --type block_trade
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_datacenter_resilient.py --type lockup --batch 200
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_datacenter_resilient.py --type block_trade --stocks 000001,600519
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Late-bound imports — resolved in main()
MarketWideStorage = None
BlockTradeSource = None
ShareholderSource = None
LockupExpirySource = None
DividendSource = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


def get_missing_stocks(data_dir, storage_key):
    """Return stock codes that have daily data but no {storage_key} data yet."""
    daily_dir = Path(data_dir) / "a_shares" / "daily"
    target_dir = Path(data_dir) / "a_shares" / storage_key

    daily_codes = set()
    for f in daily_dir.glob("**/*.parquet"):
        daily_codes.add(f.stem)

    existing_codes = set()
    if target_dir.exists():
        for f in target_dir.glob("*.parquet"):
            existing_codes.add(f.stem)

    missing = sorted(daily_codes - existing_codes)
    logger.info("%s: %d/%d stocks already have data, %d missing",
                storage_key, len(existing_codes), len(daily_codes), len(missing))
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Resilient batch download of datacenter data"
    )
    parser.add_argument("--type", type=str, required=True,
                        choices=["block_trade", "shareholder", "lockup", "dividend"],
                        help="Data type to download")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all missing)")
    parser.add_argument("--batch", type=int, default=200,
                        help="Stocks per batch (default: 200)")
    parser.add_argument("--sleep", type=float, default=0.6,
                        help="Seconds between API calls (default: 0.6)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.market_wide_storage import MarketWideStorage as _MWS
    from stoke_ml.data.sources.a_shares.datacenter_sources import (
        BlockTradeSource as _BTS, ShareholderSource as _SHS,
        LockupExpirySource as _LES, DividendSource as _DS,
    )
    global MarketWideStorage, BlockTradeSource, ShareholderSource, LockupExpirySource, DividendSource
    MarketWideStorage = _MWS
    BlockTradeSource = _BTS
    ShareholderSource = _SHS
    LockupExpirySource = _LES
    DividendSource = _DS

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.stocks:
        stock_list = [c.strip() for c in args.stocks.split(",")]
    else:
        type_to_key = {
            "block_trade": "block_trade",
            "shareholder": "shareholder",
            "lockup": "lockup",
            "dividend": "dividend",
        }
        storage_key = type_to_key[args.type]
        stock_list = get_missing_stocks(data_dir, storage_key)

    if not stock_list:
        logger.info("Nothing to download — all stocks covered.")
        return 0

    type_to_cls = {
        "block_trade": BlockTradeSource,
        "shareholder": ShareholderSource,
        "dividend": DividendSource,
    }

    if args.type == "lockup":
        _run_lockup_batches(stock_list, data_dir, args)
    else:
        source_cls = type_to_cls[args.type]
        method_name = "fetch"
        storage_key = args.type
        _run_per_stock_batches(
            args.type, storage_key, source_cls, method_name,
            stock_list, data_dir, args,
        )

    return 0


def _run_per_stock_batches(dtype, storage_key, source_cls, method_name,
                           stock_list, data_dir, args):
    """Download in small batches, saving after each batch."""
    batch_size = args.batch
    n_batches = (len(stock_list) + batch_size - 1) // batch_size
    total_saved = 0
    total_failed = 0
    t_start = time.time()

    logger.info("=== %s: %d stocks in %d batches of %d ===",
                dtype, len(stock_list), n_batches, batch_size)

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(stock_list))
        batch = stock_list[start:end]
        batch_t0 = time.time()

        source = source_cls(min_interval=args.sleep)
        storage = MarketWideStorage(data_dir, storage_key)
        fetch_fn = getattr(source, method_name)

        frames = []
        batch_failed = 0
        for code in batch:
            try:
                sdf = fetch_fn(code)
                if not sdf.empty:
                    frames.append(sdf)
            except Exception:
                batch_failed += 1

        source.close()

        if frames:
            df = pd.concat(frames, ignore_index=True)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= pd.Timestamp(args.start)) & \
                       (df["date"] <= pd.Timestamp(args.end))
                df = df[mask]
            if not df.empty:
                storage.save(df)
                total_saved += len(df)

        total_failed += batch_failed
        batch_elapsed = time.time() - batch_t0
        pct = (bi + 1) / n_batches * 100
        elapsed = time.time() - t_start
        rate = (bi + 1) * batch_size / elapsed
        logger.info("  [%d/%d] %s: %d rows saved, %d failed (%.1fs, %.0f%% @ %.1f stk/s)",
                    bi + 1, n_batches, dtype, total_saved, total_failed,
                    batch_elapsed, pct, rate)

    elapsed = time.time() - t_start
    logger.info("%s: DONE — %d rows total in %d batches (%.1f min)",
                dtype, total_saved, n_batches, elapsed / 60)


def _run_lockup_batches(stock_list, data_dir, args):
    """Special handling for lockup (history + upcoming)."""

    batch_size = args.batch
    n_batches = (len(stock_list) + batch_size - 1) // batch_size

    hist_storage = MarketWideStorage(data_dir, "lockup")
    upcoming_storage = MarketWideStorage(data_dir, "lockup_upcoming")

    logger.info("=== lockup: %d stocks in %d batches of %d ===",
                len(stock_list), n_batches, batch_size)

    t_start = time.time()
    total_hist = 0
    total_upcoming = 0

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(stock_list))
        batch = stock_list[start:end]

        source = LockupExpirySource(min_interval=args.sleep)
        hist_frames = []
        upcoming_frames = []

        for code in batch:
            try:
                data = source.fetch_all(code, trade_date=args.end)
                if not data["history"].empty:
                    hist_frames.append(data["history"])
                if not data["upcoming"].empty:
                    uc = data["upcoming"].copy()
                    uc["is_upcoming"] = True
                    upcoming_frames.append(uc)
            except Exception:
                pass

        source.close()

        if hist_frames:
            df = pd.concat(hist_frames, ignore_index=True)
            hist_storage.save(df)
            total_hist += len(df)
        if upcoming_frames:
            df = pd.concat(upcoming_frames, ignore_index=True)
            upcoming_storage.save(df)
            total_upcoming += len(df)

        pct = (bi + 1) / n_batches * 100
        rate = (bi + 1) * batch_size / (time.time() - t_start)
        logger.info("  [%d/%d] lockup: %d hist + %d upcoming (%.0f%% @ %.1f stk/s)",
                    bi + 1, n_batches, total_hist, total_upcoming, pct, rate)

    elapsed = time.time() - t_start
    logger.info("lockup: DONE — %d history + %d upcoming rows (%.1f min)",
                total_hist, total_upcoming, elapsed / 60)


if __name__ == "__main__":
    sys.exit(main())
