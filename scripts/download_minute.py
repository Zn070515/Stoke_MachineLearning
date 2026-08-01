"""Download minute K-line data for A-share stocks via Sina Finance.

Sina provides up to ~1970 bars per call. Coverage varies by frequency:
  - 5min:  ~2 months
  - 15min: ~6 months
  - 30min: ~1 year
  - 60min: ~2 years (recommended for maximum history)

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_minute.py --frequency 60 --stocks 100
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_minute.py --frequency 30 --all
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_minute.py --frequency 60 --stock-list 600519,000001

Note: single-threaded because AKShare's Sina backend uses py_mini_racer (V8),
which cannot be shared across threads.
"""
import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.minute_source import MinuteSource
from stoke_ml.data.sources.a_shares.minute_source_sina_direct import SinaDirectMinuteSource
from stoke_ml.data.sources.a_shares.minute_source_tencent import TencentMinuteSource
from stoke_ml.data.minute_storage import MinuteStorage

_SOURCE_FACTORY = {
    "akshare": MinuteSource,
    "sina-direct": SinaDirectMinuteSource,
    "tencent": TencentMinuteSource,
}

_LOG_FILE = None


def _log(msg: str) -> None:
    """Write to stderr (immediate) AND log file (line-buffered)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def discover_stocks(data_dir: str, limit: int | None = None) -> list[str]:
    """Discover available stock codes from existing daily data.

    Uses the flat daily parquet files as the authoritative stock universe,
    ensuring we only download minute data for stocks with verified daily data.
    """
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.isdir(daily_dir):
        # Fall back to AKShare index discovery
        return _discover_via_index(limit)
    stocks = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(daily_dir)
        if f.endswith(".parquet")
    )
    return stocks[:limit] if limit else stocks


def _discover_via_index(limit: int | None = None) -> list[str]:
    """Discover stocks via AKShare CSI 300 + 500 index components."""
    try:
        import akshare as ak
        codes = set()
        for symbol in ["000300", "000905"]:
            try:
                df = ak.index_stock_cons_csindex(symbol=symbol)
                codes.update(df["成分券代码"].tolist())
            except Exception:
                _log("Failed to fetch index %s", symbol)
        stocks = sorted(codes)
        return stocks[:limit] if limit else stocks
    except Exception:
        _log("Cannot discover stocks — no daily data and AKShare unavailable")
        return []


def download_stock(
    code: str,
    source: MinuteSource,
    storage: MinuteStorage,
    frequency: str,
) -> tuple[str, int, str]:
    """Download minute data for one stock. Returns (code, n_bars, status)."""
    try:
        df = source.fetch(code, period=frequency, adjust="qfq")
        if df.empty:
            return (code, 0, "empty")
        storage.save(df, frequency=frequency)
        n = len(df)
        dt_min = df["datetime"].min().strftime("%Y-%m-%d")
        dt_max = df["datetime"].max().strftime("%Y-%m-%d")
        return (code, n, f"OK [{dt_min} → {dt_max}]")
    except Exception as e:
        return (code, 0, f"error: {str(e)[:80]}")


def main():
    parser = argparse.ArgumentParser(
        description="Download A-share minute K-line data"
    )
    parser.add_argument(
        "--frequency", type=str, default="60",
        choices=["5", "15", "30", "60"],
        help="Bar frequency in minutes (default: 60)",
    )
    parser.add_argument(
        "--stocks", type=int, default=None,
        help="Limit to first N stocks (default: all available)",
    )
    parser.add_argument(
        "--stock-list", type=str, default=None,
        help="Comma-separated stock codes (overrides --stocks)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available stocks (no limit)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Only retry previously failed stocks (checks log)",
    )
    parser.add_argument(
        "--shard", type=int, default=None,
        help="Shard index (0-based) for parallel download",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards (required with --shard)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip stocks already downloaded (checked via disk)",
    )
    parser.add_argument(
        "--source", type=str, default="akshare",
        choices=["akshare", "sina-direct", "tencent"],
        help="API data source (default: akshare)",
    )
    args = parser.parse_args()

    if (args.shard is not None) != (args.num_shards is not None):
        parser.error("--shard and --num-shards must be used together")
    if args.shard is not None and not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards})")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    log_path = f"download_minute_{args.frequency}{suffix}.log"
    global _LOG_FILE
    _LOG_FILE = open(log_path, "w", encoding="utf-8", buffering=1)

    if args.stock_list:
        codes = [c.strip() for c in args.stock_list.split(",")]
    else:
        limit = None if args.all else (args.stocks or 500)
        codes = discover_stocks(data_dir, limit)

    if args.shard is not None:
        total = len(codes)
        size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * size
        end = min(start + size, total)
        codes = codes[start:end]
        if not codes:
            _log(f"[ERROR] Shard {args.shard}: no stocks in range [{start}:{end}]")
            sys.exit(1)

    if not codes:
        _log("[ERROR] No stock codes found")
        sys.exit(1)

    source = _SOURCE_FACTORY[args.source]()
    storage = MinuteStorage(data_dir)

    if args.resume:
        existing = set(storage.list_stocks(frequency=args.frequency))
        before = len(codes)
        codes = [c for c in codes if c not in existing]
        skipped = before - len(codes)
        if skipped > 0:
            _log(f"Resume: skipping {skipped} already-downloaded stocks "
                 f"({len(codes)} remaining)")

    shard_tag = f"_shard{args.shard}" if args.shard is not None else ""
    _log(f"Downloading {len(codes)} stocks @ {args.frequency}min bars "
         f"via {args.source}{shard_tag}")

    success, empty_count, error_count = 0, 0, 0
    total_bars = 0
    t0 = time.time()

    report_interval = max(1, min(50, len(codes) // 20))

    for i, code in enumerate(codes):
        _, n_bars, status = download_stock(code, source, storage, args.frequency)
        total_bars += n_bars

        if status == "empty":
            empty_count += 1
        elif status.startswith("error"):
            error_count += 1
        else:
            success += 1

        if (i + 1) % report_interval == 0 or (i + 1) == len(codes):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta = (len(codes) - i - 1) / max(rate, 0.1)
            _log(f"[{i + 1}/{len(codes)}] {rate:.0f} stk/min | "
                 f"{success} OK {empty_count} empty {error_count} err | "
                 f"{total_bars} bars | ETA {eta:.0f} min")

    elapsed = time.time() - t0
    _log(f"Done: {success} success, {empty_count} empty, {error_count} error, "
         f"{total_bars} total bars in {elapsed / 60:.1f} min")

    if success > 0:
        stored = storage.list_stocks(frequency=args.frequency)
        _log(f"Stored minute data for {len(stored)} stocks @ {args.frequency}min")

    _LOG_FILE.close()


if __name__ == "__main__":
    main()
