"""Download daily data for A-share stock universe via 4-source failover."""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import akshare as ak

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.sources.a_shares.failover import AShareDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stock_codes(indices: list[str] | None = None) -> list[str]:
    """Fetch stock codes for the given index symbols via AKShare.

    Args:
        indices: List like ['000300', '000905']. Default: CSI 300 + CSI 500.
    """
    if indices is None:
        indices = ["000300", "000905"]

    codes = set()
    for symbol in indices:
        name = {"000300": "CSI 300", "000905": "CSI 500"}.get(symbol, symbol)
        try:
            df = ak.index_stock_cons_csindex(symbol=symbol)
            new_codes = set(df["成分券代码"].tolist())
            codes.update(new_codes)
            logger.info("Fetched %d stocks from %s (%s)", len(new_codes), name, symbol)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", name, e)

    return sorted(codes)


def get_all_a_share_codes() -> list[str]:
    """Fetch ALL A-share stock codes via AKShare stock_info_a_code_name."""
    logger.info("Fetching full A-share stock list (may take ~5s)...")
    df = ak.stock_info_a_code_name()
    codes = sorted(df["code"].tolist())
    logger.info("Total A-share stocks: %d", len(codes))
    return codes


def filter_existing(codes: list[str], data_dir: str) -> tuple[list[str], set[str]]:
    """Filter out stocks already on disk. Returns (to_download, already_have)."""
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.isdir(daily_dir):
        return codes, set()

    existing = set()
    for f in os.listdir(daily_dir):
        if f.endswith(".parquet"):
            existing.add(f.replace(".parquet", ""))

    to_download = [c for c in codes if c not in existing]
    return to_download, existing


def main():
    parser = argparse.ArgumentParser(description="Download A-share daily data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: config)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: from config universe)")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Seconds between stocks (default: 1.5)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated AKShare index symbols")
    parser.add_argument("--all", action="store_true", dest="all_stocks",
                        help="Download ALL A-shares (~5500 stocks)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stocks already in data/a_shares/daily/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    storage = DataStorage(cfg.project.data_dir)
    downloader = AShareDownloader()

    start_date = args.start or cfg.markets.a_shares.start_date
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    elif args.all_stocks:
        codes = get_all_a_share_codes()
    elif args.indices:
        codes = get_stock_codes(args.indices.split(","))
    else:
        codes = get_stock_codes()

    if not codes:
        logger.error("No stock codes to download.")
        sys.exit(1)

    n_skipped = 0
    if args.skip_existing:
        codes, existing = filter_existing(codes, cfg.project.data_dir)
        n_skipped = len(existing)
        logger.info("Skipping %d already-downloaded stocks, %d to download",
                     n_skipped, len(codes))

    logger.info("Downloading %d stocks from %s to %s", len(codes), start_date, end_date)
    success, fail = 0, 0

    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        logger.info("[%d/%d] Fetching %s ...", i + 1, len(codes), code)
        df = downloader.fetch_daily(code, start_date, end_date)

        if df.empty:
            logger.warning("  %s: EMPTY (all sources failed)", code)
            fail += 1
            continue

        storage.save_daily(df)
        dates = pd.to_datetime(df["date"])
        logger.info("  %s: %d rows [%s → %s]", code, len(df),
                     dates.min().strftime("%Y-%m-%d"),
                     dates.max().strftime("%Y-%m-%d"))
        success += 1

    logger.info("Done: %d success, %d fail, %d skip", success, fail, n_skipped)


if __name__ == "__main__":
    import pandas as pd
    main()
