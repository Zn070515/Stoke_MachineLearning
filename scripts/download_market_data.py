"""Download market-wide data: dragon-tiger, margin, northbound.

Usage:
  python scripts/download_market_data.py --type all
  python scripts/download_market_data.py --type margin --start 2020-01-01
  python scripts/download_market_data.py --type northbound --stocks 600519
  python scripts/download_market_data.py --type all --concurrent
"""
import argparse
import logging
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.margin_source import MarginTradingSource
from stoke_ml.data.sources.a_shares.dragon_tiger_source import DragonTigerSource
from stoke_ml.data.sources.a_shares.northbound_source import NorthboundSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TYPE_MAP = {
    "margin": ("dragon_tiger", None),  # placeholder, handled separately below
    "dragon_tiger": ("dragon_tiger", DragonTigerSource),
    "northbound": ("northbound", NorthboundSource),
    "all": None,
}


def main():
    parser = argparse.ArgumentParser(description="Download A-share market data")
    parser.add_argument("--type", type=str, default="all",
                        choices=["margin", "dragon_tiger", "northbound", "all"],
                        help="Data type to download (default: all)")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    args = parser.parse_args()

    if args.end is None:
        from datetime import datetime
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    to_download = []
    if args.type == "all":
        to_download = [
            ("margin", MarginTradingSource, "margin_trading"),
            ("dragon_tiger", DragonTigerSource, "dragon_tiger"),
            ("northbound", NorthboundSource, "northbound"),
        ]
    elif args.type == "margin":
        to_download = [("margin", MarginTradingSource, "margin_trading")]
    elif args.type == "dragon_tiger":
        to_download = [("dragon_tiger", DragonTigerSource, "dragon_tiger")]
    elif args.type == "northbound":
        to_download = [("northbound", NorthboundSource, "northbound")]

    for label, source_cls, storage_type in to_download:
        logger.info("=== Downloading %s (%s to %s) ===", label, args.start, args.end)
        t0 = time.time()

        try:
            source = source_cls()
        except Exception as e:
            logger.error("Failed to init %s source: %s", label, e)
            continue

        storage = MarketWideStorage(data_dir, storage_type)

        if label == "margin":
            df = source.fetch_daily(args.start, args.end)
        elif label == "dragon_tiger":
            if args.stocks:
                frames = []
                for code in [c.strip() for c in args.stocks.split(",")]:
                    sdf = source.fetch_by_stock(code, args.start, args.end)
                    if not sdf.empty:
                        frames.append(sdf)
                    time.sleep(args.sleep)
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df = source.fetch_daily(args.start, args.end)
        elif label == "northbound":
            df = source.fetch_individual(args.start, args.end)
        else:
            df = pd.DataFrame()

        if df is not None and not df.empty:
            storage.save(df)
            logger.info(
                "  %s: %d rows saved (%.1fs)",
                label, len(df), time.time() - t0,
            )
        else:
            logger.warning("  %s: no data returned", label)

    logger.info("Done.")


if __name__ == "__main__":
    main()
