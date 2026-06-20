"""Download sector ETF fund flow data.

Usage:
  python scripts/download_etf_flow.py
  python scripts/download_etf_flow.py --start 2020-01-01 --end 2024-12-31
  python scripts/download_etf_flow.py --sector 半导体,券商
"""
import argparse
import logging
import time

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.etf_flow_source import SectorETFFlowSource
from stoke_ml.data.etf_storage import ETFStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download sector ETF flow data")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--sector", type=str, default=None,
                        help="Specific sector name(s), comma-separated (default: all)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between ETF fetches (default: 0.3)")
    args = parser.parse_args()

    if args.end is None:
        from datetime import datetime
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    source = SectorETFFlowSource()
    storage = ETFStorage(data_dir)

    sector_names = None
    if args.sector:
        sector_names = [s.strip() for s in args.sector.split(",")]

    logger.info("Downloading sector ETF flow from %s to %s", args.start, args.end)

    total_rows = 0
    for sector_name, sector_info in source._sector_map.items():
        if sector_names and sector_name not in sector_names:
            continue

        etf_codes = sector_info.get("etf_codes", [])
        logger.info(
            "  %s (%s): %d ETFs",
            sector_name, sector_info.get("name_en", ""), len(etf_codes),
        )

        df = source.fetch_sector_flow(sector_name, args.start, args.end)
        if not df.empty:
            storage.save(df)
            logger.info("    %d daily rows saved", len(df))
            total_rows += len(df)
        else:
            logger.warning("    no data returned")

        time.sleep(args.sleep)

    logger.info("Done: %d total rows", total_rows)


if __name__ == "__main__":
    main()
