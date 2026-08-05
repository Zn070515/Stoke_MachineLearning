"""Download sector ETF fund flow data.

Usage:
  python scripts/production/download_etf_flow.py
  python scripts/production/download_etf_flow.py --start 2020-01-01 --end 2024-12-31
  python scripts/production/download_etf_flow.py --sector 半导体,券商
"""
import argparse
import logging
import time

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.etf_flow_source import SectorETFFlowSource
from stoke_ml.data.etf_storage import ETFStorage
from stoke_ml.data.download_manifest import write_run_manifest

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

    requested_sectors = [
        s for s in source._sector_map
        if not sector_names or s in sector_names
    ]
    total_rows = 0
    done_sectors: set[str] = set()
    failed_sectors: list[str] = []
    for sector_name in requested_sectors:
        sector_info = source._sector_map[sector_name]

        etf_codes = sector_info.get("etf_codes", [])
        logger.info(
            "  %s (%s): %d ETFs",
            sector_name, sector_info.get("name_en", ""), len(etf_codes),
        )

        try:
            df = source.fetch_sector_flow(sector_name, args.start, args.end)
            if not df.empty:
                storage.save(df)
                logger.info("    %d daily rows saved", len(df))
                total_rows += len(df)
                done_sectors.add(sector_name)
            else:
                logger.warning("    no data returned")
                failed_sectors.append(sector_name)
        except Exception as e:
            logger.warning("    %s: error: %s", sector_name, str(e)[:120])
            failed_sectors.append(sector_name)

        time.sleep(args.sleep)

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/etf_flow",
            start_date=args.start, end_date=args.end,
            requested=requested_sectors,
            failed=failed_sectors, complete=done_sectors,
            success_count=len(done_sectors),
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)

    logger.info("Done: %d total rows", total_rows)


if __name__ == "__main__":
    main()
