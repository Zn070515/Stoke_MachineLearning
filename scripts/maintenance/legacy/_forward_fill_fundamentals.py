# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""One-shot: forward-fill quarterly fundamentals to daily for all available stocks.
Saves to: data/a_shares/fundamentals_daily/{code}.parquet
Usage: PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_forward_fill_fundamentals.py [shard_k/N]
"""
import logging
import os
import sys

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.fundamental_storage import FundamentalStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    data_dir = cfg.project.data_dir
    calendar = TradingCalendar("a_shares")
    storage = FundamentalStorage(data_dir, calendar)

    # Get all stocks with fundamentals data (flat + partitioned)
    fundamentals_dir = os.path.join(data_dir, "a_shares", "fundamentals")
    codes = set()
    for root, _dirs, files in os.walk(fundamentals_dir):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    codes = sorted(codes)

    # Sharding
    if len(sys.argv) > 1:
        k, n = sys.argv[1].split("/")
        k, n = int(k), int(n)
        codes = [c for i, c in enumerate(codes) if i % n == k]
        logger.info("Shard %d/%d: %d stocks", k, n, len(codes))

    out_dir = os.path.join(data_dir, "a_shares", "fundamentals_daily")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Forward-filling %d stocks to daily", len(codes))
    success, empty, fail = 0, 0, 0

    for i, code in enumerate(codes):
        # Skip if already done
        out_path = os.path.join(out_dir, f"{code}.parquet")
        if os.path.exists(out_path):
            success += 1
            continue

        try:
            daily = storage.forward_fill_to_daily(code, "2015-01-01", "2026-07-24")
            if daily.empty:
                empty += 1
            else:
                daily.to_parquet(out_path, index=False, compression='lz4')
                success += 1
        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)
            fail += 1

        if (i + 1) % 200 == 0:
            logger.info("Progress: %d/%d (ok=%d, empty=%d, fail=%d)",
                        i + 1, len(codes), success, empty, fail)

    logger.info("Done: %d success, %d empty, %d fail", success, empty, fail)


if __name__ == "__main__":
    main()
