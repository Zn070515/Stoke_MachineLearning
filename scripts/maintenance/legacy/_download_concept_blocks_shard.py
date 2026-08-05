# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Download concept_blocks for a shard of stocks, with resume support.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_download_concept_blocks_shard.py 0 4
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_download_concept_blocks_shard.py 1 4
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_download_concept_blocks_shard.py 2 4
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_download_concept_blocks_shard.py 3 4
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stocks_from_daily(data_dir: str) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    codes = set()
    for root, _dirs, files in os.walk(daily_dir):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <shard_index> <total_shards> [sleep]")
        sys.exit(1)

    shard_idx = int(sys.argv[1])
    total_shards = int(sys.argv[2])
    sleep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2

    from stoke_ml.config import load_config
    from stoke_ml.data.sources.a_shares.sector_source import ConceptBlockSource
    from stoke_ml.data.market_wide_storage import MarketWideStorage

    cfg = load_config()
    data_dir = cfg.project.data_dir
    all_stocks = get_stocks_from_daily(data_dir)
    if not all_stocks:
        logger.error("No daily stocks found.")
        sys.exit(1)

    # Shard
    stocks = [s for i, s in enumerate(all_stocks) if i % total_shards == shard_idx]
    out_dir = os.path.join(data_dir, "a_shares", "concept_blocks")

    # Resume: skip already-downloaded stocks
    todo = []
    skipped = 0
    for code in stocks:
        if os.path.isfile(os.path.join(out_dir, f"{code}.parquet")):
            skipped += 1
        else:
            todo.append(code)

    if not todo:
        logger.info("Shard %d/%d: all %d stocks cached, nothing to do.",
                    shard_idx, total_shards, len(stocks))
        return

    logger.info("Shard %d/%d: %d stocks (%d cached, %d to fetch)",
                shard_idx, total_shards, len(stocks), skipped, len(todo))

    source = ConceptBlockSource(min_interval=sleep_s)
    storage = MarketWideStorage(data_dir, "concept_blocks")
    t0 = time.time()
    done = 0

    for code in todo:
        try:
            df = source.fetch(code)
            if not df.empty:
                storage.save(df)
            done += 1
        except Exception:
            logger.debug("concept_blocks fetch failed for %s", code)
        if done % 200 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(todo) - done) / rate if rate > 0 else 0
            logger.info("  shard %d: %d/%d done (%.1f stk/min, ETA %.0fs)",
                        shard_idx, done, len(todo), rate * 60, eta)

    source.close()
    elapsed = time.time() - t0
    logger.info("Shard %d/%d: %d stocks done in %.1fs (%.1f stk/min)",
                shard_idx, total_shards, done, elapsed, done / elapsed * 60)


if __name__ == "__main__":
    main()
