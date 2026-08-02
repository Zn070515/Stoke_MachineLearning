"""Rebuild news_silver (PIT-aligned) from news_raw for all stocks.

Silver lags raw because it is only written during live downloads; stocks whose
raw was refreshed afterward keep a stale silver (missing the newest week).
This PIT-aligns every raw file and merge-saves it into silver (dedup on
title+aligned_date), so the Gold layer can then be rebuilt to full freshness.

Pure local computation — no network. Uses stored sentiment_* columns, so no
NLP re-run.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_rebuild_news_silver.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/_rebuild_news_silver.py --shard 0/4
"""
import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.news_storage import NewsStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default="0/1")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))

    data_dir = PROJECT / "data"
    raw_files = sorted(glob.glob(str(data_dir / "a_shares" / "news_raw" / "*.parquet")))
    mine = [f for i, f in enumerate(raw_files) if i % n == k]
    logger.info("shard %d/%d: %d raw files (total %d)", k, n, len(mine), len(raw_files))

    storage = NewsStorage(str(data_dir), TradingCalendar("a_shares"))
    t0 = time.time()
    ok = err = skipped = 0
    for i, f in enumerate(mine):
        code = Path(f).stem
        try:
            raw = storage.load_raw_news(code)
            if raw is None or raw.empty:
                skipped += 1
                continue
            silver = storage.bronze_to_silver(code)
            if silver is None or silver.empty:
                skipped += 1
                continue
            storage.save_silver_news(code, silver)
            ok += 1
        except Exception as e:
            err += 1
            logger.error("%s: %s", code, e)
        done = ok + err + skipped
        if done % 200 == 0:
            logger.info("  %d/%d done (%.1f stk/s)", done, len(mine), done / max(time.time() - t0, 1e-9))
    logger.info("shard %d/%d done: %d ok, %d err, %d skipped (%.1fs)",
                k, n, ok, err, skipped, time.time() - t0)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
