# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Backfill capital_flow history 2010-03 ~ now via Sina (coverage heterogeneous).

Existing per-stock flat files start in scattered years (2015..2026); many
stocks are missing 2021+ capital_flow entirely. This re-fetches each stock's
full history (days=6000 ~ back to 2010-03) and merges into the existing files
(MarketWideStorage.save dedups identical rows; Sina only provides main_net,
tier columns are 0 by design).

Resume: a stock is skipped when its existing earliest date already reaches
~2010-06 (the Sina API floor). Everything else is re-fetched.
"""
import argparse
import logging
import os
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.capital_flow_source import CapitalFlowSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FLOOR = pd.Timestamp("2010-06-01")


def collect_codes(data_dir):
    base = os.path.join(data_dir, "a_shares", "capital_flow")
    if not os.path.isdir(base):
        return []
    return sorted(p[:-8] for p in os.listdir(base) if p.endswith(".parquet"))


def already_full(path, floor):
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:
        return False
    if df.empty:
        return False
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return False
    return d.min() <= floor


def main():
    ap = argparse.ArgumentParser(description="Backfill capital_flow 2010-03 ~ now")
    ap.add_argument("--days", type=int, default=6000)
    ap.add_argument("--interval", type=float, default=1.2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="limit to first N stocks")
    ap.add_argument("--shard", type=str, default="0/1",
                    help="process k/n of stocks (parallel processes, e.g. 0/4)")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares", "capital_flow")
    storage = MarketWideStorage(data_dir, "capital_flow")
    source = CapitalFlowSource(min_interval=args.interval)

    codes = collect_codes(data_dir)
    k, n = (int(x) for x in args.shard.split("/"))
    codes = [c for i, c in enumerate(codes) if i % n == k]
    if args.smoke:
        codes = codes[: args.smoke]
    if args.force:
        pending = list(codes)
    else:
        pending = [c for c in codes if not already_full(os.path.join(base, f"{c}.parquet"), FLOOR)]
    logger.info("%d stocks, %d need backfill", len(codes), len(pending))

    t0 = time.time()
    ok = fail = empty = 0
    for i, code in enumerate(pending):
        try:
            df = source.fetch_daily(code, days=args.days)
        except Exception as e:
            logger.error("%s: fetch ERR %s", code, e)
            fail += 1
            continue
        if df is None or df.empty:
            logger.warning("%s: empty response", code)
            empty += 1
            continue
        storage.save(df)
        ok += 1
        done = ok + fail + empty
        if done % 100 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d done (%.2f s/stk, ok=%d fail=%d empty=%d)",
                        done, len(pending), elapsed / max(done, 1), ok, fail, empty)
    logger.info("backfill done: %d ok, %d fail, %d empty (%.1fs)", ok, fail, empty,
                time.time() - t0)
    source.close()


if __name__ == "__main__":
    main()
