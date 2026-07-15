"""Download EastMoney datacenter + capital flow + limit-up data.

Covers 8 new data types added in the crawler improvement plan:

  Capital flow (资金流向):
    capital_flow    — per-stock daily main/super/large/mid/small net flow

  Limit-up board (打板数据):
    limit_up_zt     — daily limit-up pool (涨停池)
    limit_up_zb     — daily busted pool (炸板池)
    limit_up_dt     — daily limit-down pool (跌停池)
    limit_up_yzt    — yesterday's ZT performance (昨日涨停池)
    limit_up_sentiment — daily board sentiment summary

  Datacenter (大宗/股东/解禁/分红):
    block_trade     — per-stock block trade records (大宗交易)
    shareholder     — per-stock shareholder count changes (股东户数)
    lockup          — per-stock lockup expiry calendar (限售解禁)
    dividend        — per-stock dividend history (分红送转)

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type all
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type capital_flow --stocks 600519,000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type block_trade --start 2024-01-01
  PYTHONPATH=. ./.venv/Scripts/python scripts/download_datacenter.py --type limit_up --start 2026-06-01
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.capital_flow_source import CapitalFlowSource
from stoke_ml.data.sources.a_shares.limit_up_source import LimitUpSource
from stoke_ml.data.sources.a_shares.datacenter_sources import (
    BlockTradeSource, ShareholderSource, LockupExpirySource, DividendSource,
)
from stoke_ml.data.sources.a_shares.sector_source import (
    IndustryRankingSource, ConceptBlockSource,
)
from stoke_ml.data.sources.a_shares.backup_sources import SinaFundFlowSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LIMIT_UP_POOLS = ["zt", "zb", "dt", "yzt"]


def get_stocks_from_disk(data_dir: str) -> list[str]:
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
    parser = argparse.ArgumentParser(
        description="Download EastMoney datacenter + capital flow + limit-up data",
    )
    parser.add_argument(
        "--type", type=str, default="all",
        choices=[
            "all", "capital_flow", "limit_up", "limit_up_zt", "limit_up_zb",
            "limit_up_dt", "limit_up_yzt", "limit_up_sentiment",
            "block_trade", "shareholder", "lockup", "dividend",
            "industry_ranking", "concept_blocks",
        ],
        help="Data type to download (default: all)",
    )
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all from disk)")
    parser.add_argument("--sleep", type=float, default=1.2,
                        help="Seconds between API calls (default: 1.2)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent workers (default: 2, keep low for EastMoney)")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    # Resolve stock list
    if args.stocks:
        stock_list = [c.strip() for c in args.stocks.split(",")]
    else:
        stock_list = get_stocks_from_disk(data_dir)
        if not stock_list:
            logger.error("No stock codes found. Run download_data.py first.")
            sys.exit(1)

    # Determine which types to download
    if args.type == "all":
        to_download = [
            "capital_flow", "block_trade", "shareholder", "lockup", "dividend",
            "limit_up_zt", "limit_up_zb", "limit_up_dt", "limit_up_yzt",
            "limit_up_sentiment",
            "industry_ranking", "concept_blocks",
        ]
    elif args.type == "limit_up":
        to_download = [
            "limit_up_zt", "limit_up_zb", "limit_up_dt", "limit_up_yzt",
            "limit_up_sentiment",
        ]
    else:
        to_download = [args.type]

    # ── Per-stock sources (capital_flow, block_trade, shareholder, etc.) ──
    per_stock_types = {
        "capital_flow": ("capital_flow", CapitalFlowSource, "fetch_daily"),
        "block_trade": ("block_trade", BlockTradeSource, "fetch"),
        "shareholder": ("shareholder", ShareholderSource, "fetch"),
        "dividend": ("dividend", DividendSource, "fetch"),
        "sina_fund_flow": ("sina_fund_flow", SinaFundFlowSource, "fetch"),
    }

    for dtype in to_download:
        if dtype in per_stock_types:
            storage_key, source_cls, method_name = per_stock_types[dtype]
            _download_per_stock(
                dtype, storage_key, source_cls, method_name,
                stock_list, data_dir, args,
            )

    # ── Lockup (special: history + upcoming) ──
    if "lockup" in to_download:
        _download_lockup(stock_list, data_dir, args)

    # ── Limit-up pools (market-wide, date-based) ──
    for pool in LIMIT_UP_POOLS:
        pool_key = f"limit_up_{pool}"
        if pool_key in to_download:
            _download_limit_up_pool(pool, pool_key, data_dir, args)

    if "limit_up_sentiment" in to_download:
        _download_limit_up_sentiment(data_dir, args)

    # ── Industry ranking (market-wide, date-based) ──
    if "industry_ranking" in to_download:
        _download_industry_ranking(data_dir, args)

    # ── Concept blocks (per-stock) ──
    if "concept_blocks" in to_download:
        _download_concept_blocks(stock_list, data_dir, args)

    logger.info("Done.")


# ── Per-stock download helpers ───────────────────────────────────────────

def _download_per_stock(dtype, storage_key, source_cls, method_name,
                        stock_list, data_dir, args):
    logger.info("=== %s: %d stocks (%s to %s) ===",
                dtype, len(stock_list), args.start, args.end)
    t0 = time.time()

    source = source_cls(min_interval=args.sleep)
    storage = MarketWideStorage(data_dir, storage_key)
    fetch_fn = getattr(source, method_name)

    if args.concurrent:
        from stoke_ml.crawler.rate_limiter import RateLimiter
        from stoke_ml.crawler.concurrent import ConcurrentDownloader

        rl = RateLimiter(base_delay_sec=args.sleep, daily_quota=50000)
        downloader = ConcurrentDownloader(rate_limiter=rl, max_workers=args.workers)
        results = downloader.download_all(stock_list, lambda c: fetch_fn(c))
        frames = [d for d in results.values() if d is not None and not d.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        frames = []
        for i, code in enumerate(stock_list):
            try:
                sdf = fetch_fn(code)
                if not sdf.empty:
                    frames.append(sdf)
            except Exception:
                logger.debug("%s fetch failed for %s", dtype, code)
            if i > 0 and i % 100 == 0:
                logger.info("  %s: %d/%d stocks done", dtype, i, len(stock_list))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not df.empty:
        storage.save(df)
        logger.info("  %s: %d rows saved (%.1fs)", dtype, len(df), time.time() - t0)
    else:
        logger.warning("  %s: no data returned", dtype)

    source.close()


def _download_lockup(stock_list, data_dir, args):
    logger.info("=== lockup: %d stocks ===", len(stock_list))
    t0 = time.time()

    source = LockupExpirySource(min_interval=args.sleep)
    hist_storage = MarketWideStorage(data_dir, "lockup")
    upcoming_storage = MarketWideStorage(data_dir, "lockup_upcoming")

    hist_frames = []
    upcoming_frames = []
    for i, code in enumerate(stock_list):
        try:
            data = source.fetch_all(code, trade_date=args.end)
            if not data["history"].empty:
                hist_frames.append(data["history"])
            if not data["upcoming"].empty:
                upcoming_stock = data["upcoming"].copy()
                upcoming_stock["is_upcoming"] = True
                upcoming_frames.append(upcoming_stock)
        except Exception:
            logger.debug("lockup fetch failed for %s", code)
        if i > 0 and i % 100 == 0:
            logger.info("  lockup: %d/%d stocks done", i, len(stock_list))

    if hist_frames:
        hist_df = pd.concat(hist_frames, ignore_index=True)
        hist_storage.save(hist_df)
        logger.info("  lockup history: %d rows", len(hist_df))

    if upcoming_frames:
        upcoming_df = pd.concat(upcoming_frames, ignore_index=True)
        upcoming_storage.save(upcoming_df)
        logger.info("  lockup upcoming: %d rows", len(upcoming_df))

    if not hist_frames and not upcoming_frames:
        logger.warning("  lockup: no data returned")

    logger.info("  lockup: done (%.1fs)", time.time() - t0)
    source.close()


def _download_limit_up_pool(pool, storage_key, data_dir, args):
    logger.info("=== limit_up_%s: %s to %s ===", pool, args.start, args.end)
    t0 = time.time()

    source = LimitUpSource(min_interval=args.sleep)
    storage = MarketWideStorage(data_dir, storage_key)
    fetch_fn = getattr(source, f"fetch_{pool}_pool")

    dates = pd.date_range(start=args.start, end=args.end, freq="B")
    frames = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        try:
            df = fetch_fn(date_str)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.debug("limit_up_%s failed for %s", pool, date_str)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        storage.save(df)
        logger.info("  limit_up_%s: %d rows saved (%.1fs)",
                    pool, len(df), time.time() - t0)
    else:
        logger.warning("  limit_up_%s: no data returned", pool)

    source.close()


def _download_limit_up_sentiment(data_dir, args):
    logger.info("=== limit_up_sentiment: %s to %s ===", args.start, args.end)
    t0 = time.time()

    source = LimitUpSource(min_interval=args.sleep)
    df = source.fetch_sentiment_batch(args.start, args.end)

    if not df.empty:
        storage = MarketWideStorage(data_dir, "limit_up_sentiment")
        storage.save(df)
        logger.info("  limit_up_sentiment: %d rows saved (%.1fs)",
                    len(df), time.time() - t0)
    else:
        logger.warning("  limit_up_sentiment: no data returned")

    source.close()


def _download_industry_ranking(data_dir, args):
    logger.info("=== industry_ranking: %s to %s ===", args.start, args.end)
    t0 = time.time()

    source = IndustryRankingSource(min_interval=args.sleep)
    df = source.fetch_batch(args.start, args.end)

    if not df.empty:
        storage = MarketWideStorage(data_dir, "industry_ranking")
        storage.save(df)
        logger.info("  industry_ranking: %d rows saved (%.1fs)",
                    len(df), time.time() - t0)
    else:
        logger.warning("  industry_ranking: no data returned")

    source.close()


def _download_concept_blocks(stock_list, data_dir, args):
    logger.info("=== concept_blocks: %d stocks ===", len(stock_list))
    t0 = time.time()

    source = ConceptBlockSource(min_interval=args.sleep)
    storage = MarketWideStorage(data_dir, "concept_blocks")

    frames = []
    for i, code in enumerate(stock_list):
        try:
            df = source.fetch(code)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.debug("concept_blocks fetch failed for %s", code)
        if i > 0 and i % 100 == 0:
            logger.info("  concept_blocks: %d/%d stocks done", i, len(stock_list))

    if frames:
        df = pd.concat(frames, ignore_index=True)
        storage.save(df)
        logger.info("  concept_blocks: %d rows saved (%.1fs)",
                    len(df), time.time() - t0)
    else:
        logger.warning("  concept_blocks: no data returned")

    source.close()


if __name__ == "__main__":
    main()
