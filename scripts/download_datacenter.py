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
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.sources.a_shares.capital_flow_source import CapitalFlowSource
from stoke_ml.data.sources.a_shares.limit_up_source import LimitUpSource, SENTIMENT_COLS
from stoke_ml.data.sources.a_shares.datacenter_sources import (
    BlockTradeSource, ShareholderSource, LockupExpirySource, DividendSource,
)
try:
    from stoke_ml.data.sources.a_shares.sector_source import (
        IndustryRankingSource, ConceptBlockSource,
    )
except ImportError:
    IndustryRankingSource = None  # type: ignore[assignment]
    ConceptBlockSource = None  # type: ignore[assignment]
try:
    from stoke_ml.data.sources.a_shares.backup_sources import SinaFundFlowSource
except ImportError:
    SinaFundFlowSource = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LIMIT_UP_POOLS = ["zt", "zb", "dt", "yzt"]


def _fetch_one(source_cls, method_name: str, min_interval: float, code: str):
    """Create a fresh source instance, fetch, and close — safe for concurrent use."""
    source = source_cls(min_interval=min_interval)
    try:
        return getattr(source, method_name)(code)
    finally:
        source.close()


def _fetch_one_lockup(code: str, min_interval: float, trade_date: str):
    """Fetch lockup data for one stock — safe for concurrent use."""
    source = LockupExpirySource(min_interval=min_interval)
    try:
        data = source.fetch_all(code, trade_date=trade_date)
        hist = data.get("history", pd.DataFrame())
        upcoming = data.get("upcoming", pd.DataFrame()).copy()
        if not upcoming.empty:
            upcoming["is_upcoming"] = True
        return (hist, upcoming)
    finally:
        source.close()


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
    }
    if SinaFundFlowSource is not None:
        per_stock_types["sina_fund_flow"] = ("sina_fund_flow", SinaFundFlowSource, "fetch")

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
        if IndustryRankingSource is None:
            logger.error("industry_ranking requested but sector_source unavailable")
        else:
            _download_industry_ranking(data_dir, args)

    # ── Concept blocks (per-stock) ──
    if "concept_blocks" in to_download:
        if ConceptBlockSource is None:
            logger.error("concept_blocks requested but sector_source unavailable")
        else:
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
        results = downloader.download_all(
            stock_list,
            lambda c: _fetch_one(source_cls, method_name, args.sleep, c),
        )
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
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= pd.Timestamp(args.start)) & \
                   (df["date"] <= pd.Timestamp(args.end))
            df = df[mask]
        if not df.empty:
            storage.save(df)
            logger.info("  %s: %d rows saved (%.1fs)", dtype, len(df), time.time() - t0)
        else:
            logger.warning("  %s: no data in date range (%s to %s)",
                         dtype, args.start, args.end)
    else:
        logger.warning("  %s: no data returned", dtype)

    source.close()


def _download_lockup(stock_list, data_dir, args):
    logger.info("=== lockup: %d stocks ===", len(stock_list))
    t0 = time.time()

    hist_storage = MarketWideStorage(data_dir, "lockup")
    upcoming_storage = MarketWideStorage(data_dir, "lockup_upcoming")

    if args.concurrent:
        from stoke_ml.crawler.rate_limiter import RateLimiter
        from stoke_ml.crawler.concurrent import ConcurrentDownloader

        rl = RateLimiter(base_delay_sec=args.sleep, daily_quota=50000)
        downloader = ConcurrentDownloader(rate_limiter=rl, max_workers=args.workers)
        results = downloader.download_all(
            stock_list,
            lambda c: _fetch_one_lockup(c, args.sleep, args.end),
        )
        hist_frames = []
        upcoming_frames = []
        for data in results.values():
            if data is None:
                continue
            hist_df, upcoming_df = data
            if not hist_df.empty:
                hist_frames.append(hist_df)
            if not upcoming_df.empty:
                upcoming_frames.append(upcoming_df)
    else:
        source = LockupExpirySource(min_interval=args.sleep)
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
        source.close()

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


def _download_limit_up_pool(pool, storage_key, data_dir, args):
    logger.info("=== limit_up_%s: %s to %s ===", pool, args.start, args.end)
    t0 = time.time()

    source = LimitUpSource(min_interval=args.sleep)
    storage = MarketWideStorage(data_dir, storage_key)
    fetch_fn = getattr(source, f"fetch_{pool}_pool")

    # Build set of already-covered dates from existing parquet data
    existing_dates = set()
    base = os.path.join(data_dir, "a_shares", storage_key)
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(
                            os.path.join(root, f), columns=["date"],
                        )
                        for d in pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"):
                            existing_dates.add(d)
                    except Exception:
                        pass

    calendar = TradingCalendar("a_shares")
    all_dates = calendar.get_trading_days(args.start, args.end)
    dates_to_fetch = [
        d for d in all_dates
        if d.strftime("%Y-%m-%d") not in existing_dates
    ]

    if not dates_to_fetch:
        logger.info("  limit_up_%s: all %d days already cached, skipping",
                    pool, len(all_dates))
        source.close()
        return

    n_cached = len(all_dates) - len(dates_to_fetch)
    logger.info("  limit_up_%s: %d/%d days cached, %d to fetch",
                pool, n_cached, len(all_dates), len(dates_to_fetch))

    frames = []
    for i, d in enumerate(dates_to_fetch):
        date_str = d.strftime("%Y-%m-%d")
        try:
            df = fetch_fn(date_str)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.debug("limit_up_%s failed for %s", pool, date_str)
        if (i + 1) % 60 == 0:
            pct = (i + 1) / len(dates_to_fetch) * 100
            logger.info("  limit_up_%s: %d/%d days (%.0f%%)",
                        pool, i + 1, len(dates_to_fetch), pct)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        storage.save(df)
        logger.info("  limit_up_%s: %d new rows saved (%.1fs)",
                    pool, len(df), time.time() - t0)
    else:
        logger.info("  limit_up_%s: no new data in range", pool)

    source.close()


def _download_limit_up_sentiment(data_dir, args):
    logger.info("=== limit_up_sentiment: %s to %s ===", args.start, args.end)
    t0 = time.time()

    # Load pool data from disk instead of re-fetching via API.
    # Pool downloads (_download_limit_up_pool) run first, so data is cached.
    pool_keys = ["limit_up_zt", "limit_up_zb", "limit_up_dt", "limit_up_yzt"]
    pool_storages = {k: MarketWideStorage(data_dir, k) for k in pool_keys}

    existing_dates: set[str] = set()
    sentiment_storage = MarketWideStorage(data_dir, "limit_up_sentiment")
    base = os.path.join(data_dir, "a_shares", "limit_up_sentiment")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(os.path.join(root, f), columns=["date"])
                        for d in pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"):
                            existing_dates.add(d)
                    except Exception:
                        pass

    calendar = TradingCalendar("a_shares")
    all_dates = calendar.get_trading_days(args.start, args.end)
    dates_to_fetch = [
        d for d in all_dates
        if d.strftime("%Y-%m-%d") not in existing_dates
    ]

    if not dates_to_fetch:
        n_cached = len(all_dates)
        logger.info("  limit_up_sentiment: all %d days cached, skipping", n_cached)
        return

    n_cached = len(all_dates) - len(dates_to_fetch)
    logger.info("  limit_up_sentiment: %d/%d days cached, %d to compute",
                n_cached, len(all_dates), len(dates_to_fetch))

    rows = []
    for d in dates_to_fetch:
        date_str = d.strftime("%Y-%m-%d")
        try:
            zt_df = pool_storages["limit_up_zt"].load(date_str)
            zb_df = pool_storages["limit_up_zb"].load(date_str)
            dt_df = pool_storages["limit_up_dt"].load(date_str)
            yzt_df = pool_storages["limit_up_yzt"].load(date_str)
        except Exception:
            logger.debug("limit_up_sentiment: pool data missing for %s", date_str)
            continue

        if zt_df is None or zb_df is None or dt_df is None or yzt_df is None:
            logger.debug("limit_up_sentiment: pool data missing for %s", date_str)
            continue

        rows.append(LimitUpSource.compute_sentiment(
            date_str, zt_df, zb_df, dt_df, yzt_df,
        ))

    if rows:
        df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
        df["date"] = pd.to_datetime(df["date"])
        sentiment_storage.save(df)
        logger.info("  limit_up_sentiment: %d new rows saved (%.1fs)",
                    len(df), time.time() - t0)
    else:
        logger.info("  limit_up_sentiment: no new data in range")


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
