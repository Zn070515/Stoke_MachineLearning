"""Preprocess new data types through the multi-shape preprocessing pipeline.

Reads raw data from MarketWideStorage (downloaded by download_datacenter.py),
runs the appropriate PreprocessingChain, saves preprocessed results.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type all
  PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type flow
  PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type event --event-type block_trade
  PYTHONPATH=. ./.venv/Scripts/python scripts/preprocess_new_data.py --type concept --stocks 600519,000001
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
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Map script --type to (storage_key, pipeline_chain_name)
TYPE_MAP = {
    "flow": ("capital_flow", "flow"),
    "block_trade": ("block_trade", "event_block_trade"),
    "shareholder": ("shareholder", "event_shareholder"),
    "lockup": ("lockup", "event_lockup"),
    "dividend": ("dividend", "event_dividend"),
    "board": (None, "board"),  # needs multiple pool storages
    "sector": ("industry_ranking", "sector"),
    "concept": ("concept_blocks", "concept"),
}


def get_stocks_from_disk(data_dir: str, storage_key: str) -> list[str]:
    """Discover available stock codes from partitioned storage."""
    base = os.path.join(data_dir, "a_shares", storage_key)
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def get_stocks_from_daily(data_dir: str) -> list[str]:
    """Fall back to daily K-line directory."""
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
        description="Preprocess new data types through multi-shape pipeline",
    )
    parser.add_argument(
        "--type", type=str, default="all",
        choices=["all"] + list(TYPE_MAP.keys()),
        help="Data type to preprocess",
    )
    parser.add_argument("--event-type", type=str, default=None,
                        choices=["block_trade", "shareholder", "lockup", "dividend"],
                        help="Specific event type when --type=event")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--save-to", type=str, default=None,
                        help="Override output storage key (default: {storage_key}_processed)")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    cfg = load_config()
    data_dir = cfg.project.data_dir

    # Resolve stock list
    if args.stocks:
        stock_list = [c.strip() for c in args.stocks.split(",")]
    else:
        stock_list = get_stocks_from_daily(data_dir)
        if not stock_list:
            logger.error("No stock codes found. Run download_data.py first.")
            sys.exit(1)

    # Build pipeline from config
    pp_cfg = cfg.get("preprocessing", {}) if hasattr(cfg, "get") else {}
    pp = PreprocessingPipeline.from_config(pp_cfg)

    # Determine types to process
    if args.type == "all":
        to_process = list(TYPE_MAP.keys())
    elif args.type == "event" and args.event_type:
        to_process = [args.event_type]
    elif args.type == "event":
        logger.error("--type event requires --event-type")
        sys.exit(1)
    else:
        to_process = [args.type]

    for dtype in to_process:
        if dtype not in TYPE_MAP:
            logger.warning("Unknown type: %s", dtype)
            continue
        storage_key, chain_name = TYPE_MAP[dtype]
        chain = pp.get_chain(chain_name)
        if chain is None:
            logger.warning("Chain '%s' not configured, skipping %s", chain_name, dtype)
            continue

        if dtype == "board":
            _process_board(chain, stock_list, data_dir, args)
        elif dtype == "sector":
            _process_sector(chain, stock_list, data_dir, args)
        elif storage_key:
            _process_standard(dtype, storage_key, chain, stock_list, data_dir, args)


def _process_standard(dtype, storage_key, chain, stock_list, data_dir, args):
    """Process standard per-stock data: load → transform → save."""
    logger.info("=== %s: %d stocks (%s to %s) ===",
                dtype, len(stock_list), args.start, args.end)
    t0 = time.time()
    source = MarketWideStorage(data_dir, storage_key)
    output_key = args.save_to or f"{storage_key}_processed"
    dest = MarketWideStorage(data_dir, output_key)

    total = 0
    for code in stock_list:
        try:
            raw = source.load(code, args.start, args.end)
            if raw.empty:
                continue
            processed = chain.fit_transform(raw)
            if not processed.empty:
                dest.save(processed)
                total += len(processed)
        except Exception:
            logger.warning("%s preprocessing failed for %s", dtype, code, exc_info=True)

    logger.info("  %s: %d rows saved (%.1fs)", dtype, total, time.time() - t0)


def _process_board(chain, stock_list, data_dir, args):
    """Process board data: load limit_up pools → broadcast to stocks."""
    logger.info("=== board: %d stocks ===", len(stock_list))
    t0 = time.time()

    # Load all 4 limit-up pools
    pools = {}
    for pool_name in ["zt", "zb", "dt", "yzt"]:
        storage_key = f"limit_up_{pool_name}"
        pool_storage = MarketWideStorage(data_dir, storage_key)
        frames = []
        for code in stock_list:
            pdf = pool_storage.load(code, args.start, args.end)
            if not pdf.empty:
                frames.append(pdf)
        if frames:
            pools[pool_name] = pd.concat(frames, ignore_index=True)
        else:
            logger.warning("No %s pool data found for %s–%s", pool_name, args.start, args.end)

    # Load sentiment if available
    sentiment = None
    try:
        sent_storage = MarketWideStorage(data_dir, "limit_up_sentiment")
        frames = []
        for code in stock_list:
            sdf = sent_storage.load(code, args.start, args.end)
            if not sdf.empty:
                frames.append(sdf)
        if frames:
            sentiment = pd.concat(frames, ignore_index=True)
    except Exception:
        logger.warning("Failed to load limit_up_sentiment", exc_info=True)

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    dest = MarketWideStorage(data_dir, args.save_to or "board_processed")
    total = 0
    for code in stock_list:
        try:
            base = ds.load(code, args.start, args.end)
            if base.empty:
                continue
            processed = chain.fit_transform(base, pools=pools, sentiment=sentiment)
            if not processed.empty:
                dest.save(processed)
                total += len(processed)
        except Exception:
            logger.debug("board preprocessing failed for %s", code, exc_info=True)

    logger.info("  board: %d rows saved (%.1fs)", total, time.time() - t0)


def _process_sector(chain, stock_list, data_dir, args):
    """Process sector data: load industry ranking + sector map → broadcast to stocks."""
    logger.info("=== sector: %d stocks ===", len(stock_list))
    t0 = time.time()

    # Load industry ranking (market-wide, not per-stock)
    import json
    ir_storage = MarketWideStorage(data_dir, "industry_ranking")
    ir_frames = []
    for code in stock_list:
        ir_df = ir_storage.load(code, args.start, args.end)
        if not ir_df.empty:
            ir_frames.append(ir_df)
    if not ir_frames:
        logger.warning("No industry_ranking data found for %s–%s", args.start, args.end)
        return
    industry_ranking = pd.concat(ir_frames, ignore_index=True)

    # Load sector map from cache CSV
    sector_map = {}
    cache_path = os.path.join(data_dir, "a_shares", "stock_sector_cache.csv")
    if os.path.exists(cache_path):
        sector_df = pd.read_csv(cache_path, dtype=str)
        sector_map = dict(zip(sector_df["stock_code"], sector_df["sector"]))
    else:
        logger.warning("No stock_sector_cache.csv found — sector preprocessing skipped")
        return

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    dest = MarketWideStorage(data_dir, args.save_to or "industry_ranking_processed")
    total = 0
    for code in stock_list:
        try:
            base = ds.load(code, args.start, args.end)
            if base.empty:
                continue
            processed = chain.fit_transform(
                base, industry_ranking=industry_ranking, sector_map=sector_map,
            )
            if not processed.empty:
                dest.save(processed)
                total += len(processed)
        except Exception:
            logger.debug("sector preprocessing failed for %s", code, exc_info=True)

    logger.info("  sector: %d rows saved (%.1fs)", total, time.time() - t0)


if __name__ == "__main__":
    main()
