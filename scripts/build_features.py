"""Pre-build features for all stocks and save to parquet.

Run once before training to decouple expensive feature engineering from
the training loop.  Training scripts can then use ``--prebuilt`` to skip
straight to sequence slicing.

Supports multiprocessing (``--jobs N``, ``--jobs 0`` = auto cpu_count) and
sharding (``--shard k/n``) for parallel builds across processes/machines.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --stock 000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --no-guba --no-comment
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --jobs 8 --shard 0/4
"""
import argparse
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.fundamental_storage import FundamentalStorage
from stoke_ml.data.etf_storage import ETFStorage
from stoke_ml.data.stock_sector_mapper import StockSectorMapper
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.comment_storage import CommentStorage
from stoke_ml.data.announcement_storage import AnnouncementStorage
from stoke_ml.features.pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# storage_key -> True once a channel has emitted a one-time load-failure warning
_reported_load_fail: set[str] = set()


def available_stocks(storage: DataStorage, market: str = "a_shares") -> list[str]:
    base = os.path.join(storage._root, market, "daily")
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def _load_stock_parquet(directory: str, code: str) -> pd.DataFrame:
    """Load a per-stock parquet file, returning empty DataFrame if missing/corrupt."""
    path = os.path.join(directory, f"{code}.parquet")
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.warning("Corrupted parquet, skipping: %s", path)
        return pd.DataFrame()


def _load_opt(args: dict, storage_key: str, method: str, code: str):
    """Call a per-stock storage loader, mapping missing/empty to None."""
    obj = args.get(storage_key)
    if obj is None:
        return None
    try:
        out = getattr(obj, method)(code, args["start"], args["end"])
        return out if not out.empty else None
    except Exception:
        if storage_key not in _reported_load_fail:
            logger.warning(
                "channel %s failed to load for %s (suppressing further warnings)",
                storage_key, code,
            )
            _reported_load_fail.add(storage_key)
        return None


def _load_etf(args: dict, code: str):
    try:
        sector = args["sector_mapper"].get_sector(code)
        if not sector:
            return None
        out = args["etf_storage"].load_sector_flow(sector, args["start"], args["end"])
        return out if not out.empty else None
    except Exception:
        return None


def _is_valid_feature(path: str) -> bool:
    """A 0-byte (interrupted-write) parquet is not a usable feature file."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def build_one(args: dict) -> tuple[str, str]:
    """Build features for one stock. args carries ALL inputs; returns (code, status)."""
    code = args["code"]
    try:
        pipeline = FeaturePipeline(
            seq_len=args["seq_len"],
            horizon=args["horizon"],
            flat_mode=False,
            use_technical=args["use_technical"],
            use_scoring=args["use_scoring"],
            use_temporal=args["use_temporal"],
            use_sentiment=args["use_sentiment"],
            use_guba=args["use_guba"],
            use_comment=args["use_comment"],
            use_limit_up=args["use_limit_up"],  # False (deferred, top scope note)
            use_pledge=args["use_pledge"],
            use_market_env=args["use_market_env"],
            use_market_env_refine=args["use_market_env_refine"],
            use_index_membership=args["use_index_membership"],
        )
        output_path = os.path.join(args["output_dir"], f"{code}.parquet")
        if not args["force"] and _is_valid_feature(output_path):
            return code, "exists"
        df = args["storage"].load_daily(code, start_date=args["start"], end_date=args["end"])
        if df.empty:
            return code, "empty"
        # limit_up_df intentionally absent (family deferred, top scope note)
        a_shares = os.path.join(args["data_dir"], "a_shares")
        pledge_df = _load_stock_parquet(os.path.join(a_shares, "pledge_processed"), code)
        index_membership_df = _load_stock_parquet(
            os.path.join(a_shares, "index_membership_processed"), code)
        pipeline.save_features(
            output_path, df,
            sentiment_df=_load_opt(args, "news_storage", "load_daily_sentiment", code),
            margin_df=_load_opt(args, "margin_storage", "load", code),
            northbound_df=_load_opt(args, "nb_storage", "load", code),
            dragon_tiger_df=_load_opt(args, "dt_storage", "load", code),
            fundamental_df=_load_opt(args, "fund_storage", "forward_fill_to_daily", code),
            valuation_df=_load_stock_parquet(os.path.join(a_shares, "valuation"), code),
            capital_flow_df=_load_stock_parquet(os.path.join(a_shares, "capital_flow_processed"), code),
            board_df=_load_stock_parquet(os.path.join(a_shares, "board_processed"), code),
            sector_df=_load_stock_parquet(os.path.join(a_shares, "industry_ranking_processed"), code),
            block_trade_df=_load_stock_parquet(os.path.join(a_shares, "block_trade_processed"), code),
            dividend_df=_load_stock_parquet(os.path.join(a_shares, "dividend_processed"), code),
            lockup_df=_load_stock_parquet(os.path.join(a_shares, "lockup_processed"), code),
            shareholder_df=_load_stock_parquet(os.path.join(a_shares, "shareholder_processed"), code),
            concept_df=_load_stock_parquet(os.path.join(a_shares, "concept_blocks_processed"), code),
            guba_df=_load_opt(args, "guba_storage", "load_daily_sentiment", code),
            comment_df=_load_opt(args, "comment_storage", "build_features", code),
            announcement_df=_load_opt(args, "ann_storage", "load_daily_sentiment", code),
            etf_flow_df=_load_etf(args, code),
            limit_up_df=None,  # deferred (top scope note)
            pledge_df=pledge_df if not pledge_df.empty else None,
            index_membership_df=index_membership_df if not index_membership_df.empty else None,
        )
        return code, "built"
    except Exception:
        logger.exception("[%s] failed", code)
        return code, "failed"


def main():
    parser = argparse.ArgumentParser(description="Pre-build features for all stocks")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--stock", type=str, default=None,
        help="Build for a single stock (default: all available)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Features output directory (default: data/features/)",
    )
    parser.add_argument(
        "--no-guba", action="store_true", help="Exclude Guba forum sentiment",
    )
    parser.add_argument(
        "--no-comment", action="store_true", help="Exclude AKShare comment sentiment",
    )
    parser.add_argument(
        "--shard", type=str, default=None,
        help="k/n shard over codes, e.g. 0/4",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="parallel worker processes (0 = cpu_count)",
    )
    # limit-up ecology family is DEFERRED (top scope note) — no --limit-up flag
    parser.add_argument("--pledge", dest="use_pledge", action="store_true", default=True,
                        help="Include pledge risk (default)")
    parser.add_argument("--no-pledge", dest="use_pledge", action="store_false",
                        help="Exclude pledge risk")
    # --market-env disables the market-breadth channel (7 market-env cols);
    # --no-market-env-refine disables the macro-regime refiner (menv_* factors).
    # NOTE: the market_ prefix is shared — the board channel's market_state_*
    # columns (use_board) are NOT affected by either flag.
    parser.add_argument("--market-env", dest="use_market_env", action="store_true", default=True,
                        help="Include market breadth (default)")
    parser.add_argument("--no-market-env", dest="use_market_env", action="store_false",
                        help="Exclude market breadth (7 market-env cols); use "
                             "--no-market-env-refine for menv_* macro-regime factors. "
                             "market_state_* (board) is a separate channel")
    parser.add_argument("--market-env-refine", dest="use_market_env_refine", action="store_true",
                        default=True,
                        help="Include MarketEnvRefiner macro-regime factors (menv_*, default)")
    parser.add_argument("--no-market-env-refine", dest="use_market_env_refine", action="store_false",
                        help="Exclude MarketEnvRefiner macro-regime factors (menv_*)")
    parser.add_argument("--index-membership", dest="use_index_membership", action="store_true",
                        default=True, help="Include index membership (default)")
    parser.add_argument("--no-index-membership", dest="use_index_membership", action="store_false",
                        help="Exclude index membership")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing feature files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = cfg.project.data_dir
    storage = DataStorage(data_dir)
    news_storage = NewsStorage(data_dir)
    margin_storage = MarketWideStorage(data_dir, "margin")
    nb_storage = MarketWideStorage(data_dir, "northbound")
    dt_storage = MarketWideStorage(data_dir, "dragon_tiger")
    fund_storage = FundamentalStorage(data_dir)
    etf_storage = ETFStorage(data_dir)
    guba_storage = GubaStorage(data_dir)
    comment_storage = CommentStorage(data_dir)
    ann_storage = AnnouncementStorage(data_dir)
    sector_mapper = StockSectorMapper()

    codes = [args.stock] if args.stock else available_stocks(storage)

    if not codes:
        logger.error("No stock data found. Run a data downloader first.")
        sys.exit(1)

    # k/n shard over the code list (applies to serial AND parallel dispatch)
    if args.shard:
        try:
            k, n = map(int, args.shard.split("/"))
        except ValueError:
            parser.error("--shard must be k/n")
        if n < 1 or k < 0 or k >= n:
            parser.error("--shard: k in [0, n) and n >= 1")
        codes = [c for i, c in enumerate(codes) if i % n == k]

    output_dir = args.output_dir or os.path.join(data_dir, "features")
    os.makedirs(output_dir, exist_ok=True)

    use_gb = not args.no_guba
    use_cm = not args.no_comment

    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    worker_args = {
        "code": None,
        "seq_len": cfg.features.seq_len,
        "horizon": cfg.features.target_horizon,
        "use_technical": cfg.features.technical_indicators,
        "use_scoring": cfg.features.rule_based_scoring,
        "use_temporal": cfg.features.temporal_features,
        "use_sentiment": cfg.features.get("use_sentiment", True),
        "use_guba": use_gb,
        "use_comment": use_cm,
        "use_limit_up": False,  # limit-up deferred (top scope note)
        "use_pledge": args.use_pledge,
        "use_market_env": args.use_market_env,
        "use_market_env_refine": args.use_market_env_refine,
        "use_index_membership": args.use_index_membership,
        "storage": storage,
        "news_storage": news_storage,
        "margin_storage": margin_storage,
        "nb_storage": nb_storage,
        "dt_storage": dt_storage,
        "fund_storage": fund_storage,
        "etf_storage": etf_storage,
        "guba_storage": guba_storage,
        "comment_storage": comment_storage,
        "ann_storage": ann_storage,
        "sector_mapper": sector_mapper,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "start": date_start,
        "end": date_end,
        "force": args.force,
    }

    tasks = [{**worker_args, "code": c} for c in codes]

    if args.jobs < 0:
        parser.error("--jobs must be >= 0")

    if args.jobs == 1:
        results = [build_one(t) for t in tasks]
    else:
        workers = args.jobs if args.jobs > 1 else min(32, (os.cpu_count() or 1) + 2)
        logger.info("Building %d stocks across %d workers", len(codes), workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(build_one, tasks))

    counts = Counter(s for _, s in results)
    logger.info("Done: %s (out of %d stocks)", dict(counts), len(codes))


if __name__ == "__main__":
    main()
