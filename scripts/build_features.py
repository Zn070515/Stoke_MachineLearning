"""Pre-build features for all stocks and save to parquet.

Run once before training to decouple expensive feature engineering from
the training loop.  Training scripts can then use ``--prebuilt`` to skip
straight to sequence slicing.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --stock 000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/build_features.py --no-guba --no-comment
"""
import argparse
import logging
import os
import sys
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
    # limit-up ecology family is DEFERRED (top scope note) — no --no-limit-up flag
    parser.add_argument("--no-pledge", action="store_true", help="Exclude pledge risk")
    parser.add_argument("--no-market-env", action="store_true", help="Exclude market env")
    parser.add_argument("--no-index-membership", action="store_true",
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

    # Directories for per-stock processed parquet files
    _a = os.path.join(data_dir, "a_shares")
    valuation_dir = os.path.join(_a, "valuation")
    capital_flow_dir = os.path.join(_a, "capital_flow_processed")
    board_dir = os.path.join(_a, "board_processed")
    sector_dir = os.path.join(_a, "industry_ranking_processed")
    block_trade_dir = os.path.join(_a, "block_trade_processed")
    dividend_dir = os.path.join(_a, "dividend_processed")
    lockup_dir = os.path.join(_a, "lockup_processed")
    shareholder_dir = os.path.join(_a, "shareholder_processed")
    concept_dir = os.path.join(_a, "concept_blocks_processed")
    limit_up_dir = os.path.join(_a, "limit_up_processed")
    pledge_dir = os.path.join(_a, "pledge_processed")
    index_membership_dir = os.path.join(_a, "index_membership_processed")
    # market_env_daily is global -> auto-loaded by _merge_market_env internally

    # Macro and industry are auto-loaded by the pipeline internally
    # (they're global, not per-stock, and _merge_macro/_merge_industry
    #  handle disk loading + caching)
    codes = [args.stock] if args.stock else available_stocks(storage)

    if not codes:
        logger.error("No stock data found. Run a data downloader first.")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(data_dir, "features")
    os.makedirs(output_dir, exist_ok=True)

    use_gb = not args.no_guba
    use_cm = not args.no_comment

    pipeline = FeaturePipeline(
        seq_len=cfg.features.seq_len,
        horizon=cfg.features.target_horizon,
        flat_mode=False,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=cfg.features.get("use_sentiment", True),
        use_guba=use_gb,
        use_comment=use_cm,
        use_limit_up=False,  # limit-up family deferred (top scope note)
        use_pledge=not args.no_pledge,
        use_market_env=not args.no_market_env,
        use_index_membership=not args.no_index_membership,
    )

    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    built = 0
    skipped = 0
    failed = 0

    for code in codes:
        output_path = os.path.join(output_dir, f"{code}.parquet")
        if os.path.exists(output_path) and not args.force:
            logger.info("[%s] already exists, skipping (use --force to rebuild)", code)
            skipped += 1
            continue

        logger.info("=== [%s] building features ===", code)

        try:
            # Load K-line
            df = storage.load_daily(code, start_date=date_start, end_date=date_end)
            if df.empty:
                logger.warning("[%s] No K-line data, skipping", code)
                skipped += 1
                continue

            # Load daily sentiment
            sentiment_df = news_storage.load_daily_sentiment(code, date_start, date_end)

            # Load market-wide data
            margin_df = margin_storage.load(code, date_start, date_end)
            nb_df = nb_storage.load(code, date_start, date_end)
            dt_df = dt_storage.load(code, date_start, date_end)

            # Load fundamentals
            fundamental_df = fund_storage.forward_fill_to_daily(code, date_start, date_end)

            # Load ETF flow
            etf_df = pd.DataFrame()
            sector = sector_mapper.get_sector(code)
            if sector:
                etf_df = etf_storage.load_sector_flow(sector, date_start, date_end)

            # Load Guba + Comment
            guba_df = guba_storage.load_daily_sentiment(code, date_start, date_end)
            comment_df = comment_storage.build_features(code, date_start, date_end)

            # Load announcement sentiment
            announcement_df = ann_storage.load_daily_sentiment(code, date_start, date_end)

            # Load per-stock processed data
            valuation_df = _load_stock_parquet(valuation_dir, code)
            capital_flow_df = _load_stock_parquet(capital_flow_dir, code)
            board_df = _load_stock_parquet(board_dir, code)
            sector_df = _load_stock_parquet(sector_dir, code)
            block_trade_df = _load_stock_parquet(block_trade_dir, code)
            dividend_df = _load_stock_parquet(dividend_dir, code)
            lockup_df = _load_stock_parquet(lockup_dir, code)
            shareholder_df = _load_stock_parquet(shareholder_dir, code)
            concept_df = _load_stock_parquet(concept_dir, code)
            # limit_up_df intentionally NOT loaded (family deferred, top scope note)
            pledge_df = _load_stock_parquet(pledge_dir, code)
            index_membership_df = _load_stock_parquet(index_membership_dir, code)

            loaded_parts = [f"K={len(df)}"]
            for label, d in [
                ("S", sentiment_df), ("M", margin_df), ("N", nb_df),
                ("DT", dt_df), ("F", fundamental_df), ("ETF", etf_df),
                ("GB", guba_df), ("CM", comment_df), ("Ann", announcement_df),
                ("Val", valuation_df), ("CF", capital_flow_df), ("Brd", board_df),
                ("Sec", sector_df), ("BT", block_trade_df), ("Div", dividend_df),
                ("LU", lockup_df), ("SH", shareholder_df), ("Conc", concept_df),
                ("Pledge", pledge_df), ("IdxM", index_membership_df),
            ]:
                if not d.empty:
                    loaded_parts.append(f"{label}={len(d)}")
            logger.info("[%s] loaded: %s", code, " ".join(loaded_parts))

            # Engineer + save
            pipeline.save_features(
                output_path,
                df,
                sentiment_df=sentiment_df if not sentiment_df.empty else None,
                margin_df=margin_df if not margin_df.empty else None,
                northbound_df=nb_df if not nb_df.empty else None,
                dragon_tiger_df=dt_df if not dt_df.empty else None,
                fundamental_df=fundamental_df if not fundamental_df.empty else None,
                valuation_df=valuation_df if not valuation_df.empty else None,
                etf_flow_df=etf_df if not etf_df.empty else None,
                announcement_df=announcement_df if not announcement_df.empty else None,
                guba_df=guba_df if not guba_df.empty else None,
                comment_df=comment_df if not comment_df.empty else None,
                capital_flow_df=capital_flow_df if not capital_flow_df.empty else None,
                block_trade_df=block_trade_df if not block_trade_df.empty else None,
                shareholder_df=shareholder_df if not shareholder_df.empty else None,
                lockup_df=lockup_df if not lockup_df.empty else None,
                dividend_df=dividend_df if not dividend_df.empty else None,
                board_df=board_df if not board_df.empty else None,
                sector_df=sector_df if not sector_df.empty else None,
                concept_df=concept_df if not concept_df.empty else None,
                limit_up_df=None,  # deferred (top scope note)
                pledge_df=pledge_df if not pledge_df.empty else None,
                index_membership_df=index_membership_df if not index_membership_df.empty else None,
            )
            built += 1

        except Exception:
            logger.exception("[%s] failed", code)
            failed += 1

    logger.info(
        "Done: %d built, %d skipped, %d failed (out of %d stocks)",
        built, skipped, failed, len(codes),
    )


if __name__ == "__main__":
    main()
