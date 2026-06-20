"""Phase 1: XGBoost baseline training with walk-forward validation."""
import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.fundamental_storage import FundamentalStorage
from stoke_ml.data.etf_storage import ETFStorage
from stoke_ml.data.stock_sector_mapper import StockSectorMapper
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_financial_metrics,
)
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

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


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--stock", type=str, default=None,
        help="Train on a single stock (default: all available)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Model output directory (default: config model_dir)"
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
    sector_mapper = StockSectorMapper()
    codes = [args.stock] if args.stock else available_stocks(storage)

    if not codes:
        logger.error(
            "No stock data found. Run a data downloader first."
        )
        sys.exit(1)

    logger.info("Training XGBoost on %d stock(s)", len(codes))

    model_params = dict(cfg.model.params)
    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)

    pipeline = FeaturePipeline(
        seq_len=cfg.features.get("flat_seq_len", cfg.features.seq_len),
        horizon=cfg.features.target_horizon,
        flat_mode=True,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=cfg.features.get("use_sentiment", True),
    )
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    )

    all_fold_scores = []
    best_mcc = -1.0
    best_model_path = ""

    for code in codes:
        logger.info("=== Processing %s ===", code)
        df = storage.load_daily(
            code,
            start_date=cfg.markets.a_shares.start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        if df.empty:
            logger.warning("No data for %s, skipping", code)
            continue

        date_start = cfg.markets.a_shares.start_date
        date_end = datetime.now().strftime("%Y-%m-%d")

        # Load daily sentiment
        sentiment_df = news_storage.load_daily_sentiment(code, date_start, date_end)
        if not sentiment_df.empty:
            logger.info(
                "  %s: loaded %d sentiment days (%d with news)",
                code, len(sentiment_df), sentiment_df["has_news"].sum(),
            )

        # Load market-wide data
        margin_df = margin_storage.load(code, date_start, date_end)
        nb_df = nb_storage.load(code, date_start, date_end)
        dt_df = dt_storage.load(code, date_start, date_end)

        # Load fundamentals (forward-filled to daily)
        fundamental_df = fund_storage.forward_fill_to_daily(code, date_start, date_end)

        # Load ETF flow for this stock's sector
        etf_df = pd.DataFrame()
        sector = sector_mapper.get_sector(code)
        if sector:
            etf_df = etf_storage.load_sector_flow(sector, date_start, date_end)

        loaded = [f"K={len(df)}"]
        if not sentiment_df.empty:
            loaded.append(f"S={len(sentiment_df)}")
        if not margin_df.empty:
            loaded.append(f"M={len(margin_df)}")
        if not nb_df.empty:
            loaded.append(f"N={len(nb_df)}")
        if not fundamental_df.empty:
            loaded.append(f"F={len(fundamental_df)}")
        if not etf_df.empty:
            loaded.append(f"ETF={len(etf_df)}")
        logger.info("  %s: %s", code, " ".join(loaded))

        X, y, aligned_close = pipeline.build_features(
            df,
            sentiment_df=sentiment_df if not sentiment_df.empty else None,
            margin_df=margin_df if not margin_df.empty else None,
            northbound_df=nb_df if not nb_df.empty else None,
            dragon_tiger_df=dt_df if not dt_df.empty else None,
            fundamental_df=fundamental_df if not fundamental_df.empty else None,
            etf_flow_df=etf_df if not etf_df.empty else None,
        )
        if len(X) == 0:
            logger.warning("Not enough features for %s, skipping", code)
            continue

        n_samples = len(X)
        pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
        folds = list(splitter.split(pseudo_dates))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
                break

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = XGBoostBaseline(**model_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            cls_metrics = compute_classification_metrics(y_val, preds)

            # n_val predictions need n_val+1 close prices for n_val returns
            close_prices = aligned_close[val_idx[0]:val_idx[-1] + 2]
            fin_metrics = compute_financial_metrics(close_prices, preds)

            mcc = cls_metrics["mcc"]
            all_fold_scores.append({
                "stock": code,
                "fold": fold_idx,
                "mcc": mcc,
                "sharpe": fin_metrics["sharpe"],
                **cls_metrics,
            })

            logger.info(
                "  %s fold %d | MCC=%.4f Acc=%.4f Sharpe=%.2f",
                code, fold_idx, mcc,
                cls_metrics["accuracy"], fin_metrics["sharpe"],
            )

            if mcc > best_mcc:
                best_mcc = mcc
                best_model_path = os.path.join(
                    output_dir, f"xgboost_{code}_best.json"
                )
                model.save(best_model_path)

    if all_fold_scores:
        scores_df = pd.DataFrame(all_fold_scores)
        summary = scores_df.groupby("stock")["mcc"].agg(["mean", "std", "count"])
        logger.info(
            "\n=== XGBoost Summary ===\nMean MCC=%.4f ± %.4f across %d folds",
            scores_df["mcc"].mean(), scores_df["mcc"].std(), len(scores_df),
        )
        scores_df.to_csv(
            os.path.join(output_dir, "xgboost_scores.csv"), index=False,
        )
        logger.info(
            "Scores saved to %s", os.path.join(output_dir, "xgboost_scores.csv")
        )

    logger.info("Best model: %s (MCC=%.4f)", best_model_path, best_mcc)


if __name__ == "__main__":
    main()
