"""Preprocessing A/B benchmark — old pipeline vs new numeric preprocessing chain.

Compares 2 configurations:
  1. old          — current FeaturePipeline (no new preprocessing)
  2. new_numeric  — numeric chain (outlier → missing → robust scaling) on K-line

The numeric chain is applied per-stock BEFORE feature engineering.
The text chain (DailyAggregator bipolar features) requires per-source silver data
re-processing and will be benchmarked separately.
"""

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
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.comment_storage import CommentStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _load_aux_data(code, date_start, date_end, data_dir):
    """Load all auxiliary data for a stock. Returns dict of DataFrames."""
    empty = pd.DataFrame()
    result = {}
    for key, loader in [
        ("sentiment", lambda: NewsStorage(data_dir).load_daily_sentiment(
            code, date_start, date_end)),
        ("guba", lambda: GubaStorage(data_dir).load_daily_sentiment(
            code, date_start, date_end)),
        ("comment", lambda: CommentStorage(data_dir).build_features(
            code, date_start, date_end)),
        ("margin", lambda: MarketWideStorage(data_dir, "margin").load(
            code, date_start, date_end)),
        ("northbound", lambda: MarketWideStorage(data_dir, "northbound").load(
            code, date_start, date_end)),
        ("dragon_tiger", lambda: MarketWideStorage(data_dir, "dragon_tiger").load(
            code, date_start, date_end)),
        ("fundamental", lambda: FundamentalStorage(data_dir).forward_fill_to_daily(
            code, date_start, date_end)),
    ]:
        try:
            df = loader()
            result[key] = df if (isinstance(df, pd.DataFrame) and not df.empty) else None
        except Exception:
            result[key] = None

    try:
        sector = StockSectorMapper().get_sector(code)
        ef = ETFStorage(data_dir).load_sector_flow(sector, date_start, date_end) if sector else empty
        result["etf_flow"] = ef if not ef.empty else None
    except Exception:
        result["etf_flow"] = None

    return result


def _build_features(df, aux, cfg, use_new_preprocessing, pp):
    """Build features with optional numeric preprocessing."""
    pipeline = FeaturePipeline(
        seq_len=cfg.features.get("flat_seq_len", cfg.features.seq_len),
        horizon=cfg.features.target_horizon,
        flat_mode=True,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=True,
        use_announcements=False,
        use_guba=True,
        use_comment=False,
        use_margin=False,
        use_northbound=False,
        use_dragon_tiger=False,
        use_fundamental=False,
        use_etf_flow=False,
        use_interaction=False,
        use_new_preprocessing=use_new_preprocessing,
        preprocessing_config=pp if use_new_preprocessing else None,
    )

    X, y, _ = pipeline.build_features(
        df,
        sentiment_df=aux.get("sentiment"),
        guba_df=aux.get("guba"),
        comment_df=aux.get("comment"),
        margin_df=aux.get("margin"),
        northbound_df=aux.get("northbound"),
        dragon_tiger_df=aux.get("dragon_tiger"),
        fundamental_df=aux.get("fundamental"),
        etf_flow_df=aux.get("etf_flow"),
    )
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing A/B benchmark (numeric chain)"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    storage = DataStorage(data_dir)
    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    # ---- Stock selection ----
    all_codes = sorted([
        f.replace(".parquet", "")
        for f in os.listdir(os.path.join(data_dir, "a_shares", "daily"))
        if f.endswith(".parquet")
    ])
    if args.stocks and args.stocks < len(all_codes):
        step = max(len(all_codes) // args.stocks, 1)
        codes = [all_codes[i * step] for i in range(args.stocks)]
    else:
        codes = all_codes

    logger.info("Benchmark: %d stocks x 2 configs (old vs new_numeric)", len(codes))

    # ---- Preprocessing config for numeric chain ----
    pp_config = {
        "numeric": {
            "outlier": {"threshold": 5.0, "clip": True},
            "missing": {"short_gap_max": 2, "medium_gap_max": 10},
            "scaling": {"window_days": 252, "winsorize_sigma": 3.0, "min_periods": 63},
        }
    }
    pp = PreprocessingPipeline.from_config(pp_config)

    model_params = dict(cfg.model.params)
    model_params["n_estimators"] = 50
    model_params["max_depth"] = 4
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
        step_months=6,
    )

    all_results = []
    stock_count = 0

    for code in codes:
        df = storage.load_daily(code, date_start, date_end)
        if df.empty or len(df) < 200:
            continue

        aux = _load_aux_data(code, date_start, date_end, data_dir)

        stock_features = {}
        for cfg_name, use_pp in [("old", False), ("new_numeric", True)]:
            try:
                X, y = _build_features(df, aux, cfg, use_pp, pp_config)
                if len(X) > 0:
                    stock_features[cfg_name] = (X, y)
                    n_feat = X.shape[1]
                    logger.info("  %s %-12s: %d samples, %d features",
                                code, cfg_name, len(X), n_feat)
            except Exception as e:
                logger.warning("  %s %s: FAILED — %s", code, cfg_name, e)

        if len(stock_features) < 2:
            continue

        stock_count += 1

        ref_X = next(iter(stock_features.values()))[0]
        n_samples = len(ref_X)
        pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
        folds = list(splitter.split(pseudo_dates))[:5]

        for cfg_name, (X_all, y_all) in stock_features.items():
            n_samples_cfg = len(X_all)

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                if train_idx[-1] >= n_samples_cfg or val_idx[-1] >= n_samples_cfg:
                    break

                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]

                try:
                    model = XGBoostBaseline(**model_params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    metrics = compute_classification_metrics(y_val, preds)
                except Exception as e:
                    logger.warning("  %s %s fold %d: training failed — %s",
                                   code, cfg_name, fold_idx, e)
                    continue

                all_results.append({
                    "stock": code,
                    "config": cfg_name,
                    "fold": fold_idx,
                    "n_features": X_train.shape[1],
                    **metrics,
                })

                logger.debug("  %s %-12s fold %d: MCC=%.4f Acc=%.4f feat=%d",
                             code, cfg_name, fold_idx,
                             metrics["mcc"], metrics["accuracy"], X_train.shape[1])

        logger.info("  [%d/%d] %s done (2 configs x %d folds)",
                    stock_count, len(codes), code, len(folds))

    # ==================================================================
    # Results
    # ==================================================================
    if not all_results:
        logger.error("No results")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    logger.info("\n%s", "=" * 64)
    logger.info("PREPROCESSING A/B BENCHMARK (%d stocks)", stock_count)
    logger.info("%s", "=" * 64)

    summary = results_df.groupby("config").agg(
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        acc_mean=("accuracy", "mean"),
        n_folds=("mcc", "count"),
        n_features=("n_features", "median"),
    ).round(4)
    logger.info("\n%s", summary.to_string())

    best = summary["mcc_mean"].idxmax()
    best_mcc = summary.loc[best, "mcc_mean"]
    logger.info("\nBest: %s (MCC=%.4f)", best, best_mcc)

    if "old" in summary.index and "new_numeric" in summary.index:
        old_mcc = summary.loc["old", "mcc_mean"]
        new_mcc = summary.loc["new_numeric", "mcc_mean"]
        delta = new_mcc - old_mcc
        logger.info("\nDelta (new_numeric - old): %+.4f", delta)

    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "preprocessing_benchmark.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved to %s", out_path)


if __name__ == "__main__":
    main()
