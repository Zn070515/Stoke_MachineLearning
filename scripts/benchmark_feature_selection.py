"""Feature selection benchmark — tests whether MI/SFS fixes dimension explosion.

Compares 4 configurations:
  1. technical-only      — baseline (K-line indicators only)
  2. +sentiment          — current best known config
  3. ALL dimensions      — all 9 auxiliary sources (tests dimension explosion)
  4. ALL + MI filter     — MI to 200 features (tests if filtering fixes explosion)
  5. ALL + MI + SFS      — MI(200) → SFS(50) (tests if greedy selection helps)

Feature selection is applied PER FOLD (fit on train, transform on val) to
prevent information leakage.
"""

import argparse
import logging
import os
import sys
import time
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
from stoke_ml.data.xueqiu_storage import XueqiuStorage
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.comment_storage import CommentStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.features.selection import FeatureSelector
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _load_aux_data(code, date_start, date_end, data_dir):
    """Load all auxiliary data for a stock. Returns dict of DataFrames.

    Returns empty DataFrames when data is unavailable (FeaturePipeline
    handles them gracefully with ZI fill).
    """
    empty = pd.DataFrame()
    try:
        sentiment = NewsStorage(data_dir).load_daily_sentiment(
            code, date_start, date_end
        )
    except Exception:
        sentiment = empty

    try:
        xueqiu = XueqiuStorage(data_dir).load_daily_sentiment(
            code, date_start, date_end
        )
    except Exception:
        xueqiu = empty

    try:
        guba = GubaStorage(data_dir).load_daily_sentiment(
            code, date_start, date_end
        )
    except Exception:
        guba = empty

    try:
        comment = CommentStorage(data_dir).build_features(code, date_start, date_end)
    except Exception:
        comment = empty

    try:
        margin = MarketWideStorage(data_dir, "margin").load(
            code, date_start, date_end
        )
    except Exception:
        margin = empty

    try:
        northbound = MarketWideStorage(data_dir, "northbound").load(
            code, date_start, date_end
        )
    except Exception:
        northbound = empty

    try:
        dragon_tiger = MarketWideStorage(data_dir, "dragon_tiger").load(
            code, date_start, date_end
        )
    except Exception:
        dragon_tiger = empty

    try:
        fundamental = FundamentalStorage(data_dir).forward_fill_to_daily(
            code, date_start, date_end
        )
    except Exception:
        fundamental = empty

    try:
        sector = StockSectorMapper().get_sector(code)
        etf_flow = ETFStorage(data_dir).load_sector_flow(
            sector, date_start, date_end
        ) if sector else empty
    except Exception:
        etf_flow = empty

    return {
        "sentiment": sentiment if not sentiment.empty else None,
        "xueqiu": xueqiu if not xueqiu.empty else None,
        "guba": guba if not guba.empty else None,
        "comment": comment if not comment.empty else None,
        "margin": margin if not margin.empty else None,
        "northbound": northbound if not northbound.empty else None,
        "dragon_tiger": dragon_tiger if not dragon_tiger.empty else None,
        "fundamental": fundamental if not fundamental.empty else None,
        "etf_flow": etf_flow if not etf_flow.empty else None,
    }


def _build_features_for_config(df, aux, cfg_name, cfg):
    """Build features for one configuration.

    cfg: dict with use_* booleans
    """
    pipeline = FeaturePipeline(
        seq_len=cfg.features.get("flat_seq_len", cfg.features.seq_len),
        horizon=cfg.features.target_horizon,
        flat_mode=True,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=cfg_name in ("sentiment", "all", "all_mi", "all_mi_sfs"),
        use_announcements=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_guba=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_comment=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_xueqiu=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_margin=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_northbound=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_dragon_tiger=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_fundamental=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_etf_flow=cfg_name in ("all", "all_mi", "all_mi_sfs"),
        use_interaction=cfg_name in ("all", "all_mi", "all_mi_sfs"),
    )

    X, y, _ = pipeline.build_features(
        df,
        sentiment_df=aux.get("sentiment"),
        xueqiu_df=aux.get("xueqiu"),
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
        description="Feature selection benchmark"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--configs", type=str, default="technical,sentiment,all,all_mi",
                        help="Comma-separated configs to run")
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

    logger.info(
        "Benchmark: %d stocks × %d configs", len(codes),
        len(args.configs.split(",")),
    )

    # ---- Per-stock data loading + feature building ----
    # Build ALL-dimension features once per stock, then subset for
    # technical/sentiment configs by rebuilding with different pipelines.
    # This is correct because different configs produce different feature
    # matrices (different columns).

    config_names = [c.strip() for c in args.configs.split(",")]
    model_params = dict(cfg.model.params)
    model_params["n_estimators"] = 50  # speed over maximal accuracy for benchmark
    model_params["max_depth"] = 4
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
        step_months=6,  # wider step → fewer folds for benchmark speed
    )

    all_results = []
    stock_count = 0

    for code in codes:
        # Load K-line
        df = storage.load_daily(code, date_start, date_end)
        if df.empty or len(df) < 200:
            continue

        # Load auxiliary data once
        aux = _load_aux_data(code, date_start, date_end, data_dir)

        # Build features for EACH config (different column counts)
        stock_features = {}
        for cfg_name in config_names:
            try:
                X, y = _build_features_for_config(df, aux, cfg_name, cfg)
                if len(X) > 0:
                    stock_features[cfg_name] = (X, y)
                    n_feat = X.shape[1]
                    logger.info(
                        "  %s %-12s: %d samples, %d features",
                        code, cfg_name, len(X), n_feat,
                    )
            except Exception as e:
                logger.warning("  %s %s: FAILED — %s", code, cfg_name, e)

        if not stock_features:
            continue

        stock_count += 1

        # Walk-forward folds
        # Use the ALL config's sample count for pseudo-dates
        # (all configs produce same number of samples for the same stock)
        ref_X = next(iter(stock_features.values()))[0]
        n_samples = len(ref_X)
        pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
        folds = list(splitter.split(pseudo_dates))[:5]  # max 5 folds for speed

        for cfg_name, (X_all, y_all) in stock_features.items():
            n_samples_cfg = len(X_all)

            use_mi = cfg_name == "all_mi"
            use_mi_sfs = cfg_name == "all_mi_sfs"

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                if train_idx[-1] >= n_samples_cfg or val_idx[-1] >= n_samples_cfg:
                    break

                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]

                # Feature selection (fit on train ONLY, transform val)
                if use_mi or use_mi_sfs:
                    mi_k = 200
                    sfs_k = 50 if use_mi_sfs else 0
                    selector = FeatureSelector(
                        mi_k=min(mi_k, X_train.shape[1]),
                        sfs_k=min(sfs_k, min(mi_k, X_train.shape[1])),
                        model_type="lgbm",
                    )
                    try:
                        X_train = selector.fit_transform(X_train, y_train)
                        X_val = selector.transform(X_val)
                    except Exception as e:
                        logger.warning(
                            "  %s %s fold %d: selection failed — %s",
                            code, cfg_name, fold_idx, e,
                        )
                        continue

                # Train
                try:
                    model = XGBoostBaseline(**model_params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    metrics = compute_classification_metrics(y_val, preds)
                except Exception as e:
                    logger.warning(
                        "  %s %s fold %d: training failed — %s",
                        code, cfg_name, fold_idx, e,
                    )
                    continue

                all_results.append({
                    "stock": code,
                    "config": cfg_name,
                    "fold": fold_idx,
                    "n_features": X_train.shape[1],
                    **metrics,
                })

                logger.debug(
                    "  %s %-12s fold %d: MCC=%.4f Acc=%.4f feat=%d",
                    code, cfg_name, fold_idx,
                    metrics["mcc"], metrics["accuracy"], X_train.shape[1],
                )

        logger.info(
            "  [%d/%d] %s done (%d configs × %d folds)",
            stock_count, len(codes), code, len(stock_features), len(folds),
        )

    # ==================================================================
    # Results
    # ==================================================================
    if not all_results:
        logger.error("No results")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    logger.info("\n%s", "=" * 64)
    logger.info("FEATURE SELECTION BENCHMARK (%d stocks)", stock_count)
    logger.info("%s", "=" * 64)

    # Summary by config
    summary = results_df.groupby("config").agg(
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        acc_mean=("accuracy", "mean"),
        n_folds=("mcc", "count"),
        n_features=("n_features", "median"),
    ).round(4)
    logger.info("\n%s", summary.to_string())

    # Find best config
    best = summary["mcc_mean"].idxmax()
    best_mcc = summary.loc[best, "mcc_mean"]
    logger.info("\nBest: %s (MCC=%.4f)", best, best_mcc)

    # Delta vs technical baseline
    if "technical" in summary.index:
        baseline_mcc = summary.loc["technical", "mcc_mean"]
        logger.info("\nDeltas vs technical baseline (MCC=%.4f):", baseline_mcc)
        for cfg_name in summary.index:
            if cfg_name != "technical":
                delta = summary.loc[cfg_name, "mcc_mean"] - baseline_mcc
                logger.info("  %-12s: %+.4f", cfg_name, delta)

    # Save
    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_selection_benchmark.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved to %s", out_path)

    # Per-config feature counts
    logger.info("\nFeature dimensions (median across folds):")
    for cfg_name, row in summary.iterrows():
        logger.info("  %-15s: %d", cfg_name, int(row["n_features"]))


if __name__ == "__main__":
    main()
