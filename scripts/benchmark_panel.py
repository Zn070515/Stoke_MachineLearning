"""Panel-mode vs single-stock training benchmark.

Compares XGBoost performance with and without cross-sectional normalization
on the same stocks, date ranges, and model parameters.

PANEL mode:  PanelBuilder → build_features_from_panel(cross_sectional=True)
SINGLE mode: per-stock FeaturePipeline (no cross-sectional normalization)
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
from stoke_ml.data.panel_builder import PanelBuilder
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _trading_date_splits(dates, train_years=2, val_months=3, step_months=3):
    """Generate (train_start, train_end, val_start, val_end) date tuples."""
    unique_dates = sorted(pd.DatetimeIndex(dates).unique())
    n = len(unique_dates)
    train_days = train_years * 252
    val_days = val_months * 21
    step_days = step_months * 21

    start = 0
    while True:
        train_end = start + train_days
        val_end = train_end + val_days
        if val_end > n:
            break
        if train_end - start >= 200:
            yield (
                unique_dates[start],
                unique_dates[train_end - 1],
                unique_dates[train_end],
                unique_dates[val_end - 1],
            )
        start += step_days


def _build_single_features(codes, date_start, date_end, storage, pipeline):
    """Build per-stock features for the full date range.

    Returns lists of (X, y, sample_dates) tuples — one per stock.
    Sample dates track the prediction date after dropna, so downstream
    code can split by date without leakage.
    """
    results = []
    for code in codes:
        df = storage.load_daily(code, date_start, date_end)
        if df.empty or len(df) < 120:
            continue

        # Run feature engineering to get dates
        feats = pipeline._engineer_features(df)
        if feats.empty:
            continue

        # Track dates through dropna (same as _create_sequences)
        drop_cols = ["date", "stock_code", "sector", "size_proxy"]
        feat_df = feats.drop(columns=[c for c in drop_cols if c in feats.columns])
        valid_mask = feat_df.notna().all(axis=1)
        feat_df = feat_df.dropna()
        valid_dates = feats.loc[valid_mask.values, "date"].values

        target_col = "close"
        close = feat_df[target_col].values
        horizon = pipeline.horizon
        seq_len = pipeline.seq_len

        if len(close) < seq_len + horizon + 10:
            continue

        target = (close[horizon:] > close[:-horizon]).astype(int)
        price_cols = ["open", "high", "low", "close", "amount"]
        X_cols = [c for c in feat_df.columns if c not in price_cols]
        X_data = feat_df[X_cols].values.astype(np.float32)

        n_samples = len(X_data) - seq_len - horizon + 1
        if n_samples <= 0:
            continue

        if pipeline.flat_mode:
            X = np.array(
                [X_data[i : i + seq_len].flatten() for i in range(n_samples)],
                dtype=np.float32,
            )
        else:
            X = np.array(
                [X_data[i : i + seq_len] for i in range(n_samples)],
                dtype=np.float32,
            )

        y = target[seq_len - 1 : seq_len - 1 + n_samples]

        # sample i predicts price change ending at valid_dates[seq_len-1+i+horizon]
        sample_dates = pd.DatetimeIndex(
            valid_dates[seq_len - 1 + horizon : seq_len - 1 + horizon + n_samples]
        )

        results.append((X, y, sample_dates, code))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Panel vs single-stock training benchmark"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", choices=["panel", "single", "both"], default="both")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=8,
                        help="Limit number of folds (default: 8)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    storage = DataStorage(data_dir)

    # ---- Stock selection ----
    codes = sorted([
        f.replace(".parquet", "")
        for f in os.listdir(os.path.join(data_dir, "a_shares", "daily"))
        if f.endswith(".parquet")
    ])
    if args.stocks and args.stocks < len(codes):
        step = max(len(codes) // args.stocks, 1)
        codes = [codes[i * step] for i in range(args.stocks)]

    logger.info("Using %d stocks: %s ... %s", len(codes), codes[:3], codes[-3:])

    # ---- Pipeline (technical-only for clean comparison) ----
    pipeline = FeaturePipeline(
        seq_len=args.seq_len,
        horizon=cfg.features.target_horizon,
        flat_mode=True,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=False, use_announcements=False,
        use_guba=False, use_comment=False, use_xueqiu=False,
        use_margin=False, use_northbound=False,
        use_dragon_tiger=False, use_fundamental=False,
        use_etf_flow=False, use_interaction=False,
    )

    model_params = dict(cfg.model.params)
    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    # ---- Date-based folds ----
    panel_dates = pd.DatetimeIndex(
        pd.date_range(date_start, date_end, freq="B")
    )
    all_folds = list(_trading_date_splits(
        panel_dates,
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    ))
    folds = all_folds[: args.max_folds]
    logger.info("Folds: %d (of %d total)", len(folds), len(all_folds))

    all_results = []

    # ==================================================================
    # PANEL mode
    # ==================================================================
    if args.mode in ("panel", "both"):
        logger.info("=== PANEL MODE (cross-sectional normalization) ===")
        t0 = time.time()

        pb = PanelBuilder(data_dir)
        panel = pb.build(codes, date_start, date_end, min_rows_per_stock=100)
        logger.info("Panel: %d stocks × %d rows",
                     panel["stock_code"].nunique(), len(panel))

        for fold_idx, (tr_s, tr_e, va_s, va_e) in enumerate(folds):
            train_panel = panel[
                (panel["date"] >= pd.Timestamp(tr_s))
                & (panel["date"] <= pd.Timestamp(tr_e))
            ]
            val_panel = panel[
                (panel["date"] >= pd.Timestamp(va_s))
                & (panel["date"] <= pd.Timestamp(va_e))
            ]

            if train_panel.empty or val_panel.empty:
                continue

            X_train, y_train, _, _ = pipeline.build_features_from_panel(
                train_panel, cross_sectional=True,
            )
            X_val, y_val, _, _ = pipeline.build_features_from_panel(
                val_panel, cross_sectional=True,
            )

            if len(X_train) < 100 or len(X_val) < 10:
                continue

            model = XGBoostBaseline(**model_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metrics = compute_classification_metrics(y_val, preds)

            logger.info(
                "  Fold %d [%s→%s | %s→%s]: MCC=%.4f Acc=%.4f "
                "n_train=%d n_val=%d",
                fold_idx, str(tr_s)[:10], str(tr_e)[:10],
                str(va_s)[:10], str(va_e)[:10],
                metrics["mcc"], metrics["accuracy"],
                len(X_train), len(X_val),
            )
            all_results.append({
                "mode": "panel", "fold": fold_idx, **metrics,
            })

        logger.info("Panel mode: %.0fs", time.time() - t0)

    # ==================================================================
    # SINGLE mode
    # ==================================================================
    if args.mode in ("single", "both"):
        logger.info("=== SINGLE MODE (no cross-sectional normalization) ===")
        t0 = time.time()

        # Build features ONCE for full range, track dates per sample
        stock_data = _build_single_features(
            codes, date_start, date_end, storage, pipeline,
        )
        logger.info("Built features for %d stocks", len(stock_data))

        for fold_idx, (tr_s, tr_e, va_s, va_e) in enumerate(folds):
            X_train_parts, y_train_parts = [], []
            X_val_parts, y_val_parts = [], []

            tr_s_ts = pd.Timestamp(tr_s)
            tr_e_ts = pd.Timestamp(tr_e)
            va_s_ts = pd.Timestamp(va_s)
            va_e_ts = pd.Timestamp(va_e)

            for X_all, y_all, sample_dates, code in stock_data:
                train_mask = (sample_dates >= tr_s_ts) & (sample_dates <= tr_e_ts)
                val_mask = (sample_dates >= va_s_ts) & (sample_dates <= va_e_ts)

                if train_mask.sum() >= 50:
                    X_train_parts.append(X_all[train_mask])
                    y_train_parts.append(y_all[train_mask])
                if val_mask.sum() >= 5:
                    X_val_parts.append(X_all[val_mask])
                    y_val_parts.append(y_all[val_mask])

            if not X_train_parts or not X_val_parts:
                continue

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            X_val = np.concatenate(X_val_parts, axis=0)
            y_val = np.concatenate(y_val_parts, axis=0)

            model = XGBoostBaseline(**model_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metrics = compute_classification_metrics(y_val, preds)

            logger.info(
                "  Fold %d [%s→%s | %s→%s]: MCC=%.4f Acc=%.4f "
                "n_train=%d n_val=%d",
                fold_idx, str(tr_s)[:10], str(tr_e)[:10],
                str(va_s)[:10], str(va_e)[:10],
                metrics["mcc"], metrics["accuracy"],
                len(X_train), len(X_val),
            )
            all_results.append({
                "mode": "single", "fold": fold_idx, **metrics,
            })

        logger.info("Single mode: %.0fs", time.time() - t0)

    # ==================================================================
    # Comparison
    # ==================================================================
    if not all_results:
        logger.error("No results — check data availability")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby("mode")["mcc"].agg(["mean", "std", "count"])
    logger.info("\n%s", "=" * 60)
    logger.info("PANEL vs SINGLE (%d stocks, %d folds)", len(codes), len(folds))
    logger.info("%s", "=" * 60)
    logger.info("\n%s", summary.to_string())

    modes_present = results_df["mode"].unique()
    if "panel" in modes_present and "single" in modes_present:
        panel_mcc = results_df[results_df["mode"] == "panel"]["mcc"]
        single_mcc = results_df[results_df["mode"] == "single"]["mcc"]
        delta = panel_mcc.mean() - single_mcc.mean()
        logger.info(
            "\nDelta MCC (panel - single): %+.4f  (panel=%.4f, single=%.4f)",
            delta, panel_mcc.mean(), single_mcc.mean(),
        )

        pivot = results_df.pivot_table(
            index="fold", columns="mode", values="mcc", aggfunc="mean"
        )
        if "panel" in pivot.columns and "single" in pivot.columns:
            pivot["delta"] = pivot["panel"] - pivot["single"]
            logger.info("\nPer-fold delta (panel - single):")
            for fold, row in pivot.iterrows():
                logger.info(
                    "  Fold %d: panel=%.4f single=%.4f delta=%+.4f",
                    fold, row["panel"], row["single"], row["delta"],
                )

    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "panel_vs_single.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved to %s", out_path)


if __name__ == "__main__":
    main()
