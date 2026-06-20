"""Sentiment ablation: compare with/without news on representative stocks.

Runs XGBoost and LSTM on each stock, with and without sentiment features,
to quantify the marginal value of news data.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/run_ablation.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/run_ablation.py --stocks 600519,000858,601318
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
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import mcc_score, compute_classification_metrics, compute_financial_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# High-news stocks recommended for ablation
DEFAULT_STOCKS = ["600519", "000858", "601318", "000001", "600036"]


def run_xgboost(X, y, aligned_close, folds, model_params, output_dir, code, tag):
    """Run XGBoost on given features."""
    n_samples = len(X)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
            break
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = XGBoostBaseline(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        cls_m = compute_classification_metrics(y_val, preds)
        close_prices = aligned_close[val_idx[0]:val_idx[-1] + 2]
        fin_m = compute_financial_metrics(close_prices, preds)
        results.append({"fold": fold_idx, "mcc": cls_m["mcc"],
                        "accuracy": cls_m["accuracy"], "sharpe": fin_m["sharpe"]})
    return results


def run_lstm(X, y, aligned_close, folds, n_features, cfg, output_dir, code, tag):
    """Run LSTM on given features."""
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from stoke_ml.models.dl.dataset import StockDataset
    from stoke_ml.models.dl.lightning_module import StockLightningModule

    batch_size = cfg.training.batch_size
    max_epochs = cfg.training.epochs
    n_samples = len(X)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
            break
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = StockDataset(X_train, y_train)
        val_ds = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        class_weight = [1.0, n_neg / n_pos] if n_pos > 0 and n_neg > 0 else None

        lit_module = StockLightningModule(
            input_dim=n_features, hidden_dim=128, num_layers=2, dropout=0.3,
            learning_rate=cfg.training.learning_rate, class_weight=class_weight,
        )
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(output_dir, "ablation_checkpoints"),
            filename=f"{code}_{tag}_fold{fold_idx}",
            monitor="val_mcc", mode="max", save_top_k=1,
        )
        early_stop_cb = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.training.early_stopping_patience, mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="auto", devices=1,
            callbacks=[checkpoint_cb, early_stop_cb],
            enable_progress_bar=False, log_every_n_steps=10,
        )
        trainer.fit(lit_module, train_loader, val_loader)

        best_path = checkpoint_cb.best_model_path
        if best_path:
            best_module = StockLightningModule.load_from_checkpoint(
                best_path, weights_only=False,
            )
            best_module.eval()
            best_module.to("cuda" if torch.cuda.is_available() else "cpu")
            all_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(best_module.device)
                    logits = best_module(xb)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.append(preds)
            val_preds = np.concatenate(all_preds)
            val_mcc = mcc_score(y_val, val_preds)
            close_prices = aligned_close[val_idx[0]:val_idx[-1] + 2]
            fin_m = compute_financial_metrics(close_prices, val_preds)
            results.append({"fold": fold_idx, "mcc": val_mcc,
                           "sharpe": fin_m["sharpe"]})

    return results


def main():
    parser = argparse.ArgumentParser(description="Sentiment ablation experiment")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: high-news stocks)")
    parser.add_argument("--model", type=str, default="both",
                        choices=["xgboost", "lstm", "both"],
                        help="Model to ablate (default: both)")
    args = parser.parse_args()

    cfg = load_config()
    storage = DataStorage(cfg.project.data_dir)
    news_storage = NewsStorage(cfg.project.data_dir)
    output_dir = cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)

    stocks = [c.strip() for c in args.stocks.split(",")] if args.stocks else DEFAULT_STOCKS
    logger.info("Ablation on %d stocks: %s", len(stocks), stocks)

    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    )

    all_results = []

    for code in stocks:
        logger.info("=== %s ===", code)
        df = storage.load_daily(
            code,
            start_date=cfg.markets.a_shares.start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        if df.empty:
            logger.warning("  No K-line data, skipping")
            continue

        sentiment_df = news_storage.load_daily_sentiment(
            code,
            start_date=cfg.markets.a_shares.start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        news_days = sentiment_df["has_news"].sum() if not sentiment_df.empty else 0
        logger.info("  News days: %d", news_days)

        for use_sent in [True, False]:
            label = "with_sent" if use_sent else "tech_only"
            logger.info("  --- %s ---", label)

            # XGBoost (flat mode)
            if args.model in ("xgboost", "both"):
                pipeline = FeaturePipeline(
                    seq_len=cfg.features.seq_len, horizon=cfg.features.target_horizon,
                    flat_mode=True,
                    use_technical=cfg.features.technical_indicators,
                    use_scoring=cfg.features.rule_based_scoring,
                    use_temporal=cfg.features.temporal_features,
                    use_sentiment=use_sent,
                )
                sd = sentiment_df if (use_sent and not sentiment_df.empty) else None
                X, y, ac = pipeline.build_features(df, sentiment_df=sd)
                if len(X) > 0:
                    pseudo_dates = pd.date_range("2000-01-01", periods=len(X), freq="B")
                    folds = list(splitter.split(pseudo_dates))
                    t0 = time.time()
                    model_params = dict(cfg.model.params)
                    xgb_results = run_xgboost(X, y, ac, folds, model_params, output_dir, code, label)
                    mccs = [r["mcc"] for r in xgb_results]
                    logger.info("  XGBoost %s | MCC=%.4f ± %.4f [%d folds] %.1fs",
                                label, np.mean(mccs), np.std(mccs), len(mccs), time.time() - t0)
                    for r in xgb_results:
                        all_results.append({"stock": code, "model": "XGBoost",
                                            "sentiment": use_sent, **r})
                else:
                    logger.warning("  XGBoost %s | No features generated, skipping", label)

            # LSTM (sequence mode)
            if args.model in ("lstm", "both"):
                pipeline = FeaturePipeline(
                    seq_len=cfg.features.seq_len, horizon=cfg.features.target_horizon,
                    flat_mode=False,
                    use_technical=cfg.features.technical_indicators,
                    use_scoring=cfg.features.rule_based_scoring,
                    use_temporal=cfg.features.temporal_features,
                    use_sentiment=use_sent,
                )
                sd = sentiment_df if (use_sent and not sentiment_df.empty) else None
                X, y, ac = pipeline.build_features(df, sentiment_df=sd)
                if len(X) > 0:
                    n_features = X.shape[2]
                    pseudo_dates = pd.date_range("2000-01-01", periods=len(X), freq="B")
                    folds = list(splitter.split(pseudo_dates))
                    t0 = time.time()
                    lstm_results = run_lstm(X, y, ac, folds, n_features, cfg, output_dir, code, label)
                    if lstm_results:
                        mccs = [r["mcc"] for r in lstm_results]
                        logger.info("  LSTM %s    | MCC=%.4f ± %.4f [%d folds] %.1fs",
                                    label, np.mean(mccs), np.std(mccs), len(mccs), time.time() - t0)
                        for r in lstm_results:
                            all_results.append({"stock": code, "model": "LSTM",
                                                "sentiment": use_sent, **r})
                else:
                    logger.warning("  LSTM %s | No features generated, skipping", label)

    # Summary
    if all_results:
        rdf = pd.DataFrame(all_results)
        rdf.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)
        logger.info("\n=== Ablation Summary ===")
        for model in rdf["model"].unique():
            for sent in [True, False]:
                sub = rdf[(rdf["model"] == model) & (rdf["sentiment"] == sent)]
                if len(sub) > 0:
                    logger.info("%s %s: MCC=%.4f ± %.4f",
                                model, "w/sent" if sent else "no sent",
                                sub["mcc"].mean(), sub["mcc"].std())


if __name__ == "__main__":
    main()
