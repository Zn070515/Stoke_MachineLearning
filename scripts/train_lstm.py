"""Phase 2: LSTM training with PyTorch Lightning."""
import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import mcc_score, compute_financial_metrics
from stoke_ml.models.dl.dataset import StockDataset
from stoke_ml.models.dl.lstm_model import LSTMModel
from stoke_ml.models.dl.lightning_module import StockLightningModule

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
    parser = argparse.ArgumentParser(description="Train LSTM model")
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
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs (default: config training.epochs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: config training.batch_size)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    storage = DataStorage(cfg.project.data_dir)
    news_storage = NewsStorage(cfg.project.data_dir)
    codes = [args.stock] if args.stock else available_stocks(storage)

    if not codes:
        logger.error("No stock data found. Run a data downloader first.")
        sys.exit(1)

    logger.info("Training LSTM on %d stock(s)", len(codes))
    logger.info("CUDA available: %s", torch.cuda.is_available())

    output_dir = args.output or cfg.project.model_dir
    max_epochs = args.epochs if args.epochs is not None else cfg.training.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.training.batch_size
    os.makedirs(output_dir, exist_ok=True)

    pipeline = FeaturePipeline(
        seq_len=cfg.features.seq_len,
        horizon=cfg.features.target_horizon,
        flat_mode=False,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=cfg.features.get("use_sentiment", True),
    )
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    )

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

        # Load daily sentiment if available
        sentiment_df = news_storage.load_daily_sentiment(
            code,
            start_date=cfg.markets.a_shares.start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        if not sentiment_df.empty:
            logger.info(
                "  %s: loaded %d sentiment days (%d with news)",
                code, len(sentiment_df), sentiment_df["has_news"].sum(),
            )

        X, y, aligned_close = pipeline.build_features(
            df, sentiment_df=sentiment_df if not sentiment_df.empty else None,
        )
        if len(X) == 0:
            logger.warning("Not enough features for %s, skipping", code)
            continue

        n_features = X.shape[2]
        logger.info(
            "  Data: X=%s y=%s n_features=%d", X.shape, y.shape, n_features
        )

        n_samples = len(X)
        pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
        folds = list(splitter.split(pseudo_dates))

        all_val_mcc = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
                break

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            train_ds = StockDataset(X_train, y_train)
            val_ds = StockDataset(X_val, y_val)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            )

            # Class weight for imbalance
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            if n_pos > 0 and n_neg > 0:
                class_weight = [1.0, n_neg / n_pos]
            else:
                class_weight = None

            lit_module = StockLightningModule(
                input_dim=n_features,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3,
                learning_rate=cfg.training.learning_rate,
                class_weight=class_weight,
            )

            checkpoint_cb = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, "lstm_checkpoints"),
                filename=f"{code}_fold{fold_idx}",
                monitor="val_mcc",
                mode="max",
                save_top_k=1,
            )
            early_stop_cb = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                callbacks=[checkpoint_cb, early_stop_cb],
                enable_progress_bar=True,
                log_every_n_steps=10,
            )

            trainer.fit(lit_module, train_loader, val_loader)

            # Evaluate best checkpoint
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
                # financial metrics
                if len(val_idx) > 1:
                    close_prices = aligned_close[val_idx[0]:val_idx[-1] + 2]
                    fin_metrics = compute_financial_metrics(close_prices, val_preds)
                else:
                    fin_metrics = {"sharpe": 0.0, "max_drawdown": 0.0,
                                   "win_rate": 0.0, "profit_factor": 0.0}
                all_val_mcc.append(val_mcc)
                logger.info(
                    "  %s fold %d | Val MCC=%.4f Sharpe=%.2f",
                    code, fold_idx, val_mcc, fin_metrics["sharpe"],
                )
            else:
                logger.warning("  %s fold %d | No checkpoint saved", code, fold_idx)

        if all_val_mcc:
            logger.info(
                "  %s | Avg MCC=%.4f ± %.4f",
                code, np.mean(all_val_mcc), np.std(all_val_mcc),
            )

            # Retrain on all data for final model
            # Keep last 20% as validation for early stopping,
            # preserve temporal order (no shuffle in time series)
            n_total = len(X)
            n_val = max(int(n_total * 0.2), 1)
            X_final_train, X_final_val = X[:-n_val], X[-n_val:]
            y_final_train, y_final_val = y[:-n_val], y[-n_val:]

            logger.info("  Retraining final model for %s (%d train / %d val)",
                        code, len(X_final_train), len(X_final_val))

            train_ds = StockDataset(X_final_train, y_final_train)
            val_ds = StockDataset(X_final_val, y_final_val)
            final_train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            )
            final_val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            )

            n_neg = (y_final_train == 0).sum()
            n_pos = (y_final_train == 1).sum()
            if n_pos > 0 and n_neg > 0:
                final_class_weight = [1.0, n_neg / n_pos]
            else:
                final_class_weight = None

            final_module = StockLightningModule(
                input_dim=n_features,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3,
                learning_rate=cfg.training.learning_rate,
                class_weight=final_class_weight,
            )
            final_checkpoint_cb = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, "lstm_checkpoints"),
                filename=f"{code}_final",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
            final_early_stop_cb = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
            )
            final_trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                callbacks=[final_checkpoint_cb, final_early_stop_cb],
                enable_progress_bar=True,
                log_every_n_steps=10,
            )
            final_trainer.fit(final_module, final_train_loader, final_val_loader)

            model_path = os.path.join(output_dir, f"lstm_{code}_final.ckpt")
            final_trainer.save_checkpoint(model_path)
            logger.info("  Final model saved to %s", model_path)


if __name__ == "__main__":
    main()
