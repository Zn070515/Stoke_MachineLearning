"""Batch run LSTM on all 798 stocks with logging.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/run_all_lstm.py           # all stocks
  PYTHONPATH=. ./.venv/Scripts/python scripts/run_all_lstm.py --top 50  # top 50 by news count
  PYTHONPATH=. ./.venv/Scripts/python scripts/run_all_lstm.py --dry-run  # list stocks only
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
from stoke_ml.evaluation.metrics import mcc_score, compute_financial_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_stocks_ranked(data_dir: str, top_n: int | None = None) -> tuple[list[str], dict[str, int]]:
    """Get stocks ranked by news coverage (most news first).

    Returns (ordered_stocks, counts_dict) to avoid reloading raw news later.
    """
    news = NewsStorage(data_dir)
    stocks = news.list_stocks_with_raw_news()
    if not stocks:
        return [], {}

    counts = {}
    for code in stocks:
        df = news.load_raw_news(code)
        counts[code] = len(df)

    ranked = sorted(counts.items(), key=lambda x: -x[1])
    if top_n:
        ranked = ranked[:top_n]
    return [code for code, _ in ranked], counts


def run_single_stock(code: str, cfg, storage, news_storage, output_dir: str) -> dict:
    """Run LSTM on a single stock, return summary dict."""
    from stoke_ml.features.pipeline import FeaturePipeline
    from stoke_ml.evaluation.splitter import WalkForwardSplitter
    from stoke_ml.models.dl.dataset import StockDataset
    from stoke_ml.models.dl.lightning_module import StockLightningModule

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    df = storage.load_daily(
        code,
        start_date=cfg.markets.a_shares.start_date,
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )
    if df.empty:
        return {"stock": code, "status": "no_kline"}

    sentiment_df = news_storage.load_daily_sentiment(
        code,
        start_date=cfg.markets.a_shares.start_date,
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )
    news_days = sentiment_df["has_news"].sum() if not sentiment_df.empty else 0

    pipeline = FeaturePipeline(
        seq_len=cfg.features.seq_len,
        horizon=cfg.features.target_horizon,
        flat_mode=False,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=cfg.features.get("use_sentiment", True),
    )
    X, y, aligned_close = pipeline.build_features(
        df, sentiment_df=sentiment_df if not sentiment_df.empty else None,
    )
    if len(X) == 0:
        return {"stock": code, "status": "no_features"}

    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
    )

    n_samples = len(X)
    n_features = X.shape[2]
    pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
    folds = list(splitter.split(pseudo_dates))

    batch_size = cfg.training.batch_size
    max_epochs = cfg.training.epochs
    all_val_mcc = []
    all_val_sharpe = []

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
            dirpath=os.path.join(output_dir, "lstm_checkpoints"),
            filename=f"{code}_fold{fold_idx}",
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
            all_val_mcc.append(val_mcc)
            # Financial metrics per fold
            close_prices = aligned_close[val_idx[0]:val_idx[-1] + 2]
            fin_m = compute_financial_metrics(close_prices, val_preds)
            all_val_sharpe.append(fin_m["sharpe"])

    if not all_val_mcc:
        return {"stock": code, "status": "no_folds"}

    return {
        "stock": code,
        "status": "ok",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_folds": len(all_val_mcc),
        "news_days": int(news_days),
        "mcc_mean": float(np.mean(all_val_mcc)),
        "mcc_std": float(np.std(all_val_mcc)),
        "mcc_max": float(np.max(all_val_mcc)),
        "mcc_min": float(np.min(all_val_mcc)),
        "sharpe_mean": float(np.mean(all_val_sharpe)),
        "sharpe_std": float(np.std(all_val_sharpe)),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch LSTM training on all stocks")
    parser.add_argument("--top", type=int, default=None,
                        help="Only run top N stocks by news count")
    parser.add_argument("--dry-run", action="store_true",
                        help="List stocks and exit")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume from this stock code")
    args = parser.parse_args()

    # Add file logging inside main() so import-time failures don't crash
    try:
        file_handler = logging.FileHandler("run_all_lstm.log")
        logging.getLogger().addHandler(file_handler)
    except (PermissionError, OSError) as e:
        logger.warning("Cannot write log file: %s — logging to console only", e)

    cfg = load_config()
    storage = DataStorage(cfg.project.data_dir)
    news_storage = NewsStorage(cfg.project.data_dir)
    output_dir = cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)

    stocks, counts = get_stocks_ranked(cfg.project.data_dir, args.top)
    if not stocks:
        logger.error("No stocks found.")
        sys.exit(1)

    logger.info("Stocks to process: %d", len(stocks))
    for i, code in enumerate(stocks):
        logger.info("  %3d. %s — %d articles", i + 1, code, counts.get(code, 0))

    if args.dry_run:
        return

    skip = args.start_from is not None
    results = []
    t_start = time.time()

    for i, code in enumerate(stocks):
        if skip:
            if code == args.start_from:
                skip = False
            else:
                logger.info("[%d/%d] %s — skipping", i + 1, len(stocks), code)
                continue

        t0 = time.time()
        logger.info("[%d/%d] %s ...", i + 1, len(stocks), code)
        result = run_single_stock(code, cfg, storage, news_storage, output_dir)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)

        if result["status"] == "ok":
            logger.info(
                "  %s | MCC=%.4f ± %.4f [%d folds, %d news_days] %.1fs",
                result["stock"], result["mcc_mean"], result["mcc_std"],
                result["n_folds"], result["news_days"], elapsed,
            )
        else:
            logger.info("  %s | status=%s", result["stock"], result["status"])

        results.append(result)

        # Save incremental results
        pd.DataFrame(results).to_csv(
            os.path.join(output_dir, "lstm_all_results.csv"), index=False,
        )

    total_elapsed = time.time() - t_start
    logger.info("Done: %d stocks in %.1f min", len(results), total_elapsed / 60)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        mccs = [r["mcc_mean"] for r in ok_results]
        logger.info(
            "Overall MCC: %.4f ± %.4f (min=%.4f max=%.4f)",
            np.mean(mccs), np.std(mccs), np.min(mccs), np.max(mccs),
        )


if __name__ == "__main__":
    main()
