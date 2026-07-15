"""Train TFT panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stocks 20 --epochs 30
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stock-list 600519,000001,000858
"""
import argparse
import logging
import sys
import time
from datetime import datetime

import torch
import numpy as np

from stoke_ml.config import load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.tft import TFTConfig
from stoke_ml.models.tft.train import train_tft

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TFT panel model")
    parser.add_argument("--stocks", type=int, default=None,
                        help="Limit to first N stocks (for quick testing)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load config and data
    cfg = load_config()
    data_dir = cfg.project.data_dir

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    if args.stock_list:
        stock_list = [c.strip() for c in args.stock_list.split(",")]
    else:
        stock_list = ds.list_stocks()
        if args.stocks:
            stock_list = stock_list[:args.stocks]

    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    # Build per-stock DataFrames into a panel
    import pandas as pd
    frames = []
    for code in stock_list:
        df = ds.load_daily(code, args.start, args.end)
        if df is not None and not df.empty:
            df["stock_code"] = code
            frames.append(df)
    if not frames:
        logger.error("No data loaded for any stock")
        sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    logger.info("Panel shape: %s", panel.shape)

    # Build features
    fp = FeaturePipeline(
        seq_len=252,
        use_sentiment=True, use_announcements=False,
        use_guba=True, use_comment=False, use_margin=True,
        use_northbound=False, use_dragon_tiger=False,
        use_fundamental=True, use_etf_flow=False, use_xueqiu=True,
        use_capital_flow=False, use_block_trade=False,
        use_shareholder=False, use_lockup=False, use_dividend=False,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(panel)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    logger.info("Panel data: %d stocks × %d timesteps", n_stocks, n_timesteps)

    # TFT config — infer input dims from built data
    config = TFTConfig(
        seq_len=252,
        static_dim=panel_data["static_features"].shape[1],
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
    )
    logger.info("TFT config: hidden=%d, total params ~%dM",
                config.hidden_dim,
                config.hidden_dim * config.hidden_dim * 4 // 1_000_000)

    # Purged walk-forward splits
    train_len = 504  # ~2 years
    val_len = 63  # ~3 months
    step = 63  # ~3 months
    purge = 5
    all_sharpes = []

    fold = 0
    train_start = 0
    while train_start + train_len + purge + val_len < n_timesteps:
        fold += 1
        train_end = train_start + train_len
        val_start = train_end + purge
        val_end = min(val_start + val_len, n_timesteps)

        train_slice = slice(train_start, train_end)
        val_slice = slice(val_start, val_end)

        train_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, train_slice],
            "past_observed": panel_data["past_observed"][:, train_slice],
            "y_direction": panel_data["y_direction"][:, train_slice],
            "y_return": panel_data["y_return"][:, train_slice],
            "y_volatility": panel_data["y_volatility"][:, train_slice],
        }
        val_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, val_slice],
            "past_observed": panel_data["past_observed"][:, val_slice],
            "y_direction": panel_data["y_direction"][:, val_slice],
            "y_return": panel_data["y_return"][:, val_slice],
            "y_volatility": panel_data["y_volatility"][:, val_slice],
        }

        logger.info("Fold %d: train [%d:%d], val [%d:%d]",
                    fold, train_start, train_end, val_start, val_end)

        t0 = time.time()
        model, history = train_tft(config, train_data, val_data, device)
        elapsed = time.time() - t0

        if history["val_sharpe"]:
            best_sharpe = max(history["val_sharpe"])
            all_sharpes.append(best_sharpe)
            logger.info("  Fold %d best Sharpe: %.4f (%.1fs)", fold, best_sharpe, elapsed)
        else:
            logger.warning("  Fold %d: no valid Sharpe (%.1fs)", fold, elapsed)

        train_start += step

    if all_sharpes:
        logger.info("Mean Sharpe across %d folds: %.4f", len(all_sharpes), np.mean(all_sharpes))
    else:
        logger.warning("No valid folds completed")


if __name__ == "__main__":
    main()
