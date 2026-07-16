"""Train TFT panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stocks 20 --epochs 10 --max-folds 1
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stock-list 600519,000001,000858
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --no-aux  # skip auxiliary data for quick test
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from stoke_ml.config import load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.tft import TFTConfig
from stoke_ml.models.tft.train import train_tft

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _discover_stocks(data_dir: str, limit: int | None = None) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.isdir(daily_dir):
        return []
    stocks = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(daily_dir) if f.endswith(".parquet")
    )
    return stocks[:limit] if limit else stocks


def load_aux_data(
    stock_list: list[str],
    data_dir: str,
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load auxiliary data (sentiment, guba, margin, etc.) per stock.

    Returns: {stock_code: {"sentiment": df, "guba": df, ...}}
    Only loads data types that exist on disk.
    """
    from stoke_ml.data.news_storage import NewsStorage
    from stoke_ml.data.guba_storage import GubaStorage
    from stoke_ml.data.xueqiu_storage import XueqiuStorage
    from stoke_ml.data.market_wide_storage import MarketWideStorage
    from stoke_ml.data.fundamental_storage import FundamentalStorage
    from stoke_ml.data.comment_storage import CommentStorage

    result: dict[str, dict[str, pd.DataFrame]] = {c: {} for c in stock_list}

    # --- Sentiment (news) ---
    try:
        ns = NewsStorage(data_dir)
        for code in stock_list:
            df = ns.load_daily_sentiment(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["sentiment"] = df
    except Exception:
        logger.warning("Sentiment data not available, skipping")

    # --- Announcements ---
    try:
        from stoke_ml.data.announcement_storage import AnnouncementStorage
        a_store = AnnouncementStorage(data_dir)
        for code in stock_list:
            df = a_store.load_daily_sentiment(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["announcement"] = df
    except Exception:
        logger.warning("Announcement data not available, skipping")

    # --- Guba ---
    try:
        gs = GubaStorage(data_dir)
        for code in stock_list:
            df = gs.load_daily_sentiment(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["guba"] = df
    except Exception:
        logger.warning("Guba data not available, skipping")

    # --- Xueqiu ---
    try:
        xs = XueqiuStorage(data_dir)
        for code in stock_list:
            df = xs.load_daily_sentiment(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["xueqiu"] = df
    except Exception:
        logger.warning("Xueqiu data not available, skipping")

    # --- Comment ---
    try:
        cs = CommentStorage(data_dir)
        for code in stock_list:
            df = cs.build_features(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["comment"] = df
    except Exception:
        logger.warning("Comment data not available, skipping")

    # --- Margin ---
    try:
        margin_storage = MarketWideStorage(data_dir, "margin")
        for code in stock_list:
            df = margin_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["margin"] = df
    except Exception:
        logger.warning("Margin data not available, skipping")

    # --- Fundamental ---
    try:
        fs = FundamentalStorage(data_dir)
        for code in stock_list:
            df = fs.load(code, "2010-01-01", end_date)
            if df is not None and not df.empty:
                result[code]["fundamental"] = df
    except Exception:
        logger.warning("Fundamental data not available, skipping")

    # --- Northbound ---
    try:
        nb_storage = MarketWideStorage(data_dir, "northbound")
        for code in stock_list:
            df = nb_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["northbound"] = df
    except Exception:
        logger.warning("Northbound data not available, skipping")

    # --- Dragon Tiger ---
    try:
        dt_storage = MarketWideStorage(data_dir, "dragon_tiger")
        for code in stock_list:
            df = dt_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["dragon_tiger"] = df
    except Exception:
        logger.warning("Dragon Tiger data not available, skipping")

    # --- ETF Flow (sector-level, aggregated to market-wide per date) ---
    try:
        from stoke_ml.data.etf_storage import ETFStorage
        etf = ETFStorage(data_dir)
        etf_base = os.path.join(data_dir, "a_shares", "etf_flow")
        etf_frames = []
        if os.path.isdir(etf_base):
            for f in os.listdir(etf_base):
                if f.startswith("sector_") and f.endswith(".parquet"):
                    sector_df = pd.read_parquet(os.path.join(etf_base, f))
                    etf_frames.append(sector_df)
        if etf_frames:
            etf_all = pd.concat(etf_frames, ignore_index=True)
            etf_all["date"] = pd.to_datetime(etf_all["date"])
            etf_agg = etf_all.groupby("date").agg(
                etf_flow_sum=("etf_flow_sum", "sum"),
                etf_amount_sum=("etf_amount_sum", "sum"),
            ).reset_index()
            for code in stock_list:
                result[code]["etf_flow"] = etf_agg
            logger.info("ETF flow aggregated from %d sector files", len(etf_frames))
    except Exception:
        logger.warning("ETF flow data not available, skipping")

    # --- Capital Flow ---
    try:
        cf_storage = MarketWideStorage(data_dir, "capital_flow")
        for code in stock_list:
            df = cf_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["capital_flow"] = df
    except Exception:
        logger.warning("Capital flow data not available, skipping")

    # --- Block Trade ---
    try:
        bt_storage = MarketWideStorage(data_dir, "block_trade")
        for code in stock_list:
            df = bt_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["block_trade"] = df
    except Exception:
        logger.warning("Block trade data not available, skipping")

    # --- Shareholder ---
    try:
        sh_storage = MarketWideStorage(data_dir, "shareholder")
        for code in stock_list:
            df = sh_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["shareholder"] = df
    except Exception:
        logger.warning("Shareholder data not available, skipping")

    # --- Lockup ---
    try:
        lu_storage = MarketWideStorage(data_dir, "lockup")
        for code in stock_list:
            df = lu_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["lockup"] = df
    except Exception:
        logger.warning("Lockup data not available, skipping")

    # --- Dividend ---
    try:
        dv_storage = MarketWideStorage(data_dir, "dividend")
        for code in stock_list:
            df = dv_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["dividend"] = df
    except Exception:
        logger.warning("Dividend data not available, skipping")

    # --- Valuation (daily PE/PB/PS/PCF from Baostock) ---
    try:
        val_storage = MarketWideStorage(data_dir, "valuation")
        for code in stock_list:
            df = val_storage.load(code, start_date, end_date)
            if df is not None and not df.empty:
                result[code]["valuation"] = df
    except Exception:
        logger.warning("Valuation data not available, skipping")

    loaded = sum(1 for v in result.values() if v)
    logger.info("Aux data loaded for %d/%d stocks", loaded, len(stock_list))
    return result


def main():
    parser = argparse.ArgumentParser(description="Train TFT panel model")
    parser.add_argument("--stocks", type=int, default=100,
                        help="Limit to first N stocks")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if args.stock_list:
        stock_list = [c.strip() for c in args.stock_list.split(",")]
    else:
        stock_list = _discover_stocks(data_dir, args.stocks)

    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)

    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
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

    # Load auxiliary data (unless --no-aux)
    aux_data = None
    if not args.no_aux:
        logger.info("Loading auxiliary data...")
        t_aux = time.time()
        aux_data = load_aux_data(stock_list, data_dir, args.start, args.end)
        logger.info("Aux data loaded in %.1fs", time.time() - t_aux)

    # Build features
    fp = FeaturePipeline(
        seq_len=60,
        use_sentiment=True, use_announcements=True,
        use_guba=True, use_comment=True, use_margin=True,
        use_northbound=True, use_dragon_tiger=True,
        use_fundamental=True, use_etf_flow=True, use_xueqiu=True,
        use_capital_flow=True, use_block_trade=True,
        use_shareholder=True, use_lockup=True, use_dividend=True,
        use_valuation=True,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(panel, aux_data=aux_data)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    dims = f"S={panel_data['static_features'].shape[1]} " \
           f"PK={panel_data['past_known'].shape[2]} " \
           f"PO={panel_data['past_observed'].shape[2]}"
    logger.info("Panel data: %d stocks × %d timesteps  dims: %s",
                n_stocks, n_timesteps, dims)

    config = TFTConfig(
        seq_len=60,
        static_dim=panel_data["static_features"].shape[1],
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
        num_workers=8,
    )
    logger.info("TFT config: hidden=%d layers=%d heads=%d batch=%d lr=%.1e",
                config.hidden_dim, config.lstm_layers, config.attention_heads,
                config.batch_size, config.learning_rate)

    # Purged walk-forward splits
    train_len = 504
    val_len = 63
    step = 63
    purge = config.seq_len  # must be >= seq_len to prevent context overlap
    all_sharpes = []

    fold = 0
    train_start = 0
    while train_start + train_len + purge + val_len < n_timesteps:
        if args.max_folds and fold >= args.max_folds:
            break
        fold += 1
        train_end = train_start + train_len
        val_start = train_end + purge
        val_end = min(val_start + val_len, n_timesteps)

        train_slice = slice(train_start, train_end)
        val_context_start = max(0, val_start - config.seq_len)
        val_slice = slice(val_context_start, val_end)

        train_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, train_slice],
            "past_observed": panel_data["past_observed"][:, train_slice],
            "y_direction": panel_data["y_direction"][:, train_slice],
            "y_return": panel_data["y_return"][:, train_slice].copy(),
            "y_volatility": panel_data["y_volatility"][:, train_slice].copy(),
        }
        val_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, val_slice],
            "past_observed": panel_data["past_observed"][:, val_slice],
            "y_direction": panel_data["y_direction"][:, val_slice],
            "y_return": panel_data["y_return"][:, val_slice].copy(),
            "y_volatility": panel_data["y_volatility"][:, val_slice].copy(),
        }

        # Per-stock z-score normalization of regression targets.
        # Different stocks have different return/vol distributions;
        # normalising per-stock gives each stock equal weight in the MSE
        # loss and keeps the MSE baseline ≈ 1.0 (balanced with CE ~1.0).
        ret_mean = train_data["y_return"].mean(axis=1, keepdims=True)
        ret_std = np.maximum(train_data["y_return"].std(axis=1, keepdims=True), 1e-8)
        vol_mean = train_data["y_volatility"].mean(axis=1, keepdims=True)
        vol_std = np.maximum(train_data["y_volatility"].std(axis=1, keepdims=True), 1e-8)
        train_data["y_return"] = (train_data["y_return"] - ret_mean) / ret_std
        train_data["y_volatility"] = (train_data["y_volatility"] - vol_mean) / vol_std
        val_data["y_return"] = (val_data["y_return"] - ret_mean) / ret_std
        val_data["y_volatility"] = (val_data["y_volatility"] - vol_mean) / vol_std
        # Clip normalized targets to [-5, 5] — regime changes can make
        # validation returns several sigma larger than training, which
        # would otherwise dominate the loss and destabilise training.
        np.clip(val_data["y_return"], -5.0, 5.0, out=val_data["y_return"])
        np.clip(val_data["y_volatility"], -5.0, 5.0, out=val_data["y_volatility"])

        logger.info("Fold %d/%d: train [%d:%d], val [%d:%d]",
                    fold, args.max_folds or "∞",
                    train_start, train_end, val_start, val_end)

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
