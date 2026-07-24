"""Train VSN+xLSTM panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --stocks 500 --epochs 30 --max-folds 1
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --stock-list 600519,000001,000858
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --no-aux  # skip auxiliary data for quick test
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
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.train import train_panel

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

    # --- Announcements (CNINFO PDF body sentiment preferred, EastMoney fallback) ---
    try:
        cninfo_dir = os.path.join(data_dir, "a_shares", "cninfo_announcements", "sentiment")
        em_loaded = False
        if os.path.isdir(cninfo_dir):
            for code in stock_list:
                path = os.path.join(cninfo_dir, f"{code}.parquet")
                if os.path.isfile(path):
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    if start_date:
                        df = df[df["date"] >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df["date"] <= pd.Timestamp(end_date)]
                    if not df.empty:
                        result[code]["announcement"] = df.sort_values("date").reset_index(drop=True)
            cninfo_count = sum(1 for c in result if "announcement" in result[c])
            if cninfo_count > 0:
                logger.info("CNINFO announcements loaded for %d stocks", cninfo_count)
                em_loaded = True

        # Fallback: EastMoney for stocks without CNINFO data
        if not em_loaded or len([c for c in stock_list if "announcement" not in result.get(c, {})]) > 0:
            from stoke_ml.data.announcement_storage import AnnouncementStorage
            a_store = AnnouncementStorage(data_dir)
            for code in stock_list:
                if "announcement" in result.get(code, {}):
                    continue
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


def _filter_quality(stock_list: list[str], data_dir: str) -> list[str]:
    """Filter out stocks with corrupted price data.

    Checks: close > 0, daily-return std < 50 %, no obviously bogus prices.
    Returns only the codes that pass all checks.
    """
    import pandas as pd
    import numpy as np
    from stoke_ml.data.storage import DataStorage

    ds = DataStorage(data_dir)
    ok: list[str] = []
    n_neg, n_vol, n_nan, n_low, n_fwd = 0, 0, 0, 0, 0
    for code in stock_list:
        df = ds.load_daily(code, "2015-01-01", "2099-12-31")
        if df is None or df.empty:
            continue
        close = df["close"].values
        if np.isnan(close).all():
            n_nan += 1
            continue
        if (close <= 0).any():
            n_neg += 1
            continue
        if close.min() < 0.001:
            n_low += 1
            continue
        ret = np.diff(close) / (close[:-1] + 1e-8)
        if np.nanstd(ret) > 0.50:  # >50 % daily vol = data error
            n_vol += 1
            continue
        if len(close) > 5:
            fwd_ret = (close[5:] - close[:-5]) / (close[:-5] + 1e-8)
            if np.nanmax(np.abs(fwd_ret)) > 10.0:
                n_fwd += 1
                continue
        ok.append(code)
    n_total = n_neg + n_vol + n_nan + n_low + n_fwd
    if n_total:
        logger.warning(
            "Data quality: %d stocks filtered out "
            "(negative=%d, hi_vol=%d, all_nan=%d, low_close=%d, extreme_fwd=%d) -> %d kept",
            n_total, n_neg, n_vol, n_nan, n_low, n_fwd, len(ok),
        )
    return ok


def _cross_sectional_normalize(
    y_arr: np.ndarray,
    mask_arr: np.ndarray,
    min_stocks: int = 5,
) -> np.ndarray:
    """Z-score normalize returns across stocks within each date.

    Preserves cross-sectional ordering while giving each date's return
    distribution zero mean and unit variance.  Dates with too few valid
    stocks are left unchanged.

    Returns a new array (does not mutate input).
    """
    y_out = y_arr.copy()
    n_stocks, n_dates = y_arr.shape
    for t in range(n_dates):
        valid = mask_arr[:, t] if mask_arr is not None else np.ones(n_stocks, dtype=bool)
        if valid.sum() < min_stocks:
            continue
        vals = y_arr[valid, t]
        mean_t = float(np.nanmean(vals))
        std_t = max(float(np.nanstd(vals)), 1e-8)
        y_out[valid, t] = (y_arr[valid, t] - mean_t) / std_t
    return y_out


def _augment_sequence(
    pk: np.ndarray,
    po: np.ndarray,
    noise_std: float = 0.01,
    mask_prob: float = 0.05,
    feat_dropout: float = 0.02,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight time-series augmentation for financial data.

    Three independent augmentations:
    1. Gaussian noise ~ N(0, noise_std) — improves robustness
    2. Time masking — zero out random contiguous segments (simulates missing data)
    3. Feature dropout — zero out random feature dimensions

    All augmentations are conservative (small magnitudes) to avoid
    distorting the financial signal.
    """
    if rng is None:
        rng = np.random.RandomState()

    pk_aug = pk.copy()
    po_aug = po.copy()

    # 1. Gaussian noise (per-element, independent)
    if noise_std > 0:
        pk_aug += rng.randn(*pk.shape).astype(np.float32) * noise_std
        po_aug += rng.randn(*po.shape).astype(np.float32) * noise_std

    # 2. Time masking: zero out a random contiguous block of length 1-5
    if mask_prob > 0 and pk.shape[1] >= 3:
        T = pk.shape[1]
        mask_len = rng.randint(1, min(6, T // 2 + 1))
        if rng.random() < mask_prob:
            start = rng.randint(0, T - mask_len)
            pk_aug[:, start:start + mask_len, :] = 0.0
            po_aug[:, start:start + mask_len, :] = 0.0

    # 3. Feature dropout: zero out random feature dimensions
    if feat_dropout > 0:
        for arr in [pk_aug, po_aug]:
            if arr.shape[2] > 0:
                mask = rng.random(arr.shape[2]) < feat_dropout
                arr[:, :, mask] = 0.0

    return pk_aug, po_aug


def main():
    parser = argparse.ArgumentParser(description="Train VSN+xLSTM panel model")
    parser.add_argument("--stocks", type=int, default=500,
                        help="Limit to first N stocks (default: 500)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in days (1/5/20)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Model hidden dimension (default: 128)")
    parser.add_argument("--xlstm-blocks", type=int, default=2,
                        help="Number of xLSTM blocks (default: 2)")
    parser.add_argument("--rank-weight", type=float, default=0.1,
                        help="Ranking loss weight (0=disable, default: 0.1)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable time-series data augmentation")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--minute", action="store_true",
                        help="Use minute-frequency K-line data instead of daily")
    parser.add_argument("--minute-frequency", type=str, default="60",
                        choices=["5", "15", "30", "60"],
                        help="Bar frequency for minute mode (default: 60)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60 daily, 64 minute)")
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
    elif args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        stock_list = MinuteStorage(data_dir).list_stocks(args.minute_frequency)
        if args.stocks:
            stock_list = stock_list[:args.stocks]
    else:
        stock_list = _discover_stocks(data_dir, args.stocks)

    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)

    # Data quality filter (daily only — minute data validated at download time)
    if not args.minute:
        stock_list = _filter_quality(stock_list, data_dir)
        if len(stock_list) < 20:
            logger.error("Too few stocks pass quality filter (%d)", len(stock_list))
            sys.exit(1)

    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    if args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        ms = MinuteStorage(data_dir)
        frames = []
        for code in stock_list:
            df = ms.load(code, args.start, args.end, args.minute_frequency)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date
                df["stock_code"] = code
                frames.append(df)
        if not frames:
            logger.error("No minute data loaded for any stock — run download_minute.py first")
            sys.exit(1)
        logger.info("Minute mode: %d stocks @ %s-min, %d available in storage",
                    len(frames), args.minute_frequency,
                    len(ms.list_stocks(args.minute_frequency)))
    else:
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
    seq_len = args.seq_len or (64 if args.minute else 60)
    fp = FeaturePipeline(
        seq_len=seq_len,
        minute_mode=args.minute,
        use_sentiment=True, use_announcements=True,
        use_guba=True, use_comment=True, use_margin=True,
        use_northbound=True, use_dragon_tiger=True,
        use_fundamental=True, use_etf_flow=True,
        use_capital_flow=True, use_block_trade=True,
        use_shareholder=True, use_lockup=True, use_dividend=True,
        use_valuation=True,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(panel, aux_data=aux_data, horizon=args.horizon)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    dims = f"S={panel_data['static_features'].shape[1]} " \
           f"PK={panel_data['past_known'].shape[2]} " \
           f"PO={panel_data['past_observed'].shape[2]}"
    logger.info("Panel data: %d stocks × %d timesteps  dims: %s  horizon=%d",
                n_stocks, n_timesteps, dims, args.horizon)

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=panel_data["static_features"].shape[1],
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        hidden_dim=args.hidden_dim,
        xlstm_num_blocks=args.xlstm_blocks,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
        num_workers=0,
        horizon=args.horizon,
        rank_loss_weight=args.rank_weight,
    )
    logger.info("VSN+xLSTM config: hidden=%d blocks=%d heads=%d batch=%d lr=%.1e rank_w=%.2f",
                config.hidden_dim, config.xlstm_num_blocks, config.xlstm_num_heads,
                config.batch_size, config.learning_rate, config.rank_loss_weight)

    # Purged walk-forward splits
    if args.minute:
        train_len = 1500   # ~375 trading days at 4 bars/day
        val_len = 250      # ~62 trading days
        step = 125          # ~31 trading days
    else:
        train_len = 756    # ~3 years daily
        val_len = 126      # ~6 months daily
        step = 63          # ~3 months daily
    purge = config.seq_len
    all_sharpes = []
    fold_histories = []

    rng = np.random.RandomState(42)
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

        # Build a validity mask: position (i,t) is valid if y_direction != -100
        # (i.e. not tail-padded and not limit-up/down masked).
        train_mask = panel_data["y_direction"][:, train_slice] != -100
        val_mask = panel_data["y_direction"][:, val_slice] != -100

        train_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, train_slice],
            "past_observed": panel_data["past_observed"][:, train_slice],
            "y_direction": panel_data["y_direction"][:, train_slice],
            "y_return": panel_data["y_return"][:, train_slice].copy(),
            "y_volatility": panel_data["y_volatility"][:, train_slice].copy(),
            "date_indices": panel_data["date_indices"][:, train_slice].copy(),
        }
        val_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, val_slice],
            "past_observed": panel_data["past_observed"][:, val_slice],
            "y_direction": panel_data["y_direction"][:, val_slice],
            "y_return": panel_data["y_return"][:, val_slice].copy(),
            "y_volatility": panel_data["y_volatility"][:, val_slice].copy(),
            "date_indices": panel_data["date_indices"][:, val_slice].copy(),
        }

        # Cross-sectional z-score normalization per date.
        # Preserves relative ordering across stocks (unlike per-stock norm)
        # so ranking losses and IC evaluation work on consistent scales.
        train_data["y_return"] = _cross_sectional_normalize(
            train_data["y_return"], train_mask,
        )
        train_data["y_volatility"] = _cross_sectional_normalize(
            train_data["y_volatility"], train_mask,
        )
        # Save raw returns BEFORE normalization for portfolio evaluation.
        raw_val_y_return = val_data["y_return"].copy()
        val_data["y_return"] = _cross_sectional_normalize(
            val_data["y_return"], val_mask,
        )
        val_data["y_volatility"] = _cross_sectional_normalize(
            val_data["y_volatility"], val_mask,
        )
        # Clip normalized targets to [-5, 5].
        np.clip(val_data["y_return"], -5.0, 5.0, out=val_data["y_return"])
        np.clip(val_data["y_volatility"], -5.0, 5.0, out=val_data["y_volatility"])

        # Time-series data augmentation on training data.
        # Each stock's sequence gets independent noise/masking/dropout
        # — increases effective dataset size and improves robustness.
        if not args.no_augment:
            pk_aug, po_aug = _augment_sequence(
                train_data["past_known"],
                train_data["past_observed"],
                noise_std=0.005,
                mask_prob=0.03,
                feat_dropout=0.01,
                rng=rng,
            )
            train_data["past_known"] = pk_aug
            train_data["past_observed"] = po_aug

        logger.info("Fold %d/%d: train [%d:%d], val [%d:%d]",
                    fold, args.max_folds or "∞",
                    train_start, train_end, val_start, val_end)

        t0 = time.time()
        model, history = train_panel(
            config, train_data, val_data, device,
            raw_val_returns=raw_val_y_return,
        )
        elapsed = time.time() - t0

        if history["val_ls_sharpe"]:
            best_ls = max(history["val_ls_sharpe"])
            if history.get("val_metrics"):
                last = history["val_metrics"][-1]
            else:
                last = {}
            all_sharpes.append(best_ls)
            fold_histories.append(history)
            logger.info(
                "  Fold %d: best LS_Sharpe=%.2f IC=%.4f(IR=%.2f) "
                "Long_Sharpe=%.2f Q5-Q1=%.1fbp EW_Sharpe=%.2f (%.1fs)",
                fold, best_ls,
                last.get("ic_mean", 0), last.get("ic_ir", 0),
                last.get("long_sharpe", 0),
                last.get("q5mq1_ret", 0) * 10000,
                last.get("ew_sharpe", 0),
                elapsed,
            )
        else:
            logger.warning("  Fold %d: no valid metrics (%.1fs)", fold, elapsed)

        train_start += step

    if all_sharpes:
        logger.info("=== %d-Fold Summary ===", len(all_sharpes))
        logger.info("LS_Sharpe mean: %.2f ± %.2f", np.mean(all_sharpes), np.std(all_sharpes))
        all_ics = [h["val_ic"][-1] for h in fold_histories if h.get("val_ic")]
        if all_ics:
            logger.info("IC mean: %.4f ± %.4f", np.mean(all_ics), np.std(all_ics))
    else:
        logger.warning("No valid folds completed")


if __name__ == "__main__":
    main()
