"""Train VSN+xLSTM panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --stocks 500 --epochs 30 --max-folds 1
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --universe csi300 --stocks 300 --outdir reports/exp/csi300
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --stock-list 600519,000001,000858
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_panel.py --no-aux  # skip auxiliary data for quick test

Universe modes (--universe): first / random / stratified / all / csi300 / csi500 / csi800.
Artifacts (args.json, universe_resolved.txt, universe_used.txt, summary.json)
are saved to --outdir (default reports/experiments/<timestamp>).
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stoke_ml.config import load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.train import train_panel
from stoke_ml.models.panel.evaluate import evaluate_portfolio

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


def _exchange_group(stock_code: str) -> str:
    """Exchange bucket by code prefix: SH=6, SZ=0/3, BJ=4/8."""
    if stock_code.startswith("6"):
        return "SH"
    if stock_code.startswith(("0", "3")):
        return "SZ"
    return "BJ"


def _load_index_universe(data_dir: str, index_codes: set[str]) -> list[str]:
    """Stocks ever in the given indices (PIT in_date/out_date union).

    Any stock that was a member at any point in history is included — the
    training span runs 2000-2026, so a fixed-date snapshot would silently
    drop stocks that left the index mid-period.
    """
    path = os.path.join(
        data_dir, "a_shares", "index_constituents_hist", "membership.parquet",
    )
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    members = df[df["index_code"].isin(index_codes)]["stock_code"].unique()
    return sorted(members)


def _resolve_universe(
    all_stocks: list[str],
    universe: str,
    limit: int | None,
    seed: int,
    data_dir: str,
) -> tuple[list[str], str]:
    """Select the training universe by name.

    Returns (resolved_stocks, description).  The description records exactly
    what was selected (mode / seed / count) for the experiment artifacts.

    Modes:
      first      — first N by code (legacy behaviour, alphabetical bias)
      random     — seeded sample of N from all stocks (no code-order bias)
      stratified — seeded sample targeting N, balanced across SH/SZ/BJ
      all        — every stock (ignores --stocks)
      csi300/500/800 — index constituents (PIT union), --stocks caps count
    """
    if limit is None:
        limit = len(all_stocks)
    all_sorted = sorted(all_stocks)
    if universe == "all":
        return all_sorted, f"all ({len(all_sorted)} stocks)"
    if universe == "first":
        chosen = all_sorted[:limit]
        return chosen, f"first {len(chosen)} (sorted by code)"
    if universe == "random":
        rng = np.random.RandomState(seed)
        k = min(limit, len(all_stocks))
        chosen = rng.choice(all_sorted, size=k, replace=False).tolist()
        return chosen, f"random {len(chosen)} (seed={seed})"
    if universe == "stratified":
        rng = np.random.RandomState(seed)
        by_group: dict[str, list[str]] = {"SH": [], "SZ": [], "BJ": []}
        for code in all_sorted:
            by_group[_exchange_group(code)].append(code)
        chosen: list[str] = []
        remaining = limit
        for group, codes in by_group.items():
            share = min(remaining // 3, len(codes))
            chosen.extend(rng.choice(codes, size=share, replace=False).tolist())
            remaining -= share
        if remaining > 0:
            used = set(chosen)
            leftover = [c for codes in by_group.values() for c in codes if c not in used]
            chosen.extend(
                rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist()
            )
        rng.shuffle(chosen)
        return chosen, f"stratified {len(chosen)} (seed={seed}, SH/SZ/BJ balanced)"
    if universe in ("csi300", "csi500", "csi800"):
        idx_map = {"csi300": {"000300"}, "csi500": {"000905"}, "csi800": {"000300", "000905"}}
        members = _load_index_universe(data_dir, idx_map[universe])
        if not members:
            return [], f"{universe}: no membership data"
        # Keep only members we actually have daily K-line for — the membership
        # file includes codes with no data (delisted / not downloaded).
        have = set(all_stocks)
        members = [c for c in members if c in have]
        chosen = members[:limit] if limit else members
        return chosen, f"{universe} {len(chosen)} (PIT union, cap={limit})"
    raise ValueError(f"unknown --universe: {universe}")


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


def _slice_panel(panel_data: dict, tslice: slice, price_pad: int = 0) -> dict:
    """Slice every time-axis array of the panel by `tslice`.

    Static features are (N, T, D) PIT since review v4 §五 — sliced on the time
    axis like every other panel array.  Arrays that downstream code mutates in
    place (y_return z-score + clip, and their neighbours) are copied so one
    fold's normalization never corrupts the shared panel for later folds.

    `price_pad`: extend the close/open price columns by this many beyond
    `tslice.stop` (capped at the panel end).  The sleeve-account evaluation
    (review v4 §四) needs open[t+h] to liquidate a position entered at open[t],
    so the last `price_pad` sleeves get a real exit instead of a forced carry.

    `y_return_raw` is a copy of the RAW open-to-open return saved BEFORE the
    caller z-scores/clips `y_return` (review v5 §五): clean IC and quintile
    spreads must be computed on raw returns, not on the normalized model target.
    """
    stop = tslice.stop
    out = {
        "static_features": panel_data["static_features"][:, tslice, :],
        "past_known": panel_data["past_known"][:, tslice],
        "past_observed": panel_data["past_observed"][:, tslice],
        "y_direction": panel_data["y_direction"][:, tslice],
        "y_return_raw": panel_data["y_return"][:, tslice].copy(),
        "y_return": panel_data["y_return"][:, tslice].copy(),
        "y_volatility": panel_data["y_volatility"][:, tslice].copy(),
        "observation_mask": panel_data["observation_mask"][:, tslice],
        "entry_eligible_mask": panel_data["entry_eligible_mask"][:, tslice],
        "return_target_mask": panel_data["return_target_mask"][:, tslice],
        "vol_target_mask": panel_data["vol_target_mask"][:, tslice],
        "realized_return": panel_data["realized_return"][:, tslice].copy(),
        "date_indices": panel_data["date_indices"][:, tslice].copy(),
        "decision_eligible_mask": panel_data["decision_eligible_mask"][:, tslice],
        "history_eligible_mask": panel_data["history_eligible_mask"][:, tslice],
    }
    # Price paths feed the sleeve-account evaluation; a stale prebuilt panel
    # without them just falls back to the legacy path in evaluate_portfolio.
    if "close_price" in panel_data and "open_price" in panel_data:
        max_T = panel_data["close_price"].shape[1]
        pstop = min(stop + price_pad, max_T) if price_pad > 0 else stop
        out["close_price"] = panel_data["close_price"][:, tslice.start:pstop]
        out["open_price"] = panel_data["open_price"][:, tslice.start:pstop]
    return out


def _fmt_date(global_dates, idx):
    """Global-calendar position → 'YYYY-MM-DD'.  Out of range → None."""
    if global_dates is None or idx < 0 or idx >= len(global_dates):
        return None
    return str(np.datetime_as_string(global_dates[idx], unit="D"))


def _augment_sequence(
    pk: np.ndarray,
    po: np.ndarray,
    obs_mask: np.ndarray | None = None,
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
    distorting the financial signal.  Gaussian noise is gated by `obs_mask`
    (True = real observation) so zero-padded history of new listings stays
    exactly zero instead of gaining fake noise that the model would read as
    real data.
    """
    if rng is None:
        rng = np.random.RandomState()

    pk_aug = pk.copy()
    po_aug = po.copy()

    # 1. Gaussian noise (per-element, independent), only on real-observation days
    if noise_std > 0:
        noise_pk = rng.randn(*pk.shape).astype(np.float32) * noise_std
        noise_po = rng.randn(*po.shape).astype(np.float32) * noise_std
        if obs_mask is not None:
            obs_b = obs_mask[..., None].astype(np.float32)
            noise_pk *= obs_b
            noise_po *= obs_b
        pk_aug += noise_pk
        po_aug += noise_po

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


def _save_artifacts(
    outdir: str,
    args: argparse.Namespace,
    resolved: list[str],
    used: list[str],
    universe_desc: str,
    summary: dict | None,
) -> str:
    """Persist the experiment: args, resolved/used universes, fold summary."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "universe_resolved.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n# n={len(resolved)}\n")
        f.write("\n".join(resolved))
        f.write("\n")
    with open(os.path.join(outdir, "universe_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc} (after quality filter)\n# n={len(used)}\n")
        f.write("\n".join(used))
        f.write("\n")
    if summary is not None:
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Experiment artifacts saved to %s", outdir)
    return outdir


def _predict_outer(model, outer_data, config, device) -> np.ndarray | None:
    """Run the deployed checkpoint over the outer-test panel (review v4 §十三.2).

    Uses the same no-sampler DataLoader as evaluate_portfolio, so every
    (stock, window) pair is emitted in index order and the flattened return
    predictions reshape back to (n_stocks, n_windows).  Window d enters at
    panel column seq_len + d — i.e. global column val_start + d of the full
    panel.  Returns None only when the outer panel has no windows.
    """
    val_ds = PanelDataset(outer_data, seq_len=config.seq_len,
                          min_history=config.min_history)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )
    model.eval()
    preds_parts = []
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_ = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            preds_parts.append(pred_ret.cpu().squeeze(-1))
    if not preds_parts:
        return None
    preds = torch.cat(preds_parts)
    n_stocks = outer_data["static_features"].shape[0]
    return preds.reshape(n_stocks, val_ds.n_windows).numpy()


def _best_eval_metrics(history: dict) -> tuple[dict, int]:
    """Metrics of the inner-val eval nearest the deployed checkpoint.

    Returns (metrics_dict, eval_epoch) for the evaluation whose 1-based epoch
    sits closest to best_epoch_idx+1 — NOT the post-hoc max, which would
    double-count hindsight.  Histories without val_eval_epochs (legacy) are
    assumed to have evaluated on the 5,10,15,... grid.  Empty histories yield
    ({}, 0).
    """
    metrics = history.get("val_metrics") or []
    if not metrics:
        return {}, 0
    best = history.get("best_epoch_idx", 0) + 1  # 1-based deployed epoch
    eval_epochs = history.get("val_eval_epochs")
    if not eval_epochs:
        eval_epochs = [5 + 5 * i for i in range(len(metrics))]
    nearest = min(range(len(metrics)), key=lambda i: abs(eval_epochs[i] - best))
    return metrics[nearest], eval_epochs[nearest]


def main():
    parser = argparse.ArgumentParser(description="Train VSN+xLSTM panel model")
    parser.add_argument("--stocks", type=int, default=500,
                        help="Universe size / cap (default: 500; with "
                             "--universe first: first N sorted; random/stratified: "
                             "N sampled; csi*: N cap)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"],
                        help="Stock universe selection (default: random; "
                             "csi* = index constituents, PIT ever-held union)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for universe sampling and data "
                             "augmentation (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Experiment artifact dir (default: "
                             "reports/experiments/<timestamp>)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--lockbox-months", type=int, default=12,
                        help="Reserve the last N months as an untouched lockbox "
                             "(review v4 §十三.3) — no fold trains on or "
                             "evaluates it; kept for a single final run once "
                             "the design freezes (default: 12)")
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
    parser.add_argument("--log-gradient-flow", action="store_true",
                        help="Log per-parameter-group gradient norms each epoch "
                             "(after optimizer.step, before zero_grad)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--prebuilt", type=str, default=None,
                        help="Load panel-mode prebuilt features from this dir "
                             "(built via build_features.py --panel-mode). "
                             "Skips aux data loading and live feature "
                             "engineering — the panel is built from the "
                             "prebuilt parquets")
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
        universe_desc = f"stock-list (explicit, n={len(stock_list)})"
    elif args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        stock_list = MinuteStorage(data_dir).list_stocks(args.minute_frequency)
        if args.stocks:
            stock_list = stock_list[:args.stocks]
        universe_desc = f"minute-mode (n={len(stock_list)})"
    else:
        all_stocks = _discover_stocks(data_dir, None)
        stock_list, universe_desc = _resolve_universe(
            all_stocks, args.universe, args.stocks, args.seed, data_dir,
        )

    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)

    universe_resolved = list(stock_list)

    # Data quality filter (daily only — minute data validated at download time)
    if not args.minute:
        stock_list = _filter_quality(stock_list, data_dir)
        if len(stock_list) < 20:
            logger.error("Too few stocks pass quality filter (%d)", len(stock_list))
            sys.exit(1)
    universe_used = list(stock_list)

    logger.info("Universe: %s", universe_desc)
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
    # Panel stock order is sorted-unique (build_panel_features sorts codes);
    # derive the code list from the panel itself so OOS artifacts stay aligned
    # with the panel arrays even if a stock's data was dropped as empty.
    panel_stocks = sorted(panel["stock_code"].unique())

    # Load auxiliary data (unless --no-aux or --prebuilt)
    aux_data = None
    if not args.no_aux and not args.prebuilt:
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
    panel_data = fp.build_panel_features(
        panel, aux_data=aux_data, horizon=args.horizon, prebuilt_dir=args.prebuilt,
    )

    # Union trading calendar (datetime64[ns]) — fold boundaries in index space
    # map back to real dates for the summary (review v3 §十五).
    global_dates = panel_data.get("global_dates")

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    # Static features are (N, T, D) PIT (review v4 §五) — feature dim is axis 2.
    static_dim = panel_data["static_features"].shape[2]
    dims = f"S={static_dim} " \
           f"PK={panel_data['past_known'].shape[2]} " \
           f"PO={panel_data['past_observed'].shape[2]}"
    logger.info("Panel data: %d stocks × %d timesteps  dims: %s  horizon=%d",
                n_stocks, n_timesteps, dims, args.horizon)

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=static_dim,
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
        seed=args.seed,
        log_gradient_flow=args.log_gradient_flow,
    )
    logger.info("VSN+xLSTM config: hidden=%d blocks=%d heads=%d batch=%d lr=%.1e rank_w=%.2f",
                config.hidden_dim, config.xlstm_num_blocks, config.xlstm_num_heads,
                config.batch_size, config.learning_rate, config.rank_loss_weight)

    # Purged walk-forward splits
    if args.minute:
        val_len = 250      # ~62 trading days
    else:
        val_len = 126      # ~6 months daily
    # Review v4 §十三: OOS folds are NON-OVERLAPPING — step == val_len, so
    # adjacent folds evaluate disjoint test windows.  The old step < val_len
    # made every fold share test days with its neighbours, inflating fold
    # count and letting mean±std masquerade as independent dispersion.
    step = val_len
    purge = config.seq_len
    all_sharpes = []
    fold_histories = []

    # Review v4 §十三.3: reserve the last N months as an untouched lockbox.
    # No fold trains on or evaluates it; it is kept for a single final run
    # once the design freezes.  Daily ≈ 21 bars/month; minute mode scales by
    # bars per day so lockbox_months spans the same wall-clock time.
    bars_per_day = {"5": 48, "15": 16, "30": 8, "60": 4}[args.minute_frequency]
    lockbox_len = int(args.lockbox_months * 21 * (bars_per_day if args.minute else 1))
    lockbox_start = max(0, n_timesteps - lockbox_len)
    if lockbox_start <= 0:
        logger.error("Lockbox (%d steps) leaves no trainable panel "
                     "(n_timesteps=%d) — reduce --lockbox-months",
                     lockbox_len, n_timesteps)
        sys.exit(1)
    logger.info("Lockbox [%d:%d] %d steps (%.1f months) — %s .. %s",
                lockbox_start, n_timesteps, lockbox_len, args.lockbox_months,
                _fmt_date(global_dates, lockbox_start),
                _fmt_date(global_dates, n_timesteps - 1))

    outdir = args.outdir or os.path.join(
        "reports", "experiments", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    oos_dir = os.path.join(outdir, "oos_preds")
    os.makedirs(oos_dir, exist_ok=True)
    oos_dates_all: list[str] = []
    oos_stocks_all: list[str] = []
    oos_preds_all: list[np.ndarray] = []

    rng = np.random.RandomState(args.seed)
    fold = 0
    # Walk BACKWARD from the lockbox boundary so the (max_folds) validation
    # windows cover the newest period instead of the earliest.  The training
    # window GROWS from position 0 out to (val_start - purge) each fold, so
    # the 2000-2015 history is genuinely in the training set — the old
    # fixed-width 756-day scheme left [0, n_timesteps-train_len-purge-val_len)
    # permanently unused and put short-history stocks' data entirely before
    # every fold window.
    last_val_start = n_timesteps - val_len - lockbox_len
    val_start = last_val_start
    while val_start >= 0:
        if args.max_folds and fold >= args.max_folds:
            break
        train_end = val_start - purge
        if train_end < config.seq_len + 1:
            # PanelDataset needs at least seq_len+1 rows for one window
            # (n_windows = n_timesteps - seq_len must be >= 1).
            break
        fold += 1
        train_start = 0
        val_end = min(val_start + val_len, n_timesteps)

        # Carve the last ~15% of the trainable span as inner_val — used ONLY
        # for checkpoint selection inside train_panel.  The outer test (the old
        # val window) is fully held out of training and evaluated exactly once
        # on the deployed checkpoint.
        n_train_targets = train_end - config.seq_len
        inner_val_len = max(1, int(round(0.15 * n_train_targets)))
        inner_val_context_start = train_end - inner_val_len - config.seq_len
        if inner_val_context_start < config.seq_len + 1:
            # not enough rows left for one inner_train + one inner_val window
            break
        inner_train_end = inner_val_context_start
        val_context_start = val_start - config.seq_len

        inner_train_slice = slice(0, inner_train_end)
        inner_val_slice = slice(inner_val_context_start, train_end)
        outer_test_slice = slice(val_context_start, val_end)

        inner_train_data = _slice_panel(panel_data, inner_train_slice, price_pad=config.horizon)
        inner_val_data = _slice_panel(panel_data, inner_val_slice, price_pad=config.horizon)
        outer_test_data = _slice_panel(panel_data, outer_test_slice, price_pad=config.horizon)

        # y_return: cross-sectional z-score per date — preserves relative
        # ordering across stocks so ranking loss and IC evaluation work on a
        # consistent scale.  Normalize using the RETURN-target mask (clean
        # open-to-open returns) so dirty/missing positions don't skew the
        # z-score.  y_volatility: kept as the raw positive future-vol target
        # (std of the next-horizon daily returns).  VolatilityHead outputs
        # softplus > 0, so z-scoring the target would reintroduce the
        # negative-target-vs-positive-output contradiction — it must stay in
        # original units.
        inner_train_data["y_return"] = _cross_sectional_normalize(
            inner_train_data["y_return"], inner_train_data["return_target_mask"],
        )
        inner_val_data["y_return"] = _cross_sectional_normalize(
            inner_val_data["y_return"], inner_val_data["return_target_mask"],
        )
        outer_test_data["y_return"] = _cross_sectional_normalize(
            outer_test_data["y_return"], outer_test_data["return_target_mask"],
        )
        # Clip normalized targets to [-5, 5] — same band for train and val so
        # the model is never asked to fit z-scores beyond the eval range.
        # (Only y_return is z-scored; y_volatility stays in original units,
        # well below 5, so the clip applies to y_return only.)
        for dd in (inner_train_data, inner_val_data, outer_test_data):
            np.clip(dd["y_return"], -5.0, 5.0, out=dd["y_return"])

        # Time-series data augmentation on the inner-training data.
        # Each stock's sequence gets independent noise/masking/dropout —
        # Gaussian noise gated by observation_mask so zero-padded history
        # (new listings) stays exactly zero instead of gaining fake noise.
        if not args.no_augment:
            pk_aug, po_aug = _augment_sequence(
                inner_train_data["past_known"],
                inner_train_data["past_observed"],
                obs_mask=inner_train_data["observation_mask"],
                noise_std=0.005,
                mask_prob=0.03,
                feat_dropout=0.01,
                rng=rng,
            )
            inner_train_data["past_known"] = pk_aug
            inner_train_data["past_observed"] = po_aug

        logger.info("Fold %d/%d: inner_train [%d:%d] inner_val [%d:%d] "
                    "outer_test [%d:%d]",
                    fold, args.max_folds or "∞",
                    0, inner_train_end,
                    inner_val_context_start, train_end,
                    val_context_start, val_end)

        t0 = time.time()
        # Checkpoint selection runs on inner_val inside train_panel; the
        # returned model is the best-inner-val checkpoint.
        model, history = train_panel(
            config, inner_train_data, inner_val_data, device,
            raw_val_returns=inner_val_data["realized_return"],
        )
        elapsed = time.time() - t0

        # Evaluate the exact deployed checkpoint ONCE on the held-out outer
        # test — the honest out-of-sample number, never used for selection.
        outer_m = evaluate_portfolio(
            model, outer_test_data, config, device,
            horizon=config.horizon,
            raw_returns=outer_test_data["realized_return"],
            # review v6 §十五.2: formal training must use the chronological
            # sleeve account — a prebuilt panel without price paths is a data
            # bug, not a reason to silently downgrade to the legacy estimator.
            require_price_path=True,
        )
        best_epoch = history.get("best_epoch_idx", 0) + 1

        # Daily OOS predictions (review v4 §十三.2): one return forecast per
        # (stock, entry day).  A window's entry is global column val_start+d,
        # so entry dates run global_dates[val_start .. val_start+val_len-1].
        oos_preds = _predict_outer(model, outer_test_data, config, device)
        if oos_preds is not None:
            n_w = oos_preds.shape[1]
            entry_dates = [_fmt_date(global_dates, val_start + d) for d in range(n_w)]
            # Entry-day eligibility on the same window-day grid so downstream
            # sleeve-account construction can filter exactly what was tradable.
            elig = outer_test_data["entry_eligible_mask"][
                :, config.seq_len:config.seq_len + n_w]
            np.savez(
                os.path.join(oos_dir, f"fold_{fold:03d}.npz"),
                preds=oos_preds,
                dates=np.array(entry_dates),
                stocks=np.array(panel_stocks),
                entry_eligible=elig,
            )
            for d in range(n_w):
                oos_dates_all.extend([entry_dates[d]] * len(panel_stocks))
                oos_stocks_all.extend(panel_stocks)
                oos_preds_all.append(oos_preds[:, d])

        if outer_m["n_periods"] >= 2:
            best_ls = outer_m["ls_sharpe"]
            all_sharpes.append(best_ls)
            # Inner-val eval nearest the deployed checkpoint — what selection
            # actually saw, reported honestly alongside the held-out outer
            # metrics (review v4 §十一: never report a post-hoc max).
            inner_eval_m, inner_eval_epoch = _best_eval_metrics(history)
            # Input-context date bounds of each segment — column t of the panel
            # is global_dates[t], so a slice [a,b) covers dates [a, b-1].
            # Semantic dates (review v4 §十三.2): entry day e buys at open[e],
            # the signal is produced after close[e-1], and the input context is
            # the seq_len days [e-seq_len, e).
            fold_histories.append({
                "history": history,
                "outer_metrics": outer_m,
                "best_epoch": best_epoch,
                "inner_eval_epoch": inner_eval_epoch,
                "inner_eval_ls_sharpe": inner_eval_m.get("ls_sharpe"),
                "inner_eval_ic": inner_eval_m.get("ic_mean"),
                "train_start": _fmt_date(global_dates, 0),
                "train_end": _fmt_date(global_dates, inner_train_end - 1),
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "context_start": _fmt_date(global_dates, val_context_start),
                "signal_start": _fmt_date(global_dates, val_start - 1),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "exit_end": _fmt_date(
                    global_dates,
                    min(val_start + val_len - 1 + config.horizon, n_timesteps - 1)),
                "test_start": _fmt_date(global_dates, val_context_start),
                "test_end": _fmt_date(global_dates, val_end - 1),
            })
            logger.info(
                "  Fold %d: best@epoch%d OUTER-TEST LS_Sharpe=%.2f IC=%.4f(IR=%.2f) "
                "Long_Sharpe=%.2f Q5-Q1=%.1fbp EW_Sharpe=%.2f (%.1fs)",
                fold, best_epoch, best_ls,
                outer_m.get("ic_mean", 0), outer_m.get("ic_ir", 0),
                outer_m.get("long_sharpe", 0),
                outer_m.get("q5mq1_ret", 0) * 10000,
                outer_m.get("ew_sharpe", 0),
                elapsed,
            )
        else:
            logger.warning(
                "  Fold %d: outer-test too short for metrics (%.1fs)", fold, elapsed,
            )

        val_start -= step

    # Combined daily OOS series (review v4 §十三.2): one row per (stock, entry
    # day) across all non-overlapping folds — the input to the sleeve-account
    # backtest, kept separate from fold-level aggregates.
    if oos_preds_all:
        oos_series = pd.DataFrame({
            "entry_date": oos_dates_all,
            "stock_code": oos_stocks_all,
            "pred": np.concatenate(oos_preds_all),
        })
        oos_series_path = os.path.join(outdir, "oos_series.parquet")
        oos_series.to_parquet(oos_series_path)
        logger.info("OOS series: %d rows -> %s", len(oos_series), oos_series_path)

    summary_data = None
    if all_sharpes:
        logger.info("=== %d-Fold Summary ===", len(all_sharpes))
        logger.info("LS_Sharpe mean: %.2f ± %.2f", np.mean(all_sharpes), np.std(all_sharpes))
        # IC comes from the outer-test evaluation of the exact deployed
        # checkpoint (outer_metrics) — never an in-loop proxy.
        all_ics = [
            f["outer_metrics"].get("ic_mean", float("nan"))
            for f in fold_histories if f.get("outer_metrics")
        ]
        all_ics = [x for x in all_ics if not np.isnan(x)]
        if all_ics:
            logger.info("IC mean: %.4f ± %.4f", np.mean(all_ics), np.std(all_ics))
        summary_data = {
            "n_folds": len(all_sharpes),
            "ls_sharpe_mean": float(np.mean(all_sharpes)),
            "ls_sharpe_std": float(np.std(all_sharpes)),
            "ic_mean": float(np.mean(all_ics)) if all_ics else None,
            "ic_std": float(np.std(all_ics)) if all_ics else None,
            "universe": universe_desc,
            # Review v4 §十三: non-overlapping folds (step == val_len) — each
            # fold's test days are disjoint, so mean±std is the dispersion of
            # genuinely independent OOS windows.
            "folds_overlap": False,
            "fold_note": None,
            "lockbox": {
                "months": args.lockbox_months,
                "start": _fmt_date(global_dates, lockbox_start),
                "end": _fmt_date(global_dates, n_timesteps - 1),
                "n_steps": lockbox_len,
                "note": "Reserved for a single final run once the design "
                        "freezes — no fold trains on or evaluates it.",
            },
            "oos_series": "oos_series.parquet",
            "folds": [],
        }
        for i, f in enumerate(fold_histories):
            m = f["outer_metrics"]
            summary_data["folds"].append({
                "fold": i + 1,
                "best_epoch": f["best_epoch"],
                "eval_epoch": f["best_epoch"],
                "inner_eval_epoch": f.get("inner_eval_epoch"),
                "inner_eval_ls_sharpe": f.get("inner_eval_ls_sharpe"),
                "inner_eval_ic": f.get("inner_eval_ic"),
                "train_start": f.get("train_start"),
                "train_end": f.get("train_end"),
                "inner_val_start": f.get("inner_val_start"),
                "inner_val_end": f.get("inner_val_end"),
                "context_start": f.get("context_start"),
                "signal_start": f.get("signal_start"),
                "entry_start": f.get("entry_start"),
                "entry_end": f.get("entry_end"),
                "exit_end": f.get("exit_end"),
                "test_start": f.get("test_start"),
                "test_end": f.get("test_end"),
                "ls_sharpe": m.get("ls_sharpe"),
                "ic_mean": m.get("ic_mean"),
                "ic_ir": m.get("ic_ir"),
                "long_sharpe": m.get("long_sharpe"),
                "q5mq1_ret": m.get("q5mq1_ret"),
                "ew_sharpe": m.get("ew_sharpe"),
            })
    else:
        logger.warning("No valid folds completed")

    _save_artifacts(
        outdir, args, universe_resolved, universe_used, universe_desc, summary_data,
    )


if __name__ == "__main__":
    main()
