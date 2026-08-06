"""Portfolio-evaluation entry points for the panel evaluator (§二十一).

Extracted from ``stoke_ml.models.panel.evaluate`` — ``evaluate_portfolio`` /
``evaluate_sharpe`` plus the quintile analysis, empty-result scaffold and raw-
actual reconstruction they share.  ``evaluate`` re-exports these for backward
compatibility.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.evaluate_metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_daily_return_profit_factor,
    compute_equity_curve,
    compute_bootstrap_sharpe_ci,
)
from stoke_ml.models.panel.evaluate_ic import (
    _compute_daily_ic,
    _raw_clean_rank_ic,
    compute_ic_summary,
    _candidate_pool,
)
from stoke_ml.models.panel.evaluate_account import (
    _build_portfolio_returns,
    _sleeve_account_metrics,
)

logger = logging.getLogger(__name__)


def evaluate_sharpe(
    model: nn.Module,
    val_data: dict,
    config: PanelConfig,
    device: torch.device,
    top_fraction: float = 0.1,
    horizon: int = 1,
    return_metrics: bool = False,
    raw_returns: np.ndarray | None = None,
) -> float | tuple[float, dict]:
    """Time-varying top-K long-only portfolio evaluation (backward-compatible).

    Prefer evaluate_portfolio() for the full multi-angle report.
    """
    result = evaluate_portfolio(
        model, val_data, config, device,
        top_fraction=top_fraction, horizon=horizon,
        raw_returns=raw_returns,
    )
    if not return_metrics:
        return result["long_sharpe"]
    return result["long_sharpe"], {
        "sharpe": result["long_sharpe"],
        "sortino": result["long_sortino"],
        "calmar": result["long_calmar"],
        "max_drawdown": result["long_maxdd"],
        "daily_return_profit_factor": result["long_daily_return_pf"],
        "ic": result["ic_mean"],
        "n_periods": result["n_periods"],
    }


def evaluate_portfolio(
    model: nn.Module,
    val_data: dict,
    config: PanelConfig,
    device: torch.device,
    top_fraction: float = 0.1,
    horizon: int = 5,
    raw_returns: np.ndarray | None = None,
    n_boot: int = 2000,
    require_price_path: bool = False,
    return_ledger: bool = False,
    delist_day: np.ndarray | None = None,
    n_trials: int | None = None,
) -> dict:
    """Multi-angle portfolio evaluation.

    Candidate pool = DECISION & HISTORY eligible stocks (close[t-1] real, input
    window covered).  Fills then gate on real open[t]; a
    selected stock with no real open stays unfilled (cash), never backfilled.

    Exec P&L (when close_price/open_price are present) is a chronological
    sleeve account: every calendar day enters a new top-K
    sleeve held `horizon` days, each with weight 1/h of NAV, marked to close
    daily → a TRUE daily return series annualized by sqrt(252).  exit_status
    classifies each position clean/carry/delist/unfilled and costs are applied
    per side.  Clean IC is computed separately over y_return ×
    return_target_mask.

    Without price paths (synthetic tests) it falls back to the legacy
    phase-concatenated top-K over realized returns.  Formal training MUST pass
    require_price_path=True: the two estimators measure
    different things, and two experiments that silently used different ones
    would not be comparable — missing price paths should then fail loudly
    instead of quietly downgrading to the legacy estimator.

    Returns a flat dict with these keys:
      — IC (clean): ic_mean, ic_std, ic_ir, ic_pos_rate, ic_newey_west_t
      — Long-only top-K: long_sharpe, long_gross_sharpe, long_sharpe_lo/hi,
          long_sortino, long_calmar, long_maxdd, long_daily_return_pf
      — Market-neutral long-short: ls_sharpe, ls_gross_sharpe, ls_sharpe_lo/hi,
          ls_sortino, ls_calmar, ls_maxdd
      — Costs/turnover: long_turnover, ls_turnover, ew_turnover
      — Exits: exit_status {counts, pnl_share}
      — Quintile: q1_ret … q5_ret, q5mq1_ret, q_monotonic
      — Eligible candidate-pool equal-weight: eligible_ew_sharpe
      — selected-universe equal-weight proxy: selected_universe_ew_sharpe
      — Metadata: n_periods, n_stocks, n_days
    """
    model.eval()
    n_stocks = val_data["static_features"].shape[0]
    val_ds = PanelDataset(
        val_data, seq_len=config.seq_len, min_history=config.min_history,
        max_stocks_per_date=None, training=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    n_windows = val_ds.n_windows
    seq_len = val_ds.seq_len
    preds = torch.full((n_stocks, n_windows), float("nan"))
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_y, date_idx_t, _dm, _rm, _vm, stock_indices = batch
            if stock_indices.numel() == 0:
                continue
            # Per-stock window indices (supports mixed-date batches).
            window_idx = date_idx_t - seq_len
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            preds[stock_indices, window_idx] = pred_ret.cpu().squeeze(-1)

    if torch.isnan(preds).all():
        return _empty_result()
    preds_np = preds.numpy()

    # Selection pool: decision & history, with entry/future-label
    # fallbacks for synthetic/legacy data.
    pool = _candidate_pool(val_data, n_windows, config.seq_len)
    pool_np = pool.numpy()

    # ── Clean IC: rank correlation of preds vs the CLEAN
    # open->open return, over decision & history & return_target.  This is the
    # pure-signal diagnostic.  The exec P&L below (sleeve account) uses fills +
    # costs + delayed exits instead — the two are deliberately separated so a
    # suspension bias in realized returns can't masquerade as signal.
    #
    # The actuals must be the RAW open->open return: train_panel
    # z-scores + clips y_return per fold for the model, and Spearman on the
    # clipped tails produces spurious ties while quintile "bp" would be in
    # cross-sectional-std units.  y_return_raw (saved before the z-score) is
    # preferred; fall back to y_return for data that has no raw copy.  This is
    # the SAME helper train._compute_val_loss uses for checkpoint selection,
    # so the selection metric and the report metric are
    # one quantity and share the min_stocks_per_day threshold.
    daily_ics, clean_actuals = _raw_clean_rank_ic(
        val_data, preds_np, n_windows, config.seq_len,
        min_stocks=config.min_stocks_per_day,
    )
    ic_pool = pool
    if "return_target_mask" in val_data:
        rt = torch.as_tensor(val_data["return_target_mask"])
        ic_pool = ic_pool & rt[:, config.seq_len:config.seq_len + n_windows]
    if daily_ics is None:
        # Legacy fallback: no y_return_raw / y_return in val_data — reconstruct
        # the raw open->open actuals from realized_return / raw_returns.
        clean_actuals = None
        if "realized_return" in val_data:
            act = _build_raw_actuals(torch.as_tensor(val_data["realized_return"]),
                                     n_stocks, n_windows, config.seq_len)
        elif raw_returns is not None:
            act = _build_raw_actuals(raw_returns, n_stocks, n_windows, config.seq_len)
        else:
            logger.warning(
                "evaluate_portfolio called without y_return/realized_return/"
                "raw_returns - returning empty result."
            )
            return _empty_result()
        daily_ics = _compute_daily_ic(
            preds_np, act.numpy(), ic_pool.numpy(),
            min_stocks=config.min_stocks_per_day,
        )
    ic_summary = compute_ic_summary(daily_ics, horizon=horizon)

    # ── Exec P&L ──
    has_prices = "close_price" in val_data and "open_price" in val_data
    if has_prices:
        # Price columns are global-calendar indexed like every mask; a window
        # prediction is for ENTRY at panel column seq_len+d, so slice prices to
        # the same window-day grid the pool / preds use.  Take
        # horizon EXTRA columns so the sleeve entered on the last signal day
        # W-1 can still liquidate at open[W-1+horizon] — the
        # pad comes from _slice_panel(price_pad=horizon).  NumPy clips the stop
        # to the array width, so data without the pad simply runs unresolved.
        p0 = config.seq_len
        close_np = np.asarray(val_data["close_price"], dtype=np.float32)[
            :, p0:p0 + n_windows + horizon]
        open_np = np.asarray(val_data["open_price"], dtype=np.float32)[
            :, p0:p0 + n_windows + horizon]
        pm = _sleeve_account_metrics(
            preds_np, close_np, open_np, pool_np,
            horizon, top_fraction, config.txn_cost, n_boot,
            return_ledger=return_ledger, delist_day=delist_day,
            n_trials=n_trials,
        )
        n_periods = pm["n_periods"]
        if n_periods < 2:
            return _empty_result()
        quintile_actuals = clean_actuals if clean_actuals is not None else act
        quintile_metrics = _quintile_analysis(
            preds, quintile_actuals, n_windows, horizon, n_stocks, mask=ic_pool,
        )
        return {
            "n_periods": n_periods,
            "n_stocks": n_stocks,
            "n_days": n_windows,
            "ic_mean": ic_summary["ic_mean"],
            "ic_std": ic_summary["ic_std"],
            "ic_ir": ic_summary["ic_ir"],
            "ic_pos_rate": ic_summary["ic_pos_rate"],
            "ic_newey_west_t": ic_summary["ic_newey_west_t"],
            "long_sharpe": pm["long_sharpe"],
            "long_gross_sharpe": pm["long_gross_sharpe"],
            "long_sharpe_lo": pm["long_sharpe_lo"],
            "long_sharpe_hi": pm["long_sharpe_hi"],
            "long_sortino": pm["long_sortino"],
            "long_calmar": pm["long_calmar"],
            "long_maxdd": pm["long_maxdd"],
            "long_daily_return_pf": pm["long_daily_return_pf"],
            "ls_sharpe": pm["ls_sharpe"],
            "ls_gross_sharpe": pm["ls_gross_sharpe"],
            "ls_sharpe_lo": pm["ls_sharpe_lo"],
            "ls_sharpe_hi": pm["ls_sharpe_hi"],
            "ls_sortino": pm["ls_sortino"],
            "ls_calmar": pm["ls_calmar"],
            "ls_maxdd": pm["ls_maxdd"],
            "ls2x_sharpe": pm["ls2x_sharpe"],
            "ls2x_gross_sharpe": pm["ls2x_gross_sharpe"],
            "ls2x_maxdd": pm["ls2x_maxdd"],
            "exposure": pm["exposure"],
            "eligible_ew_sharpe": pm["eligible_ew_sharpe"],
            "selected_universe_ew_sharpe": pm["selected_universe_ew_sharpe"],
            "long_psr": pm["long_psr"],
            "long_dsr": pm["long_dsr"],
            "ls_psr": pm["ls_psr"],
            "ls_dsr": pm["ls_dsr"],
            "dsr_n_trials": pm["dsr_n_trials"],
            "bbmm_stat": pm["bbmm_stat"],
            "bbmm_p_value": pm["bbmm_p_value"],
            "bbmm_n_strategies": pm["bbmm_n_strategies"],
            "long_turnover": pm["long_turnover"],
            "ls_turnover": pm["ls_turnover"],
            "ew_turnover": pm["ew_turnover"],
            "exit_status": pm["exit_status"],
            "long_exit_status": pm["long_exit_status"],
            "short_exit_status": pm["short_exit_status"],
            "eligible_ew_exit_status": pm["eligible_ew_exit_status"],
            # §十四-1: forward every leg's ledger, not just the long one.
            "long_ledger": pm.get("long_ledger"),
            "short_ledger": pm.get("short_ledger"),
            "ls_ledger": pm.get("ls_ledger"),
            "ew_ledger": pm.get("ew_ledger"),
            "selected_universe_ew_ledger": pm.get("selected_universe_ew_ledger"),
            **quintile_metrics,
        }

    # ── Legacy path (no price paths — synthetic tests): phase-concatenated
    # long/short over realized/raw actuals, annualized by sqrt(252/h) as before.
    if require_price_path:
        raise ValueError(
            "evaluate_portfolio(require_price_path=True) called without "
            "close_price/open_price price paths — formal training must use the "
            "chronological sleeve account, not the legacy phase-concatenation "
            "fallback")
    if "realized_return" in val_data:
        actuals = _build_raw_actuals(
            torch.as_tensor(val_data["realized_return"]),
            n_stocks, n_windows, config.seq_len,
        )
    elif raw_returns is not None:
        actuals = _build_raw_actuals(raw_returns, n_stocks, n_windows, config.seq_len)
    else:
        logger.warning(
            "evaluate_portfolio called without realized_return/raw_returns - "
            "returning empty result."
        )
        return _empty_result()

    long_rets, short_rets, spread_rets = _build_portfolio_returns(
        preds, actuals, n_windows, horizon, top_fraction, mask=pool,
    )
    n_periods = len(spread_rets)
    if n_periods < 2:
        return _empty_result()

    long_t = torch.tensor(long_rets, dtype=torch.float32)
    long_equity = compute_equity_curve(long_t)
    long_sharpe = compute_sharpe(long_t, horizon=horizon)
    long_lo, long_hi = compute_bootstrap_sharpe_ci(
        np.array(long_rets, dtype=np.float64), horizon=horizon, n_boot=n_boot,
    )
    ls_t = torch.tensor(spread_rets, dtype=torch.float32)
    ls_equity = compute_equity_curve(ls_t)
    ls_sharpe = compute_sharpe(ls_t, horizon=horizon)
    ls_lo, ls_hi = compute_bootstrap_sharpe_ci(
        np.array(spread_rets, dtype=np.float64), horizon=horizon, n_boot=n_boot,
    )
    quintile_metrics = _quintile_analysis(
        preds, actuals, n_windows, horizon, n_stocks, mask=pool,
    )
    ew_rets = []
    sel_uni_ew_rets = []
    for offset in range(horizon):
        for t in range(offset, n_windows, horizon):
            col = actuals[:, t]
            keep = pool[:, t] & torch.isfinite(col)
            r = col[keep].mean().item()
            if np.isfinite(r):
                ew_rets.append(r)
            # Selected-universe equal-weight proxy: every
            # stock with a realized return — no eligibility gate — as the naive
            # "buy everything" reference.  Zero-filled padding
            # (_build_raw_actuals) is excluded.
            keep_uni = torch.isfinite(col) & (col != 0)
            ru = col[keep_uni].mean().item()
            if np.isfinite(ru):
                sel_uni_ew_rets.append(ru)
    eligible_ew_sharpe = compute_sharpe(
        torch.tensor(ew_rets, dtype=torch.float32), horizon=horizon,
    ) if len(ew_rets) >= 2 else 0.0
    selected_universe_ew_sharpe = compute_sharpe(
        torch.tensor(sel_uni_ew_rets, dtype=torch.float32), horizon=horizon,
    ) if len(sel_uni_ew_rets) >= 2 else 0.0

    return {
        "n_periods": n_periods,
        "n_stocks": n_stocks,
        "n_days": n_windows,
        "ic_mean": ic_summary["ic_mean"],
        "ic_std": ic_summary["ic_std"],
        "ic_ir": ic_summary["ic_ir"],
        "ic_pos_rate": ic_summary["ic_pos_rate"],
        "ic_newey_west_t": ic_summary["ic_newey_west_t"],
        "long_sharpe": long_sharpe,
        "long_sharpe_lo": long_lo,
        "long_sharpe_hi": long_hi,
        "long_sortino": compute_sortino(long_t, horizon=horizon),
        "long_calmar": compute_calmar(long_t, horizon=horizon),
        "long_maxdd": compute_max_drawdown(long_equity),
        "long_daily_return_pf": compute_daily_return_profit_factor(long_t),
        "ls_sharpe": ls_sharpe,
        "ls_sharpe_lo": ls_lo,
        "ls_sharpe_hi": ls_hi,
        "ls_sortino": compute_sortino(ls_t, horizon=horizon),
        "ls_calmar": compute_calmar(ls_t, horizon=horizon),
        "ls_maxdd": compute_max_drawdown(ls_equity),
        **quintile_metrics,
        "eligible_ew_sharpe": eligible_ew_sharpe,
        "selected_universe_ew_sharpe": selected_universe_ew_sharpe,
        "long_psr": float("nan"), "long_dsr": float("nan"),
        "ls_psr": float("nan"), "ls_dsr": float("nan"),
        "dsr_n_trials": 0,
        "bbmm_stat": float("nan"), "bbmm_p_value": float("nan"),
        "bbmm_n_strategies": 0,
    }


def _quintile_analysis(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    n_windows: int,
    horizon: int,
    n_stocks: int,
    mask: torch.Tensor | None = None,
) -> dict:
    """Group stocks into 5 equal-sized quintiles each day, track mean return.

    Q1 = lowest predicted, Q5 = highest predicted.
    A healthy signal shows monotonic increase from Q1→Q5.

    mask: (n_stocks, n_windows) bool — only valid (non-padded) stocks are
    bucketed; days with <5 valid stocks are skipped. Without the mask, padded
    stocks (zero predictions) all land in Q1 and fake the spread.
    """
    q_rets = {1: [], 2: [], 3: [], 4: [], 5: []}

    for offset in range(horizon):
        for t in range(offset, n_windows, horizon):
            if mask is not None:
                valid_idx = mask[:, t].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() < 5:
                    continue
                p_day = preds[valid_idx, t]
                a_day = actuals[valid_idx, t]
            else:
                p_day = preds[:, t]
                a_day = actuals[:, t]

            # Same NaN guard as _build_portfolio_returns — a NaN prediction must
            # not silently bucket into Q1.
            keep = torch.isfinite(p_day) & torch.isfinite(a_day)
            p_day = p_day[keep]
            a_day = a_day[keep]
            n_valid = p_day.numel()
            if n_valid < 5:
                continue

            sorted_idx = torch.argsort(p_day, descending=False)
            # np.array_split distributes the remainder evenly instead of
            # silently dropping the last n_valid % 5 stocks (old q_size=//5
            # bias) and keeps ascending index order → Q1 low, Q5 high.
            for qi, q_idx in enumerate(np.array_split(sorted_idx.numpy(), 5), 1):
                if q_idx.size == 0:
                    continue
                q_r = a_day[torch.from_numpy(q_idx)].mean().item()
                if np.isfinite(q_r):
                    q_rets[qi].append(q_r)

    result = {}
    for qi in range(1, 6):
        arr = np.array(q_rets[qi], dtype=np.float64)
        result[f"q{qi}_ret"] = float(arr.mean()) if len(arr) > 0 else 0.0

    # Q5−Q1 spread
    if len(q_rets[5]) > 0 and len(q_rets[1]) > 0:
        spread_arr = np.array(q_rets[5]) - np.array(q_rets[1])
        result["q5mq1_ret"] = float(spread_arr.mean())
    else:
        result["q5mq1_ret"] = 0.0

    # Monotonicity: fraction of adjacent pairs that increase
    monotone = 0
    for i in range(1, 5):
        if result[f"q{i+1}_ret"] >= result[f"q{i}_ret"]:
            monotone += 1
    result["q_monotonic"] = monotone / 4.0

    return result


_EMPTY_EXPOSURE = {
    "target": {"gross_exposure": 1.0, "net_exposure": 0.0,
               "long_exposure": 0.5, "short_exposure": 0.5},
    "realized": {
        "mean_gross_exposure": 0.0, "max_gross_exposure": 0.0,
        "p95_gross_exposure": 0.0, "mean_net_exposure": 0.0,
        "days_off_target_gross": 0, "mean_excess_over_target_gross": 0.0,
        "mean_delayed_capital": 0.0, "max_delayed_capital": 0.0,
        "daily": {
            "long_market_value": [], "short_market_value": [], "cash": [],
            "gross_exposure": [], "net_exposure": [],
            "active_long_sleeves": [], "active_short_sleeves": [],
            "delayed_capital": [],
        },
    },
    "bankrupt": {"long_day": None, "short_day": None},
    "note": ("no price path — no realized exposure measured; target = nominal "
             "book (100% gross, 50% long / 50% short, net 0)."),
}


def _empty_result() -> dict:
    empty_exit = {
        "counts": {"clean": 0, "delayed": 0, "delisted": 0,
                   "unresolved": 0, "unfilled": 0},
        "pnl": {"clean": 0.0, "delayed": 0.0, "delisted": 0.0,
                "unresolved": 0.0, "unfilled": 0.0},
        "pnl_share": {"clean": 0.0, "delayed": 0.0, "delisted": 0.0,
                      "unresolved": 0.0, "unfilled": 0.0},
        "abs_pnl_share": {"clean": 0.0, "delayed": 0.0, "delisted": 0.0,
                          "unresolved": 0.0, "unfilled": 0.0},
        "capital_days": {"clean": 0.0, "delayed": 0.0, "delisted": 0.0,
                         "unresolved": 0.0, "unfilled": 0.0},
        "avg_delayed_days": 0.0,
    }
    return {
        "n_periods": 0, "n_stocks": 0, "n_days": 0,
        "ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_pos_rate": 0.0,
        "ic_newey_west_t": 0.0,
        "long_sharpe": 0.0, "long_gross_sharpe": 0.0,
        "long_sharpe_lo": float("nan"), "long_sharpe_hi": float("nan"),
        "long_sortino": 0.0, "long_calmar": 0.0, "long_maxdd": 0.0, "long_daily_return_pf": 0.0,
        "ls_sharpe": 0.0, "ls_gross_sharpe": 0.0,
        "ls_sharpe_lo": float("nan"), "ls_sharpe_hi": float("nan"),
        "ls_sortino": 0.0, "ls_calmar": 0.0, "ls_maxdd": 0.0,
        "ls2x_sharpe": 0.0, "ls2x_gross_sharpe": 0.0, "ls2x_maxdd": 0.0,
        "exposure": _EMPTY_EXPOSURE,
        "long_turnover": 0.0, "ls_turnover": 0.0, "ew_turnover": 0.0,
        "exit_status": empty_exit,
        "long_exit_status": empty_exit,
        "short_exit_status": empty_exit,
        "eligible_ew_exit_status": empty_exit,
        "q1_ret": 0.0, "q2_ret": 0.0, "q3_ret": 0.0, "q4_ret": 0.0, "q5_ret": 0.0,
        "q5mq1_ret": 0.0, "q_monotonic": 0.0,
        "eligible_ew_sharpe": 0.0,
        "selected_universe_ew_sharpe": 0.0,
        "long_psr": float("nan"), "long_dsr": float("nan"),
        "ls_psr": float("nan"), "ls_dsr": float("nan"),
        "dsr_n_trials": 0,
        "bbmm_stat": float("nan"), "bbmm_p_value": float("nan"),
        "bbmm_n_strategies": 0,
        "long_ledger": None,
        "short_ledger": None,
        "ls_ledger": None,
        "ew_ledger": None,
        "selected_universe_ew_ledger": None,
    }


def compute_prediction_diversity(predictions: np.ndarray) -> float:
    return float(np.std(predictions) / (abs(np.mean(predictions)) + 1e-8))


def _build_raw_actuals(
    raw_returns: np.ndarray | torch.Tensor,
    n_stocks: int,
    n_windows: int,
    seq_len: int,
) -> torch.Tensor:
    if isinstance(raw_returns, torch.Tensor):
        raw_returns = raw_returns.detach().cpu().numpy()
    end = min(seq_len + n_windows, raw_returns.shape[1])
    n_valid = end - seq_len
    if n_valid < n_windows:
        logger.warning(
            "_build_raw_actuals: raw_returns has %d cols, need %d — "
            "%d trailing columns zero-filled. Sharpe/IC may be deflated.",
            raw_returns.shape[1], seq_len + n_windows, n_windows - n_valid,
        )
    actuals = np.zeros((n_stocks, n_windows), dtype=np.float32)
    actuals[:, :n_valid] = raw_returns[:, seq_len:end]
    return torch.from_numpy(actuals)
