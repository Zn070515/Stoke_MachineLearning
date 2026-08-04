import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate

logger = logging.getLogger(__name__)

# Version stamp for the panel evaluator — experiments freeze the evaluator
# that produced their numbers.  Bump on any behavioral change to
# the sleeve-account / IC / quintile logic.
EVALUATOR_VERSION = "2026-08-04"


def compute_sharpe(
    daily_returns: torch.Tensor,
    annualize: bool = True,
    horizon: int = 1,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = daily_returns.mean().item()
    std = daily_returns.std().item()
    if std < 1e-8:
        return 0.0
    sharpe = mean / std
    if annualize:
        sharpe *= np.sqrt(252 / horizon)
    return float(sharpe)


def compute_sortino(
    daily_returns: torch.Tensor,
    annualize: bool = True,
    horizon: int = 1,
    target: float = 0.0,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = daily_returns.mean().item()
    downside = daily_returns[daily_returns < target]
    if len(downside) < 2:
        # With < 2 downside samples the Sortino is undefined.
        # NaN (not inf) so JSON summaries / means / charts don't treat a lucky
        # no-downside stretch as an unbounded score.
        return float("nan") if mean > target else 0.0
    down_std = downside.std().item()
    if down_std < 1e-8:
        return 0.0
    sortino = (mean - target) / down_std
    if annualize:
        sortino *= np.sqrt(252 / horizon)
    return float(sortino)


def compute_max_drawdown(equity_curve: torch.Tensor) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = torch.cummax(equity_curve, dim=0).values
    drawdowns = (peak - equity_curve) / (peak + 1e-8)
    return float(drawdowns.max().item())


def compute_calmar(
    daily_returns: torch.Tensor,
    annualize: bool = True,
    horizon: int = 1,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    equity = torch.cat([torch.tensor([1.0]), 1.0 + daily_returns]).cumprod(0)
    mdd = compute_max_drawdown(equity)
    if mdd < 1e-8:
        return 0.0
    # Use the actual CAGR of the NAV curve (geometric), not
    # the arithmetic mean*252 — under volatility the two diverge materially.
    final_nav = float(equity[-1].item())
    if final_nav <= 0:
        return 0.0
    cagr = final_nav ** (252 / (horizon * len(daily_returns))) - 1.0
    return float(cagr / mdd)


def compute_daily_return_profit_factor(daily_returns: torch.Tensor) -> float:
    """Gross profit / gross loss over *daily* return samples.

    This is NOT a trade-level profit factor — it aggregates per-day returns of
    a strategy equity curve, not closed-trade P&L.  A trade-level PF would need
    per-sleeve / per-position realized P&L.  Kept distinct by name so it is not
    mistaken for one.
    """
    profits = daily_returns[daily_returns > 0].sum().item()
    losses = abs(daily_returns[daily_returns < 0].sum().item())
    if losses < 1e-8:
        return float("inf") if profits > 0 else 0.0
    return float(profits / losses)


def compute_equity_curve(
    daily_returns: torch.Tensor,
    initial_capital: float = 1.0,
) -> torch.Tensor:
    return torch.cat([torch.tensor([initial_capital]), 1.0 + daily_returns]).cumprod(0)


def compute_bootstrap_sharpe_ci(
    returns: np.ndarray,
    horizon: int = 1,
    n_boot: int = 2000,
    seed: int | None = 42,
    block_len: int | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for annualized Sharpe.

    Uses a MOVING-BLOCK bootstrap (Künsch 1989): daily returns carry
    autocorrelation and volatility clustering, so resampling single points
    (iid) destroys that structure and the CI comes out too narrow — worse for
    overlapping-horizon returns.  Block length defaults to
    the classical ceil(n^(1/3)); the horizon floor keeps the block at least
    as long as the return overlap.
    """
    n = len(returns)
    if n < 5:
        return float("nan"), float("nan")
    L = block_len or max(2, int(np.ceil(n ** (1 / 3))), horizon)
    L = min(L, n)
    n_blocks = int(np.ceil(n / L))
    rng = np.random.RandomState(seed)
    sharpes = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.randint(0, n - L + 1, size=n_blocks)
        blocks = [returns[s:s + L] for s in starts]
        sample = np.concatenate(blocks)[:n]
        m = sample.mean()
        s = sample.std(ddof=1)
        sharpes[i] = (m / s) * np.sqrt(252 / horizon) if s > 1e-8 else 0.0
    lo = float(np.percentile(sharpes, 2.5))
    hi = float(np.percentile(sharpes, 97.5))
    return lo, hi


def _compute_daily_ic(
    preds_np: np.ndarray,
    actuals_np: np.ndarray,
    mask_np: np.ndarray | None = None,
    min_stocks: int = 10,
) -> list[float]:
    """Per-day Spearman rank IC.

    mask_np: (n_stocks, n_windows) bool — candidate pool (entry-eligible) on
    each day. Without it, zero-feature padded rows (short listing history) get
    garbage predictions that drag down the rank correlation.  A day is skipped
    unless it holds >= `min_stocks` eligible stocks — the unified threshold
    so checkpoint selection and the formal report agree.
    Days whose predictions are constant (std < eps) have a degenerate Spearman
    and are skipped.
    """
    daily_ics = []
    n_windows = preds_np.shape[1]
    for t in range(n_windows):
        p = preds_np[:, t]
        a = actuals_np[:, t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask_np is not None:
            mask = mask & mask_np[:, t]
        if mask.sum() >= min_stocks:
            pv = p[mask]
            if pv.std() < 1e-8:
                continue  # constant predictions → rank correlation undefined
            ic, _ = spearmanr(pv, a[mask])
            if np.isfinite(ic):
                daily_ics.append(ic)
    return daily_ics


def _raw_clean_rank_ic(
    val_data: dict,
    preds_np: np.ndarray,
    n_windows: int,
    seq_len: int,
    min_stocks: int = 10,
    diag: dict | None = None,
) -> tuple[list[float] | None, torch.Tensor | None]:
    """Per-day Spearman IC of predictions vs the RAW open->open return.

    The single shared clean-IC definition used BOTH by the formal report
    (evaluate_portfolio) and by checkpoint selection (train._compute_val_loss).  Ranking against the
    z-scored + clipped [-5,5] model
    target manufactures ties and can select a different checkpoint than the
    report; both must rank the raw clean return (y_return_raw, saved before
    the fold z-score; falls back to y_return for data without
    a raw copy).

    The candidate pool is decision & history & return-target,
    so weak cross-sections are filtered identically to the report.  Returns
    (None, None) when val_data has no y_return_raw / y_return — the caller
    falls back to the legacy realized-return reconstruction.

    `diag` (optional mutable dict) receives the per-window pool statistics the
    failure path needs: valid_days / avg_stocks_per_day /
    mask_retention.
    """
    pool = _candidate_pool(val_data, n_windows, seq_len).numpy()
    if "return_target_mask" in val_data:
        rt = np.asarray(val_data["return_target_mask"], dtype=bool)
        pool = pool & rt[:, seq_len:seq_len + n_windows]
    if diag is not None:
        per_day = pool.sum(axis=0)
        diag["valid_days"] = int((per_day >= min_stocks).sum())
        diag["avg_stocks_per_day"] = float(
            per_day[per_day > 0].mean()) if per_day.any() else 0.0
        diag["mask_retention"] = float(pool.mean())
    clean_key = "y_return_raw" if "y_return_raw" in val_data else "y_return"
    if clean_key not in val_data:
        return None, None
    clean_actuals = torch.as_tensor(val_data[clean_key], dtype=torch.float32)[
        :, seq_len:seq_len + n_windows]
    daily_ics = _compute_daily_ic(
        preds_np, clean_actuals.numpy(), pool, min_stocks=min_stocks)
    return daily_ics, clean_actuals


def _newey_west_t(series: np.ndarray, lag: int) -> float:
    """Autocorrelation-robust t-stat (Newey & West 1987, Bartlett kernel)."""
    n = len(series)
    if n < 2:
        return float("nan")
    mean = float(series.mean())
    x = series - mean
    gamma0 = float(np.dot(x, x) / n)
    if gamma0 < 1e-12:
        if abs(mean) < 1e-12:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(x[:-k], x[k:]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    var = max(var, gamma0 * 1e-8)  # guard against negative NW variance
    return float(mean / np.sqrt(var / n))


def compute_ic_summary(daily_ics: list[float], horizon: int = 1) -> dict:
    """IC mean, std, IR, positivity rate, and a Newey-West t-stat.

    The plain mean/std ratio ignores serial correlation —
    with overlapping-horizon labels the per-day IC series is autocorrelated,
    so a naive IR overstates signal.  `ic_newey_west_t` uses the NW-1994
    automatic lag truncation (Bartlett kernel), floored to horizon-1 so the
    overlap is actually captured.
    """
    if not daily_ics:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
                "ic_pos_rate": 0.0, "ic_newey_west_t": 0.0}
    arr = np.array(daily_ics, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    n = len(arr)
    lag = max(horizon - 1, int(np.floor(4 * (n / 100.0) ** (2 / 9.0))))
    lag = min(lag, n - 1)
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 1e-8 else 0.0,
        "ic_pos_rate": float((arr > 0).mean()),
        "ic_newey_west_t": _newey_west_t(arr, lag),
    }


def _build_portfolio_returns(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    n_windows: int,
    horizon: int,
    top_fraction: float,
    mask: torch.Tensor | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Build long-only top-K, short bottom-K, and long-short spread returns.

    All `horizon` entry phases (offset 0..horizon-1) are evaluated and
    concatenated into ONE return series, so no in-sample entry days are wasted
    (the old single-phase loop discarded 4/5 of the data at horizon=5).

    mask: (n_stocks, n_windows) bool — the candidate pool on day t.  Only
    stocks in the pool are selectable; a day with <2 pool members is skipped.
    """
    long_rets, short_rets, spread_rets = [], [], []
    for offset in range(horizon):
        for t in range(offset, n_windows, horizon):
            if mask is not None:
                day_mask = mask[:, t]
                valid_idx = day_mask.nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() < 2:
                    continue
                p_day = preds[valid_idx, t]
                a_day = actuals[valid_idx, t]
            else:
                p_day = preds[:, t]
                a_day = actuals[:, t]

            # Drop non-finite predictions.  Head clamps bound magnitudes but pass
            # NaN through, and descending argsort sorts NaN to the head — that
            # would inject garbage rows into the top-K long / bottom-K short books.
            keep = torch.isfinite(p_day) & torch.isfinite(a_day)
            p_day = p_day[keep]
            a_day = a_day[keep]
            n_candidates = p_day.numel()
            if n_candidates < 2:
                continue

            k = max(1, min(n_candidates // 2, int(round(top_fraction * n_candidates))))
            sorted_idx = torch.argsort(p_day, descending=True)
            top_idx = sorted_idx[:k]
            bot_idx = sorted_idx[-k:]

            long_r = a_day[top_idx].mean().item()
            short_r = a_day[bot_idx].mean().item()
            if np.isfinite(long_r) and np.isfinite(short_r):
                long_rets.append(long_r)
                short_rets.append(short_r)
                spread_rets.append(long_r - short_r)

    return long_rets, short_rets, spread_rets


def _candidate_pool(data: dict, n_windows: int, seq_len: int) -> torch.Tensor:
    """Selection pool for window day t (column t ↔ panel column seq_len+t).

    Rank over the DECISION pool — close[t-1] is real
    (signal after close[t-1]) AND the seq_len-window ending at t-1 holds >=
    min_history real observations.  Both masks are aligned to the ENTRY column
    t.  Falls back to entry-eligibility, then to future-label validity, for
    synthetic/legacy data without the decision/history masks.
    """
    has_dh = "decision_eligible_mask" in data and "history_eligible_mask" in data
    if has_dh:
        dec = torch.as_tensor(data["decision_eligible_mask"])
        hist = torch.as_tensor(data["history_eligible_mask"])
        return dec[:, seq_len:seq_len + n_windows] & hist[:, seq_len:seq_len + n_windows]
    if "entry_eligible_mask" in data:
        entry = torch.as_tensor(data["entry_eligible_mask"])
        return entry[:, seq_len:seq_len + n_windows]
    y_dir = torch.as_tensor(data["y_direction"])
    return (y_dir[:, seq_len:seq_len + n_windows] != -100)


def _ffill_last_np(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Forward-fill the last known value along axis 1 (per stock).

    NaN before a stock's first real price becomes `fill` (0) so a position in
    a never-traded stock marks to zero instead of poisoning NAV with NaN.
    """
    a = np.array(arr, dtype=np.float64)
    n, m = a.shape
    mask = np.isfinite(a)
    idx = np.where(mask, np.arange(m)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1)
    row_has_valid = mask.any(axis=1)
    first_valid = np.where(row_has_valid, np.argmax(mask, axis=1), m)
    col = np.arange(m)[None, :]
    out = np.where(col < first_valid[:, None], fill, out)
    return np.nan_to_num(out, nan=fill)


def _run_sleeve_sim(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    mode: str = "long",
    return_ledger: bool = False,
) -> dict:
    """Chronological sleeve-account backtest, ONE run at a fixed cost.

    Core simulator shared by `_simulate_sleeve_account`, which runs it twice —
    once at the real cost (net NAV series) and once at cost=0 (the TRUE
    cost-free counterfactual used for gross returns).  This function returns
    the net daily series plus the realized P&L / exit ledger for THIS run; it
    does NOT compute a gross series (the old `gross_daily = net + cost_paid`
    approximation was not a real counterfactual — a zero-cost account would
    reallocate different sleeve sizes, so its daily returns differ day-by-day,
    not just by an additive cost term).

    Every signal day d (< W) enters a new sleeve that buys the top-K (long),
    shorts the bottom-K (short), or buys every pool member (ew), at open[d],
    and is scheduled to liquidate at open[d+horizon].  Each sleeve carries
    weight 1/h of the account's NAV at entry, so h sleeves are concurrently
    invested; the NAV is marked to close each day, giving a TRUE daily return
    series annualized with sqrt(252) — fixing the phase-concatenation bug.

    The simulation runs to the END of the price path (Wp columns), NOT just to
    the last signal day: the sleeve entered on the last signal
    day W-1 must still be liquidated at open[W-1+horizon], with its exit cost
    booked and the book reaching zero active sleeves.  Entry only happens on
    days d < W (no signal beyond W).

    Exits are TRUE delayed exits: a position whose scheduled
    exit open is missing (suspension) is NOT sold at its stale close — it keeps
    holding, marks to its last known close, and retries the exit on every later
    day until a real open appears (DELAYED) or the price path ends with the
    position still open (UNRESOLVED_AT_END, booked at the last carried close,
    capital left locked — no fake exit, no fictitious sell cost).  A carried
    close <= 0 is treated as DATA-MISSING, NOT an executable delist: after
    ffill a zero price cannot
    distinguish a real delisting from a dead data row, so the position is never
    force-sold at 0.  exit_status covers every successfully entered position.

    Fills respect entry eligibility: a selected stock with no real open[d]
    stays unfilled (its weight remains cash) and is NEVER backfilled.  Long/EW
    entries are additionally capped so total spend (notional
    + entry cost) never exceeds available cash — the old cap
    booked the cost AFTER sizing, so a full-buy at 10bp went cash-negative.

    The account starts at nav 1.0 on the close BEFORE the first price day, so
    the returned daily series INCLUDES day 0.  An internal
    assertion enforces the account identity np.prod(1+daily) == final_nav.

    preds_np: (N, W) predictions for entry at window column d.  close_np /
    open_np: (N, Wp) price paths aligned to the SAME window-day grid (column d
    = entry day d = panel column seq_len+d), Wp >= W.  The exit at open[d+h]
    is the sleeve popped on day d+h reading open_np[:, d+h].  If Wp < W+horizon
    the trailing sleeves end up UNRESOLVED.
    """
    N, W = preds_np.shape
    Wp = close_np.shape[1]
    if Wp < W:
        raise ValueError(f"close path {Wp} columns < preds {W} columns")
    side = {"long": 1.0, "ew": 1.0, "short": -1.0}[mode]
    carried = _ffill_last_np(close_np)

    nav_prev = 1.0
    cash = 1.0
    sleeves: dict[int, dict] = {}
    net_daily = np.zeros(Wp)
    cost_paid = np.zeros(Wp)
    buy_notional = np.zeros(Wp)
    sell_notional = np.zeros(Wp)
    nav_before_day = np.zeros(Wp)
    # Daily exposure ledger: held notional / held value / cash /
    # active sleeves / delayed capital, each normalized by start-of-day NAV so
    # the ratios are scale-free and comparable across sub-accounts.
    gross_exposure = np.zeros(Wp)
    net_exposure = np.zeros(Wp)
    cash_ratio = np.zeros(Wp)
    active_sleeves = np.zeros(Wp, dtype=int)
    delayed_capital = np.zeros(Wp)
    exit_counts = {"clean": 0, "delayed": 0, "delisted": 0,
                   "unresolved": 0, "unfilled": 0}
    exit_pnl = {k: 0.0 for k in exit_counts}
    exit_days = {k: 0.0 for k in exit_counts}
    # Per-position ledger: one record per FILLED position,
    # with entry/exit price, scheduled/actual exit day, exit status, gross PnL
    # and attributed costs.  Sum(net_pnl) == final_nav - 1 by construction —
    # each position's gross PnL matches exit_pnl aggregation and entry/exit
    # costs match cost_paid day-by-day.  Built only when requested.
    ledger: list[dict] = []

    for d in range(Wp):
        nav_before_day[d] = nav_prev
        # ── Liquidate positions whose scheduled exit is due (or overdue) ──
        for c, sl in list(sleeves.items()):
            held = sl["mask"]
            if not held.any():
                del sleeves[c]
                continue
            if d < sl["scheduled_exit_day"]:
                continue
            ex_open = open_np[:, d]
            ex_open_ok = np.isfinite(ex_open) & (ex_open > 0)
            exitable = held & ex_open_ok
            if d == sl["scheduled_exit_day"]:
                # First attempt: positions that cannot exit become PENDING and
                # are held (marked to last close) until a real open appears.
                sl["pending"] |= held & (~ex_open_ok) & (carried[:, d] > 0)
            # A held position whose carried close is <= 0 is DATA-MISSING, not
            # an executable delist: price<=0 after ffill
            # cannot tell a real delisting from a dead data row, so it is never
            # force-sold at 0.  It stays held at its (zero) carried mark and
            # resolves as UNRESOLVED_AT_END at the path end.
            liquidate = exitable
            if liquidate.any():
                ex_price = np.where(ex_open_ok, ex_open, carried[:, d])
                ex_value = sl["shares"] * ex_price
                proceeds = side * ex_value[liquidate].sum()
                cost_here = cost * np.abs(ex_value[liquidate]).sum()
                cash += proceeds - cost_here
                cost_paid[d] += cost_here
                sell_notional[d] += np.abs(ex_value[liquidate]).sum()
                is_clean = (d == sl["scheduled_exit_day"]) & ex_open_ok & (~sl["pending"])
                for cls, cm in (("clean", liquidate & is_clean),
                                ("delayed", liquidate & ~is_clean)):
                    if cm.any():
                        cls_pnl = side * (ex_value[cm].sum() - sl["entry_val"][cm].sum())
                        exit_pnl[cls] += float(cls_pnl)
                        exit_counts[cls] += int(cm.sum())
                        exit_days[cls] += float((d - c) * int(cm.sum()))
                if return_ledger:
                    for j in np.nonzero(liquidate)[0]:
                        ej = float(sl["entry_val"][j])
                        xj = float(ex_value[j])
                        ledger.append({
                            "entry_day": c,
                            "stock": int(j),
                            "mode": mode,
                            "entry_price": float(open_np[j, c]),
                            "entry_value": ej,
                            "executed_weight": float(ej),
                            "shares": float(sl["shares"][j]),
                            "scheduled_exit_day": int(sl["scheduled_exit_day"]),
                            "actual_exit_day": d,
                            "exit_status": "clean" if is_clean[j] else "delayed",
                            "exit_price": float(ex_price[j]),
                            "realized_return": float((xj - ej) / ej) if ej > 0 else 0.0,
                            "gross_pnl": float(side * (xj - ej)),
                            "entry_cost": cost * ej,
                            "exit_cost": cost * abs(xj),
                            "net_pnl": float(side * (xj - ej) - cost * ej - cost * abs(xj)),
                        })
                sl["mask"] = held & ~liquidate
            if not sl["mask"].any():
                del sleeves[c]

        # ── Enter the sleeve at signal day d ──
        if d < W:
            w = nav_prev / horizon
            pool_d = select_pool[:, d] & np.isfinite(preds_np[:, d])
            pool_idx = np.nonzero(pool_d)[0]
            shares = np.zeros(N)
            entry_val = np.zeros(N)
            fillable = np.zeros(N, dtype=bool)  # empty pool → empty (cash) sleeve
            if pool_idx.size > 0:
                if mode == "ew":
                    chosen = pool_idx
                else:
                    k = max(1, min(pool_idx.size, int(round(top_fraction * pool_idx.size))))
                    order = pool_idx[np.argsort(preds_np[pool_idx, d])]
                    chosen = order[-k:] if mode == "long" else order[:k]
                stock_mask = np.zeros(N, dtype=bool)
                stock_mask[chosen] = True
                fillable = stock_mask & np.isfinite(open_np[:, d]) & (open_np[:, d] > 0)
                exit_counts["unfilled"] += int((stock_mask & ~fillable).sum())
                n_fill = int(fillable.sum())
                if n_fill > 0:
                    # Fixed per-stock weight w/|chosen| — the unfilled fraction
                    # stays cash (NO backfill / 递补).  Long/EW
                    # cap the whole sleeve at available cash so leverage cannot
                    # drift past what the account holds; the
                    # short leg is a theoretical factor book and needs no cap.
                    per_stock = w / chosen.size
                    if side > 0:
                        # Cap total spend (notional + entry cost) at available
                        # cash: per_stock*n_fill*(1+cost) <= cash
                        # so the entry cost can never push cash negative.
                        per_stock = min(per_stock, cash / ((1.0 + cost) * n_fill))
                    shares[fillable] = per_stock / open_np[fillable, d]
                    entry_val[fillable] = per_stock
                    spent = per_stock * n_fill
                    # The entry cost is deducted from cash along with the
                    # notional — previously only the notional
                    # left cash, so net returns excluded the entry cost.
                    cash -= side * spent + cost * spent
                    cost_paid[d] += cost * spent
                    buy_notional[d] = spent
            sleeves[d] = {"shares": shares, "mask": fillable, "entry_val": entry_val,
                          "scheduled_exit_day": d + horizon,
                          "pending": np.zeros(N, dtype=bool)}

        # ── Mark to market at close d ──
        pos_value = 0.0
        gross_value = 0.0
        active_count = 0
        delayed_value = 0.0
        for c, sl in sleeves.items():
            if sl["mask"].any():
                mv = float((sl["shares"] * carried[:, d]).sum())
                pos_value += side * mv
                gross_value += abs(mv)
                active_count += 1
                if d > sl["scheduled_exit_day"]:
                    delayed_value += abs(mv)
        nav_close = cash + pos_value
        # The account starts at nav 1.0 on the close BEFORE the
        # first price day, so day-0's open->close P&L IS part of the return
        # series — dropping it broke the identity prod(1+daily) == final_nav.
        if nav_prev > 0:
            net_daily[d] = nav_close / nav_prev - 1.0
            # Exposure ratios scale the held notional by the day's capital base:
            # the nominal 1/h-per-sleeve book is only an
            # approximation — unfilled slots stay cash (gross < target) while
            # delayed/unresolved sleeves push gross above it.
            gross_exposure[d] = gross_value / nav_prev
            net_exposure[d] = pos_value / nav_prev
            cash_ratio[d] = cash / nav_prev
        active_sleeves[d] = active_count
        delayed_capital[d] = delayed_value / nav_prev if nav_prev > 0 else 0.0
        nav_prev = nav_close

    # ── End of price path: any holding that never found a real exit is
    # UNRESOLVED_AT_END, booked at its last carried close: the
    # capital stays locked in the position — no fake exit, no fictitious sell
    # cost. ──
    for c, sl in sleeves.items():
        held = sl["mask"]
        if held.any():
            # Book the unresolved hold at the FINAL carried mark:
            # carried[:, Wp-1] is what final_nav's mark-to-market used on the
            # last day, so the P&L ledger reconciles with the NAV series.  The
            # old min(c+horizon, Wp-1) booked at the scheduled exit day, which
            # drifts from final_nav whenever price moved after that day.
            last_close = carried[:, Wp - 1]
            ex_value = sl["shares"] * last_close
            pnl = side * (ex_value[held].sum() - sl["entry_val"][held].sum())
            exit_pnl["unresolved"] += float(pnl)
            exit_counts["unresolved"] += int(held.sum())
            exit_days["unresolved"] += float((Wp - 1 - c) * int(held.sum()))
            if return_ledger:
                for j in np.nonzero(held)[0]:
                    ej = float(sl["entry_val"][j])
                    xj = float(ex_value[j])
                    ledger.append({
                        "entry_day": c,
                        "stock": int(j),
                        "mode": mode,
                        "entry_price": float(open_np[j, c]),
                        "entry_value": ej,
                        "executed_weight": float(ej),
                        "shares": float(sl["shares"][j]),
                        "scheduled_exit_day": int(sl["scheduled_exit_day"]),
                        "actual_exit_day": int(Wp - 1),
                        "exit_status": "unresolved",
                        "exit_price": float(last_close[j]),
                        "realized_return": float((xj - ej) / ej) if ej > 0 else 0.0,
                        "gross_pnl": float(side * (xj - ej)),
                        "entry_cost": cost * ej,
                        "exit_cost": 0.0,
                        "net_pnl": float(side * (xj - ej) - cost * ej),
                    })

    total_pnl = sum(exit_pnl.values())
    abs_total = sum(abs(v) for v in exit_pnl.values())

    # P&L reconciliation: every sleeve's realized P&L net of
    # ALL entry/exit costs must equal the account's total NAV change.  Booked
    # at a mark different from the final one (or costs missing from cost_paid)
    # would silently break this identity, so assert it explicitly.
    if np.isfinite(nav_close) and np.isfinite(total_pnl):
        reconciled = float(total_pnl) - float(cost_paid.sum())
        target = float(nav_close) - 1.0
        if not np.isclose(reconciled, target, rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"sleeve P&L reconciliation violated: sum(pnl) - sum(cost) = "
                f"{reconciled:.8f} != final_nav - 1 = {target:.8f}")
    # Turnover is the traded notional scaled by the NAV the day opened on —
    # a ratio, invariant to the initial capital.
    turnover_daily = (buy_notional + sell_notional) / np.maximum(nav_before_day, 1e-8)

    # The signed PnL share explodes when the book's total
    # PnL is near zero, so alongside the signed share (kept for compat) report
    # the signed PnL, an ABSOLUTE-PnL share, capital-occupancy days, and the
    # mean delay of delayed exits.
    avg_delayed_days = (exit_days["delayed"] / exit_counts["delayed"]
                        if exit_counts["delayed"] else 0.0)

    # Enforce the account identity without relying on manual inspection:
    # np.prod(1+daily) must equal final_nav / initial_nav.
    if np.isfinite(nav_close) and nav_close > 0 and np.all(nav_before_day > 0):
        cum = float(np.prod(1.0 + net_daily))
        if not np.isclose(cum, nav_close, rtol=1e-5, atol=1e-8):
            raise AssertionError(
                f"sleeve account identity violated: prod(1+daily)={cum:.8f} != "
                f"final_nav={nav_close:.8f}")

    return {
        "daily": net_daily,
        # The account's final mark, exposed so a consumer can verify the
        # identity prod(1+daily) == final_nav WITHOUT recomputing nav_close —
        # the strongest cross-check a backtest can carry.
        "final_nav": float(nav_close) if np.isfinite(nav_close) else None,
        "exit_stats": {
            "counts": exit_counts,
            "pnl": {k: float(v) for k, v in exit_pnl.items()},
            "pnl_share": {k: (float(v) / total_pnl if total_pnl != 0.0 else 0.0)
                          for k, v in exit_pnl.items()},
            "abs_pnl_share": {k: (float(abs(v)) / abs_total if abs_total != 0.0 else 0.0)
                              for k, v in exit_pnl.items()},
            "capital_days": {k: float(v) for k, v in exit_days.items()},
            "avg_delayed_days": float(avg_delayed_days),
        },
        "turnover": {
            "daily_avg": float(turnover_daily.mean()) if Wp > 0 else 0.0,
        },
        # Daily exposure ledger: ratios normalized by NAV.
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "cash_ratio": cash_ratio,
        "active_sleeves": active_sleeves,
        "delayed_capital": delayed_capital,
        # Per-position ledger; empty unless return_ledger.
        "ledger": ledger,
    }


def _simulate_sleeve_account(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    mode: str = "long",
    return_ledger: bool = False,
) -> dict:
    """Sleeve-account backtest with a TRUE cost-free gross counterfactual.

    Runs `_run_sleeve_sim` twice — once at the real `cost` (net NAV series,
    realized exit ledger, turnover) and once at cost=0 (the independent
    cost-free account).  `daily` is the NET series; `gross_daily` is the
    cost=0 series.  Because the zero-cost run re-sizes sleeve positions, its
    daily returns differ from the net series day-by-day in a way the old
    `net_daily + cost_paid/nav` additive approximation could not capture.  Both
    runs trigger the internal account-identity and P&L-reconciliation asserts,
    so each returned series is internally consistent.  Return keys are
    backward-compatible with the pre-refactor shape.

    `return_ledger` surfaces the NET run's per-position ledger so a caller can
    persist every filled position's entry/exit prices,
    status and attributed costs — the audit trail that lets the OOS account be
    reconstructed offline from a tape.
    """
    net = _run_sleeve_sim(preds_np, close_np, open_np, select_pool,
                          horizon, top_fraction, cost, mode=mode,
                          return_ledger=return_ledger)
    gross = _run_sleeve_sim(preds_np, close_np, open_np, select_pool,
                            horizon, top_fraction, 0.0, mode=mode)
    out = {
        "daily": net["daily"],
        "gross_daily": gross["daily"],
        "exit_stats": net["exit_stats"],
        "turnover": net["turnover"],
        # Exposure ledger is measured on the REAL-cost run —
        # it describes the account as actually traded, not the counterfactual.
        "gross_exposure": net["gross_exposure"],
        "net_exposure": net["net_exposure"],
        "cash_ratio": net["cash_ratio"],
        "active_sleeves": net["active_sleeves"],
        "delayed_capital": net["delayed_capital"],
    }
    if return_ledger:
        out["ledger"] = net["ledger"]
    return out


def _ls_exposure_ledger(long_a: dict, short_a: dict) -> dict:
    """Combined LS-book daily exposure ledger + summary.

    The LS book holds the long and short sub-accounts at 0.5 weight each.  Each
    sub-account reports NAV-relative ratios from its own simulation; weighting
    them by each leg's realized NAV path (cumprod of the daily series) yields
    the combined book's true ratios even when the two sub-accounts drift apart.
    `long_market_value` / `short_market_value` / `cash` are value shares of the
    combined NAV (initial capital = 1.0, so share ≈ money units at inception).

    `target` is the nominal book construction (100% gross / 0% net); `realized`
    is what actually traded — unfilled cash slots push gross below target while
    delayed/unresolved sleeves push it above, so the two must not be conflated.
    """
    long_d = np.asarray(long_a["daily"], dtype=np.float64)
    short_d = np.asarray(short_a["daily"], dtype=np.float64)
    long_nav = np.cumprod(1.0 + long_d)
    short_nav = np.cumprod(1.0 + short_d)
    comb_nav = 0.5 * long_nav + 0.5 * short_nav
    safe = np.where(comb_nav > 0, comb_nav, 1.0)

    lg = np.asarray(long_a["gross_exposure"], dtype=np.float64)
    sg = np.asarray(short_a["gross_exposure"], dtype=np.float64)
    ln = np.asarray(long_a["net_exposure"], dtype=np.float64)
    sn = np.asarray(short_a["net_exposure"], dtype=np.float64)
    lc = np.asarray(long_a["cash_ratio"], dtype=np.float64)
    sc = np.asarray(short_a["cash_ratio"], dtype=np.float64)
    ld = np.asarray(long_a["delayed_capital"], dtype=np.float64)
    sd = np.asarray(short_a["delayed_capital"], dtype=np.float64)
    la = np.asarray(long_a["active_sleeves"], dtype=np.int64)
    sa = np.asarray(short_a["active_sleeves"], dtype=np.int64)

    long_mv = (0.5 * lg * long_nav) / safe
    short_mv = (0.5 * sg * short_nav) / safe
    ls_gross = long_mv + short_mv
    ls_net = (0.5 * ln * long_nav + 0.5 * sn * short_nav) / safe
    ls_cash = (0.5 * lc * long_nav + 0.5 * sc * short_nav) / safe
    ls_delayed = (0.5 * ld * long_nav + 0.5 * sd * short_nav) / safe

    g = ls_gross[np.isfinite(ls_gross)]
    n = ls_net[np.isfinite(ls_net)]
    dly = ls_delayed[np.isfinite(ls_delayed)]
    # Target gross is 1.0; count days deviating >5% either way and the mean
    # over-leverage (gross above 1.0) — the excess delayed exits can create.
    off_target = int(np.sum(np.abs(g - 1.0) > 0.05)) if g.size else 0
    excess = float(np.mean(np.maximum(g - 1.0, 0.0))) if g.size else 0.0

    return {
        "target": {
            "gross_exposure": 1.0,
            "net_exposure": 0.0,
            "long_exposure": 0.5,
            "short_exposure": 0.5,
        },
        "realized": {
            "mean_gross_exposure": float(g.mean()) if g.size else 0.0,
            "max_gross_exposure": float(g.max()) if g.size else 0.0,
            "p95_gross_exposure": float(np.percentile(g, 95)) if g.size else 0.0,
            "mean_net_exposure": float(n.mean()) if n.size else 0.0,
            "days_off_target_gross": off_target,
            "mean_excess_over_target_gross": excess,
            "mean_delayed_capital": float(dly.mean()) if dly.size else 0.0,
            "max_delayed_capital": float(dly.max()) if dly.size else 0.0,
            "daily": {
                "long_market_value": long_mv.tolist(),
                "short_market_value": short_mv.tolist(),
                "cash": ls_cash.tolist(),
                "gross_exposure": ls_gross.tolist(),
                "net_exposure": ls_net.tolist(),
                "active_long_sleeves": la.tolist(),
                "active_short_sleeves": sa.tolist(),
                "delayed_capital": ls_delayed.tolist(),
            },
        },
        "note": ("target = nominal book (ls_sharpe is 100% gross, 50% long / "
                 "50% short, net 0; ls2x_* is 200% gross); realized = daily "
                 "measured exposure from the sleeve simulation."),
    }


def _combine_book_daily(
    a_daily: np.ndarray,
    b_daily: np.ndarray,
    w_a: float,
    w_b: float,
    subtract: float = 0.0,
) -> np.ndarray:
    """Daily returns of a NO-rebalance blend of two sub-account NAVs.

    Each sub-account starts at ``w_a`` / ``w_b`` of unit capital and is never
    rebalanced, so the combined daily return is the NAV ratio of the weighted
    NAV blend — the same NAV ``_ls_exposure_ledger`` reports, keeping the
    metric and the ledger on one policy (方案 A).  ``subtract`` folds in
    borrowed capital for leverage >100% gross: the 2x book holds a full unit
    long AND a full unit short, so its NAV = long_nav + short_nav - 1.0 (the
    short leg's margin loan).  The first element is ``nav[0] - 1`` because the
    book starts at unit capital.
    """
    a = np.cumprod(1.0 + np.asarray(a_daily, dtype=np.float64))
    b = np.cumprod(1.0 + np.asarray(b_daily, dtype=np.float64))
    nav = w_a * a + w_b * b - subtract
    out = np.empty_like(nav)
    out[0] = nav[0] - 1.0
    prev = np.where(nav[:-1] > 0, nav[:-1], 1.0)
    out[1:] = nav[1:] / prev - 1.0
    return out


def _sleeve_account_metrics(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    n_boot: int,
    return_ledger: bool = False,
) -> dict:
    """Run long / short / equal-weight sleeve accounts and assemble metrics."""
    long_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                      horizon, top_fraction, cost, "long",
                                      return_ledger=return_ledger)
    short_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                       horizon, top_fraction, cost, "short")
    # Eligible candidate-pool equal-weight: every pool member
    # gets an equal slice — NOT a full-market/index benchmark.  A SEPARATE
    # selected-universe equal-weight proxy (all fold stocks, no eligibility
    # gate) is run below as the naive "buy everything" reference.
    ew_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                    horizon, top_fraction, cost, "ew")
    sel_uni_a = _simulate_sleeve_account(
        preds_np, close_np, open_np, np.ones_like(select_pool),
        horizon, top_fraction, cost, "ew")

    long_d = np.asarray(long_a["daily"], dtype=np.float64)
    short_d = np.asarray(short_a["daily"], dtype=np.float64)
    long_g = np.asarray(long_a["gross_daily"], dtype=np.float64)
    short_g = np.asarray(short_a["gross_daily"], dtype=np.float64)

    # `short_d` is already the short account's REAL daily return
    # (side=-1 applied inside), so the long-short book ADDS the legs — NOT
    # long_d - short_d, which cancels them when both sides profit.
    # The LS book does NOT rebalance daily.  The two legs each
    # start at their target weight of unit capital and are left alone, so the
    # combined daily return is the NAV ratio of the weighted NAV blend (方案 A)
    # — NOT the arithmetic mean of the legs' daily returns, which silently
    # assumes a daily 50/50 reallocation and drifts from the ledger NAV.  All
    # Sharpe/MDD/CAGR now derive from the same combined NAV the exposure
    # ledger uses (_ls_exposure_ledger).  Primary metric = 100% gross (50% long
    # + 50% short, net 0); 200% gross (100% long + 100% short) is reported
    # alongside as ls2x (NAV = long_nav + short_nav - 1).  Both are explicit so
    # two different-leverage books are never compared as one.
    ls_d = _combine_book_daily(long_d, short_d, 0.5, 0.5)
    ls2x_d = _combine_book_daily(long_d, short_d, 1.0, 1.0, subtract=1.0)
    ls_g = _combine_book_daily(long_g, short_g, 0.5, 0.5)
    ls2x_g = _combine_book_daily(long_g, short_g, 1.0, 1.0, subtract=1.0)

    def _met(daily, gross):
        t = torch.tensor(daily, dtype=torch.float32)
        eq = compute_equity_curve(t)
        return {
            "sharpe": compute_sharpe(t, horizon=1),
            "gross_sharpe": compute_sharpe(
                torch.tensor(gross, dtype=torch.float32), horizon=1),
            "sortino": compute_sortino(t, horizon=1),
            "calmar": compute_calmar(t, horizon=1),
            "maxdd": compute_max_drawdown(eq),
            "daily_return_pf": compute_daily_return_profit_factor(t),
        }

    long_m = _met(long_d, long_g)
    ls_m = _met(ls_d, ls_g)
    ls2x_m = _met(ls2x_d, ls2x_g)
    eligible_ew_sharpe = compute_sharpe(
        torch.tensor(ew_a["daily"], dtype=torch.float32), horizon=1)
    selected_universe_ew_sharpe = compute_sharpe(
        torch.tensor(sel_uni_a["daily"], dtype=torch.float32), horizon=1)

    # No forced block_len=horizon — when horizon=1 that would
    # degenerate to iid resampling.  The default block (ceil(n^(1/3))) keeps
    # autocorrelation and volatility clustering in the CI.
    long_lo, long_hi = compute_bootstrap_sharpe_ci(
        long_d, horizon=1, n_boot=n_boot)
    ls_lo, ls_hi = compute_bootstrap_sharpe_ci(
        ls_d, horizon=1, n_boot=n_boot)

    out = {
        "n_periods": int(long_d.size),
        "long_sharpe": long_m["sharpe"],
        "long_gross_sharpe": long_m["gross_sharpe"],
        "long_sharpe_lo": long_lo,
        "long_sharpe_hi": long_hi,
        "long_sortino": long_m["sortino"],
        "long_calmar": long_m["calmar"],
        "long_maxdd": long_m["maxdd"],
        "long_daily_return_pf": long_m["daily_return_pf"],
        "ls_sharpe": ls_m["sharpe"],
        "ls_gross_sharpe": ls_m["gross_sharpe"],
        "ls_sharpe_lo": ls_lo,
        "ls_sharpe_hi": ls_hi,
        "ls_sortino": ls_m["sortino"],
        "ls_calmar": ls_m["calmar"],
        "ls_maxdd": ls_m["maxdd"],
        "ls2x_sharpe": ls2x_m["sharpe"],
        "ls2x_gross_sharpe": ls2x_m["gross_sharpe"],
        "ls2x_maxdd": ls2x_m["maxdd"],
        # The short leg is a theoretical bottom-quantile factor book (A-share
        # stocks cannot be shorted directly) — the exposure metadata keeps the
        # leverage assumption explicit.  The
        # nominal target is only a book construction — the REALIZED daily
        # exposure is measured from the sleeve simulation (unfilled slots stay
        # cash → gross below target; delayed/unresolved sleeves → gross above
        # target), so the metadata reports the measured distribution + full
        # daily ledger instead of a constant.
        "exposure": _ls_exposure_ledger(long_a, short_a),
        # Equal-weight of the ELIGIBLE candidate pool — NOT a
        # full-market/index benchmark.  `selected_universe_ew_sharpe` is the separate
        # selected-universe equal-weight (all fold stocks) proxy.
        "eligible_ew_sharpe": eligible_ew_sharpe,
        "selected_universe_ew_sharpe": selected_universe_ew_sharpe,
        "long_turnover": long_a["turnover"]["daily_avg"],
        "ls_turnover": (long_a["turnover"]["daily_avg"]
                        + short_a["turnover"]["daily_avg"]) / 2.0,
        "ew_turnover": ew_a["turnover"]["daily_avg"],
        "exit_status": long_a["exit_stats"],
        # Per-leg exit status: the long leg alone cannot
        # reveal whether the short book's abnormal returns come from more
        # unfillable or trailing positions — each leg must be auditable on its
        # own.  `exit_status` stays as the long-leg alias for backward compat.
        "long_exit_status": long_a["exit_stats"],
        "short_exit_status": short_a["exit_stats"],
        "eligible_ew_exit_status": ew_a["exit_stats"],
    }
    if return_ledger:
        out["long_ledger"] = long_a.get("ledger")
    return out


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
    val_ds = PanelDataset(val_data, seq_len=config.seq_len,
                          min_history=config.min_history)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    all_preds = []
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_ = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            all_preds.append(pred_ret.cpu().squeeze(-1))

    if not all_preds:
        return _empty_result()

    preds = torch.cat(all_preds)
    n_stocks = val_data["static_features"].shape[0]
    n_windows = val_ds.n_windows
    preds = preds.reshape(n_stocks, n_windows)
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
            return_ledger=return_ledger,
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
            "long_turnover": pm["long_turnover"],
            "ls_turnover": pm["ls_turnover"],
            "ew_turnover": pm["ew_turnover"],
            "exit_status": pm["exit_status"],
            "long_exit_status": pm["long_exit_status"],
            "short_exit_status": pm["short_exit_status"],
            "eligible_ew_exit_status": pm["eligible_ew_exit_status"],
            "long_ledger": pm.get("long_ledger"),
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
        "long_ledger": None,
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
