import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate

logger = logging.getLogger(__name__)


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
        return float("inf") if mean > target else 0.0
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
    mean = daily_returns.mean().item()
    periods_per_year = 252 / horizon
    ann_return = mean * periods_per_year
    return float(ann_return / mdd)


def compute_profit_factor(daily_returns: torch.Tensor) -> float:
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
    overlapping-horizon returns (review v3 §十三).  Block length defaults to
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
) -> list[float]:
    """Per-day Spearman rank IC.

    mask_np: (n_stocks, n_windows) bool — candidate pool (entry-eligible) on
    each day. Without it, zero-feature padded rows (short listing history) get
    garbage predictions that drag down the rank correlation.  Days whose
    predictions are constant (std < eps) have a degenerate Spearman and are
    skipped.
    """
    daily_ics = []
    n_windows = preds_np.shape[1]
    for t in range(n_windows):
        p = preds_np[:, t]
        a = actuals_np[:, t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask_np is not None:
            mask = mask & mask_np[:, t]
        if mask.sum() >= 10:
            pv = p[mask]
            if pv.std() < 1e-8:
                continue  # constant predictions → rank correlation undefined
            ic, _ = spearmanr(pv, a[mask])
            if np.isfinite(ic):
                daily_ics.append(ic)
    return daily_ics


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

    Review v3 §十三: the plain mean/std ratio ignores serial correlation —
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
        "profit_factor": result["long_pf"],
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
) -> dict:
    """Multi-angle portfolio evaluation.

    Candidate pool = ENTRY-ELIGIBLE stocks (real open at entry), evaluated on
    carry-last-close realized returns, across ALL `horizon` entry phases.
    Survivorship is avoided because every entry-eligible stock has a tradeable
    outcome (carry/0 fallback), so selection never conditions on the future
    label existing.

    Returns a flat dict with these keys:
      — IC: ic_mean, ic_std, ic_ir, ic_pos_rate
      — Long-only top-K: long_sharpe, long_sharpe_lo, long_sharpe_hi,
          long_sortino, long_calmar, long_maxdd, long_pf
      — Market-neutral long-short: ls_sharpe, ls_sharpe_lo, ls_sharpe_hi,
          ls_sortino, ls_calmar, ls_maxdd
      — Quintile: q1_ret … q5_ret, q5mq1_ret, q_monotonic
      — Equal-weight baseline: ew_sharpe
      — Metadata: n_periods, n_stocks, n_days
    """
    model.eval()
    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
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

    # Realized P&L for evaluation: carry-last-close returns — clean open-to-open
    # where the full window is observed, else the last real close in (t, t+h],
    # else flat 0.  Prefer the pipeline-computed array; fall back to the raw
    # forward-return argument for synthetic/legacy callers.
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

    # Candidate pool: ENTRY-ELIGIBLE (real open at the entry day), NOT "has a
    # future label".  The old y_direction mask conditioned selection on the
    # future outcome existing — a survivorship bias.  With carry-realized
    # returns every entry-eligible stock has a tradeable outcome.  Fallback to
    # y_direction validity for synthetic data without masks.
    if "entry_eligible_mask" in val_data:
        entry = torch.as_tensor(val_data["entry_eligible_mask"])
        pool = entry[:, config.seq_len:config.seq_len + n_windows]
    else:
        y_dir = torch.as_tensor(val_data["y_direction"])
        pool = (y_dir[:, config.seq_len:config.seq_len + n_windows] != -100)
    pool_np = pool.numpy()

    preds_np = preds.numpy()
    actuals_np = actuals.numpy()

    long_rets, short_rets, spread_rets = _build_portfolio_returns(
        preds, actuals, n_windows, horizon, top_fraction, mask=pool,
    )

    n_periods = len(spread_rets)
    if n_periods < 2:
        return _empty_result()

    # ── IC diagnostics (over the entry-eligible pool) ──
    daily_ics = _compute_daily_ic(preds_np, actuals_np, pool_np)
    ic_summary = compute_ic_summary(daily_ics, horizon=horizon)

    # ── Long-only top-K ──
    long_t = torch.tensor(long_rets, dtype=torch.float32)
    long_equity = compute_equity_curve(long_t)
    long_sharpe = compute_sharpe(long_t, horizon=horizon)
    long_lo, long_hi = compute_bootstrap_sharpe_ci(
        np.array(long_rets, dtype=np.float64), horizon=horizon, n_boot=n_boot,
    )

    # ── Market-neutral long-short ──
    ls_t = torch.tensor(spread_rets, dtype=torch.float32)
    ls_equity = compute_equity_curve(ls_t)
    ls_sharpe = compute_sharpe(ls_t, horizon=horizon)
    ls_lo, ls_hi = compute_bootstrap_sharpe_ci(
        np.array(spread_rets, dtype=np.float64), horizon=horizon, n_boot=n_boot,
    )

    # ── Quintile analysis ──
    quintile_metrics = _quintile_analysis(
        preds, actuals, n_windows, horizon, n_stocks, mask=pool,
    )

    # ── Equal-weight baseline (all entry phases) ──
    ew_rets = []
    for offset in range(horizon):
        for t in range(offset, n_windows, horizon):
            col = actuals[:, t]
            keep = pool[:, t] & torch.isfinite(col)
            r = col[keep].mean().item()
            if np.isfinite(r):
                ew_rets.append(r)
    ew_sharpe = compute_sharpe(
        torch.tensor(ew_rets, dtype=torch.float32), horizon=horizon,
    ) if len(ew_rets) >= 2 else 0.0

    return {
        "n_periods": n_periods,
        "n_stocks": n_stocks,
        "n_days": n_windows,
        # IC
        "ic_mean": ic_summary["ic_mean"],
        "ic_std": ic_summary["ic_std"],
        "ic_ir": ic_summary["ic_ir"],
        "ic_pos_rate": ic_summary["ic_pos_rate"],
        "ic_newey_west_t": ic_summary["ic_newey_west_t"],
        # Long-only
        "long_sharpe": long_sharpe,
        "long_sharpe_lo": long_lo,
        "long_sharpe_hi": long_hi,
        "long_sortino": compute_sortino(long_t, horizon=horizon),
        "long_calmar": compute_calmar(long_t, horizon=horizon),
        "long_maxdd": compute_max_drawdown(long_equity),
        "long_pf": compute_profit_factor(long_t),
        # Long-short (market-neutral alpha)
        "ls_sharpe": ls_sharpe,
        "ls_sharpe_lo": ls_lo,
        "ls_sharpe_hi": ls_hi,
        "ls_sortino": compute_sortino(ls_t, horizon=horizon),
        "ls_calmar": compute_calmar(ls_t, horizon=horizon),
        "ls_maxdd": compute_max_drawdown(ls_equity),
        # Quintile
        **quintile_metrics,
        # Benchmark
        "ew_sharpe": ew_sharpe,
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


def _empty_result() -> dict:
    return {
        "n_periods": 0, "n_stocks": 0, "n_days": 0,
        "ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_pos_rate": 0.0,
        "ic_newey_west_t": 0.0,
        "long_sharpe": 0.0, "long_sharpe_lo": float("nan"), "long_sharpe_hi": float("nan"),
        "long_sortino": 0.0, "long_calmar": 0.0, "long_maxdd": 0.0, "long_pf": 0.0,
        "ls_sharpe": 0.0, "ls_sharpe_lo": float("nan"), "ls_sharpe_hi": float("nan"),
        "ls_sortino": 0.0, "ls_calmar": 0.0, "ls_maxdd": 0.0,
        "q1_ret": 0.0, "q2_ret": 0.0, "q3_ret": 0.0, "q4_ret": 0.0, "q5_ret": 0.0,
        "q5mq1_ret": 0.0, "q_monotonic": 0.0,
        "ew_sharpe": 0.0,
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
