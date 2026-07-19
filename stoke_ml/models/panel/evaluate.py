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
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for annualized Sharpe ratio."""
    n = len(returns)
    if n < 5:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    sharpes = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = returns[rng.randint(0, n, size=n)]
        m = sample.mean()
        s = sample.std(ddof=1)
        sharpes[i] = (m / s) * np.sqrt(252 / horizon) if s > 1e-8 else 0.0
    lo = float(np.percentile(sharpes, 2.5))
    hi = float(np.percentile(sharpes, 97.5))
    return lo, hi


def _compute_daily_ic(preds_np: np.ndarray, actuals_np: np.ndarray) -> list[float]:
    """Per-day Spearman rank IC."""
    daily_ics = []
    n_windows = preds_np.shape[1]
    for t in range(n_windows):
        p = preds_np[:, t]
        a = actuals_np[:, t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() >= 10:
            ic, _ = spearmanr(p[mask], a[mask])
            if np.isfinite(ic):
                daily_ics.append(ic)
    return daily_ics


def compute_ic_summary(daily_ics: list[float]) -> dict:
    """IC mean, std, information ratio, and positivity rate."""
    if not daily_ics:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_pos_rate": 0.0}
    arr = np.array(daily_ics, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 1e-8 else 0.0,
        "ic_pos_rate": float((arr > 0).mean()),
    }


def _build_portfolio_returns(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    n_windows: int,
    horizon: int,
    top_k: int,
) -> tuple[list[float], list[float], list[float]]:
    """Build long-only top-K, short bottom-K, and long-short spread returns."""
    n_stocks = preds.shape[0]
    k = min(top_k, max(1, n_stocks // 2))

    long_rets, short_rets, spread_rets = [], [], []
    for t in range(0, n_windows, horizon):
        sorted_idx = torch.argsort(preds[:, t], descending=True)
        top_idx = sorted_idx[:k]
        bot_idx = sorted_idx[-k:]

        long_r = actuals[top_idx, t].mean().item()
        short_r = actuals[bot_idx, t].mean().item()
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
    top_k: int = 20,
    horizon: int = 1,
    return_metrics: bool = False,
    raw_returns: np.ndarray | None = None,
) -> float | tuple[float, dict]:
    """Time-varying top-K long-only portfolio evaluation (backward-compatible).

    Prefer evaluate_portfolio() for the full multi-angle report.
    """
    result = evaluate_portfolio(
        model, val_data, config, device,
        top_k=top_k, horizon=horizon,
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
    top_k: int = 20,
    horizon: int = 5,
    raw_returns: np.ndarray | None = None,
    n_boot: int = 2000,
) -> dict:
    """Multi-angle portfolio evaluation.

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
        for static, pk, po, _, _y_ret, _ in val_loader:
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

    if raw_returns is not None:
        actuals = _build_raw_actuals(raw_returns, n_stocks, n_windows, config.seq_len)
    else:
        logger.warning(
            "evaluate_portfolio called without raw_returns — "
            "financial metrics computed from z-scored targets "
            "are NOT interpretable. Pass raw_returns= for real metrics."
        )
        actuals = preds  # fallback: z-scored, financial metrics invalid

    preds_np = preds.numpy()
    actuals_np = actuals.numpy()

    k = min(top_k, max(1, n_stocks // 2))
    long_rets, short_rets, spread_rets = _build_portfolio_returns(
        preds, actuals, n_windows, horizon, k,
    )

    n_periods = len(spread_rets)
    if n_periods < 2:
        return _empty_result()

    # ── IC diagnostics ──
    daily_ics = _compute_daily_ic(preds_np, actuals_np)
    ic_summary = compute_ic_summary(daily_ics)

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
    quintile_metrics = _quintile_analysis(preds, actuals, n_windows, horizon, n_stocks)

    # ── Equal-weight baseline ──
    ew_rets = []
    for t in range(0, n_windows, horizon):
        col = actuals[:, t]
        r = col[torch.isfinite(col)].mean().item()
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
) -> dict:
    """Group stocks into 5 equal-sized quintiles each day, track mean return.

    Q1 = lowest predicted, Q5 = highest predicted.
    A healthy signal shows monotonic increase from Q1→Q5.
    """
    q_rets = {1: [], 2: [], 3: [], 4: [], 5: []}
    q_size = max(1, n_stocks // 5)

    for t in range(0, n_windows, horizon):
        sorted_idx = torch.argsort(preds[:, t], descending=False)
        for qi, q_start in enumerate([0, q_size, 2*q_size, 3*q_size, 4*q_size], 1):
            q_idx = sorted_idx[q_start:q_start + q_size]
            q_r = actuals[q_idx, t].mean().item()
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
    raw_returns: np.ndarray,
    n_stocks: int,
    n_windows: int,
    seq_len: int,
) -> torch.Tensor:
    end = min(seq_len + n_windows, raw_returns.shape[1])
    n_valid = end - seq_len
    actuals = np.zeros((n_stocks, n_windows), dtype=np.float32)
    actuals[:, :n_valid] = raw_returns[:, seq_len:end]
    return torch.from_numpy(actuals)
