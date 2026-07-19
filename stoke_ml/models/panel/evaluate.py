import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate


def compute_sharpe(
    daily_returns: torch.Tensor,
    annualize: bool = True,
    horizon: int = 1,
) -> float:
    """Sharpe ratio from non-overlapping period returns.

    Args:
        daily_returns: period returns (each covers `horizon` trading days).
        annualize: multiply by sqrt(periods_per_year) if True.
        horizon: number of trading days each return covers.
    """
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
    """Sortino ratio — uses only downside deviation (returns < target)."""
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
    """Maximum drawdown from peak — worst peak-to-trough decline (as positive fraction)."""
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
    """Calmar ratio = annualized return / max drawdown."""
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
    """Profit factor = gross profit / gross loss (absolute value)."""
    profits = daily_returns[daily_returns > 0].sum().item()
    losses = abs(daily_returns[daily_returns < 0].sum().item())
    if losses < 1e-8:
        return float("inf") if profits > 0 else 0.0
    return float(profits / losses)


def compute_equity_curve(
    daily_returns: torch.Tensor,
    initial_capital: float = 1.0,
) -> torch.Tensor:
    """Cumulative equity curve from period returns."""
    return torch.cat([torch.tensor([initial_capital]), 1.0 + daily_returns]).cumprod(0)


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
    """Time-varying top-K portfolio evaluation.

    For each validation day (subsampled by horizon to avoid overlap),
    ranks stocks by predicted return, selects top-K, and computes
    equal-weight portfolio return using non-overlapping forward returns.

    Args:
        raw_returns: (N_stocks, T_total) raw forward returns in percent.
            If provided, used for Sharpe/IC computation instead of the
            z-scored returns in val_data.  Without this, Sharpe is
            computed on normalised returns and is NOT a valid financial
            metric — only useful for model comparison, not P&L estimation.
    """
    model.eval()
    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    all_preds = []
    all_actuals = [] if raw_returns is None else None
    with torch.no_grad():
        for static, pk, po, _, y_ret, _ in val_loader:
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            all_preds.append(pred_ret.cpu().squeeze(-1))
            if all_actuals is not None:
                all_actuals.append(y_ret.cpu())

    if not all_preds:
        return 0.0

    preds = torch.cat(all_preds)
    n_stocks = val_data["static_features"].shape[0]
    n_windows = val_ds.n_windows
    preds = preds.reshape(n_stocks, n_windows)

    # Use raw returns for financial metrics when available; otherwise
    # the z-scored targets produce a Sharpe that is only meaningful
    # for model comparison, not strategy P&L.
    if raw_returns is not None:
        actuals = _build_raw_actuals(raw_returns, n_stocks, n_windows, config.seq_len)
    else:
        actuals = torch.cat(all_actuals).reshape(n_stocks, n_windows)

    k = min(top_k, n_stocks)
    if k == 0:
        return 0.0

    portfolio_returns = []
    for t in range(0, n_windows, horizon):
        _, top_idx = torch.topk(preds[:, t], k)
        ret = actuals[top_idx, t].mean().item()
        if np.isfinite(ret):
            portfolio_returns.append(ret)

    if not portfolio_returns:
        if return_metrics:
            return 0.0, {"ic": 0.0}
        return 0.0
    portfolio_daily = torch.tensor(portfolio_returns, dtype=torch.float32)
    sharpe = compute_sharpe(portfolio_daily, horizon=horizon)

    if not return_metrics:
        return sharpe

    # Compute Spearman rank IC (cross-sectional, per-day, then average)
    daily_ics = []
    for t in range(n_windows):
        p = preds[:, t].numpy()
        a = actuals[:, t].numpy()
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() >= 10:
            ic, _ = spearmanr(p[mask], a[mask])
            if np.isfinite(ic):
                daily_ics.append(ic)
    mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

    # Full financial metrics panel
    p_ret = torch.tensor(portfolio_returns, dtype=torch.float32)
    equity = compute_equity_curve(p_ret)
    metrics = {
        "sharpe": sharpe,
        "sortino": compute_sortino(p_ret, horizon=horizon),
        "calmar": compute_calmar(p_ret, horizon=horizon),
        "max_drawdown": compute_max_drawdown(equity),
        "profit_factor": compute_profit_factor(p_ret),
        "ic": mean_ic,
        "n_periods": len(portfolio_returns),
    }
    return sharpe, metrics


def compute_prediction_diversity(predictions: np.ndarray) -> float:
    """Prediction diversity: std(preds) / (|mean(preds)| + 1e-8).

    FinFusion (2024) found this metric positively correlated with directional
    accuracy while val_loss is anti-correlated (r=-0.46).  Low diversity
    (< 0.1) after epoch 5 signals gradient collapse — the model is
    producing near-constant predictions regardless of input.

    Args:
        predictions: 1-D array of predicted values (returns or logits).

    Returns:
        diversity ratio (higher = more diverse predictions, generally better).
    """
    return float(np.std(predictions) / (abs(np.mean(predictions)) + 1e-8))


def _build_raw_actuals(
    raw_returns: np.ndarray,
    n_stocks: int,
    n_windows: int,
    seq_len: int,
) -> torch.Tensor:
    """Build (n_stocks, n_windows) matrix of raw forward returns.

    PanelDataset returns (stock i, window w) → y_return at position
    w + seq_len.  Vectorized slice instead of per-window Python loop.
    """
    end = min(seq_len + n_windows, raw_returns.shape[1])
    n_valid = end - seq_len
    actuals = np.zeros((n_stocks, n_windows), dtype=np.float32)
    actuals[:, :n_valid] = raw_returns[:, seq_len:end]
    return torch.from_numpy(actuals)
