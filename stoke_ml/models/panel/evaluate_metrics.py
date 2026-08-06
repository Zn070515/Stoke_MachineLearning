"""Pure risk/return metrics for the panel evaluator (§二十一).

Extracted from ``stoke_ml.models.panel.evaluate`` — the standalone Sharpe /
Sortino / max-drawdown / Calmar / daily profit-factor / equity-curve /
bootstrap-CI helpers.  ``evaluate`` re-exports these for backward
compatibility.
"""
import numpy as np
import torch


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
