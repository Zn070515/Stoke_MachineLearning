import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate


def compute_sharpe(daily_returns: torch.Tensor, annualize: bool = True) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = daily_returns.mean().item()
    std = daily_returns.std().item()
    if std < 1e-8:
        return 0.0  # flat returns → zero Sharpe, not inf
    sharpe = mean / std
    if annualize:
        sharpe *= np.sqrt(252)
    # Clamp to reasonable range for early-stopping compatibility
    return float(max(min(sharpe, 10.0), -10.0))


def evaluate_sharpe(
    model: nn.Module,
    val_data: dict,
    config: TFTConfig,
    device: torch.device,
    top_k: int = 20,
    return_metrics: bool = False,
) -> float | tuple[float, dict]:
    """Time-varying top-K portfolio evaluation.

    For each validation day, ranks stocks by predicted return, selects
    top-K, and computes equal-weight portfolio return. This correctly
    measures a daily-rebalanced strategy rather than a static ranking.
    """
    model.eval()
    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for static, pk, po, _, y_ret, _ in val_loader:
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            all_preds.append(pred_ret.cpu().squeeze(-1))
            all_actuals.append(y_ret.cpu())

    if not all_preds:
        return 0.0

    preds = torch.cat(all_preds)
    actuals = torch.cat(all_actuals)
    n_stocks = val_data["static_features"].shape[0]
    n_windows = val_ds.n_windows

    # Reshape to (n_stocks, n_windows) for per-day stock ranking
    preds = preds.reshape(n_stocks, n_windows)
    actuals = actuals.reshape(n_stocks, n_windows)

    k = min(top_k, n_stocks)
    if k == 0:
        return 0.0

    portfolio_returns = []
    for t in range(n_windows):
        _, top_idx = torch.topk(preds[:, t], k)
        ret = actuals[top_idx, t].mean().item()
        if np.isfinite(ret):
            portfolio_returns.append(ret)

    if not portfolio_returns:
        if return_metrics:
            return 0.0, {"ic": 0.0}
        return 0.0
    portfolio_daily = torch.tensor(portfolio_returns, dtype=torch.float32)
    sharpe = compute_sharpe(portfolio_daily)

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

    return sharpe, {"ic": mean_ic}
