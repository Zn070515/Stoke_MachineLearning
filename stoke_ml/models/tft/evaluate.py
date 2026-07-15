import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate


def compute_sharpe(daily_returns: torch.Tensor, annualize: bool = True) -> float:
    """Compute Sharpe ratio from daily returns.

    Args:
        daily_returns: (T,) tensor of daily portfolio returns.
        annualize: if True, multiply by sqrt(252).

    Returns:
        float Sharpe ratio.
    """
    if len(daily_returns) < 2:
        return 0.0
    mean = daily_returns.mean().item()
    std = daily_returns.std().item()
    if std < 1e-8:
        return 0.0 if abs(mean) < 1e-8 else (float("inf") if mean > 0 else float("-inf"))
    sharpe = mean / std
    if annualize:
        sharpe *= np.sqrt(252)
    return sharpe


def evaluate_sharpe(
    model: nn.Module,
    val_data: dict,
    config: TFTConfig,
    device: torch.device,
    top_k: int = 20,
) -> float:
    """Evaluate model by top-K portfolio Sharpe on validation set.

    1. Predict expected return for all stocks in val set.
    2. Sort by expected return, select top-K.
    3. Simulate equal-weight portfolio, compute Sharpe.
    """
    model.eval()
    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    all_returns = []
    sample_idx = 0
    with torch.no_grad():
        for static, pk, po, _, _, _ in val_loader:
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            all_returns.append(pred_ret.cpu())
            sample_idx += static.shape[0]

    if not all_returns:
        return 0.0

    all_returns = torch.cat(all_returns).squeeze(-1)

    # Aggregate: mean predicted return per stock
    n_stocks = val_data["static_features"].shape[0]
    n_windows = val_ds.n_windows
    stock_returns = torch.full((n_stocks,), -float("inf"))

    for i in range(min(len(all_returns), n_stocks * n_windows)):
        s = i // n_windows
        if s < n_stocks:
            val = all_returns[i].item()
            if stock_returns[s] == -float("inf"):
                stock_returns[s] = val
            else:
                stock_returns[s] = (stock_returns[s] + val) / 2  # running avg

    # Select top-K stocks
    k = min(top_k, n_stocks)
    _, top_indices = torch.topk(stock_returns, k)

    # Simulate portfolio using actual returns from val data
    actual_returns = val_data["y_return"]  # (N, T)
    top_returns = actual_returns[top_indices.numpy()]
    eval_window = min(50, top_returns.shape[1] - config.seq_len)
    t_start = top_returns.shape[1] - eval_window
    portfolio_daily = top_returns[:, t_start:].mean(axis=0)

    sharpe = compute_sharpe(torch.from_numpy(portfolio_daily.astype(np.float32)))
    return sharpe
