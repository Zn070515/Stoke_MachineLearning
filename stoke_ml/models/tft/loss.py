import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting (Kendall et al. 2018).

    Each task i has a learned log-variance parameter log_var_i.
    Total loss = 0.5 * Σ_i (task_loss_i / exp(log_var_i) + log_var_i)

    The log_var regularizer prevents the model from driving σ → ∞ to
    zero out losses. Higher task noise σ → lower weight for that task.

    Args:
        num_tasks: number of tasks (typically 3: CE, MSE_r, MSE_v).
        init_log_var: initial log-variance values (default 0 → σ=1).
    """

    def __init__(self, num_tasks: int = 3, init_log_var: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(
            torch.full((num_tasks,), init_log_var)
        )

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        assert len(task_losses) == self.num_tasks
        log_vars = torch.clamp(self.log_vars, -10.0, 10.0)
        total = torch.tensor(0.0, device=log_vars.device, dtype=log_vars.dtype)
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-log_vars[i])
            total = total + 0.5 * (precision * loss + log_vars[i])
        return total


class AdjMSELoss(nn.Module):
    """Sign-aware MSE — penalises wrong-sign predictions more heavily.

    From ml-quant-trading (Du 2025):
      - Same sign as target:  loss = gamma * (pred - target)^2
      - Wrong sign:           loss = (1 + gamma) * (pred - target)^2

    With gamma=0.1, wrong-sign predictions are penalised 11× more
    than right-sign predictions of equal magnitude. This aligns
    the loss with trading P&L where sign errors cost money.
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = (pred - target) ** 2
        same_sign = (pred * target) > 0
        weight = torch.where(
            same_sign,
            torch.full_like(squared, self.gamma),
            torch.full_like(squared, 1.0 + self.gamma),
        )
        return (squared * weight).mean()
