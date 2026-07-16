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
        # Clamp log_vars to keep exp(-log_var) in fp16 safe range [-10, 12]
        log_vars = torch.clamp(self.log_vars, -10.0, 12.0)
        total = torch.tensor(0.0, device=log_vars.device, dtype=log_vars.dtype)
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-log_vars[i])
            total = total + 0.5 * (precision * loss + log_vars[i])
        return total
