import torch
import torch.nn as nn
import torch.nn.functional as F


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
        log_vars = torch.clamp(self.log_vars, -2.0, 10.0)
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
        same_sign = (pred * target) >= 0
        weight = torch.where(
            same_sign,
            torch.full_like(squared, self.gamma),
            torch.full_like(squared, 1.0 + self.gamma),
        )
        return (squared * weight).mean()


class PairwiseRankingLoss(nn.Module):
    """Differentiable pairwise ranking loss for cross-sectional ordering.

    For each pair of stocks (i,j) on the SAME date, the loss penalises
    predictions whose relative ordering disagrees with the actual returns:

        loss = mean_{i,j} max(0, margin - sign(ret_i - ret_j) * (pred_i - pred_j))

    This is a hinge-loss variant of RankNet — it directly optimises for
    the ranking that IC and long-short Sharpe evaluate on.

    A `date_idx` tensor maps each sample to its date position so that
    pairwise comparisons are only computed within the same date group.

    Temperature τ controls the soft-sign steepness for gradient flow.
    """

    def __init__(self, margin: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.margin = margin
        self.tau = tau

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        date_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Args:
            pred: (B,) predicted returns.
            target: (B,) actual returns.
            mask: (B,) valid-position mask (1.0 = valid, 0.0 = ignore).
            date_idx: (B,) integer date index for same-date grouping.
        """
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        valid = mask > 0.5
        if valid.sum() < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Same-date mask: (B, B) boolean — True where dates match
        same_date = date_idx.unsqueeze(0) == date_idx.unsqueeze(1)  # (B, B)
        same_date = same_date & valid.unsqueeze(0) & valid.unsqueeze(1)

        # Upper triangular (avoid double-counting and self-pairs)
        triu = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1).bool()
        eligible = same_date & triu

        n_pairs = eligible.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Scale-invariant pairwise differences: normalize predictions
        # within the batch so the loss magnitude doesn't depend on
        # model output scale (which is tiny at initialization).
        pred_std = pred[valid].std() + 1e-8
        pd = (pred.unsqueeze(0) - pred.unsqueeze(1)) / pred_std    # pd[i,j]
        td = (target.unsqueeze(0) - target.unsqueeze(1))           # td[i,j]

        # Soft sign for gradient flow: sign(td) ≈ tanh(td / τ)
        sign_td = torch.tanh(td / (self.tau + 1e-8))

        # Hinge: max(0, margin - sign(td) * pd), masked element-wise.
        # Element-wise masking avoids boolean advanced-indexing which can
        # trigger illegal-memory-access with CUDA AMP on some drivers.
        pair_loss = F.relu(self.margin - sign_td * pd)
        eligible_f = eligible.float()
        return (pair_loss * eligible_f).sum() / eligible_f.sum().clamp(min=1)
