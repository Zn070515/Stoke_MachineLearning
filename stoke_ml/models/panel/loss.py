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

    def forward(
        self,
        task_losses: list[torch.Tensor],
        task_active_mask: list[bool] | None = None,
    ) -> torch.Tensor:
        assert len(task_losses) == self.num_tasks
        if task_active_mask is None:
            task_active_mask = [True] * self.num_tasks
        log_vars = torch.clamp(self.log_vars, -2.0, 10.0)
        total = torch.tensor(0.0, device=log_vars.device, dtype=log_vars.dtype)
        for i, loss in enumerate(task_losses):
            # A task with no labels in this batch must not
            # contribute even its log_var regularizer — doing so pushes
            # inactive weights toward the clamp floor and distorts the
            # weights of batches where that task IS active.
            if not task_active_mask[i]:
                continue
            precision = torch.exp(-log_vars[i])
            total = total + 0.5 * (precision * loss + log_vars[i])
        return total


class FixedTaskWeights(nn.Module):
    """Fixed equal-weight multi-task loss — the UncertaintyLoss ablation.

    Kendall-style learned log-variances are one weighting scheme; fixing all
    active tasks to equal weight tests whether the learned re-weighting is
    where performance comes from (§十一.3).  Carries NO learnable parameters
    (nothing in the optimizer's loss group), and matches UncertaintyLoss's
    ``forward(losses, task_active_mask)`` signature so train.py swaps one for
    the other without branching.
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(
        self,
        task_losses: list[torch.Tensor],
        task_active_mask: list[bool] | None = None,
    ) -> torch.Tensor:
        assert len(task_losses) == self.num_tasks
        if task_active_mask is None:
            task_active_mask = [True] * self.num_tasks
        active = [l for l, a in zip(task_losses, task_active_mask) if a]
        if not active:
            return torch.zeros(
                (), device=task_losses[0].device, dtype=task_losses[0].dtype)
        # Equal weight: the mean over active tasks, not their sum, so the
        # combined loss scale is independent of how many tasks are enabled.
        return torch.stack(active).sum() / len(active)


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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        squared = (pred - target) ** 2
        same_sign = (pred * target) >= 0
        weight = torch.where(
            same_sign,
            torch.full_like(squared, self.gamma),
            torch.full_like(squared, 1.0 + self.gamma),
        )
        elem = squared * weight
        if reduction == "mean":
            return elem.mean()
        if reduction == "sum":
            return elem.sum()
        if reduction == "none":
            return elem
        raise ValueError(f"unsupported reduction: {reduction!r}")


class PairwiseRankingLoss(nn.Module):
    """Differentiable pairwise ranking loss for cross-sectional ordering.

    For each pair of stocks (i,j) on the SAME date, the loss penalises
    predictions whose relative ordering disagrees with the actual returns:

        loss = mean_{i,j} max(0, margin - sign(ret_i - ret_j) * (pred_i - pred_j))

    This is a hinge-loss variant of RankNet — it directly optimises for
    the ranking that IC and long-short Sharpe evaluate on.

    Date-centric contract (§七/§十六): ``date_idx`` maps each sample to its
    date position so pairwise comparisons are only computed within the same
    date group.  With the production DataLoader (``batch_size=1`` +
    ``DateSampler``) a batch IS one calendar date's complete cross-section — or
    a ``max_stocks_per_date``-capped random sample of it — so ALL intra-date
    pairs are present and none are lost to batch boundaries (the old
    stock-centric ``DateGroupedSampler`` problem).  With ``batch_size>1``
    (mixed-date batches via ``panel_collate`` concatenation, or a future
    gradient-accumulation scheme) ``date_idx`` still groups same-date stocks,
    so pairs only ever form within a date.  When a date is cap-sampled, pairs
    form over the sampled subset — a deliberate tradeoff (fewer pairs per
    batch, bounded batch size).

    Per-date prediction normalization is scale-invariant within each date:
    predictions are re-scaled by their own date's std before the pairwise
    differences, so multiplying one date's predictions by a positive constant
    leaves that date's hinge term unchanged.

    Gradient accumulation: each micro-batch is one date, and the per-date loss
    is already normalized by its own pair count, so summing ``l_rank`` with
    unit weight per micro-batch (train.py) weights every date equally
    regardless of how many stocks/pairs it held.

    Temperature τ controls the soft-sign steepness for gradient flow.
    """

    def __init__(
        self,
        margin: float = 0.0,
        tau: float = 1.0,
        spread_target: float = 1.0,
        spread_weight: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.tau = tau
        self.spread_target = spread_target
        self.spread_weight = spread_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        date_idx: torch.Tensor,
        stats: list[dict] | None = None,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Args:
            pred: (B,) predicted returns.
            target: (B,) actual returns.
            mask: (B,) valid-position mask (1.0 = valid, 0.0 = ignore).
            date_idx: (B,) integer date index for same-date grouping.
            stats: optional list — when given, appended a dict with
                {n_dates, stocks_per_date, n_pairs} so the caller can detect
                when ranking signal rests on very few pairs.
        """
        B = pred.shape[0]
        if B < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        valid = mask > 0.5
        if valid.sum() < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        # Guard against non-finite predictions: a padded/zero window can
        # overflow the heads under fp16 AMP, and one Inf/NaN must not poison
        # the whole batch (NaN * 0 == NaN propagates through the masked sum).
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Predictions are normalized PER DATE — not over the mixed batch — and
        # each date's pairwise loss is weighted by its own pair count.  With
        # batch_size=1 (production) this loop runs once per date batch; the
        # per-date grouping also handles batch_size>1 mixed-date batches.
        unique_dates = torch.unique(date_idx[valid])
        hinge_sum = torch.zeros((), device=pred.device, dtype=pred.dtype)
        spread_sum = torch.zeros((), device=pred.device, dtype=pred.dtype)
        pair_count = 0
        stocks_per_date: list[int] = []
        for d in unique_dates:
            idx = (date_idx == d) & valid
            n = int(idx.sum().item())
            if n < 2:
                continue
            stocks_per_date.append(n)
            p = pred[idx]
            t = target[idx]

            # Scale-invariant pairwise differences within THIS date.
            pred_std = p.std() + 1e-8
            pd = (p.unsqueeze(0) - p.unsqueeze(1)) / pred_std  # pd[i,j]
            td = t.unsqueeze(0) - t.unsqueeze(1)             # td[i,j]

            # Soft sign for gradient flow: sign(td) ≈ tanh(td / τ)
            sign_td = torch.tanh(td / (self.tau + 1e-8))

            # Hinge: max(0, margin - sign(td) * pd), upper-triangular pairs.
            pair_loss = F.relu(self.margin - sign_td * pd)
            triu = torch.triu(torch.ones(n, n, device=pred.device), diagonal=1).bool()
            n_pairs = int(triu.sum().item())
            hinge_sum = hinge_sum + (pair_loss * triu).sum()

            # Spread-preservation penalty: the scale-invariant normalization
            # has a trivial minimum at constant predictions — the model can
            # shrink |pred| toward 0 and the margin=0 hinge still vanishes
            # (pd ≈ 0), so rank IC collapses while the scalar losses keep
            # falling.  Threshold tied to this date's target_std rather than a
            # fixed 1.0 because cross-sectional z-scored targets have std ~1
            # but sparse-date raw returns (std ~0.02) are left unnormalized.
            target_std = t.std() + 1e-8
            spread = F.relu(self.spread_target * target_std - pred_std)
            spread_sum = spread_sum + spread * n_pairs
            pair_count += n_pairs

        if pair_count == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        if stats is not None:
            stats.append({
                "n_dates": int(len(unique_dates)),
                "stocks_per_date": stocks_per_date,
                "n_pairs": pair_count,
            })

        return hinge_sum / pair_count + self.spread_weight * (spread_sum / pair_count)
