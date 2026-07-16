import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionHead(nn.Module):
    """Binary direction classifier: takes last timestep, outputs logits."""

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — take last timestep
        last = x[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class ReturnHead(nn.Module):
    """Return % regressor: predicts next-day return as a scalar.

    Initialized with small weights so that early predictions are near-zero
    (~daily-return scale), preventing the MSE term in the uncertainty loss
    from dominating at the start of training.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc.weight, std=1e-4)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class VolatilityHead(nn.Module):
    """Volatility regressor: softplus gate ensures positive output.

    Initialized with small weights so early predictions are near the
    typical daily-vol range (0.01–0.05), keeping the uncertainty-loss
    MSE term stable.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc.weight, std=1e-4)
        # Bias ≈ softplus^{-1}(0.02) so initial pred ≈ 0.02 (typical daily vol)
        nn.init.constant_(self.fc.bias, -3.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = self.dropout(last)
        raw = self.fc(last)
        return F.softplus(raw) + 1e-6  # strictly positive
