import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionHead(nn.Module):
    """3-class direction classifier with bottleneck + ELU for gradient stability.

    Head dropout higher than backbone dropout — output layers are the primary
    overfitting site in financial deep learning models (gradient collapse research, 2024).
    """

    def __init__(self, hidden_dim: int, num_classes: int = 3, dropout: float = 0.35):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        # Extreme small init: prevent early-epoch gradient explosion on output layer
        nn.init.normal_(self.fc1.weight, std=1e-4)
        nn.init.normal_(self.fc2.weight, std=1e-4)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = F.elu(self.fc1(self.dropout(last)))
        return self.fc2(self.dropout(last))


class ReturnHead(nn.Module):
    """Return % regressor with bottleneck for gradient stability.

    Predicts next-day return as a scalar.  Bottleneck layer + ELU gives
    the head enough nonlinearity to decode complex temporal representations
    without the extreme gradient variance of a single linear layer.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.35):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        nn.init.normal_(self.fc1.weight, std=1e-4)
        nn.init.normal_(self.fc2.weight, std=1e-4)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = F.elu(self.fc1(self.dropout(last)))
        return self.fc2(self.dropout(last))


class VolatilityHead(nn.Module):
    """Volatility regressor with bottleneck.  Softplus enforces positivity.

    Bias initialized near softplus⁻¹(0.02) so early predictions ≈ typical
    daily vol, keeping the uncertainty-loss MSE term stable.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.35):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        nn.init.normal_(self.fc1.weight, std=1e-4)
        nn.init.normal_(self.fc2.weight, std=1e-4)
        nn.init.zeros_(self.fc1.bias)
        # Bias ≈ softplus^{-1}(0.02) so initial pred ≈ 0.02 (typical daily vol)
        nn.init.constant_(self.fc2.bias, -3.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = F.elu(self.fc1(self.dropout(last)))
        raw = self.fc2(self.dropout(last))
        return F.softplus(raw) + 1e-6
