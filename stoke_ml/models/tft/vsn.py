import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from stoke_ml.models.tft.components import GRN


class VariableSelectionNetwork(nn.Module):
    """Per-timestep feature selection via softmax-weighted GRN.

    Input shape: (B, T, N_features, input_dim) where N_features is the number
    of input variables and input_dim is the dimension of each variable.

    Each variable is independently scored for importance via a shared weight
    network, then softmax-gated across variables, weighted-sum combined, and
    passed through a final GRN.

    Args:
        input_dim: dimension of each input variable.
        hidden_dim: GRN hidden/output dimension.
        num_features: number of input variables to select from.
        dropout: dropout rate.
        context_dim: optional context vector dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_features: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_features = num_features

        # Shared feature transformation: each variable → hidden_dim
        self.feature_grn = GRN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            context_dim=context_dim,
        )

        # Scalar weight per variable
        self.weight_fc = nn.Linear(input_dim, hidden_dim)
        self.weight_out = nn.Linear(hidden_dim, 1)

        # Post-selection GRN
        self.output_grn = GRN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # Flatten batch and time for per-variable processing
        x_flat = x.reshape(B * T * N, D)  # (B*T*N, D)

        # Compute scalar importance per variable
        w = self.weight_out(F.relu(self.weight_fc(x_flat)))  # (B*T*N, 1)
        w = w.reshape(B * T, N)  # (B*T, N)
        weights = F.softmax(w, dim=-1)  # (B*T, N)

        # Transform each variable to hidden_dim
        if context is not None:
            ctx_flat = context.reshape(B * T, 1, -1).expand(-1, N, -1).reshape(B * T * N, -1)
        else:
            ctx_flat = None
        features = self.feature_grn(x_flat, context=ctx_flat)  # (B*T*N, hidden_dim)
        features = features.reshape(B * T, N, -1)  # (B*T, N, hidden_dim)

        # Weighted sum across variables
        combined = (features * weights.unsqueeze(-1)).sum(dim=1)  # (B*T, hidden_dim)

        # Final GRN
        output = self.output_grn(combined)  # (B*T, hidden_dim)
        output = output.reshape(B, T, -1)  # (B, T, hidden_dim)
        weights = weights.reshape(B, T, N)  # (B, T, N)

        return output, weights
