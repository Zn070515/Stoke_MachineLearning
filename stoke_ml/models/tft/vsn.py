import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from stoke_ml.models.tft.components import GRN


class VariableSelectionNetwork(nn.Module):
    """Memory-efficient feature selection via softmax-weighted embeddings.

    For scalar features (input_dim=1), uses a lightweight weight-net
    that avoids creating (B*T*N, hidden_dim) intermediate tensors.
    Feature values modulate learned per-feature embeddings via element-wise
    multiplication, then a softmax gate selects across features.

    Input:  (B, T, N_features, input_dim)
    Output: (B, T, hidden_dim), (B, T, N_features) weights
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
        self.input_dim = input_dim
        weight_hidden = max(hidden_dim // 4, 8)

        if input_dim == 1:
            # Scalar path — memory efficient
            # Weight net: per-feature learnable basis for value→weight mapping
            self.weight_basis = nn.Parameter(
                torch.randn(num_features, weight_hidden) * 0.02
            )
            self.weight_bias = nn.Parameter(torch.zeros(num_features, weight_hidden))
            self.weight_out = nn.Linear(weight_hidden, 1)
            # Feature embeddings: each scalar feature → hidden_dim via embedding * value
            self.feat_emb = nn.Parameter(
                torch.randn(num_features, hidden_dim) * 0.02
            )
        else:
            # Dense path for non-scalar inputs
            self.feat_fc = nn.Linear(input_dim, hidden_dim)
            self.weight_fc1 = nn.Linear(input_dim, weight_hidden)
            self.weight_fc2 = nn.Linear(weight_hidden, 1)

        self.output_grn = GRN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = x.shape

        if self.input_dim == 1:
            return self._forward_scalar(x, context, B, T, N)
        else:
            return self._forward_dense(x, context, B, T, N, D)

    def _forward_scalar(self, x, context, B, T, N):
        """Efficient path for D=1: no (B*T*N, hidden) intermediates."""
        x_val = x.squeeze(-1)  # (B, T, N)

        # Weight: map each scalar through per-feature basis
        w = x_val.unsqueeze(-1) * self.weight_basis  # (B,T,N, wh)
        w = w + self.weight_bias
        w = F.relu(w)
        w = self.weight_out(w).squeeze(-1)  # (B, T, N)
        weights = F.softmax(w, dim=-1)  # (B, T, N)

        # Feature representations: (B,T,N,H) via broadcasting
        features = x_val.unsqueeze(-1) * self.feat_emb  # (B,T,N,H)

        # Weighted sum → (B,T,H)
        combined = (features * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, H)

        # Output GRN
        combined_flat = combined.reshape(B * T, -1)
        if context is not None:
            ctx_flat = context.reshape(B * T, -1)
        else:
            ctx_flat = None
        output = self.output_grn(combined_flat, context=ctx_flat)
        return output.reshape(B, T, -1), weights

    def _forward_dense(self, x, context, B, T, N, D):
        """Fallback for multi-dim features — only used by static VSN."""
        x_flat = x.reshape(B * T * N, D)

        w = self.weight_fc2(F.relu(self.weight_fc1(x_flat)))  # (B*T*N, 1)
        w = w.reshape(B * T, N)
        weights = F.softmax(w, dim=-1)

        features = self.feat_fc(x_flat)  # (B*T*N, H)
        features = features.reshape(B * T, N, -1)

        combined = (features * weights.unsqueeze(-1)).sum(dim=1)  # (B*T, H)

        if context is not None:
            ctx_flat = context.reshape(B * T, -1)
        else:
            ctx_flat = None
        output = self.output_grn(combined, context=ctx_flat)
        return output.reshape(B, T, -1), weights.reshape(B, T, N)
