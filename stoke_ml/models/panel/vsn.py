import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from stoke_ml.models.panel.components import GRN


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

        # Scale initialization so the weighted sum over N features has
        # roughly unit variance regardless of N_features.
        #
        # With softmax weights ≈ 1/N at init and z-scored features x_i ~ N(0,1):
        #   combined_j = Σ_i (x_i * e_{i,j} * w_i)
        #   std(combined_j) = std(embedding) / sqrt(N_features)
        # → std(embedding) = target_std * sqrt(N_features)
        #
        # Target 0.3 for unit-ish input to downstream GRN LayerNorm.
        target_vsn_std = 0.3
        emb_std = target_vsn_std * (num_features ** 0.5) if num_features > 0 else 0.3

        if input_dim == 1:
            # Scalar path — memory efficient
            self.weight_basis = nn.Parameter(
                torch.randn(num_features, weight_hidden) * emb_std
            )
            self.weight_bias = nn.Parameter(
                torch.zeros(num_features, weight_hidden)
            )
            self.weight_out = nn.Linear(weight_hidden, 1)
            self.feat_emb = nn.Parameter(
                torch.randn(num_features, hidden_dim) * emb_std
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
        """Memory-efficient path for D=1.

        Avoids materializing the full (B,T,N,H) tensor by chunking over
        the feature dimension N.  When N is large (e.g. 200+ features),
        the naive outer product would need B*T*N*H*4 bytes — easily >20 GB
        on a 24 GB GPU.
        """
        x_val = x.squeeze(-1)  # (B, T, N)

        # Weight: map each scalar through per-feature basis
        w = x_val.unsqueeze(-1) * self.weight_basis  # (B,T,N, wh)
        w = w + self.weight_bias
        w = F.relu(w)
        w = self.weight_out(w).squeeze(-1)  # (B, T, N)
        weights = F.softmax(w, dim=-1)  # (B, T, N)

        # Chunked feature combination — each chunk adds its contribution
        # to the weighted sum directly, so we never hold all (B,T,N,H).
        H = self.feat_emb.shape[1]
        combined = torch.zeros(B, T, H, device=x.device, dtype=x.dtype)

        # Size each chunk so its intermediate fits within ~1 GB.
        # (B * T * C * H * 4) ≤ 1 GB  →  C ≤ 1GB / (B*T*H*4)
        max_chunk_bytes = 512 * 1024 ** 2  # 512 MB (safer under fragmentation)
        elem_bytes = 4
        max_chunk = max(1, max_chunk_bytes // (B * T * H * elem_bytes))
        chunk_size = min(max_chunk, 64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_c = x_val[:, :, start:end]                     # (B, T, C)
            e_c = self.feat_emb[start:end]                    # (C, H)
            f_c = x_c.unsqueeze(-1) * e_c                     # (B, T, C, H)
            w_c = weights[:, :, start:end]                    # (B, T, C)
            combined += (f_c * w_c.unsqueeze(-1)).sum(dim=2)  # (B, T, H)

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
