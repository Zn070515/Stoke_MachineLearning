import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InterpretableMultiHeadAttention(nn.Module):
    """TFT interpretable multi-head attention.

    Per the TFT paper: each head has its own Q and K projections, but
    V is shared across all heads. Head outputs are averaged (not
    concatenated), then projected back to d_model. This enables
    feature-level attention analysis by inspecting per-head weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        # Shared V: project to head_dim and broadcast across heads,
        # not split into per-head slices like standard MHA.
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(self.head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Per-head Q, K; shared V broadcast across heads
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).unsqueeze(1)  # (B, 1, T_k, head_dim) — shared

        scale = self.head_dim ** 0.5
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, H, T_q, T_k)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, T_q, head_dim), v broadcasts to all heads
        out = out.mean(dim=1)   # (B, T_q, head_dim) — average across heads
        out = self.out_proj(out)
        return out, attn_weights
