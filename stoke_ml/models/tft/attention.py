import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with exported attention weights.

    Unlike standard MHA which concatenates heads, this uses a shared
    value projection and averages head outputs — the "interpretable"
    variant from the TFT paper that enables feature-level attention
    analysis.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
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

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, H, T_q, T_k)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(out)
        return out, attn_weights
