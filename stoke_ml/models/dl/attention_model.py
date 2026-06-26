"""Lightweight attention model for stock time-series classification.

Single-layer self-attention + learned temporal pooling — far fewer
parameters than a full TransformerEncoder, designed for small per-fold
training sets (~100-200 samples).
"""
import torch
import torch.nn as nn


class SimpleAttentionModel(nn.Module):
    """Single self-attention layer with learned temporal query pooling."""

    def __init__(
        self,
        input_dim: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.input_proj = nn.Linear(input_dim, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Temporal attention pooling via learned query
        q = self.query.expand(x.size(0), -1, -1)  # (batch, 1, d_model)
        scale = x.size(-1) ** 0.5
        weights = torch.softmax(torch.bmm(q, x.transpose(1, 2)) / scale, dim=-1)
        x = torch.bmm(weights, x).squeeze(1)  # (batch, d_model)
        return self.fc(self.dropout(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)[:, 1]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs > 0.5).long()
