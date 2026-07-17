"""Transformer encoder for stock time series classification.

Multi-head self-attention over the temporal dimension, followed by
global pooling and a linear classifier. Comparable to the LSTM in
parameter count (~350K) for fair comparison.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learnable parameters)."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])


class TransformerModel(nn.Module):
    """Transformer encoder for binary stock direction classification.

    Projects input features to d_model, adds positional encoding,
    applies num_layers of multi-head self-attention, pools across time,
    and classifies.
    """

    def __init__(
        self,
        input_dim: int = 50,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10,
                                              dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)     # → (batch, seq_len, d_model)
        x = x.transpose(1, 2)   # → (batch, d_model, seq_len) for pool1d
        x = self.pool(x).squeeze(-1)  # → (batch, d_model)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
