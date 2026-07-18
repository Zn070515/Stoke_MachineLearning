import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedLinearUnit(nn.Module):
    """GLU: output = Linear(x) * sigmoid(Gate(x))."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GRN(nn.Module):
    """Gated Residual Network — TFT's core nonlinear block.

    Args:
        input_dim: input feature dimension.
        hidden_dim: intermediate dimension (ELU + GLU operate here).
        output_dim: output dimension.
        dropout: dropout rate applied twice in the block.
        context_dim: optional context vector dimension (injected after first dense).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.has_context = context_dim is not None
        if self.has_context:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.residual = input_dim == output_dim
        if not self.residual:
            self.skip = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        eta1 = self.fc1(x)
        if self.has_context and context is not None:
            eta1 = eta1 + self.context_fc(context)
        eta1 = F.elu(eta1)
        eta1 = self.dropout(eta1)
        eta2 = self.fc2(eta1)
        eta2 = self.dropout(eta2)
        gated = self.gate(eta1)
        output = gated * eta2
        if self.residual:
            output = output + x
        else:
            output = output + self.skip(x)
        return self.layer_norm(output)


class GateAddNorm(nn.Module):
    """GLU gating + residual add + LayerNorm.

    Used in the VSN+xLSTM architecture post-xLSTM and optionally
    post-static-enrichment.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.glu(x)
        out = self.dropout(out)
        return self.layer_norm(out + skip)


