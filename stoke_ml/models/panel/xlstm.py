"""sLSTM and mLSTM blocks for time-series prediction.

Based on Beck et al. "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517).
Lightweight implementation tailored for short financial sequences (60 steps),
with batch_first convention matching the rest of the codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalConv1d(nn.Module):
    """Causal 1D convolution (left-padded so position t only sees ≤t)."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for Conv1d
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.pad, 0))  # causal: pad left only
        x_t = self.conv(x_t)
        return x_t.transpose(1, 2)  # back to (B, T, D)


class sLSTMBlock(nn.Module):
    """Scalar LSTM block with exponential gating and memory mixing.

    Multiple heads process the input in parallel chunks, then
    outputs are concatenated and projected back.  Sequential
    (recurrent) — fine for our 60-step sequences.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.conv = CausalConv1d(hidden_dim, conv_kernel)
        self.dropout = nn.Dropout(dropout)
        # Input + recurrent projections per head — combined for efficiency
        in_dim = hidden_dim * 2  # x_t + h_{t-1} concatenated
        self.W = nn.Linear(in_dim, num_heads * self.head_dim * 4)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # GroupNorm for stabilization after each step
        self.norm = nn.GroupNorm(num_heads, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            x: (B, T, D) input sequence.
            state: optional (h, c, n, m) for truncated BPTT.  If None,
                   initialized to zeros.

        Returns:
            output: (B, T, D) hidden states.
            state: (h_last, c_last, n_last, m_last) for continuation.
        """
        B, T, D = x.shape
        device = x.device

        # Causal conv pre-processing
        x = self.conv(x)
        x = self.dropout(x)

        if state is not None:
            h, c, n, m = state
        else:
            h = torch.zeros(B, self.num_heads, self.head_dim, device=device)
            c = torch.zeros(B, self.num_heads, self.head_dim, device=device)
            n = torch.zeros(B, self.num_heads, self.head_dim, device=device)
            m = torch.zeros(B, self.num_heads, self.head_dim, device=device)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (B, D)
            h_flat = h.reshape(B, D)  # (B, D)
            W_in = self.W(torch.cat([x_t, h_flat], dim=-1))  # (B, 4*D)
            W_in = W_in.reshape(B, self.num_heads, 4 * self.head_dim)

            i_raw, f_raw, z, o = W_in.chunk(4, dim=-1)  # each (B, H, d_h)
            # Exponential gating with stabilization
            i = torch.exp(self._stabilize(i_raw, m))
            f = torch.sigmoid(f_raw)
            z = torch.tanh(z)
            o = torch.sigmoid(o)

            # Cell state update
            c = f * c + i * z
            # Normalizer: per-element forget + input accumulation
            n = f * n + i
            # Stabilizer: track max raw logit (log-space), detached from graph
            m = torch.max(m, i_raw).detach()

            h = o * (c / n.clamp(min=1e-8))
            outputs.append(h.reshape(B, 1, D))

        out = torch.cat(outputs, dim=1)  # (B, T, D)
        out = self.out_proj(out)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out, (h, c, n, m)

    @staticmethod
    def _stabilize(log_gate: torch.Tensor, stabilizer: torch.Tensor) -> torch.Tensor:
        return log_gate - stabilizer.detach()


class mLSTMBlock(nn.Module):
    """Matrix LSTM block with covariance memory update.

    Fully parallelizable — uses parallel scan over the sequence
    dimension.  Best for capturing global patterns across the
    full 60-step window.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        qk_dim: int = 32,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_dim = qk_dim

        self.conv = CausalConv1d(hidden_dim, conv_kernel)
        self.dropout = nn.Dropout(dropout)
        # input → (query, key, value) for each head
        self.q_proj = nn.Linear(hidden_dim, num_heads * qk_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * qk_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # Forget + input gates
        self.gate_proj = nn.Linear(hidden_dim, num_heads * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.GroupNorm(num_heads, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward pass via associative scan.

        Args:
            x: (B, T, D) input sequence.

        Returns:
            output: (B, T, D) hidden states.
        """
        B, T, D = x.shape

        x = self.conv(x)
        x = self.dropout(x)

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.qk_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.qk_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        # Forget (f) and input (i) gates
        gates = self.gate_proj(x).reshape(B, T, self.num_heads, 2)
        f = torch.sigmoid(gates[..., 0])   # (B, T, H)
        i = torch.sigmoid(gates[..., 1])   # (B, T, H)

        # Parallel scan: C_t = f_t * C_{t-1} + i_t * v_t * k_t^T
        # Implemented iteratively for T=60 (parallel scan overkill for short seqs)
        C = torch.zeros(B, self.num_heads, self.head_dim, self.qk_dim, device=x.device)
        n = torch.zeros(B, self.num_heads, device=x.device)  # per-head scalar
        outputs = []

        for t in range(T):
            f_t = f[:, t, :]  # (B, H)
            i_t = i[:, t, :]  # (B, H)
            v_t = v[:, t, :, :]  # (B, H, d_h)
            k_t = k[:, t, :, :]  # (B, H, d_k)

            # Outer product update: v_t ⊗ k_t^T = (B,H,d_h,d_k)
            update = torch.einsum("bhd,bhk->bhdk", v_t, k_t)
            C = (f_t.unsqueeze(-1).unsqueeze(-1) * C
                 + i_t.unsqueeze(-1).unsqueeze(-1) * update)
            n = f_t * n + i_t  # (B, H)

            # Query: h_t = C_t @ q_t / (n_t + eps)
            q_val = torch.einsum("bhdk,bhk->bhd", C, q[:, t, :, :])
            h_t = q_val / (n.unsqueeze(-1) + 1e-8)
            outputs.append(h_t.reshape(B, 1, D))

        out = torch.cat(outputs, dim=1)  # (B, T, D)
        out = self.out_proj(out)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class xLSTMBackbone(nn.Module):
    """Mixed sLSTM/mLSTM backbone for time-series feature extraction.

    Stacks alternating sLSTM and mLSTM blocks according to a signature.
    For financial data (low SNR, short sequences), we recommend sLSTM-heavy.

    Args:
        hidden_dim: feature dimension throughout the backbone.
        num_blocks: total number of xLSTM blocks.
        slstm_ratio: fraction of blocks that are sLSTM (rest are mLSTM).
        num_heads: heads per block.
        dropout: dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        slstm_ratio: float = 0.67,
        num_heads: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        blocks = []
        for i in range(num_blocks):
            is_slstm = i < int(num_blocks * slstm_ratio)
            if is_slstm:
                blocks.append(sLSTMBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    conv_kernel=4,
                    dropout=dropout,
                ))
            else:
                blocks.append(mLSTMBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    conv_kernel=4,
                    dropout=dropout,
                ))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through all blocks.

        Args:
            x: (B, T, D) input.
            states: optional list of sLSTM states for truncated BPTT.

        Returns:
            output: (B, T, D).
            new_states: list of (h, c, n, m) for each sLSTM block (None for mLSTM).
        """
        new_states = []
        s_idx = 0
        for block in self.blocks:
            if isinstance(block, sLSTMBlock):
                st = states[s_idx] if states is not None else None
                x, st = block(x, st)
                new_states.append(st)
                s_idx += 1
            else:
                x = block(x)
                new_states.append(None)
        return x, new_states
