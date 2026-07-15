# TFT Panel Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a panel-trained TFT model (~20M params) for 3-task stock prediction (direction/return/volatility) with 252-day sequences on 798 A-share stocks, optimized for RTX 4090 24GB.

**Architecture:** TFT with 3 VSNs (static/past-known/past-observed) → 2-layer LSTM encoder → 4-head interpretable attention → 3 GRN layers → 3 output heads. Panel Dataset feeds same-date stocks in batches. CrossSectionNormalizer does per-date Z-score. Uncertainty Weighting auto-balances multi-task loss. Purged walk-forward with top-20 portfolio Sharpe as early-stop criterion.

**Tech Stack:** PyTorch 2.x, torch.compile, AMP fp16, AdamW, CosineAnnealing, pandas, numpy. Existing code preserved; new components are additive in `stoke_ml/models/tft/`.

---

## File Structure

```
Create:
  stoke_ml/models/tft/__init__.py
  stoke_ml/models/tft/config.py              # TFTConfig dataclass
  stoke_ml/models/tft/components.py          # GRN, GLU, TimeDistributed
  stoke_ml/models/tft/vsn.py                 # VariableSelectionNetwork
  stoke_ml/models/tft/attention.py           # InterpretableMultiHeadAttention
  stoke_ml/models/tft/heads.py               # DirectionHead, ReturnHead, VolatilityHead
  stoke_ml/models/tft/model.py               # TFTModel assembly
  stoke_ml/models/tft/loss.py                # UncertaintyLoss
  stoke_ml/models/tft/dataset.py             # PanelDataset, collate_fn
  stoke_ml/models/tft/train.py               # Training loop
  stoke_ml/models/tft/evaluate.py            # Top-K portfolio simulation
  scripts/train_tft.py                       # CLI entry point

Modify:
  stoke_ml/preprocessing/numeric/cross_section.py   # Add CrossSectionNormalizer
  stoke_ml/features/pipeline.py                     # Add build_panel_features()

Test:
  tests/models/tft/test_components.py
  tests/models/tft/test_vsn.py
  tests/models/tft/test_heads.py
  tests/models/tft/test_model.py
  tests/models/tft/test_loss.py
  tests/models/tft/test_dataset.py
  tests/models/tft/test_evaluate.py
```

---

### Task 1: TFTConfig Dataclass

**Files:**
- Create: `stoke_ml/models/tft/__init__.py`
- Create: `stoke_ml/models/tft/config.py`

- [ ] **Step 1: Create TFTConfig with all hyperparameters**

```python
# stoke_ml/models/tft/config.py
from dataclasses import dataclass, field


@dataclass
class TFTConfig:
    """TFT model hyperparameters. ~20M params with defaults below."""

    # Input dimensions
    static_dim: int = 30
    past_known_dim: int = 250
    past_observed_dim: int = 120

    # Core
    hidden_dim: int = 256
    lstm_layers: int = 2
    attention_heads: int = 4
    grn_layers: int = 3
    dropout: float = 0.15

    # Training
    batch_size: int = 512
    grad_accum_steps: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    max_epochs: int = 100

    # Sequence
    seq_len: int = 252

    # Output
    num_direction_classes: int = 2

    # Hardware
    use_amp: bool = True
    compile_model: bool = True
    num_workers: int = 8
```

- [ ] **Step 2: Create __init__.py with imports**

```python
# stoke_ml/models/tft/__init__.py
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.model import TFTModel
from stoke_ml.models.tft.loss import UncertaintyLoss
from stoke_ml.models.tft.dataset import PanelDataset
```

- [ ] **Step 3: Verify import works**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "from stoke_ml.models.tft import TFTConfig; c = TFTConfig(); print(f'hidden={c.hidden_dim}, params~{c.hidden_dim*c.hidden_dim*4//1000000}M')"`
Expected: `hidden=256, params~1M` (config only, not full model)

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/models/tft/__init__.py stoke_ml/models/tft/config.py
git commit -m "feat: add TFTConfig dataclass with all hyperparameters"
```

---

### Task 2: GRN + GLU + TimeDistributed Building Blocks

**Files:**
- Create: `stoke_ml/models/tft/components.py`
- Create: `tests/models/tft/test_components.py`

TFT's core is the Gated Residual Network. GRN uses ELU activation, GLU gating, optional context vector injection, and residual skip connection.

- [ ] **Step 1: Write failing tests for GRN and GLU**

```python
# tests/models/tft/test_components.py
import torch
import pytest
from stoke_ml.models.tft.components import GatedLinearUnit, GRN, TimeDistributed


class TestGLU:
    def test_output_shape(self):
        glu = GatedLinearUnit(input_dim=16, output_dim=32)
        x = torch.randn(4, 8, 16)  # (B, T, D)
        out = glu(x)
        assert out.shape == (4, 8, 32)

    def test_values_in_range(self):
        glu = GatedLinearUnit(input_dim=8, output_dim=8)
        x = torch.randn(2, 5, 8)
        out = glu(x)
        # GLU applies sigmoid gate, output should not explode
        assert out.abs().max() < 50.0


class TestGRN:
    def test_no_context_output_shape(self):
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32)
        x = torch.randn(4, 16, 32)
        out = grn(x)
        assert out.shape == (4, 16, 32)

    def test_with_context(self):
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32, context_dim=16)
        x = torch.randn(4, 16, 32)
        ctx = torch.randn(4, 16, 16)
        out = grn(x, context=ctx)
        assert out.shape == (4, 16, 32)

    def test_optional_context(self):
        """GRN without context_dim should accept context=None."""
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32)
        x = torch.randn(4, 16, 32)
        out = grn(x)  # no context arg
        assert out.shape == (4, 16, 32)

    def test_residual_skip(self):
        """When input_dim == output_dim, residual skip should be active."""
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32, dropout=0.0)
        x = torch.randn(2, 4, 32)
        # With dropout=0 and identity init, output ≈ input when weights are small
        out = grn(x)
        assert out.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_components.py -v`
Expected: FAIL — ModuleNotFoundError or ImportError

- [ ] **Step 3: Implement GLU and GRN**

```python
# stoke_ml/models/tft/components.py
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

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        eta1 = F.elu(self.fc1(x))
        eta1 = self.dropout(eta1)
        if self.has_context and context is not None:
            eta1 = eta1 + self.context_fc(context)
        eta2 = self.fc2(eta1)
        eta2 = self.dropout(eta2)
        gated = self.gate(eta1)
        output = gated * eta2
        if self.residual:
            output = output + x
        else:
            output = output + self.skip(x)
        return output


class TimeDistributed(nn.Module):
    """Apply a module independently to each time step.

    Useful for static variable tiling: apply static context to every timestep.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        out = self.module(x_flat)
        out = out.reshape(B, T, -1)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_components.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/components.py tests/models/tft/test_components.py
git commit -m "feat: add GRN, GLU, TimeDistributed — TFT building blocks"
```

---

### Task 3: Variable Selection Network

**Files:**
- Create: `stoke_ml/models/tft/vsn.py`
- Create: `tests/models/tft/test_vsn.py`

VSN uses GRN + softmax over input features to select which features to attend to at each time step. Each feature gets a learned weight; the weighted sum is fed through a final GRN.

- [ ] **Step 1: Write failing test for VSN**

```python
# tests/models/tft/test_vsn.py
import torch
from stoke_ml.models.tft.vsn import VariableSelectionNetwork


class TestVSN:
    def test_output_shape(self):
        vsn = VariableSelectionNetwork(
            input_dim=16, hidden_dim=32, num_features=8
        )
        x = torch.randn(4, 60, 8, 16)  # (B, T, N_features, D)
        out, weights = vsn(x)
        assert out.shape == (4, 60, 32)
        assert weights.shape == (4, 60, 8)

    def test_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(
            input_dim=8, hidden_dim=16, num_features=5
        )
        x = torch.randn(2, 10, 5, 8)
        _, weights = vsn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_single_feature(self):
        """Edge case: single feature should work without errors."""
        vsn = VariableSelectionNetwork(
            input_dim=16, hidden_dim=32, num_features=1
        )
        x = torch.randn(2, 5, 1, 16)
        out, weights = vsn(x)
        assert out.shape == (2, 5, 32)
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_with_context(self):
        vsn = VariableSelectionNetwork(
            input_dim=16, hidden_dim=32, num_features=8, context_dim=24
        )
        x = torch.randn(4, 60, 8, 16)
        ctx = torch.randn(4, 60, 24)
        out, weights = vsn(x, context=ctx)
        assert out.shape == (4, 60, 32)
        assert weights.shape == (4, 60, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_vsn.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VSN**

```python
# stoke_ml/models/tft/vsn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from stoke_ml.models.tft.components import GRN


class VariableSelectionNetwork(nn.Module):
    """Per-timestep feature selection via softmax-weighted GRN.

    Input shape: (B, T, N_features, input_dim)
    Each of the N_features gets a learned weight via a shared GRN.
    Softmax across features → weighted sum → final GRN.

    Args:
        input_dim: dimension of each input feature.
        hidden_dim: GRN hidden dimension.
        num_features: number of input features to select from.
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
        # Weight network: processes each feature independently (flattened across features)
        self.weight_grn = GRN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
            context_dim=context_dim,
        )
        # Post-selection GRN: processes weighted feature sum
        self.selection_grn = GRN(
            input_dim=input_dim,
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

        # Flatten B,T together for GRN processing
        x_flat = x.reshape(B * T, N, D)

        # Compute per-feature weight via GRN
        # weight_grn processes each (N*D) row → need to broadcast correctly
        # Flatten features to treat each one independently
        w_flat = x_flat.reshape(B * T * N, D)
        if context is not None:
            ctx_flat = context.reshape(B * T, 1, context.shape[-1])
            ctx_flat = ctx_flat.expand(-1, N, -1).reshape(B * T * N, -1)
        else:
            ctx_flat = None
        w = self.weight_grn(w_flat, context=ctx_flat)
        w = w.reshape(B * T, N, N)  # GRN outputs N values per feature row
        # Take mean across the last dim to get scalar weight per feature
        w = w.mean(dim=-1)  # (B*T, N)
        weights = F.softmax(w, dim=-1)  # (B*T, N)

        # Weighted sum of features
        x_flat_sum = x_flat  # (B*T, N, D)
        weights_expanded = weights.unsqueeze(-1)  # (B*T, N, 1)
        selected = (x_flat_sum * weights_expanded).sum(dim=1)  # (B*T, D)

        # Final GRN
        selected = selected.reshape(B * T, D)
        output = self.selection_grn(selected)

        output = output.reshape(B, T, -1)
        weights = weights.reshape(B, T, N)
        return output, weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_vsn.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/vsn.py tests/models/tft/test_vsn.py
git commit -m "feat: add VariableSelectionNetwork — softmax feature gating"
```

---

### Task 4: Interpretable Multi-Head Attention

**Files:**
- Create: `stoke_ml/models/tft/attention.py`
- Create: `tests/models/tft/test_attention.py` (add to existing or separate)

Standard MHA but with attention weight export for interpretability.

- [ ] **Step 1: Write failing test**

```python
# tests/models/tft/test_attention.py
import torch
from stoke_ml.models.tft.attention import InterpretableMultiHeadAttention


class TestInterpretableMHA:
    def test_output_shape(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(2, 20, 64)
        k = torch.randn(2, 20, 64)
        v = torch.randn(2, 20, 64)
        out, attn = mha(q, k, v)
        assert out.shape == (2, 20, 64)
        assert attn.shape == (2, 4, 20, 20)

    def test_attention_weights_sum_to_one(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(1, 10, 64)
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)
        _, attn = mha(q, k, v)
        # Sum over key dimension should be 1
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mask_support(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(1, 5, 64)
        k = torch.randn(1, 5, 64)
        v = torch.randn(1, 5, 64)
        mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        out, _ = mha(q, k, v, attn_mask=mask)
        assert out.shape == (1, 5, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_attention.py -v`
Expected: FAIL

- [ ] **Step 3: Implement InterpretableMultiHeadAttention**

```python
# stoke_ml/models/tft/attention.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_attention.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/attention.py tests/models/tft/test_attention.py
git commit -m "feat: add InterpretableMultiHeadAttention with weight export"
```

---

### Task 5: Output Heads

**Files:**
- Create: `stoke_ml/models/tft/heads.py`
- Create: `tests/models/tft/test_heads.py`

Three independent heads: DirectionHead (2-class logits), ReturnHead (regression), VolatilityHead (regression with softplus).

- [ ] **Step 1: Write failing tests**

```python
# tests/models/tft/test_heads.py
import torch
from stoke_ml.models.tft.heads import DirectionHead, ReturnHead, VolatilityHead


class TestDirectionHead:
    def test_output_shape(self):
        head = DirectionHead(hidden_dim=128, num_classes=2)
        x = torch.randn(4, 20, 128)
        out = head(x)
        assert out.shape == (4, 2)  # (B, num_classes)

    def test_output_are_logits(self):
        head = DirectionHead(hidden_dim=128, num_classes=2)
        x = torch.randn(4, 20, 128)
        out = head(x)
        # Logits can be any real value
        assert out.dtype == torch.float32


class TestReturnHead:
    def test_output_shape(self):
        head = ReturnHead(hidden_dim=128)
        x = torch.randn(4, 20, 128)
        out = head(x)
        assert out.shape == (4, 1)  # (B, 1)

    def test_values_are_reasonable(self):
        head = ReturnHead(hidden_dim=128)
        x = torch.randn(100, 20, 128)
        out = head(x)
        # Daily returns should be in a reasonable range
        assert out.abs().mean() < 0.5  # not 100%


class TestVolatilityHead:
    def test_output_shape(self):
        head = VolatilityHead(hidden_dim=128)
        x = torch.randn(4, 20, 128)
        out = head(x)
        assert out.shape == (4, 1)

    def test_output_positive(self):
        head = VolatilityHead(hidden_dim=128)
        x = torch.randn(4, 20, 128)
        out = head(x)
        assert (out >= 0).all(), f"Volatility must be positive, got {out.min()}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_heads.py -v`
Expected: FAIL

- [ ] **Step 3: Implement three heads**

```python
# stoke_ml/models/tft/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionHead(nn.Module):
    """Binary direction classifier: takes last timestep, outputs logits."""

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — take last timestep
        last = x[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class ReturnHead(nn.Module):
    """Return % regressor: predicts next-day return as a scalar."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class VolatilityHead(nn.Module):
    """Volatility regressor: softplus gate ensures positive output."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        last = self.dropout(last)
        raw = self.fc(last)
        return F.softplus(raw) + 1e-6  # strictly positive
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_heads.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/heads.py tests/models/tft/test_heads.py
git commit -m "feat: add DirectionHead, ReturnHead, VolatilityHead"
```

---

### Task 6: TFTModel Assembly

**Files:**
- Create: `stoke_ml/models/tft/model.py`
- Create: `tests/models/tft/test_model.py`

Assemble all components: 3 VSNs → LSTM Encoder → MHA → GRN stack → 3 heads.

- [ ] **Step 1: Write failing test for TFTModel forward pass**

```python
# tests/models/tft/test_model.py
import torch
from stoke_ml.models.tft import TFTConfig
from stoke_ml.models.tft.model import TFTModel


class TestTFTModel:
    @classmethod
    def setup_class(cls):
        cls.config = TFTConfig(
            static_dim=8,
            past_known_dim=24,
            past_observed_dim=12,
            hidden_dim=64,
            lstm_layers=1,
            attention_heads=2,
            grn_layers=2,
            seq_len=60,
        )
        cls.model = TFTModel(cls.config)

    def test_forward_outputs(self):
        B, T = 4, 60
        static = torch.randn(B, self.config.static_dim)
        past_known = torch.randn(B, T, self.config.past_known_dim)
        past_obs = torch.randn(B, T, self.config.past_observed_dim)

        direction, ret, vol = self.model(static, past_known, past_obs)

        assert direction.shape == (B, 2)
        assert ret.shape == (B, 1)
        assert vol.shape == (B, 1)
        assert (vol >= 0).all()

    def test_batch_independence(self):
        """Same input twice should give same output."""
        static = torch.randn(2, self.config.static_dim)
        pk = torch.randn(2, 60, self.config.past_known_dim)
        po = torch.randn(2, 60, self.config.past_observed_dim)

        self.model.eval()
        with torch.no_grad():
            d1, r1, v1 = self.model(static, pk, po)
            d2, r2, v2 = self.model(static, pk, po)

        assert torch.allclose(d1, d2, atol=1e-5)
        assert torch.allclose(r1, r2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)

    def test_param_count_in_range(self):
        total = sum(p.numel() for p in self.model.parameters())
        # Small test config should be < 1M
        assert total < 1_000_000, f"Expected <1M params, got {total:,}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TFTModel**

```python
# stoke_ml/models/tft/model.py
import torch
import torch.nn as nn
from typing import Optional
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.components import GRN
from stoke_ml.models.tft.vsn import VariableSelectionNetwork
from stoke_ml.models.tft.attention import InterpretableMultiHeadAttention
from stoke_ml.models.tft.heads import DirectionHead, ReturnHead, VolatilityHead


class TFTModel(nn.Module):
    """Temporal Fusion Transformer for panel stock prediction.

    Input: static features (B, S), past_known (B, T, P), past_observed (B, T, O)
    Output: direction logits (B, 2), return % (B, 1), volatility (B, 1)
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # ── Variable Selection Networks (×3) ──
        self.vsn_static = VariableSelectionNetwork(
            input_dim=config.static_dim, hidden_dim=h,
            num_features=1, dropout=config.dropout,
        ) if config.static_dim > 0 else None

        self.vsn_past = VariableSelectionNetwork(
            input_dim=config.past_known_dim, hidden_dim=h,
            num_features=config.past_known_dim, dropout=config.dropout,
        )

        self.vsn_obs = VariableSelectionNetwork(
            input_dim=config.past_observed_dim, hidden_dim=h,
            num_features=config.past_observed_dim, dropout=config.dropout,
        )

        # ── LSTM Encoder ──
        lstm_input_dim = h  # VSN output
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )

        # ── Static enrichment ──
        if config.static_dim > 0:
            self.static_enrich_grn = GRN(
                input_dim=h, hidden_dim=h, output_dim=h,
                dropout=config.dropout, context_dim=h,
            )
        else:
            self.static_enrich_grn = None

        # ── Multi-Head Attention ──
        self.attention = InterpretableMultiHeadAttention(
            d_model=h, n_heads=config.attention_heads, dropout=config.dropout,
        )

        # ── Post-attention GRN ──
        self.post_attn_grn = GRN(
            input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout,
        )

        # ── Decoder GRN stack ──
        self.decoder_grns = nn.ModuleList([
            GRN(input_dim=h, hidden_dim=h, output_dim=h, dropout=config.dropout)
            for _ in range(config.grn_layers)
        ])

        # ── Output heads ──
        self.direction_head = DirectionHead(h, config.num_direction_classes, config.dropout)
        self.return_head = ReturnHead(h, config.dropout)
        self.volatility_head = VolatilityHead(h, config.dropout)

    def forward(
        self,
        static_features: torch.Tensor,
        past_known: torch.Tensor,
        past_observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = past_known.shape

        # ── VSN: feature selection ──
        # past_known: (B, T, D_p) → treat each feature dim as a "variable"
        pk = past_known.unsqueeze(2)  # (B, T, 1, D_p) → fake single variable
        # Actually, TFT VSN treats each feature as a variable:
        pk = past_known.unsqueeze(-2).expand(-1, -1, past_known.shape[-1], -1)
        # Hmm, that's not right. Let me re-think.
        # Each "variable" in VSN is a single scalar per timestep.
        # We need to reshape: (B, T, D_p) → (B, T, D_p, 1)
        pk_vars = past_known.unsqueeze(-1)  # (B, T, D_p, 1)
        po_vars = past_observed.unsqueeze(-1)  # (B, T, D_o, 1)

        past_selected, _ = self.vsn_past(pk_vars)  # (B, T, h)
        obs_selected, _ = self.vsn_obs(po_vars)  # (B, T, h)

        # Combine past + observed
        lstm_input = past_selected + obs_selected  # (B, T, h)

        # ── LSTM Encoder ──
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, h)

        # ── Static enrichment ──
        if self.static_enrich_grn is not None and static_features is not None:
            # VSN on static features
            s_vars = static_features.unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
            static_selected, _ = self.vsn_static(s_vars)  # (B, 1, h)
            static_tiled = static_selected.expand(-1, T, -1)  # (B, T, h)
            lstm_out = self.static_enrich_grn(lstm_out, context=static_tiled)

        # ── Multi-Head Attention ──
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # self-attention

        # ── Post-attention GRN ──
        attn_out = self.post_attn_grn(attn_out)

        # ── Decoder GRN stack ──
        decoder_out = attn_out
        for grn in self.decoder_grns:
            decoder_out = grn(decoder_out)

        # ── Output heads ──
        direction = self.direction_head(decoder_out)
        return_pct = self.return_head(decoder_out)
        volatility = self.volatility_head(decoder_out)

        return direction, return_pct, volatility
```

- [ ] **Step 4: Run tests, fix any shape mismatches, retest**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_model.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/model.py tests/models/tft/test_model.py
git commit -m "feat: assemble TFTModel — VSN×3 + LSTM + MHA + GRN + 3 heads"
```

---

### Task 7: Uncertainty Loss

**Files:**
- Create: `stoke_ml/models/tft/loss.py`
- Create: `tests/models/tft/test_loss.py`

Kendall's uncertainty weighting: each task learns a log-variance parameter. Loss = Σ task_loss / (2σ²) + log(σ).

- [ ] **Step 1: Write failing test**

```python
# tests/models/tft/test_loss.py
import torch
from stoke_ml.models.tft.loss import UncertaintyLoss


class TestUncertaintyLoss:
    def test_output_is_scalar(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        losses = [torch.tensor(0.5), torch.tensor(0.01), torch.tensor(0.02)]
        total = loss_fn(losses)
        assert total.ndim == 0  # scalar

    def test_learnable_params(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        params = list(loss_fn.parameters())
        assert len(params) == 3  # 3 log-variance params

    def test_variance_positive(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        # sigmas = exp(log_var), should be positive
        for p in loss_fn.parameters():
            sigma = torch.exp(p)
            assert sigma > 0

    def test_forward_pass_works(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        # Simulate a training step to verify gradients flow
        losses = [torch.tensor(0.7, requires_grad=False),
                  torch.tensor(0.05, requires_grad=False),
                  torch.tensor(0.03, requires_grad=False)]
        total = loss_fn(losses)
        total.backward()
        for p in loss_fn.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_loss.py -v`
Expected: FAIL

- [ ] **Step 3: Implement UncertaintyLoss**

```python
# stoke_ml/models/tft/loss.py
import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting (Kendall et al. 2018).

    Each task i has a learned log-variance parameter log_var_i.
    Total loss = Σ_i (task_loss_i / (2 * exp(log_var_i)) + 0.5 * log_var_i)

    The 0.5 factor on log_var acts as a regularizer — prevents the model
    from driving σ → ∞ to zero out losses.

    Args:
        num_tasks: number of tasks (typically 3: CE, MSE_r, MSE_v).
        init_log_var: initial log-variance values (default 0 → σ=1).
    """

    def __init__(self, num_tasks: int = 3, init_log_var: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(
            torch.full((num_tasks,), init_log_var)
        )

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss.

        Args:
            task_losses: list of per-task scalar losses, e.g.,
                [CE_loss, MSE_return_loss, MSE_vol_loss]

        Returns:
            scalar total loss with uncertainty weighting.
        """
        assert len(task_losses) == self.num_tasks
        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + 0.5 * self.log_vars[i]
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_loss.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/loss.py tests/models/tft/test_loss.py
git commit -m "feat: add UncertaintyLoss with learned task weights"
```

---

### Task 8: PanelDataset and Collate Function

**Files:**
- Create: `stoke_ml/models/tft/dataset.py`
- Create: `tests/models/tft/test_dataset.py`

Panel dataset that groups same-date stocks into batches, produces (static, past_known, past_observed, y_direction, y_return, y_volatility) tuples.

- [ ] **Step 1: Write failing tests**

```python
# tests/models/tft/test_dataset.py
import torch
import numpy as np
import pandas as pd
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate


def make_synthetic_data(n_stocks=10, n_days=100, seq_len=60):
    """Create synthetic panel data for testing."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    stocks = [f"{i:06d}" for i in range(n_stocks)]
    static = np.random.randn(n_stocks, 8).astype(np.float32)
    past_known = np.random.randn(n_stocks, n_days, 20).astype(np.float32)
    past_obs = np.random.randn(n_stocks, n_days, 12).astype(np.float32)
    y_dir = np.random.randint(0, 2, (n_stocks, n_days)).astype(np.int64)
    y_ret = np.random.randn(n_stocks, n_days).astype(np.float32) * 0.02
    y_vol = np.abs(np.random.randn(n_stocks, n_days).astype(np.float32)) * 0.01
    return {
        "static_features": torch.from_numpy(static),
        "past_known": torch.from_numpy(past_known),
        "past_observed": torch.from_numpy(past_obs),
        "y_direction": torch.from_numpy(y_dir),
        "y_return": torch.from_numpy(y_ret),
        "y_volatility": torch.from_numpy(y_vol),
        "dates": dates,
        "stock_codes": stocks,
    }


class TestPanelDataset:
    def test_len(self):
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        # n_days - seq_len windows per stock × n_stocks
        expected = (100 - 60) * 10
        assert len(ds) == expected

    def test_getitem_shapes(self):
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        static, pk, po, y_dir, y_ret, y_vol = ds[0]
        assert static.shape == (8,)
        assert pk.shape == (60, 20)
        assert po.shape == (60, 12)
        assert y_dir.ndim == 0  # scalar
        assert y_ret.ndim == 0
        assert y_vol.ndim == 0

    def test_collate_fn(self):
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        batch = [ds[i] for i in range(4)]
        static, pk, po, y_dir, y_ret, y_vol = panel_collate(batch)
        assert static.shape == (4, 8)
        assert pk.shape == (4, 60, 20)
        assert po.shape == (4, 60, 12)
        assert y_dir.shape == (4,)
        assert y_ret.shape == (4, 1)
        assert y_vol.shape == (4, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PanelDataset + panel_collate**

```python
# stoke_ml/models/tft/dataset.py
import torch
from torch.utils.data import Dataset
from typing import Optional
import numpy as np


class PanelDataset(Dataset):
    """Panel dataset for TFT training.

    Pre-built tensor data organized as (N_stocks, T_total, D_features).
    Each __getitem__ returns a single stock's sequence window.

    For Panel training, the collate function groups same-date stocks
    across different sequences — see panel_collate().
    """

    def __init__(
        self,
        data: dict,
        seq_len: int = 252,
        stride: int = 1,
    ):
        self.static_features = data["static_features"]  # (N, S)
        self.past_known = data["past_known"]  # (N, T, P)
        self.past_observed = data["past_observed"]  # (N, T, O)
        self.y_direction = data["y_direction"]  # (N, T)
        self.y_return = data["y_return"]  # (N, T)
        self.y_volatility = data["y_volatility"]  # (N, T)
        self.dates = data.get("dates", None)
        self.stock_codes = data.get("stock_codes", None)

        self.seq_len = seq_len
        self.stride = stride
        self.n_stocks = self.past_known.shape[0]
        self.n_timesteps = self.past_known.shape[1]
        self.n_windows = self.n_timesteps - seq_len

        if self.n_windows <= 0:
            raise ValueError(
                f"n_timesteps ({self.n_timesteps}) must be > seq_len ({seq_len})"
            )

    def __len__(self) -> int:
        return self.n_stocks * self.n_windows

    def __getitem__(self, idx: int) -> tuple:
        stock_idx = idx // self.n_windows
        window_idx = idx % self.n_windows

        start = window_idx
        end = start + self.seq_len

        static = self.static_features[stock_idx]  # (S,)
        pk = self.past_known[stock_idx, start:end]  # (T, P)
        po = self.past_observed[stock_idx, start:end]  # (T, O)
        y_dir = self.y_direction[stock_idx, end]  # target at t+1
        y_ret = self.y_return[stock_idx, end]
        y_vol = self.y_volatility[stock_idx, end]

        return (
            static,
            pk,
            po,
            torch.tensor(y_dir, dtype=torch.long),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_vol, dtype=torch.float32),
        )


def panel_collate(batch: list) -> tuple:
    """Collate panel samples into a batch tensor."""
    statics = torch.stack([b[0] for b in batch])
    past_knowns = torch.stack([b[1] for b in batch])
    past_observeds = torch.stack([b[2] for b in batch])
    y_dirs = torch.stack([b[3] for b in batch])
    y_rets = torch.stack([b[4] for b in batch]).unsqueeze(-1)
    y_vols = torch.stack([b[5] for b in batch]).unsqueeze(-1)
    return statics, past_knowns, past_observeds, y_dirs, y_rets, y_vols
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_dataset.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/dataset.py tests/models/tft/test_dataset.py
git commit -m "feat: add PanelDataset and panel_collate for TFT data loading"
```

---

### Task 9: CrossSectionNormalizer + build_panel_features

**Files:**
- Modify: `stoke_ml/preprocessing/numeric/cross_section.py`
- Modify: `stoke_ml/features/pipeline.py`

CrossSectionNormalizer: Z-score per date across all stocks.
build_panel_features: new method on FeaturePipeline that outputs pre-built tensors for PanelDataset.

- [ ] **Step 1: Implement CrossSectionNormalizer**

```python
# Add to stoke_ml/preprocessing/numeric/cross_section.py

class CrossSectionNormalizer:
    """Z-score per date across all stocks in panel data.

    Fit computes μ_day and σ_day for each feature. Transform applies
    (x - μ) / (σ + eps). μ and σ are computed cross-sectionally
    (across stocks) for each date independently.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._daily_mean: dict = {}
        self._daily_std: dict = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "CrossSectionNormalizer":
        """Compute per-date μ, σ across all stocks."""
        if df.empty:
            return self
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
        for dt, grp in df.groupby("_dt"):
            dt_str = dt.strftime("%Y-%m-%d")
            self._daily_mean[dt_str] = grp[feature_cols].mean().to_dict()
            self._daily_std[dt_str] = grp[feature_cols].std().to_dict()
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Apply (x - μ_day) / σ_day per date."""
        if df.empty or not self._daily_mean:
            return df
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in feature_cols:
            if col not in df.columns:
                continue
            # Map per-date stats
            mu = df["_dt"].map(lambda d: self._daily_mean.get(d, {}).get(col, 0.0))
            sigma = df["_dt"].map(lambda d: self._daily_std.get(d, {}).get(col, 1.0))
            sigma = sigma.clip(lower=self.eps)
            df[col] = (df[col] - mu) / sigma
        df.drop(columns=["_dt"], inplace=True)
        return df
```

- [ ] **Step 2: Add build_panel_features() to FeaturePipeline**

Add this method to `stoke_ml/features/pipeline.py` (in the `FeaturePipeline` class):

```python
def build_panel_features(
    self,
    stock_list: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """Build panel-format features for TFT training.

    Returns a dict with pre-built numpy arrays:
        static_features: (N_stocks, static_dim)
        past_known: (N_stocks, T, past_known_dim)
        past_observed: (N_stocks, T, past_observed_dim)
        y_direction: (N_stocks, T)
        y_return: (N_stocks, T)
        y_volatility: (N_stocks, T) — 5-day realized vol
    """
    all_static = []
    all_pk = []
    all_po = []
    all_y_dir = []
    all_y_ret = []
    all_y_vol = []

    for code in stock_list:
        # Build existing features per stock
        df = self._load_and_build_single(code, start_date, end_date)
        if df.empty or len(df) < self.seq_len + 5:
            continue

        df = df.sort_values("date").reset_index(drop=True)

        # Split features into TFT input types
        static = _extract_static_features(df)
        pk = _extract_past_known_features(df)
        po = _extract_past_observed_features(df)

        # Targets
        close = df["close"].values
        ret_1d = np.diff(close) / close[:-1]  # next-day return
        direction = (ret_1d > 0).astype(np.int64)
        # 5-day realized volatility
        ret_5d = close[5:] / close[:-5] - 1
        realized_vol = np.zeros(len(close))
        for i in range(5, len(close)):
            realized_vol[i] = np.std(ret_1d[i-5:i])

        # Align lengths to the shortest
        min_len = min(len(pk), len(po), len(direction), len(ret_1d), len(realized_vol))
        pk = pk[:min_len]
        po = po[:min_len]
        direction = direction[:min_len]
        y_ret = ret_1d[:min_len]
        y_vol = realized_vol[:min_len]

        all_static.append(static)
        all_pk.append(pk)
        all_po.append(po)
        all_y_dir.append(direction)
        all_y_ret.append(y_ret)
        all_y_vol.append(y_vol)

    if not all_pk:
        raise ValueError("No valid stocks produced features")

    # Pad to uniform length
    max_T = max(p.shape[0] for p in all_pk)
    max_T = min(max_T, 3000)  # cap at ~12 years

    N = len(all_pk)
    S_dim = all_static[0].shape[0]
    P_dim = all_pk[0].shape[1]
    O_dim = all_po[0].shape[1]

    static_arr = np.zeros((N, S_dim), dtype=np.float32)
    pk_arr = np.zeros((N, max_T, P_dim), dtype=np.float32)
    po_arr = np.zeros((N, max_T, O_dim), dtype=np.float32)
    y_dir_arr = np.zeros((N, max_T), dtype=np.int64)
    y_ret_arr = np.zeros((N, max_T), dtype=np.float32)
    y_vol_arr = np.zeros((N, max_T), dtype=np.float32)

    for i in range(N):
        T_i = all_pk[i].shape[0]
        static_arr[i] = all_static[i]
        pk_arr[i, :T_i] = all_pk[i]
        po_arr[i, :T_i] = all_po[i]
        y_dir_arr[i, :T_i] = all_y_dir[i]
        y_ret_arr[i, :T_i] = all_y_ret[i]
        y_vol_arr[i, :T_i] = all_y_vol[i]

    return {
        "static_features": static_arr,
        "past_known": pk_arr,
        "past_observed": po_arr,
        "y_direction": y_dir_arr,
        "y_return": y_ret_arr,
        "y_volatility": y_vol_arr,
    }
```

And add the helper functions at module level:

```python
STATIC_FEATURE_COLS = [
    "sector", "market_cap_quantile",
]

PAST_KNOWN_COLS = [
    "open", "high", "low", "close", "volume", "amount",
    "ma_5", "ma_10", "ma_20", "ma_60", "ma_120",
    "ema_12", "ema_26", "macd", "macd_signal", "macd_hist",
    "rsi_6", "rsi_12", "rsi_24",
    "kdj_k", "kdj_d", "kdj_j",
    "boll_pct_b", "atr_14",
    "roc_10", "willr_14", "cci_14", "obv",
    "volume_ratio_5", "volume_ratio_20",
    "day_of_week", "day_of_month", "month", "quarter",
    "days_to_earnings",
    # Fundamental (forward-filled)
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
]

PAST_OBSERVED_COLS = [
    "sentiment_mean", "sentiment_std", "news_count",
    "guba_sentiment_mean", "guba_sentiment_std",
    "xueqiu_sentiment_mean", "xueqiu_sentiment_std",
    "main_net", "margin_net", "north_net_buy",
    "lhb_net_amount", "sector_etf_flow",
    "board_momentum_mean", "board_momentum_max",
    "avg_concept_heat",
]


def _extract_static_features(df: pd.DataFrame) -> np.ndarray:
    """Extract time-invariant features."""
    feats = []
    for col in STATIC_FEATURE_COLS:
        if col in df.columns:
            feats.append(df[col].iloc[0] if not df[col].empty else 0.0)
        else:
            feats.append(0.0)
    # Add concept board multi-hot (L2 from ConceptBlockEncoder)
    cb_cols = [c for c in df.columns if c.startswith("cb_")]
    for c in cb_cols:
        feats.append(df[c].max() if not df[c].empty else 0)
    return np.array(feats, dtype=np.float32)


def _extract_past_known_features(df: pd.DataFrame) -> np.ndarray:
    """Extract known-ahead temporal features."""
    available = [c for c in PAST_KNOWN_COLS if c in df.columns]
    data = df[available].fillna(0.0).values.astype(np.float32)
    return data


def _extract_past_observed_features(df: pd.DataFrame) -> np.ndarray:
    """Extract look-back-only temporal features."""
    available = [c for c in PAST_OBSERVED_COLS if c in df.columns]
    data = df[available].fillna(0.0).values.astype(np.float32)
    return data
```

- [ ] **Step 3: Quick import test**

Run: `PYTHONPATH=. ./.venv/Scripts/python -c "from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer; print('CS normalizer OK')"`

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/preprocessing/numeric/cross_section.py stoke_ml/features/pipeline.py
git commit -m "feat: add CrossSectionNormalizer and build_panel_features()"
```

---

### Task 10: Training Loop

**Files:**
- Create: `stoke_ml/models/tft/train.py`

Full training loop with AMP, gradient accumulation, cosine annealing, uncertainty-weighted loss, and Sharpe-based early stopping.

- [ ] **Step 1: Implement training loop**

```python
# stoke_ml/models/tft/train.py
import logging
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.model import TFTModel
from stoke_ml.models.tft.loss import UncertaintyLoss
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate
from stoke_ml.models.tft.evaluate import evaluate_sharpe

logger = logging.getLogger(__name__)


def train_tft(
    config: TFTConfig,
    train_data: dict,
    val_data: dict,
    device: torch.device,
) -> tuple[TFTModel, dict]:
    """Train TFT model with purged walk-forward fold.

    Returns:
        model: best model (by validation Sharpe).
        history: dict of training metrics per epoch.
    """
    model = TFTModel(config).to(device)
    if config.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    loss_fn = UncertaintyLoss(num_tasks=3).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=config.use_amp)

    train_ds = PanelDataset(train_data, seq_len=config.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=True,
    )

    # Cosine annealing with warmup
    total_steps = config.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    best_sharpe = -float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_sharpe": []}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (static, pk, po, y_dir, y_ret, y_vol) in enumerate(train_loader):
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)

            with autocast(enabled=config.use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                l_ce = ce_loss(pred_dir, y_dir)
                l_ret = mse_loss(pred_ret.squeeze(-1), y_ret.squeeze(-1))
                l_vol = mse_loss(pred_vol.squeeze(-1), y_vol.squeeze(-1))
                total_loss = loss_fn([l_ce, l_ret, l_vol])

            total_loss = total_loss / config.grad_accum_steps
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += total_loss.item() * config.grad_accum_steps

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            sharpe = evaluate_sharpe(model, val_data, config, device)
            history["val_sharpe"].append(sharpe)
            logger.info("Epoch %d: loss=%.4f, val_sharpe=%.4f", epoch + 1, avg_loss, sharpe)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 2:  # 2 checks × 5 epochs = 10 epoch patience
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
```

- [ ] **Step 2: Commit**

```bash
git add stoke_ml/models/tft/train.py
git commit -m "feat: add TFT training loop with AMP, grad accum, Sharpe early-stop"
```

---

### Task 11: Portfolio Evaluation

**Files:**
- Create: `stoke_ml/models/tft/evaluate.py`
- Create: `tests/models/tft/test_evaluate.py`

Top-K portfolio simulation: rank stocks by predicted return, equal-weight top-K, compute Sharpe over the validation period.

- [ ] **Step 1: Write failing test**

```python
# tests/models/tft/test_evaluate.py
import torch
from stoke_ml.models.tft.evaluate import compute_sharpe


class TestSharpe:
    def test_positive_returns(self):
        daily_returns = torch.tensor([0.01, 0.02, 0.015, 0.005, 0.01])
        sharpe = compute_sharpe(daily_returns)
        assert sharpe > 0

    def test_zero_returns(self):
        daily_returns = torch.zeros(20)
        sharpe = compute_sharpe(daily_returns)
        assert sharpe == 0.0

    def test_negative_returns(self):
        daily_returns = torch.tensor([-0.01, -0.02, -0.005, -0.015])
        sharpe = compute_sharpe(daily_returns)
        assert sharpe < 0

    def test_annualization(self):
        """Sharpe with 252-day annualization."""
        daily_returns = torch.randn(252) * 0.01 + 0.0005  # slight positive drift
        sharpe = compute_sharpe(daily_returns)
        # Should be roughly sqrt(252) * (mean/std)
        assert -5.0 < sharpe < 5.0  # reasonable range
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement evaluation functions**

```python
# stoke_ml/models/tft/evaluate.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate


def compute_sharpe(daily_returns: torch.Tensor, annualize: bool = True) -> float:
    """Compute Sharpe ratio from daily returns.

    Args:
        daily_returns: (T,) tensor of daily portfolio returns.
        annualize: if True, multiply by sqrt(252).

    Returns:
        float Sharpe ratio.
    """
    if len(daily_returns) < 2:
        return 0.0
    mean = daily_returns.mean().item()
    std = daily_returns.std().item()
    if std < 1e-8:
        return 0.0 if mean == 0 else (float("inf") if mean > 0 else float("-inf"))
    sharpe = mean / std
    if annualize:
        sharpe *= np.sqrt(252)
    return sharpe


def evaluate_sharpe(
    model: nn.Module,
    val_data: dict,
    config: TFTConfig,
    device: torch.device,
    top_k: int = 20,
) -> float:
    """Evaluate model by top-K portfolio Sharpe on validation set.

    1. Predict expected return for all stocks in val set.
    2. Sort by expected return, select top-K.
    3. Simulate equal-weight portfolio, compute Sharpe.

    Args:
        model: trained TFT model.
        val_data: dict with the same structure as train_data.
        config: TFTConfig.
        device: torch device.
        top_k: number of stocks to select.

    Returns:
        float annualized Sharpe ratio.
    """
    model.eval()
    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
    )

    all_returns = []
    all_stock_indices = []

    # Map each sample back to its stock
    # We need to track which stock each prediction belongs to
    sample_idx = 0
    with torch.no_grad():
        for static, pk, po, _, _, _ in val_loader:
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            all_returns.append(pred_ret.cpu())
            # Track stock index for each sample
            batch_size = static.shape[0]
            for j in range(batch_size):
                stock_idx = sample_idx // val_ds.n_windows
                all_stock_indices.append(stock_idx)
                sample_idx += 1

    all_returns = torch.cat(all_returns).squeeze(-1)  # (N_samples,)

    # Aggregate: mean predicted return per stock
    n_stocks = val_data["static_features"].shape[0]
    stock_returns = []
    for i in range(n_stocks):
        mask = torch.tensor([s == i for s in all_stock_indices[:len(all_returns)]])
        if mask.sum() > 0:
            stock_returns.append(all_returns[mask].mean().item())
        else:
            stock_returns.append(-float("inf"))

    stock_returns = torch.tensor(stock_returns)

    # Select top-K stocks
    _, top_indices = torch.topk(stock_returns, min(top_k, n_stocks))

    # Simulate portfolio: equal-weight daily return of top-K
    # Use actual y_return from val_data for the top stocks
    actual_returns = val_data["y_return"]  # (N, T)
    top_returns = actual_returns[top_indices.numpy()]  # (K, T)
    # Use last 50 days for Sharpe calculation
    eval_window = min(50, top_returns.shape[1] - config.seq_len)
    t_start = top_returns.shape[1] - eval_window
    portfolio_daily = top_returns[:, t_start:].mean(axis=0)  # (T_eval,)

    sharpe = compute_sharpe(torch.from_numpy(portfolio_daily))
    return sharpe
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_evaluate.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/models/tft/evaluate.py tests/models/tft/test_evaluate.py
git commit -m "feat: add top-K portfolio Sharpe evaluation"
```

---

### Task 12: CLI Entry Point

**Files:**
- Create: `scripts/train_tft.py`

- [ ] **Step 1: Implement train_tft.py script**

```python
"""Train TFT panel model on all 798 A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stocks 20 --epochs 30
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_tft.py --stock-list 600519,000001,000858
"""
import argparse
import logging
import sys
import time
from datetime import datetime

import torch
import pandas as pd
import numpy as np

from stoke_ml.config import load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.tft import TFTConfig
from stoke_ml.models.tft.train import train_tft

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TFT panel model")
    parser.add_argument("--stocks", type=int, default=None,
                        help="Limit to first N stocks (for quick testing)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load config
    cfg = load_config()
    data_dir = cfg.project.data_dir

    # Resolve stock list
    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    if args.stock_list:
        stock_list = [c.strip() for c in args.stock_list.split(",")]
    else:
        stock_list = ds.list_stocks()
        if args.stocks:
            stock_list = stock_list[:args.stocks]

    logger.info("Building features for %d stocks...", len(stock_list))

    # Build panel features
    fp = FeaturePipeline.from_config(cfg, seq_len=252)
    panel_data = fp.build_panel_features(stock_list, args.start, args.end)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    logger.info("Panel data: %d stocks × %d timesteps", n_stocks, n_timesteps)

    # Purged walk-forward splits
    config = TFTConfig(
        seq_len=252,
        static_dim=panel_data["static_features"].shape[1],
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
    )

    # ── Walk-forward loop ──
    train_start = 0
    train_len = 504  # ~2 years
    val_len = 63  # ~3 months
    step = 63  # ~3 months
    purge = 5
    all_sharpes = []

    fold = 0
    while train_start + train_len + purge + val_len < n_timesteps:
        fold += 1
        train_end = train_start + train_len
        val_start = train_end + purge
        val_end = min(val_start + val_len, n_timesteps)

        train_slice = slice(train_start, train_end)
        val_slice = slice(val_start, val_end)

        train_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, train_slice],
            "past_observed": panel_data["past_observed"][:, train_slice],
            "y_direction": panel_data["y_direction"][:, train_slice],
            "y_return": panel_data["y_return"][:, train_slice],
            "y_volatility": panel_data["y_volatility"][:, train_slice],
        }
        val_data = {
            "static_features": panel_data["static_features"],
            "past_known": panel_data["past_known"][:, val_slice],
            "past_observed": panel_data["past_observed"][:, val_slice],
            "y_direction": panel_data["y_direction"][:, val_slice],
            "y_return": panel_data["y_return"][:, val_slice],
            "y_volatility": panel_data["y_volatility"][:, val_slice],
        }

        logger.info("Fold %d: train %d-%d, val %d-%d",
                    fold, train_start, train_end, val_start, val_end)

        t0 = time.time()
        model, history = train_tft(config, train_data, val_data, device)
        elapsed = time.time() - t0

        if history["val_sharpe"]:
            best_sharpe = max(history["val_sharpe"])
            all_sharpes.append(best_sharpe)
            logger.info("  Fold %d best Sharpe: %.4f (%.1fs)", fold, best_sharpe, elapsed)
        else:
            logger.warning("  Fold %d: no valid Sharpe (%.1fs)", fold, elapsed)

        train_start += step

    if all_sharpes:
        logger.info("Mean Sharpe across %d folds: %.4f", len(all_sharpes), np.mean(all_sharpes))
    else:
        logger.warning("No valid folds completed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_tft.py
git commit -m "feat: add train_tft.py CLI with purged walk-forward"
```

---

### Task 13: End-to-End Integration Test

**Files:**
- Create: `tests/models/tft/test_integration.py`

Smoke test: synthetic data → train 2 epochs → verify all components work together.

- [ ] **Step 1: Write integration test**

```python
# tests/models/tft/test_integration.py
import torch
import numpy as np
from stoke_ml.models.tft import TFTConfig
from stoke_ml.models.tft.model import TFTModel
from stoke_ml.models.tft.loss import UncertaintyLoss
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate
from torch.utils.data import DataLoader
import torch.nn as nn


def make_synthetic_panel(n_stocks=20, n_timesteps=300, seq_len=60):
    """Tiny synthetic panel for fast integration test."""
    static = np.random.randn(n_stocks, 8).astype(np.float32)
    pk = np.random.randn(n_stocks, n_timesteps, 20).astype(np.float32)
    po = np.random.randn(n_stocks, n_timesteps, 12).astype(np.float32)
    y_dir = np.random.randint(0, 2, (n_stocks, n_timesteps)).astype(np.int64)
    y_ret = (np.random.randn(n_stocks, n_timesteps) * 0.02).astype(np.float32)
    y_vol = np.abs(np.random.randn(n_stocks, n_timesteps) * 0.01).astype(np.float32)
    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": y_vol,
    }


class TestIntegration:
    def test_full_training_loop(self):
        """Train 2 epochs on synthetic data — verify no crashes."""
        data = make_synthetic_panel(n_stocks=20, n_timesteps=300, seq_len=60)
        device = torch.device("cpu")

        config = TFTConfig(
            static_dim=8, past_known_dim=20, past_observed_dim=12,
            hidden_dim=32, lstm_layers=1, attention_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=8,
            max_epochs=2,
        )
        model = TFTModel(config).to(device)
        loss_fn = UncertaintyLoss(num_tasks=3).to(device)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ds = PanelDataset(data, seq_len=config.seq_len)
        loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=panel_collate)

        model.train()
        for epoch in range(2):
            for static, pk, po, y_dir, y_ret, y_vol in loader:
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                l_ce = ce(pred_dir, y_dir)
                l_ret = mse(pred_ret.squeeze(-1), y_ret.squeeze(-1))
                l_vol = mse(pred_vol.squeeze(-1), y_vol.squeeze(-1))
                loss = loss_fn([l_ce, l_ret, l_vol])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # After training, verify model produces finite outputs
        model.eval()
        with torch.no_grad():
            d, r, v = model(
                torch.from_numpy(data["static_features"]),
                torch.from_numpy(data["past_known"][:, :60]),
                torch.from_numpy(data["past_observed"][:, :60]),
            )
            assert not torch.isnan(d).any()
            assert not torch.isnan(r).any()
            assert not torch.isnan(v).any()
            assert (v >= 0).all()

    def test_checkpoint_save_load(self):
        """Verify model can be saved and loaded."""
        config = TFTConfig(
            static_dim=8, past_known_dim=20, past_observed_dim=12,
            hidden_dim=32, lstm_layers=1, attention_heads=2,
            grn_layers=1, seq_len=60, compile_model=False,
        )
        model = TFTModel(config)
        # Save
        state = {k: v.clone() for k, v in model.state_dict().items()}
        # Load into new model
        model2 = TFTModel(config)
        model2.load_state_dict(state)

        # Verify same output
        x_s = torch.randn(2, 8)
        x_pk = torch.randn(2, 60, 20)
        x_po = torch.randn(2, 60, 12)
        with torch.no_grad():
            d1, r1, v1 = model(x_s, x_pk, x_po)
            d2, r2, v2 = model2(x_s, x_pk, x_po)
        assert torch.allclose(d1, d2, atol=1e-5)
        assert torch.allclose(r1, r2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)
```

- [ ] **Step 2: Run integration test**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/test_integration.py -v`
Expected: 2 PASS

- [ ] **Step 3: Run ALL TFT tests to confirm nothing broken**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/models/tft/ -v`
Expected: All ~20+ tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/models/tft/test_integration.py
git commit -m "test: add TFT end-to-end integration test"
```

---

## Execution Order

Tasks 2-5 are independent and can run in parallel. Task 6 requires 2-5 (serial only after all complete). Task 7-8 are independent. Task 9-10 are independent. Task 11 requires 6,7,8. Task 12 is independent. Task 13 requires all previous tasks.

Recommended order:
```
1 (config)
├── 2 (GRN/GLU) ──┐
├── 4 (MHA) ──────┤
├── 5 (heads) ────┤
└── 3 (VSN) ──────┼──► 6 (model) ──┐
                   │                 │
7 (loss) ──────────┼─────────────────┤
8 (dataset) ───────┼─────────────────┤
9 (CS norm) ───────┤                 │
10 (panel feat) ───┘                 ├──► 11 (train) ──► 12 (CLI) ──► 13 (integration)
                                     │
12 (evaluate) ───────────────────────┘
```
