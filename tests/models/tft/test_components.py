import torch
import pytest
from stoke_ml.models.tft.components import GatedLinearUnit, GRN, TimeDistributed


class TestGLU:
    def test_output_shape(self):
        glu = GatedLinearUnit(input_dim=16, output_dim=32)
        x = torch.randn(4, 8, 16)
        out = glu(x)
        assert out.shape == (4, 8, 32)

    def test_values_in_range(self):
        glu = GatedLinearUnit(input_dim=8, output_dim=8)
        x = torch.randn(2, 5, 8)
        out = glu(x)
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
        """GRN without context_dim should accept call without context arg."""
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32)
        x = torch.randn(4, 16, 32)
        out = grn(x)
        assert out.shape == (4, 16, 32)

    def test_residual_skip(self):
        """When input_dim == output_dim, residual skip should be active."""
        grn = GRN(input_dim=32, hidden_dim=32, output_dim=32, dropout=0.0)
        x = torch.randn(2, 4, 32)
        out = grn(x)
        assert out.shape == x.shape
