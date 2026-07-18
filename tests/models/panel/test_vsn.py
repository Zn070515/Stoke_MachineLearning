import torch
from stoke_ml.models.panel.vsn import VariableSelectionNetwork


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
