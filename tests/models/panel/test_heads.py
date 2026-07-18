import torch
from stoke_ml.models.panel.heads import DirectionHead, ReturnHead, VolatilityHead


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
        assert out.abs().mean() < 5.0  # random init, just check no explosion


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
