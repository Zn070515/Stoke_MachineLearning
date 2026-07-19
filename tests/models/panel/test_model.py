import torch
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.model import PanelModel


class TestPanelModel:
    @classmethod
    def setup_class(cls):
        cls.config = PanelConfig(
            static_dim=8,
            past_known_dim=65,   # > 64 triggers VSN chunking (2+ chunks)
            past_observed_dim=12,
            hidden_dim=64,
            xlstm_num_blocks=1,
            xlstm_num_heads=2,
            grn_layers=2,
            seq_len=60,
        )
        cls.model = PanelModel(cls.config)

    def test_forward_outputs(self):
        B, T = 4, 60
        static = torch.randn(B, self.config.static_dim)
        past_known = torch.randn(B, T, self.config.past_known_dim)
        past_obs = torch.randn(B, T, self.config.past_observed_dim)

        direction, ret, vol = self.model(static, past_known, past_obs)

        assert direction.shape == (B, 3)
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
        assert total < 1_000_000, f"Expected <1M params for test config, got {total:,}"
