from dataclasses import replace

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


class TestPanelModelAblations:
    """§十一.3: every architecture-ablation switch must forward and gate the
    right components off."""

    BASE = PanelConfig(
        static_dim=4, past_known_dim=8, past_observed_dim=6,
        hidden_dim=32, xlstm_num_blocks=2, xlstm_slstm_ratio=0.5,
        xlstm_num_heads=2, grn_layers=1, seq_len=10,
    )

    @staticmethod
    def _forward(cfg: PanelConfig) -> PanelModel:
        model = PanelModel(cfg).eval()
        B, T = 3, cfg.seq_len
        static = torch.randn(B, cfg.static_dim)
        pk = torch.randn(B, T, cfg.past_known_dim)
        po = torch.randn(B, T, cfg.past_observed_dim)
        with torch.no_grad():
            d, r, v = model(static, pk, po)
        assert d.shape == (B, 3)
        assert r.shape == (B, 1)
        assert v.shape == (B, 1)
        assert torch.isfinite(r).all()
        return model

    def test_plain_lstm(self):
        m = self._forward(replace(self.BASE, backbone="lstm"))
        assert m._is_xlstm is False
        assert isinstance(m.backbone, torch.nn.LSTM)

    def test_vsn_lstm(self):
        m = self._forward(replace(self.BASE, backbone="lstm", use_vsn=True))
        assert m._is_xlstm is False
        assert isinstance(m.backbone, torch.nn.LSTM)

    def test_xlstm_no_vsn(self):
        m = self._forward(replace(self.BASE, use_vsn=False))
        assert m._is_xlstm is True
        assert isinstance(m.vsn_past, torch.nn.Linear)
        assert isinstance(m.vsn_obs, torch.nn.Linear)

    def test_no_dir_head_emits_zeros(self):
        cfg = replace(self.BASE, use_dir_head=False)
        m = self._forward(cfg)
        assert m.direction_head is None
        B, T = 3, cfg.seq_len
        static = torch.randn(B, cfg.static_dim)
        pk = torch.randn(B, T, cfg.past_known_dim)
        po = torch.randn(B, T, cfg.past_observed_dim)
        with torch.no_grad():
            d, _, _ = m(static, pk, po)
        assert torch.allclose(d, torch.zeros(B, 3), atol=1e-7)

    def test_no_vol_head_emits_zeros(self):
        cfg = replace(self.BASE, use_vol_head=False)
        m = self._forward(cfg)
        assert m.volatility_head is None
        B, T = 3, cfg.seq_len
        static = torch.randn(B, cfg.static_dim)
        pk = torch.randn(B, T, cfg.past_known_dim)
        po = torch.randn(B, T, cfg.past_observed_dim)
        with torch.no_grad():
            _, _, v = m(static, pk, po)
        assert torch.allclose(v, torch.zeros(B, 1), atol=1e-7)

    def test_return_only(self):
        m = self._forward(replace(self.BASE, use_dir_head=False, use_vol_head=False))
        assert m.direction_head is None
        assert m.volatility_head is None
        assert m.return_head is not None

    def test_no_pit_static_drops_static_encoder(self):
        m = self._forward(replace(self.BASE, use_pit_static=False))
        assert m.static_proj is None
        assert m.static_enrich is None
        assert m.static_vs_context is None
        assert len(m.c_h_projections) == 0
