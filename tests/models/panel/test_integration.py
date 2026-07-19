import torch
import numpy as np
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.loss import UncertaintyLoss
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from torch.utils.data import DataLoader
import torch.nn as nn


def make_synthetic_panel(n_stocks=20, n_timesteps=300, seq_len=60):
    """Tiny synthetic panel for fast integration test."""
    static = np.random.randn(n_stocks, 8).astype(np.float32)
    pk = np.random.randn(n_stocks, n_timesteps, 20).astype(np.float32)
    po = np.random.randn(n_stocks, n_timesteps, 12).astype(np.float32)
    y_dir = np.random.randint(0, 3, (n_stocks, n_timesteps)).astype(np.int64)
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

        config = PanelConfig(
            static_dim=8, past_known_dim=20, past_observed_dim=12,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=8,
            max_epochs=2, num_workers=0,
        )
        model = PanelModel(config).to(device)
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
            assert not torch.isnan(d).any(), "Direction outputs contain NaN"
            assert not torch.isnan(r).any(), "Return outputs contain NaN"
            assert not torch.isnan(v).any(), "Volatility outputs contain NaN"
            assert (v >= 0).all(), f"Volatility must be positive, got min={v.min()}"

    def test_checkpoint_save_load(self):
        """Verify model can be saved and loaded with identical outputs."""
        config = PanelConfig(
            static_dim=8, past_known_dim=20, past_observed_dim=12,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, compile_model=False,
        )
        model = PanelModel(config)
        state = {k: v.clone() for k, v in model.state_dict().items()}

        model2 = PanelModel(config)
        model2.load_state_dict(state)

        model.eval()
        model2.eval()
        x_s = torch.randn(2, 8)
        x_pk = torch.randn(2, 60, 20)
        x_po = torch.randn(2, 60, 12)
        with torch.no_grad():
            d1, r1, v1 = model(x_s, x_pk, x_po)
            d2, r2, v2 = model2(x_s, x_pk, x_po)
        assert torch.allclose(d1, d2, atol=1e-5), "Direction mismatch after load"
        assert torch.allclose(r1, r2, atol=1e-5), "Return mismatch after load"
        assert torch.allclose(v1, v2, atol=1e-5), "Volatility mismatch after load"
