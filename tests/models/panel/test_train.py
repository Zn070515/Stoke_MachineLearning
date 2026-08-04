"""Integration tests for train_panel on a masked synthetic panel.

P1-A (review v4 §八/§九/§十) regression: the training loop must survive
all-ignore direction batches (y_direction=-100 with ret/vol still active),
the validation loss must accumulate per valid sample, and the per-task
uncertainty masks must not produce NaN.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.loss import AdjMSELoss, UncertaintyLoss
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.train import train_panel, _compute_val_loss


def _make_masked_panel(n_stocks=12, n_timesteps=120, seq_len=60, horizon=5, seed=0):
    """Synthetic panel with per-task masks and price paths.

    Two stocks get a "new listing" hole (first 20 days observation-masked) so
    the history-eligibility path is exercised; the rest are fully real.
    """
    rng = np.random.RandomState(seed)
    close = np.full((n_stocks, n_timesteps), 10.0, dtype=np.float32)
    shocks = rng.randn(n_stocks, n_timesteps - 1).astype(np.float32) * 0.02
    close[:, 1:] = close[:, :1] * np.exp(np.cumsum(shocks, axis=1))
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0]
    gaps = 1.0 + rng.randn(n_stocks, n_timesteps - 1).astype(np.float32) * 0.005
    open_[:, 1:] = close[:, :-1] * gaps

    obs = np.ones((n_stocks, n_timesteps), dtype=bool)
    obs[0, :20] = False   # new-listing hole
    obs[1, :20] = False
    entry = np.ones((n_stocks, n_timesteps), dtype=bool)
    decision = np.ones((n_stocks, n_timesteps), dtype=bool)
    decision[:, 0] = False  # no close[t-1]

    ret_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    vol_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    for t in range(n_timesteps - horizon):
        have_exit = np.isfinite(open_[:, t + horizon])
        ret_tgt[:, t] = have_exit
        if t + horizon < n_timesteps:
            fwd = close[:, t + 1:t + horizon + 1]
            vol_tgt[:, t] = have_exit & np.isfinite(fwd).all(axis=1)

    y_ret = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
    for t in range(n_timesteps - horizon):
        y_ret[:, t] = open_[:, t + horizon] / open_[:, t] - 1.0

    y_dir = np.full((n_stocks, n_timesteps), -100, dtype=np.int64)
    sel = y_ret[ret_tgt]
    y_dir[ret_tgt] = np.where(sel > 0.001, 2, np.where(sel < -0.001, 0, 1))

    y_vol = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
    for t in range(n_timesteps - horizon - 1):
        fwd = close[:, t + 1:t + horizon + 1]
        y_vol[:, t] = np.nanstd(np.diff(fwd, axis=1), axis=1)

    realized = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
    realized[:, :-1] = close[:, 1:] / close[:, :-1] - 1.0

    static = rng.randn(n_stocks, n_timesteps, 4).astype(np.float32)
    pk = rng.randn(n_stocks, n_timesteps, 12).astype(np.float32)
    po = rng.randn(n_stocks, n_timesteps, 6).astype(np.float32)
    date_indices = np.tile(np.arange(n_timesteps, dtype=np.int64)[None, :],
                           (n_stocks, 1))

    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": y_vol,
        "observation_mask": obs,
        "entry_eligible_mask": entry,
        "decision_eligible_mask": decision,
        "return_target_mask": ret_tgt,
        "vol_target_mask": vol_tgt,
        "realized_return": realized,
        "date_indices": date_indices,
        "close_price": close,
        "open_price": open_,
    }


class TestTrainPanelMasked:
    def test_trains_2_epochs_on_masked_panel(self):
        """Full train loop on a masked panel — no NaN, history populated."""
        data = _make_masked_panel()
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=16, num_workers=0,
            max_epochs=2, horizon=5, rank_loss_weight=0.1,
        )
        device = torch.device("cpu")
        model, history = train_panel(
            config, data, data, device,
            raw_val_returns=data["realized_return"],
        )
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert history["best_epoch_idx"] >= 0
        for v in history["val_loss"]:
            assert np.isfinite(v)

        # Deployed best checkpoint produces finite, sane outputs.
        model.eval()
        with torch.no_grad():
            s = torch.from_numpy(data["static_features"][:, -1])
            pk = torch.from_numpy(data["past_known"][:, -60:])
            po = torch.from_numpy(data["past_observed"][:, -60:])
            d, r, v = model(s, pk, po)
        assert not torch.isnan(d).any()
        assert not torch.isnan(r).any()
        assert not torch.isnan(v).any()
        assert (v >= 0).all()

    def test_vol_only_batch_trains(self):
        """A batch whose direction labels are all -100 still trains via ret/vol.

        Construct a panel where no clean return targets exist in the last
        columns, forcing the all-ignore-dir path in the val accumulator.
        """
        data = _make_masked_panel(seed=1)
        # Blanket-mask direction on the final columns of every stock — those
        # windows are still valid for vol/ret and must not poison the loss.
        data["y_direction"][:, -10:] = -100
        data["return_target_mask"][:, -10:] = False
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=16, num_workers=0,
            max_epochs=2, horizon=5, rank_loss_weight=0.1,
        )
        device = torch.device("cpu")
        model, history = train_panel(
            config, data, data, device,
            raw_val_returns=data["realized_return"],
        )
        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(v) for v in history["val_loss"])


class TestValLossBatchSizeInvariant:
    """Review v5 §十七 P1#9: the validation return loss is a SAMPLE-WEIGHTED
    mean (train.py `_compute_val_loss`), so re-batching the same validation
    data with batch sizes 32 / 64 / 128 must give the same v_ret.  This guards
    the checkpoint-selection metric against batch-boundary artifacts."""

    def _model_and_losses(self):
        data = _make_masked_panel(n_stocks=12, n_timesteps=150, seed=3)
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=128, num_workers=0,
            max_epochs=1, horizon=5, rank_loss_weight=0.1,
        )
        model = PanelModel(config)
        ret_loss = AdjMSELoss()
        loss_fn = UncertaintyLoss(num_tasks=3)
        return data, config, model, ret_loss, loss_fn

    def test_val_return_loss_batch_size_invariant(self):
        data, config, model, ret_loss, loss_fn = self._model_and_losses()
        device = torch.device("cpu")
        rets = []
        rankics = []
        for bs in (32, 64, 128):
            ds = PanelDataset(data, seq_len=config.seq_len,
                              min_history=config.min_history)
            loader = DataLoader(ds, batch_size=bs, shuffle=False,
                                collate_fn=panel_collate, num_workers=0)
            _, _, v_ret, _, v_rankic = _compute_val_loss(
                model, loader, ret_loss, loss_fn, device,
                use_amp=False, vol_enabled=True,
            )
            assert np.isfinite(v_ret)
            assert np.isfinite(v_rankic), "RankIC must be computable on a normal panel"
            rets.append(v_ret)
            rankics.append(v_rankic)
        assert max(rets) - min(rets) < 1e-5, rets
        # RankIC is accumulated over all valid samples and grouped by date at
        # the end, so like v_ret it must be independent of batch boundaries —
        # the primary checkpoint-selection metric must not shift with batching.
        assert max(rankics) - min(rankics) < 1e-12, rankics
