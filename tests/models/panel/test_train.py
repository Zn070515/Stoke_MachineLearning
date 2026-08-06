"""Integration tests for train_panel on a masked synthetic panel.

P1-A regression: the training loop must survive
all-ignore direction batches (y_direction=-100 with ret/vol still active),
the validation loss must accumulate per valid sample, and the per-task
uncertainty masks must not produce NaN.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.loss import AdjMSELoss, UncertaintyLoss
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.panel_store import load_panel_memmap, save_panel_memmap
from stoke_ml.models.panel.train import train_panel, _compute_val_loss
from test_panel_store import _storeable_panel


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
    @pytest.mark.slow
    def test_trains_2_epochs_on_masked_panel(self):
        """Full train loop on a masked panel — no NaN, history populated."""
        data = _make_masked_panel()
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0,
            compile_model=False, batch_size=16, num_workers=0,
            max_epochs=2, horizon=5, rank_loss_weight=0.1,
            min_stocks_per_day=5,
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

    @pytest.mark.slow
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
            min_stocks_per_day=5,
        )
        device = torch.device("cpu")
        model, history = train_panel(
            config, data, data, device,
            raw_val_returns=data["realized_return"],
        )
        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(v) for v in history["val_loss"])


class TestValLossBatchSizeInvariant:
    """The validation return loss is a SAMPLE-WEIGHTED
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
            min_stocks_per_day=5,
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
                              min_history=config.min_history, training=False)
            loader = DataLoader(ds, batch_size=bs, shuffle=False,
                                collate_fn=panel_collate, num_workers=0)
            _, _, v_ret, _, v_rankic = _compute_val_loss(
                model, loader, data, config, ret_loss, loss_fn, device,
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


class TestOffsetSlicedFoldValLoss:
    """§七 date-centric grid placement on offset-sliced folds.

    panel_builder emits GLOBAL date_indices (0..max_T-1); ``_slice_panel``
    REBASES them to LOCAL column space (start → 0).  Date-centric consumers
    place predictions at ``window_idx = date_idx - seq_len``, which is only
    in-bounds for the (N, n_windows) grid when the slice is rebased.  Without
    the rebase, every inner-val/outer-test slice with ``start > 0`` yields
    ``window_idx = start + window >= n_windows`` → IndexError in
    ``_compute_val_loss`` / ``evaluate_portfolio`` / ``_predict_outer``.  The
    fast fixtures build date_indices from 0, so only an offset slice exercises
    this path (the offset folds are slow-deselected tests).
    """

    SEQ_LEN = 60

    def _sliced_panel(self):
        from scripts.production.train_panel import _slice_panel

        data = _make_masked_panel(n_stocks=12, n_timesteps=150, seq_len=self.SEQ_LEN,
                                  horizon=5, seed=3)
        # _slice_panel requires history_eligible_mask — inject one built from
        # the observation mask (full-window real rule, like panel_builder).
        obs = data["observation_mask"]
        n, t = obs.shape
        cum = np.concatenate(
            [np.zeros((n, 1), dtype=np.int64), np.cumsum(obs, axis=1)], axis=1)
        hist = np.zeros((n, t), dtype=bool)
        for tt in range(self.SEQ_LEN, t):
            hist[:, tt] = (cum[:, tt] - cum[:, tt - self.SEQ_LEN]) >= self.SEQ_LEN
        data["history_eligible_mask"] = hist
        # Offset slice with start > 0 — the inner-val/outer-test shape.
        return _slice_panel(data, slice(30, 120))

    def test_offset_sliced_val_loss_no_indexerror(self):
        sliced = self._sliced_panel()
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=self.SEQ_LEN, dropout=0.0,
            compile_model=False, batch_size=128, num_workers=0,
            max_epochs=1, horizon=5, rank_loss_weight=0.1,
            min_stocks_per_day=5,
        )
        model = PanelModel(config)
        ret_loss = AdjMSELoss()
        loss_fn = UncertaintyLoss(num_tasks=3)
        ds = PanelDataset(sliced, seq_len=config.seq_len,
                          min_history=config.min_history, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=panel_collate, num_workers=0)
        # Must NOT raise IndexError on an offset slice, and must return a
        # computable return loss / RankIC.
        _, _, v_ret, _, v_rankic = _compute_val_loss(
            model, loader, sliced, config, ret_loss, loss_fn,
            torch.device("cpu"), use_amp=False, vol_enabled=True,
        )
        assert np.isfinite(v_ret), f"offset-slice val return loss not finite: {v_ret}"
        assert np.isfinite(v_rankic), (
            "offset-slice RankIC not computable — preds likely landed out of "
            f"bounds or all-NaN: {v_rankic}")

    def test_offset_sliced_preds_land_in_bounds(self):
        sliced = self._sliced_panel()
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=self.SEQ_LEN, dropout=0.0,
            compile_model=False, batch_size=128, num_workers=0,
            max_epochs=1, horizon=5, rank_loss_weight=0.1,
            min_stocks_per_day=5,
        )
        model = PanelModel(config).eval()
        ds = PanelDataset(sliced, seq_len=config.seq_len,
                          min_history=config.min_history, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=panel_collate, num_workers=0)
        n_stocks = sliced["static_features"].shape[0]
        preds = torch.full((n_stocks, ds.n_windows), float("nan"))
        with torch.no_grad():
            for batch in loader:
                static, pk, po, *_y, date_idx, _dm, _rm, _vm, stock_idx = batch
                if stock_idx.numel() == 0:
                    continue
                window_idx = date_idx - config.seq_len
                assert (window_idx >= 0).all() and (window_idx < ds.n_windows).all(), (
                    f"grid placement out of bounds: window_idx in "
                    f"[{window_idx.min()}, {window_idx.max()}) for "
                    f"n_windows={ds.n_windows}")
                _, pred_ret, _ = model(static, pk, po)
                preds[stock_idx, window_idx] = pred_ret.squeeze(-1)
        # Every eval-eligible cell must be predicted — not all-NaN where the
        # mask says eligible.
        eval_mask = ds.eval_mask.numpy()
        nan_frac = float(np.isnan(preds.numpy())[eval_mask].mean())
        assert nan_frac < 0.01, (
            f"{nan_frac:.1%} of eval-eligible cells unpredicted on offset slice")


class TestMemmapSlicePanel:
    """§十六: _slice_panel on a memmap-backed panel (loaded from a store)
    must slice exactly like the dense panel.  The result is a mixed bag —
    basic-sliced arrays (static/pk/po/y_direction/masks) stay lazy memmap
    views, .copy() arrays (y_return/y_volatility/realized_return/date_indices)
    materialize — and must feed a PanelDataset with elementwise-equal windows.
    This is the exact shape the --panel-store training path consumes."""

    SEQ_LEN = 60

    def _sliced_pair(self, tmp_path):
        from scripts.production.train_panel import _slice_panel

        panel = _storeable_panel(n_stocks=12, n_days=150,
                                 seq_len=self.SEQ_LEN, seed=6)
        save_panel_memmap(panel, tmp_path)
        memmap_data = load_panel_memmap(tmp_path)
        dense_sliced = _slice_panel(panel, slice(30, 120))
        mem_sliced = _slice_panel(memmap_data, slice(30, 120))
        return dense_sliced, mem_sliced

    def test_slice_values_equal(self, tmp_path):
        """Every sliced array (memmap-view and .copy() alike) matches the
        dense slice elementwise."""
        dense_sliced, mem_sliced = self._sliced_pair(tmp_path)
        assert set(mem_sliced) == set(dense_sliced)
        for key in dense_sliced:
            np.testing.assert_array_equal(np.asarray(mem_sliced[key]),
                                          np.asarray(dense_sliced[key]),
                                          err_msg=key)

    def test_sliced_memmap_feeds_dataset(self, tmp_path):
        """PanelDataset over the memmap-sliced panel yields the same windows as
        over the dense slice — the mixed lazy/eager panel is fully usable."""
        dense_sliced, mem_sliced = self._sliced_pair(tmp_path)
        ds_dense = PanelDataset(dense_sliced, seq_len=self.SEQ_LEN,
                                min_history=50, training=False)
        ds_mem = PanelDataset(mem_sliced, seq_len=self.SEQ_LEN,
                              min_history=50, training=False)
        assert len(ds_dense) == len(ds_mem)
        assert torch.equal(ds_dense.valid_mask, ds_mem.valid_mask)
        assert torch.equal(ds_dense.eval_mask, ds_mem.eval_mask)
        for w in (0, 1, 10, len(ds_dense) - 1):
            for i in range(11):
                assert torch.equal(ds_dense[w][i], ds_mem[w][i]), (
                    f"window {w} element {i} differs after memmap slice")


class TestAblationTraining:
    """§十一.3: every architecture ablation must survive the full train loop
    on the masked synthetic panel — no NaN, history populated."""

    _BASE_KWARGS = dict(
        static_dim=4, past_known_dim=12, past_observed_dim=6,
        hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
        grn_layers=1, seq_len=60, dropout=0.0,
        compile_model=False, batch_size=16, num_workers=0,
        max_epochs=2, horizon=5, rank_loss_weight=0.1,
        min_stocks_per_day=5,
    )

    @staticmethod
    def _train(seed, **overrides):
        data = _make_masked_panel(seed=seed)
        config = PanelConfig(**{**TestAblationTraining._BASE_KWARGS, **overrides})
        device = torch.device("cpu")
        model, history = train_panel(
            config, data, data, device,
            raw_val_returns=data["realized_return"],
        )
        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(v) for v in history["val_loss"])
        return model, history

    @pytest.mark.slow
    def test_plain_lstm_trains(self):
        """backbone='lstm' → nn.LSTM backbone survives training."""
        model, _ = self._train(seed=11, backbone="lstm")
        assert model._is_xlstm is False
        assert isinstance(model.backbone, torch.nn.LSTM)

    @pytest.mark.slow
    def test_return_only_trains(self):
        """No direction/vol heads → return-only objective still trains."""
        model, _ = self._train(seed=12, use_dir_head=False, use_vol_head=False)
        assert model.direction_head is None
        assert model.volatility_head is None
        assert model.return_head is not None

    @pytest.mark.slow
    def test_fixed_task_weights_trains(self):
        """FixedTaskWeights (UncertaintyLoss ablation) drives the loop."""
        model, _ = self._train(seed=13, fixed_task_weights=True)
        assert list(model.parameters())  # model still has weights

    @pytest.mark.slow
    def test_no_pit_static_trains(self):
        """use_pit_static=False drops the static encoder, still trains."""
        model, _ = self._train(seed=14, use_pit_static=False)
        assert model.static_proj is None
        assert model.static_enrich is None

    @pytest.mark.slow
    def test_no_ranking_trains(self):
        """use_ranking_loss=False disables the ranking term, still trains."""
        model, _ = self._train(seed=15, use_ranking_loss=False)
        assert model.return_head is not None
