import pytest
import torch
import numpy as np
import pandas as pd
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate


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
        expected = (100 - 60) * 10
        assert len(ds) == expected

    def test_getitem_shapes(self):
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        static, pk, po, y_dir, y_ret, y_vol, date_idx, dir_mask, ret_mask, vol_mask = ds[0]
        assert static.shape == (8,)
        assert pk.shape == (60, 20)
        assert po.shape == (60, 12)
        assert y_dir.ndim == 0  # scalar
        assert y_ret.ndim == 0
        assert y_vol.ndim == 0
        assert date_idx == 0  # default 0 when no date_indices passed
        # masks fall back to y_direction validity when target masks absent
        assert bool(dir_mask) and bool(ret_mask) and bool(vol_mask)

    def test_collate_fn(self):
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        batch = [ds[i] for i in range(4)]
        static, pk, po, y_dir, y_ret, y_vol, date_idx, dir_mask, ret_mask, vol_mask = panel_collate(batch)
        assert static.shape == (4, 8)
        assert pk.shape == (4, 60, 20)
        assert po.shape == (4, 60, 12)
        assert y_dir.shape == (4,)
        assert y_ret.shape == (4,)
        assert y_vol.shape == (4,)
        assert date_idx.shape == (4,)
        assert dir_mask.shape == (4,) and dir_mask.dtype == torch.bool
        assert ret_mask.shape == (4,)
        assert vol_mask.shape == (4,)

    def test_valid_mask_requires_history_and_entry(self):
        """A window is trainable only when its input window holds
        >= min_history real observations (new listings with mostly zero-padded
        history are excluded) AND the target day is entry-eligible."""
        data = make_synthetic_data(n_stocks=2, n_days=100, seq_len=60)
        n_stocks, n_days = data["past_known"].shape[0], data["past_known"].shape[1]
        obs = torch.zeros(n_stocks, n_days, dtype=torch.bool)
        obs[0, :] = True          # stock 0 fully observed
        obs[1, 90:] = True        # stock 1 only real from column 90 (new listing)
        data["observation_mask"] = obs
        data["entry_eligible_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        data["return_target_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        data["vol_target_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        ds = PanelDataset(data, seq_len=60, min_history=50)
        vm = ds.valid_mask  # (2, 40)
        # stock 0: window [0,60) has 60 real obs >= 50 → valid
        assert bool(vm[0, 0])
        # stock 1: window [0,60) has 0 real obs < 50 → invalid (new listing)
        assert not bool(vm[1, 0])
        # stock 1: every window holds at most 9 real obs (cols 90..98) < 50 → all invalid
        assert not bool(vm[1, :].any())

    def test_getitem_returns_per_task_masks(self):
        """Each loss applies its own mask instead of one shared
        y_direction mask — return-target valid but vol-target invalid must be
        reflected separately."""
        data = make_synthetic_data(n_stocks=2, n_days=100, seq_len=60)
        n_stocks, n_days = data["past_known"].shape[0], data["past_known"].shape[1]
        data["observation_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        data["entry_eligible_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        data["return_target_mask"] = torch.ones(n_stocks, n_days, dtype=torch.bool)
        data["vol_target_mask"] = torch.zeros(n_stocks, n_days, dtype=torch.bool)
        ds = PanelDataset(data, seq_len=60)
        _, _, _, y_dir, y_ret, y_vol, _, dir_mask, ret_mask, vol_mask = ds[0]
        assert bool(dir_mask)          # y_direction valid
        assert bool(ret_mask)          # return target valid
        assert not bool(vol_mask)      # vol target invalid


    def test_date_idx_is_target_date_not_last_feature_date(self):
        """Off-by-one: a window [start, end) is ranked by the TARGET
        date `end` (the step after the window), not the last feature date
        `end - 1`.  Ranking pairs must compare stocks' outcomes on the SAME
        future day, and with global calendar alignment date_indices is simply
        the column index tiled across stocks."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        n_stocks, n_days = data["past_known"].shape[0], data["past_known"].shape[1]
        data["date_indices"] = torch.tile(
            torch.arange(n_days, dtype=torch.long), (n_stocks, 1),
        )
        ds = PanelDataset(data, seq_len=60)
        # window [0, 60): target lives at column 60 → date_idx must be 60.
        _, _, _, _, _, _, date_idx, *_ = ds[0]
        assert date_idx == 60
        # window [5, 65): target at column 65 → not 64.
        _, _, _, _, _, _, date_idx, *_ = ds[5]
        assert date_idx == 65

    def test_date_indices_narrower_than_timesteps_raises(self):
        """date_indices width < n_timesteps means the target column `end`
        indexes out of range — the guard must reject it up front."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        n_stocks = data["past_known"].shape[0]
        data["date_indices"] = torch.zeros((n_stocks, 50), dtype=torch.long)
        with pytest.raises(ValueError):
            PanelDataset(data, seq_len=60)
