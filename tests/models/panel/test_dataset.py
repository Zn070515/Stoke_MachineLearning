import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate, DateSampler
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.panel_store import load_panel_memmap, save_panel_memmap
from test_panel_store import _storeable_panel


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
        """__len__ = n_windows (date-centric primary index, not n_stocks * n_windows)."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        expected = 100 - 60  # n_windows
        assert len(ds) == expected

    def test_getitem_shapes(self):
        """__getitem__ returns per-date (M, ...) tensors, not per-stock scalars."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        (static, pk, po, y_dir, y_ret, y_vol,
         date_idx, dir_mask, ret_mask, vol_mask, stock_indices) = ds[0]
        M = len(stock_indices)
        assert M > 0  # at least one valid stock for window 0
        assert static.shape == (M, 8)
        assert pk.shape == (M, 60, 20)
        assert po.shape == (M, 60, 12)
        assert y_dir.shape == (M,)
        assert y_ret.shape == (M,)
        assert y_vol.shape == (M,)
        assert date_idx.shape == (M,)
        assert dir_mask.shape == (M,) and dir_mask.dtype == torch.bool
        assert ret_mask.shape == (M,) and ret_mask.dtype == torch.bool
        assert vol_mask.shape == (M,) and vol_mask.dtype == torch.bool
        assert stock_indices.shape == (M,) and stock_indices.dtype == torch.long

    def test_collate_fn_single_date(self):
        """batch_size=1: collate passes through the single-element batch."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        sample = ds[0]
        batch = [sample]
        result = panel_collate(batch)
        # With batch_size=1, collate is identity
        assert len(result) == 11
        for a, b in zip(result, sample):
            assert torch.equal(a, b), f"collate pass-through mismatch"

    def test_collate_fn_multiple_dates(self):
        """batch_size>1: collate concatenates multiple dates along dim 0."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60, max_stocks_per_date=6)
        batch = [ds[0], ds[1], ds[2]]
        result = panel_collate(batch)
        assert len(result) == 11
        # Each sample has M stocks; concatenated has M0 + M1 + M2
        assert result[0].shape[0] == batch[0][0].shape[0] + batch[1][0].shape[0] + batch[2][0].shape[0]

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
        _, _, _, _, _, _, _, dir_mask, ret_mask, vol_mask, _ = ds[0]
        assert bool(dir_mask.all())         # y_direction valid for all
        assert bool(ret_mask.all())         # return target valid for all
        assert not bool(vol_mask.any())     # vol target invalid for all

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
        # window 0: target at column 60 → date_idx must be 60 for all stocks.
        _, _, _, _, _, _, date_idx0, *_ = ds[0]
        assert (date_idx0 == 60).all()
        # window 5: target at column 65 → not 64.
        _, _, _, _, _, _, date_idx5, *_ = ds[5]
        assert (date_idx5 == 65).all()

    def test_date_indices_narrower_than_timesteps_raises(self):
        """date_indices width < n_timesteps means the target column `end`
        indexes out of range — the guard must reject it up front."""
        data = make_synthetic_data(n_days=100, seq_len=60)
        n_stocks = data["past_known"].shape[0]
        data["date_indices"] = torch.zeros((n_stocks, 50), dtype=torch.long)
        with pytest.raises(ValueError):
            PanelDataset(data, seq_len=60)

    def test_max_stocks_per_date_caps_return(self):
        """When max_stocks_per_date is set, __getitem__ returns at most that many."""
        data = make_synthetic_data(n_stocks=20, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60, max_stocks_per_date=5, training=True)
        *_, stock_indices = ds[0]
        assert stock_indices.numel() <= 5

    def test_training_false_no_cap(self):
        """When training=False, max_stocks_per_date is ignored — all valid stocks."""
        data = make_synthetic_data(n_stocks=20, n_days=100, seq_len=60)
        ds_all = PanelDataset(data, seq_len=60, max_stocks_per_date=None, training=False)
        ds_capped = PanelDataset(data, seq_len=60, max_stocks_per_date=5, training=False)
        *_, si_all = ds_all[0]
        *_, si_capped = ds_capped[0]
        # training=False ignores the cap — both return all valid stocks
        assert si_all.numel() == si_capped.numel()

    def test_date_sampler_skips_empty_dates(self):
        """DateSampler skips windows with 0 valid stocks."""
        data = make_synthetic_data(n_stocks=3, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        # Manually zero out all stocks on window 0
        ds.valid_mask[:, 0] = False
        sampler = DateSampler(ds.valid_mask)
        indices = list(sampler)
        assert 0 not in indices  # window 0 has 0 valid → skipped
        assert len(indices) == ds.n_windows - 1

    def test_full_cross_section_small_date(self):
        """Dates with fewer stocks than the cap get the FULL cross-section."""
        data = make_synthetic_data(n_stocks=5, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60, max_stocks_per_date=100, training=True)
        *_, stock_indices = ds[0]
        # All 5 stocks should be valid (random labels, all valid for simple data)
        assert stock_indices.numel() == 5


class TestMemmapDataset:
    """§十六: a dataset built from np.memmap arrays (lazy window gather) must
    behave IDENTICALLY to one built from dense ndarrays (eager tensors).

    The big per-timestep arrays stay memmap references in the lazy path and
    are sliced+converted per window inside __getitem__; masks / date_indices
    are always eager.  Both paths must yield elementwise-equal windows, equal
    valid/eval masks, and equal per-window stock lists.
    """

    def _pair(self, tmp_path, seq_len=60, seed=5):
        panel = _storeable_panel(n_stocks=12, n_days=100, seq_len=seq_len,
                                 seed=seed)
        save_panel_memmap(panel, tmp_path)
        memmap_data = load_panel_memmap(tmp_path)
        ds_dense = PanelDataset(panel, seq_len=seq_len, min_history=50)
        ds_mem = PanelDataset(memmap_data, seq_len=seq_len, min_history=50)
        return ds_dense, ds_mem

    def test_dense_vs_memmap_equal_masks_and_stocks(self, tmp_path):
        """valid/eval masks and the per-window stock lists are identical."""
        ds_dense, ds_mem = self._pair(tmp_path)
        assert len(ds_dense) == len(ds_mem)
        assert torch.equal(ds_dense.valid_mask, ds_mem.valid_mask), (
            "valid_mask differs between dense and memmap datasets")
        assert torch.equal(ds_dense.eval_mask, ds_mem.eval_mask), (
            "eval_mask differs between dense and memmap datasets")
        for w, (a, b) in enumerate(
                zip(ds_dense._date_to_stocks, ds_mem._date_to_stocks)):
            assert torch.equal(a, b), f"_date_to_stocks[window {w}] differs"

    def test_dense_vs_memmap_equal_windows(self, tmp_path):
        """__getitem__ returns elementwise-equal, dtype-equal 11-tuples."""
        ds_dense, ds_mem = self._pair(tmp_path, seed=7)
        for w in (0, 1, 5, 17, len(ds_dense) - 1):
            dense_win = ds_dense[w]
            mem_win = ds_mem[w]
            for i in range(11):
                assert dense_win[i].dtype == mem_win[i].dtype, (
                    f"window {w} element {i} dtype mismatch")
                assert torch.equal(dense_win[i], mem_win[i]), (
                    f"window {w} element {i} values differ")

    def test_end_to_end_dataloader_matches_dense(self, tmp_path):
        """A DataLoader over the memmap dataset yields the same batches as the
        dense one (collate concatenates dates; tensors are fully materialized)."""
        ds_dense, ds_mem = self._pair(tmp_path, seed=11)
        dl_dense = DataLoader(ds_dense, batch_size=3, shuffle=False,
                              collate_fn=panel_collate)
        dl_mem = DataLoader(ds_mem, batch_size=3, shuffle=False,
                            collate_fn=panel_collate)
        for bi, (b_dense, b_mem) in enumerate(zip(dl_dense, dl_mem)):
            for i in range(11):
                assert torch.equal(b_dense[i], b_mem[i]), (
                    f"batch {bi} element {i} differs")

    def test_memmap_batch_is_model_consumable(self, tmp_path):
        """A memmap-derived batch feeds the model forward — proves the lazy
        tensors are fully materialized and contiguous enough for inference."""
        _, ds_mem = self._pair(tmp_path, seed=13)
        config = PanelConfig(
            static_dim=4, past_known_dim=12, past_observed_dim=6,
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=60, dropout=0.0, compile_model=False,
        )
        model = PanelModel(config).eval()
        loader = DataLoader(ds_mem, batch_size=4, shuffle=False,
                            collate_fn=panel_collate)
        seen = 0
        with torch.no_grad():
            for static, pk, po, *_rest in loader:
                d, r, v = model(static, pk, po)
                assert not torch.isnan(r).any()
                assert not torch.isnan(d).any()
                assert not torch.isnan(v).any()
                seen += 1
                if seen >= 3:
                    break
        assert seen >= 1


class TestDateSampler:
    def test_len_counts_valid_dates(self):
        data = make_synthetic_data(n_stocks=5, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        sampler = DateSampler(ds.valid_mask)
        # Each window has some valid stocks (random dir labels, all valid)
        assert len(sampler) == ds.n_windows

    def test_reproducible_with_seed(self):
        """Same seed → same date order."""
        data = make_synthetic_data(n_stocks=5, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        vm = ds.valid_mask.clone()
        torch.manual_seed(42)
        order1 = list(DateSampler(vm))
        torch.manual_seed(42)
        order2 = list(DateSampler(vm))
        assert order1 == order2

    def test_different_seed_different_order(self):
        """Different seeds → different date order."""
        data = make_synthetic_data(n_stocks=5, n_days=100, seq_len=60)
        ds = PanelDataset(data, seq_len=60)
        vm = ds.valid_mask.clone()
        torch.manual_seed(1)
        order1 = list(DateSampler(vm))
        torch.manual_seed(2)
        order2 = list(DateSampler(vm))
        assert order1 != order2
