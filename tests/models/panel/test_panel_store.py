"""Round-trip tests for the §十六 panel memmap persistence layer.

``save_panel_memmap`` writes every build_panel_features array to
``{out}/{name}.npy`` atomically (temp file + os.replace) and a
``complete.json`` marker LAST; ``load_panel_memmap`` re-opens them lazily via
``np.load(mmap_mode='r')``.  These tests pin the contract: a complete store
round-trips values exactly, an interrupted save never looks complete, and a
missing required file is reported by name.
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.models.panel import panel_store
from stoke_ml.models.panel.panel_store import (
    _PANEL_ARRAY_KEYS,
    _PANEL_JSON_KEYS,
    load_panel_memmap,
    panel_store_complete,
    save_panel_memmap,
)


def _storeable_panel(n_stocks=10, n_days=100, seq_len=60, horizon=5, seed=0):
    """A complete panel dict every _PANEL_ARRAY_KEYS/_PANEL_JSON_KEYS expects.

    Mirrors the masked-panel shape used elsewhere (3D PIT static, per-task
    masks, price paths) plus the store-only keys (global_dates, stock_codes,
    forward_vol_nobs, universe_eligible_mask, *cols).
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    stocks = [f"{i:06d}" for i in range(n_stocks)]

    static = rng.randn(n_stocks, n_days, 4).astype(np.float32)
    pk = rng.randn(n_stocks, n_days, 12).astype(np.float32)
    po = rng.randn(n_stocks, n_days, 6).astype(np.float32)

    y_dir = np.full((n_stocks, n_days), -100, dtype=np.int64)
    y_dir[:, seq_len:] = rng.randint(0, 3, (n_stocks, n_days - seq_len))
    y_ret = (rng.randn(n_stocks, n_days) * 0.02).astype(np.float32)
    y_vol = np.abs(rng.randn(n_stocks, n_days) * 0.01).astype(np.float32)

    ones_bool = np.ones((n_stocks, n_days), dtype=bool)
    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": y_vol,
        "date_indices": np.tile(np.arange(n_days, dtype=np.int64)[None, :],
                                (n_stocks, 1)),
        "global_dates": dates.to_numpy(dtype="datetime64[ns]"),
        "observation_mask": ones_bool,
        "entry_eligible_mask": ones_bool,
        "return_target_mask": ones_bool,
        "vol_target_mask": ones_bool,
        "decision_eligible_mask": ones_bool,
        "history_eligible_mask": ones_bool,
        "universe_eligible_mask": ones_bool,
        "forward_vol_nobs": np.full((n_stocks, n_days), horizon, dtype=np.int32),
        "realized_return": (rng.randn(n_stocks, n_days) * 0.02).astype(np.float32),
        "close_price": np.full((n_stocks, n_days), 10.0, dtype=np.float32),
        "open_price": np.full((n_stocks, n_days), 10.0, dtype=np.float32),
        "stock_codes": stocks,
        "past_known_cols": [f"pk_{i}" for i in range(12)],
        "past_observed_cols": [f"po_{i}" for i in range(6)],
    }


class TestPanelStoreRoundTrip:
    def test_roundtrip_values_and_keys(self, tmp_path):
        """A clean save/load returns every required key, memmap-backed, with
        elementwise-identical values."""
        panel = _storeable_panel(seed=0)
        written = save_panel_memmap(panel, tmp_path)
        loaded = load_panel_memmap(tmp_path)

        # Every required array key round-trips as a memmap-backed ndarray.
        for key in _PANEL_ARRAY_KEYS:
            assert key in loaded, f"missing required key: {key}"
            arr = loaded[key]
            assert isinstance(arr, np.ndarray), key
            assert isinstance(arr, np.memmap), (
                f"{key} not memmap-backed — load_panel_memmap must mmap")
        # JSON-list keys round-trip as plain Python lists.
        for key in _PANEL_JSON_KEYS:
            assert key in loaded and isinstance(loaded[key], list), key

        # Elementwise equality on the big arrays + masks + index grids.
        for key in ("static_features", "past_known", "past_observed",
                    "y_direction", "y_return", "y_volatility",
                    "date_indices", "observation_mask"):
            np.testing.assert_array_equal(np.asarray(loaded[key]), panel[key],
                                          err_msg=key)
        # datetime64 round-trip + row identity.
        np.testing.assert_array_equal(loaded["global_dates"],
                                      panel["global_dates"])
        assert loaded["stock_codes"] == panel["stock_codes"]

        # Returned filename list matches what was written.
        assert written == sorted(
            [f"{k}.npy" for k in _PANEL_ARRAY_KEYS]
            + [f"{k}.json" for k in _PANEL_JSON_KEYS])
        assert panel_store_complete(tmp_path) is True

    def test_complete_marker_absent_on_empty(self, tmp_path):
        """A fresh/empty dir is never a complete store."""
        assert panel_store_complete(tmp_path) is False

    def test_missing_file_error_names_file(self, tmp_path):
        """A store missing a required file raises FileNotFoundError naming it."""
        save_panel_memmap(_storeable_panel(seed=2), tmp_path)
        (tmp_path / "y_return.npy").unlink()
        with pytest.raises(FileNotFoundError) as ei:
            load_panel_memmap(tmp_path)
        assert "y_return.npy" in str(ei.value)

    def test_atomic_write_on_failure(self, tmp_path, monkeypatch):
        """An interrupted save must never leave a complete-looking store."""
        panel = _storeable_panel(seed=3)
        orig = panel_store._atomic_npy
        calls = {"n": 0}

        def flaky(out, name, arr):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated write failure")
            orig(out, name, arr)

        monkeypatch.setattr(panel_store, "_atomic_npy", flaky)
        with pytest.raises(RuntimeError):
            save_panel_memmap(panel, tmp_path)
        # The completeness marker is only written after every array is in
        # place, so the interrupted dir must not read as complete.
        assert panel_store_complete(tmp_path) is False
        assert not (tmp_path / "complete.json").exists()

    def test_memmap_slices_are_lazy(self, tmp_path):
        """A loaded big array can be window-sliced without materializing — a
        basic slice returns another memmap view, not a dense copy."""
        save_panel_memmap(_storeable_panel(seed=4), tmp_path)
        pk = load_panel_memmap(tmp_path)["past_known"]
        assert isinstance(pk, np.memmap)
        window = pk[:, 30:90]
        assert isinstance(window, np.memmap), (
            "basic slicing of a memmap should stay a lazy view")
