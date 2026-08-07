"""Unit tests for the panel_builders subpackage (§二十一 T17 review).

Focused regression net for the T8 memmap swap: ``PanelArrays`` allocation,
per-stock write via ``TargetBuilder``, ``sanitize``/``assemble`` round-trip,
and a small pure-function check for ``EligibilityBuilder``.  These do NOT
exercise ``build_panel_features`` end-to-end (that is covered elsewhere) —
they pin the builder/container seams directly.
"""

import numpy as np
import os
import pandas as pd
import pytest

from stoke_ml.features.panel_builders._arrays import PanelArrays
from stoke_ml.features.panel_builders._targets import TargetBuilder


def _tiny_panel():
    """Two stocks, 8 trading days, monotone prices -> predictable targets."""
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
    ])
    dfs = []
    for code, base in [("000001", 10.0), ("000002", 20.0)]:
        px = base + np.arange(len(dates)) * 0.1
        dfs.append(pd.DataFrame({
            "date": dates,
            "stock_code": code,
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": np.full(len(dates), 1e6),
            "amount": np.full(len(dates), 1e8),
        }))
    all_dates = sorted({d for df in dfs for d in pd.to_datetime(df["date"])})
    date_to_pos = {str(d.date()): i for i, d in enumerate(all_dates)}
    return dfs, ["000001", "000002"], len(all_dates), date_to_pos


def test_panel_arrays_round_trip():
    """TargetBuilder writes into PanelArrays; sanitize+assemble round-trips."""
    dfs, valid_codes, max_T, date_to_pos = _tiny_panel()
    N = len(dfs)

    arrays = PanelArrays(N, max_T)
    TargetBuilder(horizon=1).compute(dfs, valid_codes, max_T, date_to_pos, arrays)

    # Target arrays have the right shape; every row is a real observation.
    assert arrays.obs.shape == (N, max_T)
    assert arrays.obs.all()
    assert arrays.entry.all()
    assert arrays.y_dir.shape == (N, max_T)
    assert arrays.y_ret.shape == (N, max_T)
    # Strictly increasing prices -> 1-day forward return > dir_threshold -> up
    # (2) for every column except the last (no exit window there -> -100).
    assert (arrays.y_dir[0] == 2).sum() == max_T - 1
    assert arrays.y_dir[0, -1] == -100

    # Feature-grid round trip: write a sentinel into past_known (within the
    # [-10, 10] sanitize clip window so it survives), drop a NaN into static
    # (sanitize must zero it), then confirm both round-trip through assemble.
    arrays.alloc_features(static_dim=1, pk_dim=1, po_dim=1)
    arrays.pk[:, :, 0] = 3.5
    arrays.static[:, :, 0] = np.nan
    arrays.sanitize()

    all_dates = sorted({d for df in dfs for d in pd.to_datetime(df["date"])})
    out = arrays.assemble(
        global_dates=np.array(all_dates, dtype="datetime64[ns]"),
        decision_arr=np.ones((N, max_T), dtype=bool),
        history_arr=np.ones((N, max_T), dtype=bool),
        universe_eligible_arr=np.ones((N, max_T), dtype=bool),
        fill_prob_arr=np.zeros(max_T, dtype=np.float64),
        pk_cols=["pk0"],
        po_cols=["po0"],
        valid_codes=valid_codes,
    )

    expected_keys = {
        "static_features", "past_known", "past_observed",
        "y_direction", "y_return", "y_volatility",
        "date_indices", "global_dates",
        "observation_mask", "entry_eligible_mask",
        "return_target_mask", "vol_target_mask",
        "forward_vol_nobs", "realized_return", "fill_prob",
        "decision_eligible_mask", "history_eligible_mask",
        "universe_eligible_mask",
        "close_price", "open_price",
        "past_known_cols", "past_observed_cols", "stock_codes",
    }
    assert set(out.keys()) == expected_keys
    assert out["past_known"].shape == (N, max_T, 1)
    assert out["past_known"][0, 3, 0] == 3.5  # sentinel survived sanitize
    assert out["static_features"][0, 3, 0] == 0.0  # NaN zeroed by sanitize
    assert out["past_known_cols"] == ["pk0"]
    assert out["stock_codes"] == valid_codes
    assert out["global_dates"].dtype == "datetime64[ns]"
    assert out["date_indices"].shape == (N, max_T)


def test_eligibility_builder_universe_mask(monkeypatch):
    """EligibilityBuilder produces a sane decision/history/universe mask."""
    from stoke_ml.features.panel_builders._eligibility import EligibilityBuilder

    # Deterministic universe config — don't depend on the repo's config.yaml.
    monkeypatch.setattr(
        "stoke_ml.config.load_config",
        lambda: {"universe": {
            "long_suspension_days": 60,
            "suspension_lookback": 60,
            "min_amount_60d": 5_000_000,
        }},
    )

    N, T = 2, 5
    obs = np.ones((N, T), dtype=bool)
    first_col = np.zeros(N, dtype=np.int32)  # listed from day 0
    amt60 = np.full((N, T), 1e8, dtype=np.float32)  # very liquid
    has_amount = np.ones(N, dtype=bool)

    decision, history, universe = EligibilityBuilder(
        seq_len=2, min_history=1,
    ).compute(obs, first_col, amt60, has_amount)

    assert decision.shape == (N, T)
    assert history.shape == (N, T)
    assert universe.shape == (N, T)
    # decision[0] is False (no close[t-1] yet), decision[1:] True.
    assert not decision[:, 0].any()
    assert decision[:, 1:].all()
    # Universe eligible on columns 1+: listed from day 0, never long-suspended,
    # amt60 well above the liquidity floor.  Column 0 is legitimately False —
    # the causal 60d-turnover shift leaves no turnover known at close[-1] for
    # the entry at day 0, so the liquidity floor fails there.
    assert not universe[:, 0].any()
    assert universe[:, 1:].all()


# ═══════════════════════════════════════════════════════════════════════════
# T8: Memmap sink tests (§七-P0)
# ═══════════════════════════════════════════════════════════════════════════


class TestPanelArraysMemmap:
    """T8: ``PanelArrays`` with ``sink_dir`` writes feature grids to disk
    via ``open_memmap``; builder-write and sanitize semantics are preserved;
    the dense and memmap paths produce element-identical results."""

    def test_sink_allocation_creates_npy_files(self, tmp_path):
        """When sink_dir is set, alloc_features creates .npy files."""
        arrays = PanelArrays(3, 50, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=2, pk_dim=4, po_dim=3)
        assert (tmp_path / "static_features.npy").is_file()
        assert (tmp_path / "past_known.npy").is_file()
        assert (tmp_path / "past_observed.npy").is_file()
        # Files have .npy headers — np.load can read them.
        loaded = np.load(tmp_path / "past_known.npy", mmap_mode="r")
        assert loaded.shape == (3, 50, 4)
        assert loaded.dtype == np.float32

    def test_sink_write_and_sanitize(self, tmp_path):
        """Scatter-write onto a memmap-backed grid + sanitize produces the
        same result as dense allocation."""
        N, T = 2, 10
        # Dense path
        dense = PanelArrays(N, T)
        dense.alloc_features(static_dim=1, pk_dim=2, po_dim=1)
        dense.static[0, :5, 0] = np.nan
        dense.static[0, 5:, 0] = 100.0  # extreme — should be clipped
        dense.pk[1, :, 0] = np.inf
        dense.pk[1, :, 1] = -np.inf
        dense.sanitize()

        # Memmap path
        mmap_dir = str(tmp_path / "sink")
        mmap = PanelArrays(N, T, sink_dir=mmap_dir)
        mmap.alloc_features(static_dim=1, pk_dim=2, po_dim=1)
        mmap.static[0, :5, 0] = np.nan
        mmap.static[0, 5:, 0] = 100.0
        mmap.pk[1, :, 0] = np.inf
        mmap.pk[1, :, 1] = -np.inf
        mmap.sanitize()

        np.testing.assert_array_equal(
            np.asarray(mmap.static), np.asarray(dense.static),
            err_msg="static memmap vs dense mismatch")
        np.testing.assert_array_equal(
            np.asarray(mmap.pk), np.asarray(dense.pk),
            err_msg="pk memmap vs dense mismatch")
        np.testing.assert_array_equal(
            np.asarray(mmap.po), np.asarray(dense.po),
            err_msg="po memmap vs dense mismatch")

    def test_sink_2d_arrays_stay_dense(self, tmp_path):
        """Only the 3 big grids are memmap-backed; 2-D arrays stay dense."""
        arrays = PanelArrays(3, 50, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=2, pk_dim=4, po_dim=3)
        assert isinstance(arrays.static, np.memmap), "grid should be memmap"
        assert not isinstance(arrays.y_dir, np.memmap), "2D target should be dense"
        assert not isinstance(arrays.obs, np.memmap), "2D mask should be dense"
        assert not isinstance(arrays.close_price, np.memmap), "price should be dense"

    def test_flush_sink_closes_mappings(self, tmp_path):
        """flush_sink flushes + closes mmap; subsequent file ops succeed."""
        arrays = PanelArrays(3, 50, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=2, pk_dim=4, po_dim=3)
        arrays.pk[0, 0, 0] = 42.0
        arrays.flush_sink()
        # After flush+close, the file should be readable without lock issues.
        loaded = np.load(tmp_path / "past_known.npy", mmap_mode="r")
        assert loaded[0, 0, 0] == 42.0

    def test_close_memmap_grids_module_function(self, tmp_path):
        """The module-level close_memmap_grids is the single source of truth
        for the flush+close sequence: it returns the set of keys it closed and
        leaves dense/absent keys untouched."""
        from stoke_ml.features.panel_builders._arrays import (
            close_memmap_grids,
        )

        N, T = 2, 10
        arrays = PanelArrays(N, T, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=1, pk_dim=2, po_dim=1)
        arrays.pk[0, 0, 0] = 7.0
        arrays.flush_sink()  # close the memmap mappings

        # A dict mirroring build_panel_features output: memmap grids + a dense
        # 2-D array + metadata lists.
        panel = {
            "static_features": arrays.static,
            "past_known": arrays.pk,
            "past_observed": arrays.po,
            "y_direction": np.full((N, T), -100, dtype=np.int64),
            "stock_codes": ["A", "B"],
        }
        closed = close_memmap_grids(panel)
        # Only the three feature grids are memmaps -> exactly those returned.
        assert closed == {"static_features", "past_known", "past_observed"}
        # The closed memmaps keep their header props (used by
        # _feature_schema_hash to record the T4 schema binding).
        assert panel["past_known"].shape == (N, T, 2)
        assert panel["past_known"].dtype == np.float32
        # Dense / non-array keys untouched.
        assert panel["y_direction"][0, 0] == -100
        assert panel["stock_codes"] == ["A", "B"]

        # Calling again is a no-op (already closed) and still returns the keys.
        closed_again = close_memmap_grids(panel)
        assert closed_again == {"static_features", "past_known", "past_observed"}

    def test_close_memmap_grids_mmap_attr_canary(self, tmp_path):
        """Canary: ``close_memmap_grids`` closes the mapping via the private
        numpy ``_mmap`` attribute.  Assert the attribute exists on the venv
        numpy (2.2.6 per uv.lock) so a future dependency bump that renames it
        fails this test instead of silently leaking the Windows file lock."""
        arrays = PanelArrays(2, 10, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=1, pk_dim=1, po_dim=1)
        assert hasattr(arrays.static, "_mmap"), (
            "numpy memmap must expose _mmap (venv numpy 2.2.6) — the "
            "close-memmap cleanup relies on it to release the Windows file "
            "lock; bump the pin or update close_memmap_grids if this fails"
        )
        assert arrays.static._mmap is not None
        arrays.flush_sink()  # must not raise

    def test_assemble_returns_memmap_objects(self, tmp_path):
        """assemble() returns memmap-backed objects for the three big grids
        when a sink_dir is configured."""
        N, T = 2, 10
        arrays = PanelArrays(N, T, sink_dir=str(tmp_path))
        arrays.alloc_features(static_dim=1, pk_dim=2, po_dim=1)
        arrays.pk[:, :, 0] = 3.0
        arrays.sanitize()
        out = arrays.assemble(
            global_dates=pd.date_range(
                "2024-01-02", periods=T, freq="B",
            ).to_numpy(dtype="datetime64[ns]"),
            decision_arr=np.ones((N, T), dtype=bool),
            history_arr=np.ones((N, T), dtype=bool),
            universe_eligible_arr=np.ones((N, T), dtype=bool),
            fill_prob_arr=np.zeros(T, dtype=np.float64),
            pk_cols=["pk0", "pk1"],
            po_cols=["po0"],
            valid_codes=["A", "B"],
        )
        assert isinstance(out["static_features"], np.memmap)
        assert isinstance(out["past_known"], np.memmap)
        assert isinstance(out["past_observed"], np.memmap)
        # 2-D arrays stay dense.
        assert not isinstance(out["y_direction"], np.memmap)
        assert out["past_known"][0, 0, 0] == 3.0


class TestBuildPanelFeaturesMemmap:
    """T8: ``build_panel_features`` with ``memmap_dir`` produces results
    element-identical to the dense path, and the store round-trips
    through save/load."""

    def _build_fixture(self, tmp_path, memmap_dir=None):
        """Build a small panel via build_panel_features for testing."""
        from stoke_ml.features.pipeline import FeaturePipeline
        from unittest.mock import patch

        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        dfs = []
        for code, base in [("000001", 10.0), ("000002", 20.0), ("000003", 30.0)]:
            px = base + np.arange(len(dates)) * 0.1
            dfs.append(pd.DataFrame({
                "date": dates,
                "stock_code": code,
                "open": px,
                "high": px + 0.05,
                "low": px - 0.05,
                "close": px,
                "volume": np.full(len(dates), 1e6, dtype=np.float64),
                "amount": np.full(len(dates), 1e8, dtype=np.float64),
            }))
        panel = pd.concat(dfs, ignore_index=True)
        fp = FeaturePipeline(seq_len=10, use_sentiment=False,
                            use_guba=False, use_comment=False,
                            use_announcements=False, use_margin=False,
                            use_northbound=False, use_dragon_tiger=False,
                            use_fundamental=False, use_earnings=False,
                            use_valuation=False, use_etf_flow=False,
                            use_capital_flow=False, use_block_trade=False,
                            use_shareholder=False, use_lockup=False,
                            use_dividend=False, use_board=False,
                            use_sector=False, use_concept=False,
                            use_industry=False, use_macro=False,
                            use_pledge=False, use_index_membership=False,
                            use_market_env=False, use_market_env_refine=False,
                            use_limit_up=False, use_topic=False)
        return fp.build_panel_features(
            panel, horizon=1, memmap_dir=memmap_dir)

    def test_dense_vs_memmap_identical(self, tmp_path):
        """Both paths produce element-identical arrays."""
        dense = self._build_fixture(tmp_path, memmap_dir=None)
        memmap_out = self._build_fixture(
            tmp_path, memmap_dir=str(tmp_path / "sink"))

        for key in ("static_features", "past_known", "past_observed",
                    "y_direction", "y_return", "y_volatility",
                    "observation_mask", "entry_eligible_mask",
                    "return_target_mask", "vol_target_mask",
                    "realized_return", "close_price", "open_price",
                    "decision_eligible_mask", "history_eligible_mask",
                    "universe_eligible_mask"):
            np.testing.assert_array_equal(
                np.asarray(memmap_out[key]), np.asarray(dense[key]),
                err_msg=f"memmap vs dense mismatch on {key}")
        # Metadata identical
        assert memmap_out["stock_codes"] == dense["stock_codes"]
        assert memmap_out["past_known_cols"] == dense["past_known_cols"]
        assert memmap_out["past_observed_cols"] == dense["past_observed_cols"]
        np.testing.assert_array_equal(
            memmap_out["global_dates"], dense["global_dates"])

    def _meta(self):
        """A meta fingerprint matching this class's small fixture."""
        return {
            "horizon": 1, "seq_len": 10, "start": "2024-01-01",
            "end": "2024-12-31", "universe": "csi300", "n_stocks": 3,
            "feature_switches": {"seq_len": 10},
            "config_hash": "hash-abc", "git_commit": "abc123",
            "label_policy": "carry_to_last_close_v1",
        }

    def _flush_close(self, panel_data):
        """Flush + close the three big memmap grids, KEEPING them in the dict
        (their .dtype header props stay readable; save_panel_memmap reads only
        header props for the feature_schema_hash binding and skips rewriting
        the files via skip_npy)."""
        for key in ("static_features", "past_known", "past_observed"):
            arr = panel_data.get(key)
            if arr is not None and isinstance(arr, np.memmap):
                arr.flush()
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()

    def test_round_trip_save_load(self, tmp_path):
        """Build with memmap sink, save the remaining arrays + metadata (with a
        real meta= so the feature_schema_hash binding is recorded), then load
        everything back — values match the dense reference and the T4 schema
        binding round-trips."""
        import json
        from stoke_ml.models.panel.panel_store import (
            save_panel_memmap, load_panel_memmap, panel_store_complete,
        )

        dense = self._build_fixture(tmp_path, memmap_dir=None)
        store = str(tmp_path / "store")
        os.makedirs(store, exist_ok=True)

        # Build with memmap sink into the store dir.
        memmap_out = self._build_fixture(tmp_path, memmap_dir=store)
        self._flush_close(memmap_out)
        meta = self._meta()

        # Write remaining small arrays + metadata.  skip_npy keeps
        # save_panel_memmap from rewriting the already-sunk big grids (and
        # from touching a closed memmap's data); the arrays REMAIN in the dict
        # so _feature_schema_hash can read their .dtype and record the binding.
        save_panel_memmap(memmap_out, store, meta=meta,
                          skip_npy={"static_features", "past_known",
                                    "past_observed"})
        assert panel_store_complete(store)

        # T4 schema binding: meta.json must record a non-None feature_schema_hash
        # (the review bug: deleting the grids made this None and silently
        # disabled the tamper guard at load).
        with open(os.path.join(store, "meta.json"), encoding="utf-8") as fh:
            recorded = json.load(fh)
        assert recorded.get("feature_schema_hash"), (
            "feature_schema_hash must be recorded on the memmap-sink path — "
            "the grids must stay in panel_data for _feature_schema_hash")
        assert recorded.get("stock_order_hash"), (
            "stock_order_hash must also be recorded")

        # Reload the full store; expected_meta validates against meta.json.
        loaded = load_panel_memmap(store, expected_meta=meta)

        # All values match the dense reference.
        for key in ("static_features", "past_known", "past_observed",
                    "y_direction", "y_return", "y_volatility",
                    "observation_mask", "entry_eligible_mask",
                    "return_target_mask", "vol_target_mask",
                    "realized_return", "close_price", "open_price"):
            np.testing.assert_array_equal(
                np.asarray(loaded[key]), np.asarray(dense[key]),
                err_msg=f"round-trip mismatch on {key}")
        assert loaded["stock_codes"] == dense["stock_codes"]

    def test_sink_store_records_schema_binding_tamper_refused(self, tmp_path):
        """Regression for the review bug: the memmap-sink store path MUST record
        feature_schema_hash in meta.json, and a tampered past_known_cols.json
        is REFUSED at load (T4 schema binding preserved)."""
        import json
        from stoke_ml.models.panel.panel_store import (
            save_panel_memmap, load_panel_memmap, panel_store_complete,
        )

        store = str(tmp_path / "store")
        os.makedirs(store, exist_ok=True)
        memmap_out = self._build_fixture(tmp_path, memmap_dir=store)
        self._flush_close(memmap_out)
        save_panel_memmap(memmap_out, store, meta=self._meta(),
                          skip_npy={"static_features", "past_known",
                                    "past_observed"})
        assert panel_store_complete(store)

        # Tamper past_known_cols.json — must be refused by the self-consistency
        # guard (feature_schema_hash), not silently accepted.
        cols_path = os.path.join(store, "past_known_cols.json")
        with open(cols_path, encoding="utf-8") as fh:
            cols = json.load(fh)
        cols[-1] = "pk_tampered"
        with open(cols_path, "w", encoding="utf-8") as fh:
            json.dump(cols, fh)

        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(store, expected_meta=self._meta())
        assert "feature_schema_hash" in str(ei.value)

    def test_tracemalloc_memmap_peak_below_dense(self, tmp_path):
        """Build a small synthetic universe twice; memmap peak < dense peak."""
        import tracemalloc

        # Use a larger fixture to show a measurable difference.
        dates = pd.date_range("2024-01-02", periods=100, freq="B")
        dfs = []
        for i in range(10):
            code = f"{i:06d}"
            px = 10.0 + np.arange(len(dates)) * 0.05
            dfs.append(pd.DataFrame({
                "date": dates,
                "stock_code": code,
                "open": px,
                "high": px + 0.05,
                "low": px - 0.05,
                "close": px,
                "volume": np.full(len(dates), 1e6, dtype=np.float64),
                "amount": np.full(len(dates), 1e8, dtype=np.float64),
            }))
        panel = pd.concat(dfs, ignore_index=True)

        from stoke_ml.features.pipeline import FeaturePipeline
        fp = FeaturePipeline(seq_len=10, use_sentiment=False,
                            use_guba=False, use_comment=False,
                            use_announcements=False, use_margin=False,
                            use_northbound=False, use_dragon_tiger=False,
                            use_fundamental=False, use_earnings=False,
                            use_valuation=False, use_etf_flow=False,
                            use_capital_flow=False, use_block_trade=False,
                            use_shareholder=False, use_lockup=False,
                            use_dividend=False, use_board=False,
                            use_sector=False, use_concept=False,
                            use_industry=False, use_macro=False,
                            use_pledge=False, use_index_membership=False,
                            use_market_env=False, use_market_env_refine=False,
                            use_limit_up=False, use_topic=False)

        # Dense peak
        tracemalloc.start()
        _dense = fp.build_panel_features(panel, horizon=1)
        _, dense_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memmap peak
        sink = str(tmp_path / "sink")
        tracemalloc.start()
        _mmap = fp.build_panel_features(panel, horizon=1, memmap_dir=sink)
        _, mmap_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Clean up memmaps so tmp_path cleanup doesn't fail on Windows.
        for key in ("static_features", "past_known", "past_observed"):
            arr = _mmap.get(key)
            if arr is not None and isinstance(arr, np.memmap):
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()
                del _mmap[key]
        del _dense

        assert mmap_peak < dense_peak, (
            f"memmap peak ({mmap_peak}) should be below dense peak "
            f"({dense_peak}) — the 3 big grids must not be in RAM")
