"""Unit tests for the panel_builders subpackage (§二十一 T17 review).

Focused regression net for the T8 memmap swap: ``PanelArrays`` allocation,
per-stock write via ``TargetBuilder``, ``sanitize``/``assemble`` round-trip,
and a small pure-function check for ``EligibilityBuilder``.  These do NOT
exercise ``build_panel_features`` end-to-end (that is covered elsewhere) —
they pin the builder/container seams directly.

§T5 streaming/two-pass tests added at the bottom of
``TestBuildPanelFeaturesMemmap``.
"""

import gc
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile

from stoke_ml.features.panel_builders._arrays import PanelArrays
from stoke_ml.features.panel_builders._targets import TargetBuilder


def _noisy_px(base, n_dates, code):
    """Deterministic per-stock close price noise (seeded by stock code).

    §T5: a fixture where every stock shares the SAME linear price sequence
    makes the cross-sectional oscillator columns (cci/rsi/kdj/wr) identical
    across stocks.  On the all-sparse fixtures (count<5 on every date) the
    sparse fallback's per-date std is 0 and clips to the 1e-8 floor, which
    amplifies the ~1e-13 float64 summation-order diff between the dense
    row-level cumsum and the streaming per-date-sum cumsum to a ~1e-5
    absolute z-diff.  Adding seeded noise keeps the cross-sections
    non-constant so the amplification cannot occur (production cross-sections
    are never constant).
    """
    return (base + np.arange(n_dates) * 0.1
            + np.random.RandomState(int(code)).normal(0, 0.2, n_dates))


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
            px = _noisy_px(base, len(dates), code)
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

    def _assert_grid_matches(self, key, actual, expected):
        """Byte-identity on every grid except the §T5 controlled ULP diff.

        past_known / past_observed carry the per-date z-scored columns.  The
        streaming path accumulates stats per stock with float64 running sums
        while the dense path concats all frames then runs a pandas groupby —
        float summation order differs, so those two grids match only within
        rtol=1e-5/atol=1e-6 (§T5).  Every other grid (static features, masks,
        prices, quantile ranks) must stay bit-identical.
        """
        if key in ("past_known", "past_observed"):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected),
                rtol=1e-5, atol=1e-6,
                err_msg=f"z-scored grid mismatch on {key}")
        else:
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected),
                err_msg=f"grid mismatch on {key}")

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
            self._assert_grid_matches(
                key, memmap_out[key], dense[key],
            )
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
            self._assert_grid_matches(
                key, loaded[key], dense[key],
            )
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

    # ═══════════════════════════════════════════════════════════════════════
    # §T5: Streaming / two-pass tests
    # ═══════════════════════════════════════════════════════════════════════

    def _assert_streaming_eq_dense(self, dense, streaming, extra_grid_keys=()):
        """Assert streaming-vs-dense identity on all arrays + metadata.

        past_known / past_observed are compared with the §T5 controlled-ULP
        tolerance (rtol=1e-5/atol=1e-6); every other grid stays byte-exact
        (see _assert_grid_matches)."""
        grid_keys = (
            "static_features", "past_known", "past_observed",
            "y_direction", "y_return", "y_volatility",
            "observation_mask", "entry_eligible_mask",
            "return_target_mask", "vol_target_mask",
            "realized_return", "close_price", "open_price",
            "decision_eligible_mask", "history_eligible_mask",
            "universe_eligible_mask",
        ) + tuple(extra_grid_keys)
        for key in grid_keys:
            if key in dense and key in streaming:
                self._assert_grid_matches(
                    key, streaming[key], dense[key],
                )
        assert streaming["stock_codes"] == dense["stock_codes"]
        assert streaming["past_known_cols"] == dense["past_known_cols"]
        assert streaming["past_observed_cols"] == dense["past_observed_cols"]
        np.testing.assert_array_equal(
            streaming["global_dates"], dense["global_dates"])

    # ── daily_membership fixture ──────────────────────────────────────

    def test_streaming_vs_dense_with_membership(self, tmp_path):
        """Streaming matches dense when daily_membership restricts the
        statistical set AND produces a zero-member date."""
        from stoke_ml.features.pipeline import FeaturePipeline

        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        codes = ["000001", "000002", "000003"]
        dfs = []
        for code, base in zip(codes, [10.0, 20.0, 30.0]):
            px = _noisy_px(base, len(dates), code)
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

        # Membership: all stocks are members on dates[0:5] and dates[6:],
        # but NOT on dates[5] (zero-member date).
        membership = pd.DataFrame({
            "stock_code": [c for c in codes for _ in range(2)],
            "in_date": [
                dates[0], dates[6],  # stock 0
                dates[0], dates[6],  # stock 1
                dates[0], dates[6],  # stock 2
            ],
            "out_date": [
                dates[5], pd.NaT,   # stock 0: member D0-D4, D6-end
                dates[5], pd.NaT,   # stock 1
                dates[5], pd.NaT,   # stock 2
            ],
        })

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

        dense = fp.build_panel_features(
            panel, horizon=1, daily_membership=membership,
        )
        memmap_out = fp.build_panel_features(
            panel, horizon=1, daily_membership=membership,
            memmap_dir=str(tmp_path / "sink"),
        )
        self._assert_streaming_eq_dense(dense, memmap_out)

    # ── cross-sectional fundamental fixture ───────────────────────────

    def test_streaming_vs_dense_with_cs_fundamental(self, tmp_path):
        """Streaming matches dense when cross-sectional fundamental is active
        (use_fundamental_refine=True + sector_code in feature frames)."""
        from stoke_ml.features.pipeline import FeaturePipeline

        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        codes = ["000001", "000002", "000003"]
        dfs = []
        for code_idx, (code, base) in enumerate(
            zip(codes, [10.0, 20.0, 30.0])
        ):
            px = _noisy_px(base, len(dates), code)
            dfs.append(pd.DataFrame({
                "date": dates,
                "stock_code": code,
                "open": px,
                "high": px + 0.05,
                "low": px - 0.05,
                "close": px,
                "volume": np.full(len(dates), 1e6, dtype=np.float64),
                "amount": np.full(len(dates), 1e8, dtype=np.float64),
                "sector_code": 1000 + code_idx,
                "pe_ttm": np.full(len(dates), 15.0 + code_idx * 5,
                                  dtype=np.float64),
                "pb_mrq": np.full(len(dates), 2.0 + code_idx * 0.5,
                                  dtype=np.float64),
                "ps_ttm": np.full(len(dates), 3.0 + code_idx * 0.3,
                                  dtype=np.float64),
                "debt_ratio": np.full(len(dates), 0.5 + code_idx * 0.1,
                                      dtype=np.float64),
                "pe_percentile_252d": np.full(len(dates), 0.5,
                                              dtype=np.float64),
                "pb_percentile_252d": np.full(len(dates), 0.5,
                                              dtype=np.float64),
            }))
        panel = pd.concat(dfs, ignore_index=True)

        # use_fundamental_refine is coupled to use_fundamental (pipeline
        # forces it off when fundamental is off), so BOTH must be True.
        fp = FeaturePipeline(seq_len=10, use_sentiment=False,
                            use_guba=False, use_comment=False,
                            use_announcements=False, use_margin=False,
                            use_northbound=False, use_dragon_tiger=False,
                            use_fundamental=True, use_earnings=False,
                            use_valuation=False, use_etf_flow=False,
                            use_capital_flow=False, use_block_trade=False,
                            use_shareholder=False, use_lockup=False,
                            use_dividend=False, use_board=False,
                            use_sector=False, use_concept=False,
                            use_industry=False, use_macro=False,
                            use_pledge=False, use_index_membership=False,
                            use_market_env=False, use_market_env_refine=False,
                            use_limit_up=False, use_topic=False,
                            use_fundamental_refine=True)

        dense = fp.build_panel_features(panel, horizon=1)
        memmap_out = fp.build_panel_features(
            panel, horizon=1, memmap_dir=str(tmp_path / "sink"),
        )

        # The cs-fundamental path adds specific new columns; verify they
        # appear in the past_known_cols.
        cs_expected = {"pe_sector_ratio", "pb_sector_ratio",
                       "ps_sector_ratio", "leverage_warning",
                       "valuation_composite_z"}
        pk_set = set(dense["past_known_cols"])
        assert cs_expected.issubset(pk_set), (
            f"cs-fundamental cols {cs_expected} missing from "
            f"past_known_cols: {pk_set}")

        self._assert_streaming_eq_dense(dense, memmap_out)

    # ── sparse-date backfill fixture ───────────────────────────────────

    def test_streaming_vs_dense_sparse_backfill(self, tmp_path):
        """Streaming matches dense on a sparse-date fixture where the
        expanding-moment fallback is exercised (3 stocks with early dates,
        5 stocks starting later → count<5 on early dates)."""
        from stoke_ml.features.pipeline import FeaturePipeline

        all_dates = pd.date_range("2024-01-02", periods=30, freq="B")
        # 3 "early" stocks — full date range
        early_dates = all_dates
        # 5 "late" stocks — only last 20 dates (first 10 dates sparse)
        late_dates = all_dates[10:]

        dfs = []
        for i in range(3):
            code = f"{i:06d}"
            px = _noisy_px(10.0, len(early_dates), code)
            dfs.append(pd.DataFrame({
                "date": early_dates,
                "stock_code": code,
                "open": px,
                "high": px + 0.05,
                "low": px - 0.05,
                "close": px,
                "volume": np.full(len(early_dates), 1e6, dtype=np.float64),
                "amount": np.full(len(early_dates), 1e8, dtype=np.float64),
            }))
        for i in range(3, 8):
            code = f"{i:06d}"
            px = _noisy_px(10.0, len(late_dates), code)
            dfs.append(pd.DataFrame({
                "date": late_dates,
                "stock_code": code,
                "open": px,
                "high": px + 0.05,
                "low": px - 0.05,
                "close": px,
                "volume": np.full(len(late_dates), 1e6, dtype=np.float64),
                "amount": np.full(len(late_dates), 1e8, dtype=np.float64),
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

        dense = fp.build_panel_features(panel, horizon=1)
        memmap_out = fp.build_panel_features(
            panel, horizon=1, memmap_dir=str(tmp_path / "sink"),
        )
        self._assert_streaming_eq_dense(dense, memmap_out)

    # ── tracemalloc sublinear growth ───────────────────────────────────

    def test_streaming_tracemalloc_sublinear(self, tmp_path):
        """Streaming peak memory grows sublinearly with N (all_feat_dfs
        residence eliminated)."""
        import tracemalloc

        from stoke_ml.features.pipeline import FeaturePipeline

        def _build_n(n_stocks: int, sink_dir: str | None):
            dates = pd.date_range("2024-01-02", periods=60, freq="B")
            dfs = []
            for i in range(n_stocks):
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

            gc.collect()
            tracemalloc.start()
            result = fp.build_panel_features(
                panel, horizon=1, memmap_dir=sink_dir,
            )
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Clean up memmaps.
            if sink_dir is not None:
                for key in ("static_features", "past_known", "past_observed"):
                    arr = result.get(key)
                    if arr is not None and isinstance(arr, np.memmap):
                        if hasattr(arr, "_mmap") and arr._mmap is not None:
                            arr._mmap.close()
                        del result[key]
            del result
            gc.collect()
            return peak

        # Use two separate sink dirs so memmap files don't collide.
        N = 8
        peak_n = _build_n(N, str(tmp_path / "sink_n"))
        peak_2n = _build_n(N * 2, str(tmp_path / "sink_2n"))

        # Sublinear: doubling stocks should NOT double peak memory.
        # The stats pass uses the dense normalizer for byte-identical stats
        # (loads date+norm_cols for all stocks once), so peak grows with
        # the number of stocks — but sublinearly (well under 2×) because
        # the full feature frames are never resident.
        assert peak_2n < peak_n * 1.8, (
            f"streaming peak should grow sublinearly with N: "
            f"peak({N})={peak_n}, peak({2*N})={peak_2n}, "
            f"ratio={peak_2n/peak_n:.2f}"
        )
