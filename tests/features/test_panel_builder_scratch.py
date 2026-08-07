"""§T7 scratch-dir management for the streaming panel build.

Covers the audit §四/§十五 remediation: ``--scratch-dir`` resolution
(explicit > ``<panel-store>/scratch/<run_id>/`` > system temp), the disk
pre-check (estimate-based pre-build refusal + the builder's exact post-Pass-1
backstop), ``run_manifest.json``, crash-resume (a same-scratch re-run skips
already-engineered per-stock pickles), and the startup stale-sweep of orphan
scratch dirs.
"""

import json
import os
import re
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.panel_builder import (
    _cleanup_stale_scratch_dirs,
    _read_run_manifest,
    _scratch_run_id,
    _write_run_manifest,
)
from scripts.production.train_panel_panel import (
    _enforce_streaming_disk_space,
    _resolve_scratch_dir,
    _streaming_disk_required_gb,
)


# ── tiny streaming-build fixtures (mirror TestBuildPanelFeaturesMemmap) ──────

def _tiny_panel():
    """Three stocks, 20 business days, deterministic seeded prices."""
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    dfs = []
    for code, base in [("000001", 10.0), ("000002", 20.0), ("000003", 30.0)]:
        px = (base + np.arange(len(dates)) * 0.1
              + np.random.RandomState(int(code)).normal(0, 0.2, len(dates)))
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
    return pd.concat(dfs, ignore_index=True)


def _tiny_pipeline():
    from stoke_ml.features.pipeline import FeaturePipeline
    return FeaturePipeline(
        seq_len=10,
        use_sentiment=False, use_guba=False, use_comment=False,
        use_announcements=False, use_margin=False, use_northbound=False,
        use_dragon_tiger=False, use_fundamental=False, use_earnings=False,
        use_valuation=False, use_etf_flow=False, use_capital_flow=False,
        use_block_trade=False, use_shareholder=False, use_lockup=False,
        use_dividend=False, use_board=False, use_sector=False,
        use_concept=False, use_industry=False, use_macro=False,
        use_pledge=False, use_index_membership=False, use_market_env=False,
        use_market_env_refine=False, use_limit_up=False, use_topic=False,
    )


def _preserve_scratch(monkeypatch, scratch_path):
    """Make the builder's finally-cleanup a no-op for *scratch_path*.

    Simulates a hard-killed build (where the ``finally`` never runs) so a test
    can inspect / resume the scratch after a build.
    """
    import stoke_ml.features.panel_builder as pb
    real_rmtree = shutil.rmtree
    target = os.path.abspath(scratch_path)

    def keep(path, *a, **k):
        if os.path.abspath(path) == target:
            return
        real_rmtree(path, *a, **k)

    monkeypatch.setattr(pb.shutil, "rmtree", keep)


# ── run-id / manifest / stale-sweep primitives ─────────────────────────────

def test_scratch_run_id_convention():
    """The run id follows preprocess_new_data.py's YYYYMMDD-HHMMSS-<pid>."""
    assert re.fullmatch(r"\d{8}-\d{6}-\d+", _scratch_run_id())


def test_run_manifest_write_read_round_trip(tmp_path):
    """run_manifest.json carries the §T7 contract fields and survives a resume."""
    scratch = str(tmp_path / "scratch")
    os.makedirs(scratch)
    _write_run_manifest(scratch, "run-abc", stage="running")
    m = _read_run_manifest(scratch)
    assert m["run_id"] == "run-abc"
    assert m["stage"] == "running"
    assert "start_time" in m and "pid" in m
    # start_time is preserved across a resume re-write.
    first_start = m["start_time"]
    _write_run_manifest(scratch, "run-abc", stage="done", resumed=True)
    m2 = _read_run_manifest(scratch)
    assert m2["start_time"] == first_start
    assert m2["stage"] == "done"
    assert m2["resumed"] is True


def test_read_run_manifest_missing_or_corrupt(tmp_path):
    assert _read_run_manifest(str(tmp_path / "nope")) is None
    bad = str(tmp_path / "bad")
    os.makedirs(bad)
    with open(os.path.join(bad, "run_manifest.json"), "w") as fh:
        fh.write("not json {{{")
    assert _read_run_manifest(bad) is None


def test_cleanup_stale_scratch_dirs_removes_old_keeps_fresh(tmp_path):
    """Only old scratch-named dirs are swept; fresh / excluded / unrelated kept."""
    root = str(tmp_path / "scratch_root")
    old = os.path.join(root, "panel_stream_scratch_old")
    fresh = os.path.join(root, "panel_stream_scratch_fresh")
    current = os.path.join(root, "panel_stream_scratch_current")
    other = os.path.join(root, "unrelated")
    for p in (old, fresh, current, other):
        os.makedirs(p)
        with open(os.path.join(p, "x"), "w") as fh:
            fh.write("x")
    old_t = time.time() - 30 * 86400
    os.utime(old, (old_t, old_t))
    os.utime(current, (old_t, old_t))
    removed = _cleanup_stale_scratch_dirs(
        root, 7, prefix="panel_stream_scratch_", exclude=current)
    assert not os.path.isdir(old)       # stale + prefixed → swept
    assert os.path.isdir(current)       # excluded (the live run)
    assert os.path.isdir(fresh)         # not yet stale
    assert os.path.isdir(other)         # name mismatch
    assert set(removed) == {old}


def test_cleanup_stale_scratch_dirs_missing_root(tmp_path):
    assert _cleanup_stale_scratch_dirs(str(tmp_path / "nope"), 7) == []


# ── scratch-dir resolution ─────────────────────────────────────────────────

def _scratch_args(**overrides):
    base = {
        "scratch_dir": None,
        "panel_store": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_scratch_dir_explicit_wins():
    """Explicit --scratch-dir wins and disables the stale sweep of siblings."""
    scratch_dir, run_id, cleanup_root, prefix = _resolve_scratch_dir(
        _scratch_args(scratch_dir="C:/chosen/scratch",
                      panel_store="C:/store"))
    assert scratch_dir == "C:/chosen/scratch"
    assert cleanup_root is None and prefix is None
    assert re.fullmatch(r"\d{8}-\d{6}-\d+", run_id)


def test_resolve_scratch_dir_derived_from_panel_store():
    """Scratch lands under <panel-store>/scratch/<run_id>/, NOT system temp."""
    # A store path deliberately NOT under the system temp dir, so the "not
    # system temp" claim is verifiable (a temp-relative store would make the
    # derived scratch temp-relative too).  Pure string resolution — the
    # function creates nothing.
    store = os.path.abspath("t7_store")
    scratch_dir, run_id, cleanup_root, prefix = _resolve_scratch_dir(
        _scratch_args(panel_store=store))
    assert os.path.abspath(scratch_dir).startswith(
        os.path.abspath(os.path.join(store, "scratch")))
    assert os.path.basename(scratch_dir) == run_id
    assert not scratch_dir.startswith(tempfile.gettempdir())
    assert cleanup_root == os.path.join(store, "scratch")
    assert prefix is None


def test_resolve_scratch_dir_temp_fallback():
    """No panel_store → system temp under the panel_stream_scratch_ prefix."""
    scratch_dir, run_id, cleanup_root, prefix = _resolve_scratch_dir(
        _scratch_args())
    assert scratch_dir.startswith(
        os.path.join(tempfile.gettempdir(), "panel_stream_scratch_"))
    assert cleanup_root == tempfile.gettempdir()
    assert prefix == "panel_stream_scratch_"


# ── disk pre-check (estimate-based, pre-build) ─────────────────────────────

def test_streaming_disk_required_gb_includes_margin():
    """required == final-panel + scratch + safety margin (never under-budget)."""
    required = _streaming_disk_required_gb(100, 243, 1700)
    final_panel = 100 * 243 * 1700 * 4 / (1024 ** 3)
    scratch = 100 * 243 * 1700 * 8 * 1.3 / (1024 ** 3)
    assert required > final_panel + scratch
    assert required > 0.0


def test_enforce_streaming_disk_space_refuses(monkeypatch, tmp_path):
    """Insufficient free space → clean refusal BEFORE any build work."""
    class _FakeUsage:
        free = 1  # 1 byte free

    monkeypatch.setattr(
        "scripts.production.train_panel_panel.shutil.disk_usage",
        lambda path: _FakeUsage())
    args = _scratch_args(panel_store=str(tmp_path / "store"))
    args.prebuilt = None
    args.minute = False
    args.seq_len = None
    args.vintage_policy = "revision-safe"
    args.allow_fundamental_ablation = False
    args.start = "2024-01-01"
    args.end = "2024-12-31"
    scratch = str(tmp_path / "store" / "scratch" / "run-x")
    with pytest.raises(SystemExit) as ei:
        _enforce_streaming_disk_space(args, ["000001"] * 500,
                                      str(tmp_path), scratch)
    assert "exceeds free space" in str(ei.value)


def test_enforce_streaming_disk_space_ok(monkeypatch, tmp_path):
    """Plenty of free space → no refusal, returns (required, free)."""
    class _FakeUsage:
        free = 10 ** 15

    monkeypatch.setattr(
        "scripts.production.train_panel_panel.shutil.disk_usage",
        lambda path: _FakeUsage())
    args = _scratch_args(panel_store=str(tmp_path / "store"))
    args.prebuilt = None
    args.minute = False
    args.seq_len = None
    args.vintage_policy = "revision-safe"
    args.allow_fundamental_ablation = False
    args.start = "2024-01-01"
    args.end = "2024-12-31"
    scratch = str(tmp_path / "store" / "scratch" / "run-x")
    got = _enforce_streaming_disk_space(args, ["000001"], str(tmp_path), scratch)
    assert got is not None
    assert got[0] > 0.0


def test_enforce_streaming_disk_space_skipped_non_streaming(tmp_path):
    """No --panel-store (dense build) → the check is a no-op."""
    args = _scratch_args(panel_store=None)
    assert _enforce_streaming_disk_space(
        args, ["000001"], str(tmp_path), str(tmp_path / "s")) is None


# ── end-to-end streaming scratch behavior ─────────────────────────────────

def _build_and_preserve(monkeypatch, scratch, run_id, sink, cleanup_root=None):
    """Run a tiny streaming build into *scratch*, keeping the scratch alive."""
    _preserve_scratch(monkeypatch, scratch)
    return _tiny_pipeline().build_panel_features(
        _tiny_panel(), horizon=1, memmap_dir=sink,
        scratch_dir=scratch, run_id=run_id,
        scratch_cleanup_root=cleanup_root,
    )


def test_scratch_under_panel_store_and_manifest_written(monkeypatch, tmp_path):
    """A store-derived scratch holds run_manifest.json with the right fields."""
    store = str(tmp_path / "store")
    scratch = os.path.join(store, "scratch", "run-1")
    _build_and_preserve(monkeypatch, scratch, "run-1",
                        str(tmp_path / "sink"), cleanup_root=os.path.join(store, "scratch"))
    # scratch landed under the panel store's scratch/ subdir — NOT the
    # panel_stream_scratch_* temp-fallback location (the meaningful contrast:
    # pytest tmp_path is itself under the system temp, so a store-derived
    # scratch is inherently temp-relative when the store is a tmp_path).
    assert scratch.startswith(os.path.join(store, "scratch"))
    assert not scratch.startswith(
        os.path.join(tempfile.gettempdir(), "panel_stream_scratch_"))
    m = _read_run_manifest(scratch)
    assert m is not None
    assert m["run_id"] == "run-1"
    assert m["stage"] == "done"
    assert "start_time" in m and "pid" in m


def test_resume_skips_existing_pickles(monkeypatch, tmp_path):
    """A same-scratch re-run re-uses existing per-stock pickles (Pass-1 resume)."""
    import stoke_ml.features.panel_builder as pb

    scratch = str(tmp_path / "store" / "scratch" / "run-1")
    sink1 = str(tmp_path / "sink1")
    sink2 = str(tmp_path / "sink2")

    # Build once (simulated successful run; scratch preserved for the test).
    out1 = _build_and_preserve(monkeypatch, scratch, "run-1", sink1)

    # Simulate a hard-kill mid-Pass-1: one stock's pickle is missing and the
    # manifest is back to "running".
    os.remove(os.path.join(scratch, "000002.pkl"))
    _write_run_manifest(scratch, "run-1", stage="running")

    # Spy on _engineer_stock to count which stocks are (re-)engineered.
    engineered = []
    real_engineer = pb._engineer_stock

    def spy(pipeline, code, *a, **k):
        engineered.append(code)
        return real_engineer(pipeline, code, *a, **k)

    monkeypatch.setattr(pb, "_engineer_stock", spy)

    out2 = _tiny_pipeline().build_panel_features(
        _tiny_panel(), horizon=1, memmap_dir=sink2,
        scratch_dir=scratch, run_id="run-1",
    )
    # Only the missing stock was re-engineered; the other two pickles reused.
    assert engineered == ["000002"]
    # Resume produces the identical panel (same pickles + deterministic
    # re-engineering of the missing stock).
    assert list(out2["stock_codes"]) == list(out1["stock_codes"])
    np.testing.assert_allclose(
        np.asarray(out2["close_price"]), np.asarray(out1["close_price"]))
    # The re-run adopted the crashed run's identity + marked it resumed.
    m = _read_run_manifest(scratch)
    assert m["run_id"] == "run-1"
    assert m["stage"] == "done"
    assert m["resumed"] is True


def test_resume_corrupt_pickle_reengineered(monkeypatch, tmp_path):
    """A corrupt/partial pickle (kill during the write) is re-engineered."""
    import stoke_ml.features.panel_builder as pb

    scratch = str(tmp_path / "store" / "scratch" / "run-1")
    _build_and_preserve(monkeypatch, scratch, "run-1", str(tmp_path / "sink1"))
    # Corrupt one pickle + set the manifest back to "running".
    with open(os.path.join(scratch, "000001.pkl"), "wb") as fh:
        fh.write(b"garbage partial pickle")
    _write_run_manifest(scratch, "run-1", stage="running")

    engineered = []
    real_engineer = pb._engineer_stock

    def spy(pipeline, code, *a, **k):
        engineered.append(code)
        return real_engineer(pipeline, code, *a, **k)

    monkeypatch.setattr(pb, "_engineer_stock", spy)
    out2 = _tiny_pipeline().build_panel_features(
        _tiny_panel(), horizon=1, memmap_dir=str(tmp_path / "sink2"),
        scratch_dir=scratch, run_id="run-1",
    )
    assert "000001" in engineered   # corrupt → re-engineered
    assert len(out2["stock_codes"]) == 3


def test_builder_startup_sweeps_stale_siblings(monkeypatch, tmp_path):
    """Startup sweeps orphan scratch dirs older than the stale window."""
    root = str(tmp_path / "scratch_root")
    old = os.path.join(root, "panel_stream_scratch_old")
    os.makedirs(old)
    old_t = time.time() - 30 * 86400
    os.utime(old, (old_t, old_t))

    scratch = os.path.join(root, "panel_stream_scratch_current")
    _tiny_pipeline().build_panel_features(
        _tiny_panel(), horizon=1, memmap_dir=str(tmp_path / "sink"),
        scratch_dir=scratch, run_id="r1",
        scratch_cleanup_root=root,
        scratch_cleanup_prefix="panel_stream_scratch_",
        scratch_stale_days=7,
    )
    assert not os.path.isdir(old)        # swept at startup
    assert not os.path.isdir(scratch)    # current run removed in finally


def test_exact_disk_backstop_refuses(monkeypatch, tmp_path):
    """The builder's post-Pass-1 exact backstop refuses when the scratch drive
    cannot hold the KNOWN footprint (the hard net that cannot underestimate)."""
    class _FakeUsage:
        free = 1

    monkeypatch.setattr(
        "stoke_ml.features.panel_builder.shutil.disk_usage",
        lambda path: _FakeUsage())
    with pytest.raises(RuntimeError, match="exceeds"):
        _tiny_pipeline().build_panel_features(
            _tiny_panel(), horizon=1, memmap_dir=str(tmp_path / "sink"),
            scratch_dir=str(tmp_path / "scratch"), run_id="r1")


# ── _resolve_panel threads the scratch spec into the builder ───────────────

def _capture_pipeline():
    calls = []

    class _FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_panel_features(self, panel, **kw):
            calls.append({"panel": panel, **kw})
            return {"__dummy": True}

    return _FakePipeline, calls


def test_resolve_panel_threads_scratch_spec(monkeypatch, tmp_path):
    """_resolve_panel forwards the resolved scratch dir + run_id + cleanup spec
    into build_panel_features for a streaming (panel-store) build."""
    import scripts.production.train_panel_panel as tpp
    import stoke_ml.data.storage as storage_mod

    class _FakeStorage:
        def __init__(self, data_dir):
            self.data_dir = data_dir

        def load_daily(self, code, start, end, require_valid_manifest=True):
            return pd.DataFrame({
                "date": pd.to_datetime(["2022-01-04"]),
                "open": [1.0], "high": [1.1], "low": [0.9],
                "close": [1.0], "volume": [100], "amount": [100.0],
            })

    fake_pipe, calls = _capture_pipeline()
    monkeypatch.setattr("scripts.production.train_panel_panel.FeaturePipeline",
                        fake_pipe)
    monkeypatch.setattr(storage_mod, "DataStorage", _FakeStorage)
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.load_index_membership",
        lambda data_dir, indices: pd.DataFrame(
            columns=["stock_code", "index_code", "in_date", "out_date"]))
    monkeypatch.setattr(
        "scripts.production.train_panel_panel._enforce_streaming_disk_space",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.production.train_panel_panel._panel_store_meta",
        lambda *a, **k: {"meta": True})
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.save_panel_memmap",
        lambda *a, **k: [])
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.close_memmap_grids",
        lambda panel_data: set())
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.load_panel_memmap",
        lambda *a, **k: {"__dummy": True, "channel_coverage_manifest": {}})

    store = str(tmp_path / "store")
    args = _scratch_args(panel_store=store)
    args.vintage_policy = "revision-safe"
    args.minute = False
    args.horizon = 1
    args.start = "2022-01-01"
    args.end = "2022-01-31"
    args.universe = "csi300"
    args.prebuilt = None
    args.seq_len = None
    args.allow_high_risk_universe = False
    args.allow_fundamental_ablation = False
    args.no_formal = False
    args.no_require_quality_gate = False
    args.feature_profile = None
    args.require_aux_channels = ""
    args.no_aux = True
    args.require_feature_manifest = False

    panel_data, _ = tpp._resolve_panel(
        args, ["600000"], 60, str(tmp_path), set(), _store_load=False)
    # _resolve_panel's store path re-loads the store, which legitimately adds
    # channel_coverage_manifest into the returned dict — assert the sentinel
    # instead of dict equality.
    assert panel_data["__dummy"] is True
    assert len(calls) == 1
    assert calls[0]["scratch_dir"].startswith(
        os.path.join(store, "scratch"))
    assert calls[0]["scratch_cleanup_root"] == os.path.join(store, "scratch")
    assert calls[0]["scratch_cleanup_prefix"] is None
    assert calls[0]["run_id"]
