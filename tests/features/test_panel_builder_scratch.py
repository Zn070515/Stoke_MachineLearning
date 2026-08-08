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
    _same_filesystem,
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

def test_streaming_disk_required_gb_splits_final_vs_scratch():
    """Returns (final_panel_gb, scratch_gb) — no margin, split by destination."""
    final_gb, scratch_gb = _streaming_disk_required_gb(100, 243, 1700)
    assert final_gb == pytest.approx(100 * 243 * 1700 * 4 / (1024 ** 3))
    assert scratch_gb == pytest.approx(
        100 * 243 * 1700 * 8 * 1.3 / (1024 ** 3))
    assert final_gb > 0.0 and scratch_gb > 0.0


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


def test_enforce_streaming_disk_space_sizes_volumes_independently(
        monkeypatch, tmp_path):
    """§v18-6: scratch and panel_store on DIFFERENT volumes are sized
    independently — the panel-only volume must not be asked to also fit the
    scratch pickles (the old combined check would falsely refuse here)."""
    import scripts.production.train_panel_panel as tpp
    scratch = tmp_path / "scratch"
    panel = tmp_path / "panel"
    os.makedirs(scratch)
    os.makedirs(panel)
    monkeypatch.setattr(tpp, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(tpp, "_scratch_safety_margin_gb", lambda: 0.0)
    usage = {os.path.realpath(str(scratch)): 10 ** 15,      # huge scratch
             os.path.realpath(str(panel)): 1.0 * (1024 ** 3)}  # panel fits final only
    monkeypatch.setattr(
        tpp.shutil, "disk_usage",
        lambda p: SimpleNamespace(free=usage[os.path.realpath(str(p))]))
    args = _scratch_args(panel_store=str(panel))
    args.prebuilt = None
    args.minute = False
    args.seq_len = None
    args.vintage_policy = "revision-safe"
    args.allow_fundamental_ablation = False
    args.start = "2024-01-01"
    args.end = "2024-12-31"
    # 500 stocks x ~243 timesteps x 1700 feats → final ≈ 0.8 GB, scratch ≈ 8 GB.
    # panel free 1 GB fits final+margin but NOT final+scratch+margin — so the
    # per-fs check passes (no SystemExit) where the old combined sizing against
    # the panel volume would have refused.
    got = _enforce_streaming_disk_space(
        args, ["000001"] * 500, str(tmp_path), str(scratch))
    assert got is not None
    assert got[0] > 0.0


def test_enforce_streaming_disk_space_refuses_when_panel_volume_small(
        monkeypatch, tmp_path):
    """§v18-6: a panel_store volume too small for the FINAL grids alone is
    refused — the per-fs check must catch an over-full panel volume (the old
    scratch-only check would have falsely passed)."""
    import scripts.production.train_panel_panel as tpp
    scratch = tmp_path / "scratch"
    panel = tmp_path / "panel"
    os.makedirs(scratch)
    os.makedirs(panel)
    monkeypatch.setattr(tpp, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(tpp, "_scratch_safety_margin_gb", lambda: 0.0)
    usage = {os.path.realpath(str(scratch)): 10 ** 15,
             os.path.realpath(str(panel)): 1}   # panel free = 1 byte
    monkeypatch.setattr(
        tpp.shutil, "disk_usage",
        lambda p: SimpleNamespace(free=usage[os.path.realpath(str(p))]))
    args = _scratch_args(panel_store=str(panel))
    args.prebuilt = None
    args.minute = False
    args.seq_len = None
    args.vintage_policy = "revision-safe"
    args.allow_fundamental_ablation = False
    args.start = "2024-01-01"
    args.end = "2024-12-31"
    with pytest.raises(SystemExit) as ei:
        _enforce_streaming_disk_space(
            args, ["000001"] * 500, str(tmp_path), str(scratch))
    assert "final panel" in str(ei.value)


def test_same_filesystem_same_volume(tmp_path):
    """Two paths under the same tmp volume resolve to the same st_dev."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    os.makedirs(a)
    os.makedirs(b)
    assert _same_filesystem(str(a), str(b))


def test_same_filesystem_missing_paths_conservative(tmp_path):
    """Nonexistent paths → OSError → True (combined conservative check)."""
    assert _same_filesystem(
        str(tmp_path / "nope1"), str(tmp_path / "nope2"))


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
    # manifest is back to "running" (preserving the feature-switch fingerprint
    # — a faithful same-switches crash).
    os.remove(os.path.join(scratch, "000002.pkl"))
    _write_run_manifest(
        scratch, "run-1", stage="running",
        feature_switches_hash=pb._feature_switches_hash(_tiny_pipeline()))

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
    # Corrupt one pickle + set the manifest back to "running", preserving the
    # feature-switch fingerprint (a faithful crash: same switches, same run).
    with open(os.path.join(scratch, "000001.pkl"), "wb") as fh:
        fh.write(b"garbage partial pickle")
    _write_run_manifest(
        scratch, "run-1", stage="running",
        feature_switches_hash=pb._feature_switches_hash(_tiny_pipeline()))

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


def test_resume_refuses_feature_switch_mismatch(monkeypatch, tmp_path):
    """A scratch whose manifest fingerprint differs from the current run is
    REFUSED on resume — resuming would skip old-schema pickles and engineer
    new-schema ones into a silent hybrid panel."""
    import stoke_ml.features.panel_builder as pb

    scratch = str(tmp_path / "store" / "scratch" / "run-1")
    _build_and_preserve(monkeypatch, scratch, "run-1", str(tmp_path / "sink1"))
    # Simulate a crash (stage back to "running") under DIFFERENT feature
    # switches: corrupt the recorded fingerprint to an unrelated switch set.
    m = _read_run_manifest(scratch)
    assert m["feature_switches_hash"]           # the builder now records it
    m["stage"] = "running"
    m["feature_switches_hash"] = "0" * 64
    with open(os.path.join(scratch, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(m, fh)

    with pytest.raises(RuntimeError, match="feature switches"):
        _tiny_pipeline().build_panel_features(
            _tiny_panel(), horizon=1, memmap_dir=str(tmp_path / "sink2"),
            scratch_dir=scratch, run_id="run-1")


def test_done_manifest_not_resumed(monkeypatch, tmp_path, caplog):
    """A preserved COMPLETED run (stage=done) is NOT resumed — a fresh full
    rebuild happens with a warning, and the completed run's id is not adopted."""
    import stoke_ml.features.panel_builder as pb

    scratch = str(tmp_path / "store" / "scratch" / "run-1")
    _build_and_preserve(monkeypatch, scratch, "run-1", str(tmp_path / "sink1"))
    assert _read_run_manifest(scratch)["stage"] == "done"   # completed, not crashed

    engineered = []
    real_engineer = pb._engineer_stock

    def spy(pipeline, code, *a, **k):
        engineered.append(code)
        return real_engineer(pipeline, code, *a, **k)

    monkeypatch.setattr(pb, "_engineer_stock", spy)
    with caplog.at_level("WARNING"):
        _tiny_pipeline().build_panel_features(
            _tiny_panel(), horizon=1, memmap_dir=str(tmp_path / "sink2"),
            scratch_dir=scratch, run_id=None,
        )
    # Not resumed: every stock re-engineered (a fresh full rebuild), with a
    # warning about the completed run, and a fresh run id (not the old one).
    assert engineered == ["000001", "000002", "000003"]
    assert "COMPLETED" in caplog.text
    m = _read_run_manifest(scratch)
    assert m["run_id"] != "run-1"
    assert m.get("resumed") is not True


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


def test_exact_disk_backstop_sizes_volumes_independently(monkeypatch, tmp_path):
    """§v18-6: the exact backstop sizes the panel volume separately from the
    scratch volume — a sink that fits the final grids but NOT grids+scratch
    pickles passes the per-fs net where the old combined net would refuse."""
    import stoke_ml.features.panel_builder as pb
    scratch = str(tmp_path / "scratch")
    sink = str(tmp_path / "sink")
    os.makedirs(scratch)
    os.makedirs(sink)
    real_stat = os.stat

    def _stat_with_dev(p, dev):
        # Override only st_dev; keep every other field so pathlib / pandas
        # (st_mode / st_size) still work on these real directories.
        r = real_stat(p)
        return os.stat_result((
            r.st_mode, r.st_ino, dev, r.st_nlink,
            r.st_uid, r.st_gid, r.st_size,
            r.st_atime, r.st_mtime, r.st_ctime,
        ))

    def fake_stat(p, *a, **k):
        p = os.path.realpath(str(p))
        if p == os.path.realpath(scratch):
            return _stat_with_dev(p, 1)
        if p == os.path.realpath(sink):
            return _stat_with_dev(p, 2)
        return real_stat(p, *a, **k)

    monkeypatch.setattr(pb.os, "stat", fake_stat)
    # margin 0 so the grid/scratch bytes dominate (default 5 GB would swamp them)
    monkeypatch.setattr(pb, "_DEFAULT_SCRATCH_SAFETY_MARGIN_GB", 0.0)
    # Pad each per-stock pickle to 10 MB → scratch_bytes ≈ 30 MB (≫ grid_bytes).
    # Only the three pickles under scratch are padded; everything else is real.
    real_getsize = os.path.getsize
    scratch_real = os.path.realpath(scratch)

    def fake_getsize(p):
        rp = os.path.realpath(str(p))
        if rp.startswith(scratch_real + os.sep) and rp.endswith(".pkl"):
            return 10 * 1024 * 1024
        return real_getsize(p)

    monkeypatch.setattr(pb.os.path, "getsize", fake_getsize)
    scratch_bytes = 3 * 10 * 1024 * 1024            # 30 MB — what the backstop sees
    usage = {os.path.realpath(scratch): scratch_bytes,      # scratch: only fits the pickles
             os.path.realpath(sink): 1 * 1024 * 1024}       # sink: 1 MB — fits grids
    monkeypatch.setattr(
        pb.shutil, "disk_usage",
        lambda p: SimpleNamespace(free=usage[os.path.realpath(str(p))]))
    # per-fs: scratch 30MB ≤ 30MB ✓, grids (~KB) < 1MB ✓ → build succeeds.  Old
    # combined sum-vs-scratch-only: 30MB + grids > 30MB → would have refused.
    out = _tiny_pipeline().build_panel_features(
        _tiny_panel(), horizon=1, memmap_dir=sink,
        scratch_dir=scratch, run_id="r1")
    assert len(out["stock_codes"]) == 3


def test_exact_disk_backstop_refuses_when_panel_volume_small(monkeypatch, tmp_path):
    """§v18-6: the cross-fs exact backstop refuses when the PANEL (sink) volume
    alone is too full — the panel volume is checked independently, not only the
    scratch drive."""
    import stoke_ml.features.panel_builder as pb
    scratch = str(tmp_path / "scratch")
    sink = str(tmp_path / "sink")
    os.makedirs(scratch)
    os.makedirs(sink)
    real_stat = os.stat

    def _stat_with_dev(p, dev):
        r = real_stat(p)
        return os.stat_result((
            r.st_mode, r.st_ino, dev, r.st_nlink,
            r.st_uid, r.st_gid, r.st_size,
            r.st_atime, r.st_mtime, r.st_ctime,
        ))

    def fake_stat(p, *a, **k):
        p = os.path.realpath(str(p))
        if p == os.path.realpath(scratch):
            return _stat_with_dev(p, 1)
        if p == os.path.realpath(sink):
            return _stat_with_dev(p, 2)
        return real_stat(p, *a, **k)

    monkeypatch.setattr(pb.os, "stat", fake_stat)
    monkeypatch.setattr(pb, "_DEFAULT_SCRATCH_SAFETY_MARGIN_GB", 0.0)
    usage = {os.path.realpath(scratch): 10 ** 15,   # scratch: huge
             os.path.realpath(sink): 1}              # sink: 1 byte → grid problem
    monkeypatch.setattr(
        pb.shutil, "disk_usage",
        lambda p: SimpleNamespace(free=usage[os.path.realpath(str(p))]))
    with pytest.raises(RuntimeError, match="final grids"):
        _tiny_pipeline().build_panel_features(
            _tiny_panel(), horizon=1, memmap_dir=sink,
            scratch_dir=scratch, run_id="r1")


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
