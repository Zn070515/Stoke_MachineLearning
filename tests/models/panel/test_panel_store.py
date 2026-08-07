"""Round-trip tests for the §十六 panel memmap persistence layer.

``save_panel_memmap`` writes every build_panel_features array to
``{out}/{name}.npy`` atomically (temp file + os.replace) and a
``complete.json`` marker LAST; ``load_panel_memmap`` re-opens them lazily via
``np.load(mmap_mode='r')``.  These tests pin the contract: a complete store
round-trips values exactly, an interrupted save never looks complete, a
missing required file is reported by name, and the meta.json config guard
refuses a stale store (wrong horizon/universe/feature switches) instead of
silently training on wrong targets.
"""
import hashlib
import json
import logging
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.production.train_panel import (
    _panel_store_meta,
    _resolve_panel,
    _validate_panel_store_path,
)
from stoke_ml.models.panel import panel_store
from stoke_ml.models.panel.panel_store import (
    _PANEL_ARRAY_KEYS,
    _PANEL_JSON_KEYS,
    _META_FILE,
    load_panel_memmap,
    panel_store_complete,
    save_panel_memmap,
    verify_panel_store_chunks,
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
        "fill_prob": np.full(n_days, np.nan, dtype=np.float64),  # §T13
        "entry_fill_prob": np.full(n_days, np.nan, dtype=np.float64),  # §T10a
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

        # Returned filename list matches what was written (fill_prob and
        # entry_fill_prob are extra §T13/§T10a diagnostics persisted alongside
        # the required keys — they are NOT in _PANEL_ARRAY_KEYS).
        assert written == sorted(
            [f"{k}.npy" for k in _PANEL_ARRAY_KEYS]
            + [f"{k}.json" for k in _PANEL_JSON_KEYS]
            + ["fill_prob.npy", "entry_fill_prob.npy"])
        assert panel_store_complete(tmp_path) is True

    def test_channel_coverage_manifest_optional_roundtrip(self, tmp_path):
        """A channel_coverage_manifest persisted alongside the panel round-trips
        through the store, and a store saved WITHOUT it (legacy) still loads —
        it is an OPTIONAL JSON key, not part of the required-file contract."""
        panel = _storeable_panel(seed=1)
        panel["channel_coverage_manifest"] = {
            "sentiment": {"stock_coverage": 0.9, "cell_coverage": 0.8,
                          "status": "OK"},
            "industry": {"date_coverage": 0.95, "status": "OK"},
        }
        written = save_panel_memmap(panel, tmp_path)
        loaded = load_panel_memmap(tmp_path)
        assert loaded["channel_coverage_manifest"] == panel["channel_coverage_manifest"]
        assert "channel_coverage_manifest.json" in written
        assert panel_store_complete(tmp_path) is True

        # A store saved WITHOUT the manifest (legacy) still loads — the required
        # file check at load does NOT include the optional key.
        legacy_dir = tmp_path / "legacy"
        save_panel_memmap(_storeable_panel(seed=2), legacy_dir)
        legacy = load_panel_memmap(legacy_dir)
        assert "channel_coverage_manifest" not in legacy
        assert panel_store_complete(legacy_dir) is True

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


def _meta(**over):
    """A valid meta.json fingerprint; override any field per test."""
    m = {
        "horizon": 5, "seq_len": 60, "start": "2024-01-01",
        "end": "2024-12-31", "universe": "csi300", "n_stocks": 10,
        "feature_switches": {"seq_len": 60, "minute_mode": False,
                             "use_sentiment": True},
        "config_hash": "hash-abc", "git_commit": "abc123",
    }
    m.update(over)
    return m


class TestPanelStoreMetaGuard:
    """The meta.json config guard: a stale store is refused, never trusted."""

    def test_critical_mismatch_refuses(self, tmp_path):
        """A research-critical field (horizon) mismatch raises RuntimeError
        naming the field, so a --horizon-5 store can never feed a horizon-10 run."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(horizon=5))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta(horizon=10))
        msg = str(ei.value)
        assert "horizon" in msg
        assert "5" in msg and "10" in msg

    def test_feature_switch_mismatch_refuses(self, tmp_path):
        """A feature-switch mismatch is equally critical — the stored panel's
        column set would differ from the requested one."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(feature_switches={"use_sentiment": True}))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(
                tmp_path,
                expected_meta=_meta(feature_switches={"use_sentiment": False}))
        assert "feature_switches" in str(ei.value)

    def test_git_commit_mismatch_warns_and_proceeds(self, tmp_path, caplog):
        """Non-critical drift (git_commit) warns loudly but proceeds — model-
        layer code does not change feature values."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(git_commit="abc123"))
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path,
                                       expected_meta=_meta(git_commit="def456"))
        assert loaded["stock_codes"] == _storeable_panel()["stock_codes"]
        assert any("git commit" in r.message for r in caplog.records)

    def test_config_hash_skipped_when_one_side_none(self, tmp_path):
        """config_hash is compared only when both sides have it (None means
        config could not load, mirroring cache_manifest)."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(config_hash=None))
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta(config_hash="x"))
        assert loaded["stock_codes"] == _storeable_panel()["stock_codes"]

    def test_missing_meta_refuses(self, tmp_path):
        """A store with no meta.json cannot vouch for its config and is refused."""
        save_panel_memmap(_storeable_panel(), tmp_path)  # no meta
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta())
        assert "no meta.json" in str(ei.value)

    def test_save_with_meta_writes_meta_before_complete(self, tmp_path):
        """meta.json is persisted BEFORE the complete.json marker, and a store
        with matching meta loads cleanly."""
        written = save_panel_memmap(_storeable_panel(), tmp_path,
                                    meta=_meta())
        assert _META_FILE in written
        assert (tmp_path / _META_FILE).is_file()
        assert (tmp_path / "complete.json").is_file()
        assert panel_store_complete(tmp_path) is True
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta())
        assert loaded["stock_codes"] == _storeable_panel()["stock_codes"]


class TestPanelStoreSelfBinding:
    """§八 (T4): the store binds to ITSELF — feature schema + stock order are
    recomputed from the store's own arrays/lists at load, so a tampered
    stock_codes.json / past_known_cols.json / feature dtype is refused instead
    of silently training the WRONG stocks (the dataset's max_stocks_per_date
    randperm samples rows by position, so a misaligned stock_codes trains the
    wrong codes with no error)."""

    def test_save_records_self_fingerprints(self, tmp_path):
        """save_panel_memmap merges stock_order_hash + feature_schema_hash into
        meta.json, recomputed from the ACTUAL panel_data (authoritative), not
        inherited from the caller's expected_meta."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        with open(tmp_path / _META_FILE, encoding="utf-8") as fh:
            recorded = json.load(fh)
        # Pin the RECORDED values to the recompute-from-panel, not just presence:
        # the recorded fingerprint must be exactly what the panel itself yields.
        assert recorded["stock_order_hash"] == panel_store._stock_order_hash(panel)
        assert recorded["feature_schema_hash"] == panel_store._feature_schema_hash(panel)

    def test_save_skips_fingerprints_panel_lacks_identity(self, tmp_path):
        """A panel missing the optional identity keys (stock_codes / *_cols /
        feature arrays) saves WITHOUT KeyError, and meta drops the keys its
        panel cannot recompute — every self-consistency recompute must be
        None-safe (never raise), since save iterates the same recompute table."""
        panel = _storeable_panel()
        for key in ("stock_codes", "past_known_cols", "past_observed_cols",
                    "past_known", "past_observed", "static_features"):
            panel.pop(key, None)
        save_panel_memmap(panel, tmp_path, meta=_meta())
        with open(tmp_path / _META_FILE, encoding="utf-8") as fh:
            recorded = json.load(fh)
        assert "stock_order_hash" not in recorded
        assert "feature_schema_hash" not in recorded

    def test_clean_store_with_matching_meta_loads(self, tmp_path):
        """Regression: a store saved with self-fingerprints and loaded with a
        matching expected_meta passes BOTH the expected-vs-recorded guard and
        the self-consistency check."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta())
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert list(loaded["past_known_cols"]) == panel["past_known_cols"]

    def test_tampered_stock_order_refuses(self, tmp_path):
        """Swapping two codes in stock_codes.json desynchronizes row identity
        from the arrays — hard-fail refused, never silently trained."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        codes = json.loads((tmp_path / "stock_codes.json").read_text(encoding="utf-8"))
        codes[0], codes[1] = codes[1], codes[0]
        (tmp_path / "stock_codes.json").write_text(
            json.dumps(codes), encoding="utf-8")
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta())
        assert "stock_order_hash" in str(ei.value)

    def test_tampered_feature_schema_refuses(self, tmp_path):
        """Renaming a past_known column desynchronizes the col list from the
        array's feature axis — hard-fail refused."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        cols = json.loads(
            (tmp_path / "past_known_cols.json").read_text(encoding="utf-8"))
        cols[-1] = "pk_tampered"
        (tmp_path / "past_known_cols.json").write_text(
            json.dumps(cols), encoding="utf-8")
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta())
        assert "feature_schema_hash" in str(ei.value)

    def test_tampered_feature_dtype_refuses(self, tmp_path):
        """A dtype change on disk (float32 → int16) flips the recomputed
        feature_schema_hash via the _array_dtype path — hard-fail refused."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        arr = np.load(tmp_path / "past_known.npy")
        np.save(tmp_path / "past_known.npy", arr.astype(np.int16))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta())
        assert "feature_schema_hash" in str(ei.value)

    def test_self_check_runs_without_expected_meta(self, tmp_path):
        """The self-consistency guard is NOT gated on expected_meta — a store
        loaded bare still refuses a tampered row order."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        codes = json.loads((tmp_path / "stock_codes.json").read_text(encoding="utf-8"))
        codes.reverse()
        (tmp_path / "stock_codes.json").write_text(
            json.dumps(codes), encoding="utf-8")
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path)
        assert "stock_order_hash" in str(ei.value)


class TestPanelStoreChunkManifest:
    """§十九 (T10b): content-integrity binding via a per-chunk SHA-256 manifest.

    The header bindings (stock_order_hash / feature_schema_hash) only pin
    IDENTITY — a modified VALUE inside a large .npy array (silent disk error /
    mid-array tamper) with unchanged shape/dtype is invisible to them.  T10b
    adds a chunk manifest: one SHA-256 per _CHUNK_BYTES block of each array's
    raw data, aggregated to a root hash recorded in meta.json.  A load ONLY
    checks the manifest cheaply (recomputed root from its own digests + the
    meta binding — no array I/O); verify_panel_store_chunks() re-hashes every
    chunk on demand."""

    def test_save_with_meta_writes_manifest_and_binds_root(self, tmp_path):
        """save_panel_memmap with meta writes chunk_manifest.json over every
        content array and records the manifest root in meta.json."""
        panel = _storeable_panel(seed=9)
        save_panel_memmap(panel, tmp_path, meta=_meta())
        manifest = json.loads(
            (tmp_path / "chunk_manifest.json").read_text(encoding="utf-8"))
        assert manifest["version"] == 1
        assert manifest["chunk_bytes"] == panel_store._CHUNK_BYTES
        # every content array (canonical + §T13/§T10a diagnostics) is hashed
        assert set(manifest["arrays"]) == (
            set(_PANEL_ARRAY_KEYS) | {"fill_prob", "entry_fill_prob"})
        for entry in manifest["arrays"].values():
            assert set(entry) == {"shape", "dtype", "n_chunks", "digests", "root"}
            assert len(entry["digests"]) == entry["n_chunks"]
        # the meta binding is the manifest root
        recorded = json.loads((tmp_path / _META_FILE).read_text(encoding="utf-8"))
        assert recorded["chunk_manifest_root_hash"] == manifest["root"]
        # root recomputes canonically from the arrays section
        assert manifest["root"] == panel_store._manifest_root(manifest["arrays"])
        # round-trips through the optional-JSON glob on load
        loaded = load_panel_memmap(tmp_path)
        assert loaded["chunk_manifest"] == manifest

    def test_save_without_meta_writes_no_manifest(self, tmp_path):
        """meta=None (no meta.json) skips the entire meta block — no manifest,
        store unaffected by T10b."""
        save_panel_memmap(_storeable_panel(seed=10), tmp_path)
        assert not (tmp_path / "chunk_manifest.json").exists()
        assert not (tmp_path / _META_FILE).exists()
        assert panel_store_complete(tmp_path) is True
        load_panel_memmap(tmp_path)  # legacy store loads cleanly

    def test_chunk_hashing_matches_independent_byte_split(self, tmp_path):
        """_hash_npy_chunks with a tiny chunk size reproduces an independent
        byte-level split of the file's raw data — every byte in exactly one
        chunk, per-chunk digests exact, root = SHA-256 of the concatenated
        digests."""
        panel = _storeable_panel(seed=7)
        save_panel_memmap(panel, tmp_path)
        for name in ("past_known", "static_features", "y_return"):
            path = tmp_path / f"{name}.npy"
            offset = panel_store._npy_data_offset(path)
            data = path.read_bytes()[offset:]
            chunk = 8
            expect = [
                hashlib.sha256(data[i:i + chunk]).hexdigest()
                for i in range(0, len(data), chunk)
            ]
            digests, root = panel_store._hash_npy_chunks(path, chunk_bytes=chunk)
            assert digests == expect
            assert len(digests) == math.ceil(len(data) / chunk)
            assert root == hashlib.sha256(
                b"".join(d.encode("ascii") for d in digests)).hexdigest()
            # chunk_bytes works positionally too
            assert digests == panel_store._hash_npy_chunks(path, chunk)[0]

    def test_build_manifest_small_chunks_records_n_chunks(self, tmp_path):
        """_build_chunk_manifest with a tiny chunk size records n_chunks =
        ceil(data_bytes / chunk_bytes) plus per-array shape/dtype/root."""
        panel = _storeable_panel(seed=8)
        save_panel_memmap(panel, tmp_path)
        npy_files = sorted(tmp_path.glob("*.npy"))
        manifest = panel_store._build_chunk_manifest(npy_files, chunk_bytes=8)
        assert manifest["chunk_bytes"] == 8
        pk = manifest["arrays"]["past_known"]
        off = panel_store._npy_data_offset(tmp_path / "past_known.npy")
        data_bytes = (tmp_path / "past_known.npy").stat().st_size - off
        assert pk["n_chunks"] == math.ceil(data_bytes / 8)
        assert len(pk["digests"]) == pk["n_chunks"]
        assert pk["shape"] == [10, 100, 12]
        assert pk["dtype"] == "float32"
        assert manifest["root"] == panel_store._manifest_root(manifest["arrays"])

    def test_clean_store_deep_verify_passes(self, tmp_path):
        """verify_panel_store_chunks returns a summary on a clean store."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        result = verify_panel_store_chunks(tmp_path)
        assert result["verified"] is True
        assert result["arrays_verified"] >= len(_PANEL_ARRAY_KEYS)
        assert result["chunks_verified"] >= len(_PANEL_ARRAY_KEYS)

    def test_verify_requires_manifest(self, tmp_path):
        """A store with no chunk_manifest.json cannot be deep-verified."""
        save_panel_memmap(_storeable_panel(), tmp_path)  # no meta → no manifest
        with pytest.raises(RuntimeError) as ei:
            verify_panel_store_chunks(tmp_path)
        assert "chunk_manifest" in str(ei.value)

    def test_middle_chunk_tamper_refused_by_verify(self, tmp_path, monkeypatch):
        """A single flipped byte in the MIDDLE of an array's data region (same
        shape/dtype — the header is untouched, so the identity bindings and the
        cheap load check still pass) is caught by the deep chunk verify, which
        names the array and chunk index."""
        monkeypatch.setattr(panel_store, "_CHUNK_BYTES", 8)  # tiny chunks
        panel = _storeable_panel(seed=3)
        save_panel_memmap(panel, tmp_path, meta=_meta())
        pk = tmp_path / "past_known.npy"
        offset = panel_store._npy_data_offset(pk)
        tamper_at = offset + 257  # chunk 32 of 8-byte chunks — inside data
        with open(pk, "r+b") as fh:
            fh.seek(tamper_at)
            b = fh.read(1)
            fh.seek(tamper_at)
            fh.write(bytes([b[0] ^ 0xFF]))
        # the tamper is VALUE-level: the cheap load (header bindings + manifest
        # root) still succeeds — the grid is simply corrupted inside.
        loaded = load_panel_memmap(tmp_path)
        assert isinstance(loaded["past_known"], np.memmap)
        # the deep verify refuses, naming the array + chunk index
        with pytest.raises(RuntimeError) as ei:
            verify_panel_store_chunks(tmp_path)
        msg = str(ei.value)
        assert "past_known" in msg
        assert "chunk 32" in msg

    def test_missing_manifest_with_binding_refuses(self, tmp_path):
        """A meta that records the chunk binding but a store with no
        chunk_manifest.json is refused — the store claims a binding it cannot
        vouch for."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        (tmp_path / "chunk_manifest.json").unlink()
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path)
        assert "chunk_manifest" in str(ei.value)

    def test_tampered_manifest_refused_at_load(self, tmp_path):
        """A digest string flipped in chunk_manifest.json desynchronizes the
        manifest root from its own digests — refused at load (cheap check)."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        path = tmp_path / "chunk_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        name = next(iter(manifest["arrays"]))
        digests = manifest["arrays"][name]["digests"]
        mid = len(digests) // 2
        flipped = ("0" if digests[mid][0] != "0" else "1") + digests[mid][1:]
        manifest["arrays"][name]["digests"][mid] = flipped
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path)
        assert "chunk_manifest" in str(ei.value)

    def test_substituted_manifest_refused_at_load(self, tmp_path):
        """A manifest copied from another store is internally consistent but its
        root no longer matches THIS store's meta binding — refused at load."""
        store_a = tmp_path / "a"
        store_b = tmp_path / "b"
        save_panel_memmap(_storeable_panel(seed=1), store_a, meta=_meta())
        save_panel_memmap(_storeable_panel(seed=2), store_b, meta=_meta())
        shutil.copyfile(store_b / "chunk_manifest.json",
                        store_a / "chunk_manifest.json")
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(store_a)
        assert "chunk_manifest" in str(ei.value)

    def test_legacy_no_binding_warns_and_loads_both_modes(self, tmp_path, caplog):
        """A pre-T10b store (meta records no chunk binding, no manifest file)
        loads with a warn and proceeds in BOTH explore and formal modes — the
        audit's explicit non-disruption guarantee."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        (tmp_path / "chunk_manifest.json").unlink()
        meta = json.loads((tmp_path / _META_FILE).read_text(encoding="utf-8"))
        del meta["chunk_manifest_root_hash"]
        (tmp_path / _META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        # explore mode
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path)
        assert loaded["stock_codes"] == _storeable_panel()["stock_codes"]
        assert any("chunk" in r.message for r in caplog.records)
        caplog.clear()
        # formal mode
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(
                tmp_path, expected_meta=_meta(), strict_external_meta=True)
        assert loaded["stock_codes"] == _storeable_panel()["stock_codes"]
        assert any("chunk" in r.message for r in caplog.records)


class TestPanelStoreWarnBinding:
    """§八 (T4): warn-and-proceed bindings to the external data artifacts the
    store was built from — a drift warns loudly but proceeds (re-derivable by
    rebuilding), and a side missing a key is skipped (mirrors config_hash)."""

    def _bound_meta(self):
        return _meta(
            data_manifest_hash="manifest-abc",
            calendar_hash="cal-abc",
            universe_status_hash="uni-abc",
            membership_hash="mem-abc",
            prebuilt_feature_manifest_hash="pre-abc",
        )

    def test_warn_key_mismatch_proceeds(self, tmp_path, caplog):
        """A drifted external artifact hash warns loudly but proceeds."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=self._bound_meta())
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(
                tmp_path, expected_meta=_meta(data_manifest_hash="manifest-CHANGED"))
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert any("data_manifest_hash" in r.message for r in caplog.records)

    def test_warn_keys_skipped_when_absent(self, tmp_path, caplog):
        """A meta with no warn keys (legacy / unbinding caller) loads without a
        warning — None on either side skips the comparison."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())  # no warn keys
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path, expected_meta=_meta())
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_warn_key_mismatch_expected_missing_skips(self, tmp_path, caplog):
        """Recorded carries a warn key but the current expected_meta does not —
        skipped, no warning (a side missing the key = skip)."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=self._bound_meta())
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path, expected_meta=_meta())
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


class TestPanelStoreStrictExternalMeta:
    """T1: formal mode (strict_external_meta=True) must REFUSE a store whose
    external-artifact hashes (data manifest / calendar / universe status /
    membership / prebuilt feature manifest) drifted OR cannot be vouched for
    on either side — upstream data changed means the stored panel is stale and
    must be rebuilt, never reused.  Explore mode keeps warn-and-proceed."""

    def test_strict_mismatch_refuses(self, tmp_path):
        """A drifted external hash is a hard-fail in formal mode, naming the
        key and both values (stored vs requested)."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(membership_hash="old"))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path,
                              expected_meta=_meta(membership_hash="new"),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "membership_hash" in msg
        assert "old" in msg and "new" in msg

    def test_strict_recorded_has_expected_missing_refuses(self, tmp_path):
        """Formal mode forbids the None-skip: a key the recorded store carries
        but the requested run does not cannot be vouched for — refused."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(membership_hash="abc"))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta(),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "membership_hash" in msg
        assert "requested" in msg  # the missing side is named

    def test_strict_expected_has_recorded_missing_refuses(self, tmp_path):
        """Formal mode forbids the None-skip the other way: a key the requested
        run carries but the store never recorded is refused, not skipped."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path,
                              expected_meta=_meta(membership_hash="abc"),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "membership_hash" in msg
        assert "recorded" in msg  # the missing side is named

    def test_explore_mode_mismatch_warns_and_proceeds(self, tmp_path, caplog):
        """Default (explore) mode keeps warn-and-proceed on the SAME drift."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path,
                          meta=_meta(membership_hash="old"))
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path,
                                       expected_meta=_meta(membership_hash="new"))
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert any("membership_hash" in r.message for r in caplog.records)

    def test_strict_clean_store_loads(self, tmp_path):
        """Strict mode does not reject a VALID store: matching external hashes
        on both sides load fine."""
        panel = _storeable_panel()
        m = _meta(
            data_manifest_hash="manifest-abc",
            calendar_hash="cal-abc",
            universe_status_hash="uni-abc",
            membership_hash="mem-abc",
            prebuilt_feature_manifest_hash="pre-abc",
        )
        save_panel_memmap(panel, tmp_path, meta=m)
        loaded = load_panel_memmap(tmp_path, expected_meta=m,
                                   strict_external_meta=True)
        assert loaded["stock_codes"] == panel["stock_codes"]

    def test_strict_non_membership_store_reuse_regression(self, tmp_path):
        """T1 regression: a store whose meta OMITS membership_hash (non-membership
        universe, e.g. --universe all) must REUSE in strict mode when the
        requested meta also omits it — the key is out of scope, not a refusal."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())  # no membership_hash
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta(),
                                   strict_external_meta=True)
        assert loaded["stock_codes"] == panel["stock_codes"]

    def test_strict_both_explicit_none_calendar_refuses(self, tmp_path):
        """An EXPLICIT None on BOTH sides (calendar materialization failed) is
        still refused in strict mode — the run cannot vouch for the artifact."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta(calendar_hash=None))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path,
                              expected_meta=_meta(calendar_hash=None),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "calendar_hash" in msg
        assert "neither side" in msg

    def test_strict_both_absent_skips(self, tmp_path):
        """Both sides ABSENT a warn key (neither context consumes the artifact)
        skips even in strict mode — the both-absent skip the T1 fix relies on."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta(),
                                   strict_external_meta=True)
        assert loaded["stock_codes"] == panel["stock_codes"]

    def test_panel_store_meta_omits_membership_for_non_csi(self, tmp_path):
        """_panel_store_meta must NOT write membership_hash for a non-membership
        universe (--universe all): the None from _universe_artifact_hashes is a
        'membership not consumed' sentinel, and writing it as an explicit null
        would brick strict-mode reuse of the store (T1)."""
        args = SimpleNamespace(
            universe="all", horizon=5, start="2024-01-01", end="2024-12-31",
            vintage_policy="allow-revised", minute=False,
        )
        meta = _panel_store_meta(args, seq_len=60, stock_list=["600000"],
                                 data_dir=str(tmp_path), prebuilt_dir=None,
                                 required_set=set())
        assert "membership_hash" not in meta
        # the status hash the universe DOES consume is still recorded
        assert "universe_status_hash" in meta


class TestPanelStoreSaveEdgeCases:
    def test_save_to_existing_file_raises(self, tmp_path):
        """--panel-store pointing at an existing FILE is a clear error."""
        f = tmp_path / "not_a_dir"
        f.write_text("i am a file")
        with pytest.raises(ValueError) as ei:
            save_panel_memmap(_storeable_panel(), f)
        assert "not a directory" in str(ei.value)

    def test_drops_unknown_key_with_warning(self, tmp_path, caplog):
        """An unknown non-array, non-JSON value is dropped with a warning (so a
        missing required key surfaces immediately rather than silently)."""
        panel = _storeable_panel()
        panel["garbage_extra"] = 42
        with caplog.at_level(logging.WARNING):
            save_panel_memmap(panel, tmp_path)
        assert any("garbage_extra" in r.message for r in caplog.records)
        assert panel_store_complete(tmp_path) is True
        # The stray key is not part of the store contract.
        assert "garbage_extra" not in load_panel_memmap(tmp_path)


class TestPanelStoreFillProbAndLabelPolicy:
    """§T13: fill_prob round-trips through the store; a legacy (pre-T13) store
    WITHOUT fill_prob loads (warn + NaN, not hard-fail), and the label_policy
    critical key refuses silently reusing a pre-T13 store for a new run."""

    def test_roundtrip_carries_fill_prob(self, tmp_path):
        """A store saved WITH fill_prob round-trips it elementwise."""
        panel = _storeable_panel(seed=0)
        n_days = panel["close_price"].shape[1]
        fill = np.full(n_days, np.nan, dtype=np.float64)
        fill[: n_days // 2] = 0.5
        panel["fill_prob"] = fill
        save_panel_memmap(panel, tmp_path)
        loaded = load_panel_memmap(tmp_path)
        np.testing.assert_array_equal(
            np.asarray(loaded["fill_prob"]), fill, err_msg="fill_prob")

    def test_legacy_store_without_fill_prob_warns_and_fills_nan(
            self, tmp_path, caplog):
        """A pre-T13 store (no fill_prob.npy) loads — warned, fill_prob filled
        with NaN — instead of hard-failing."""
        panel = _storeable_panel(seed=1)
        panel.pop("fill_prob", None)  # simulate a pre-T13 store (no fill_prob)
        save_panel_memmap(panel, tmp_path)
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path)
        n_days = panel["close_price"].shape[1]
        assert loaded["fill_prob"].shape == (n_days,)
        assert np.isnan(np.asarray(loaded["fill_prob"])).all()
        assert any("fill_prob" in r.message for r in caplog.records)

    def test_roundtrip_carries_entry_fill_prob(self, tmp_path):
        """A store saved WITH entry_fill_prob round-trips it elementwise."""
        panel = _storeable_panel(seed=3)
        n_days = panel["close_price"].shape[1]
        efp = np.full(n_days, np.nan, dtype=np.float64)
        efp[: n_days // 2] = 0.8
        panel["entry_fill_prob"] = efp
        save_panel_memmap(panel, tmp_path)
        loaded = load_panel_memmap(tmp_path)
        np.testing.assert_array_equal(
            np.asarray(loaded["entry_fill_prob"]), efp, err_msg="entry_fill_prob")

    def test_legacy_store_without_entry_fill_prob_warns_and_fills_nan(
            self, tmp_path, caplog):
        """A pre-T10a store (no entry_fill_prob.npy) loads — warned,
        entry_fill_prob filled with NaN — instead of hard-failing.  It is an
        OPTIONAL diagnostic like fill_prob, not a required-array key."""
        panel = _storeable_panel(seed=4)
        panel.pop("entry_fill_prob", None)  # simulate a pre-T10a store
        save_panel_memmap(panel, tmp_path)
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path)
        n_days = panel["close_price"].shape[1]
        assert loaded["entry_fill_prob"].shape == (n_days,)
        assert np.isnan(np.asarray(loaded["entry_fill_prob"])).all()
        assert any("entry_fill_prob" in r.message for r in caplog.records)

    def test_label_policy_mismatch_refuses(self, tmp_path):
        """A pre-T13 store (meta records no label_policy) is refused by a
        current run's expected_meta — its y_return labels carry different
        semantics, so it must never be silently reused."""
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_meta())  # legacy meta: no label_policy
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(
                tmp_path,
                expected_meta=_meta(label_policy="carry_to_last_close_v1"))
        assert "label_policy" in str(ei.value)

    def test_label_policy_matching_loads(self, tmp_path):
        """A store whose meta records the SAME label_policy loads cleanly."""
        panel = _storeable_panel(seed=2)
        panel["fill_prob"] = np.full(
            panel["close_price"].shape[1], np.nan, dtype=np.float64)
        save_panel_memmap(panel, tmp_path,
                          meta=_meta(label_policy="carry_to_last_close_v1"))
        loaded = load_panel_memmap(
            tmp_path, expected_meta=_meta(label_policy="carry_to_last_close_v1"))
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert "fill_prob" in loaded


class TestResolvePanelStoreSkip:
    """train_panel._resolve_panel: a COMPLETE store must skip the K-line load +
    feature build entirely and return the mmap'd panel."""

    def _args(self, store_path):
        return SimpleNamespace(
            panel_store=store_path, minute=False, minute_frequency="60",
            start="2024-01-01", end="2024-12-31", universe="csi300", horizon=5,
            no_aux=False, prebuilt=None, require_feature_manifest=False,
            require_aux_channels=None,
            vintage_policy="allow-revised",  # §T2: reproduces the pre-T2 switch set
            no_formal=False,  # T1: _resolve_panel threads _formal_mode(args)
        )

    def test_complete_store_skips_kline_load(self, tmp_path, monkeypatch):
        panel = _storeable_panel(n_stocks=10, n_days=100)
        stock_list = list(panel["stock_codes"])
        seq_len = 60
        store_path = str(tmp_path / "store")
        args = self._args(store_path)
        save_panel_memmap(panel, store_path,
                          meta=_panel_store_meta(
                              args, seq_len, stock_list, str(tmp_path),
                              args.prebuilt, required_set=set()))

        # Any touch of the live K-line / aux pipeline is a regression: the
        # store path must return before DataStorage.load_daily or load_aux_data
        # is ever reached.
        def _boom(*_a, **_k):
            raise AssertionError("K-line/aux load must be skipped on a complete store")

        import stoke_ml.data.storage as storage_mod
        monkeypatch.setattr(storage_mod.DataStorage, "load_daily", _boom)
        monkeypatch.setattr("scripts.production.train_panel_panel.load_aux_data", _boom)

        panel_data, channel_manifest = _resolve_panel(
            args, stock_list, seq_len, str(tmp_path), set(), _store_load=True)

        assert list(panel_data["stock_codes"]) == stock_list
        np.testing.assert_array_equal(
            np.asarray(panel_data["past_known"]), panel["past_known"])
        assert isinstance(panel_data["past_known"], np.memmap)
        assert isinstance(channel_manifest, dict)

    def test_no_store_returns_live_path(self, tmp_path, monkeypatch):
        """Without a complete store, _resolve_panel builds the panel live and
        persists it (with meta) when --panel-store is set."""
        panel = _storeable_panel(n_stocks=10, n_days=100)
        stock_list = list(panel["stock_codes"])
        seq_len = 60
        store_path = str(tmp_path / "store")
        args = self._args(store_path)

        # Build a fake K-line load: DataStorage.load_daily returns a minimal
        # frame per code, and build_panel_features is stubbed to return the
        # storeable panel (so the save-with-meta path is exercised, not the
        # full feature engine).
        import pandas as pd
        import stoke_ml.data.storage as storage_mod

        def fake_load_daily(self, code, start, end, **kw):
            return pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]),
                                 "close": [10.0], "open": [10.0],
                                 "high": [10.0], "low": [10.0],
                                 "volume": [1e6], "amount": [1e7],
                                 "stock_code": code})

        monkeypatch.setattr(storage_mod.DataStorage, "load_daily", fake_load_daily)
        # Skip aux loading and the real feature build — build_panel_features is
        # stubbed to return a full storeable panel so the save-with-meta path
        # is exercised, not the (already covered elsewhere) feature engine.
        monkeypatch.setattr("scripts.production.train_panel_panel.load_aux_data",
                            lambda *a, **k: (None, {}))
        monkeypatch.setattr(
            "scripts.production.train_panel_panel.FeaturePipeline",
            lambda **kw: SimpleNamespace(
                build_panel_features=lambda _panel, **kw2: _storeable_panel()))

        panel_data, channel_manifest = _resolve_panel(
            args, stock_list, seq_len, str(tmp_path), set(), _store_load=False)

        # The live path persisted the panel to the store with meta, and a
        # subsequent store-load round-trips it.
        import pathlib
        assert panel_store_complete(store_path) is True
        assert (pathlib.Path(store_path) / _META_FILE).is_file()
        assert list(panel_data["stock_codes"]) == stock_list
        assert isinstance(channel_manifest, dict)

    def test_validate_panel_store_path_file_raises(self, tmp_path):
        f = tmp_path / "file_store"
        f.write_text("x")
        with pytest.raises(SystemExit):
            _validate_panel_store_path(str(f))
