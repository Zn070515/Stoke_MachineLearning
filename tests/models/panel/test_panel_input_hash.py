"""T6 (v17 §七): panel-input provenance binding — feature code tree hash,
aux asset-root hash, and the panel_input_hash aggregate.

A panel store's meta.json recorded ``git_commit`` at build time, so a store-
backed re-run compared git HEADs.  Uncommitted edits (the common research
workflow) leave git_commit unchanged, so a store built from edited feature code
was silently reused — the provenance was fake.  T6 closes that gap with CONTENT
hashes:

* ``feature_code_tree_hash`` (code_tree_hash.py) — SHA-256 aggregate of every
  ``.py`` file under ``stoke_ml/`` + ``scripts/production/`` (the source trees
  that compute the panel's feature values).  A code edit — committed or not —
  changes the hash, so a store built from old code is refused.
* ``aux_asset_root_hash`` — content hash of the CONSUMED channels' live asset
  manifests (``*.manifest.json`` sidecars, per-write bookkeeping keys
  ``written_at`` / ``updated`` / ``run_id`` excluded).  The §七
  guard: a live-mode store binds today's aux roots; changed aux tomorrow makes
  the formal load refuse.  (Required ⊂ Consumed — the required subset is the
  extra coverage-threshold layer, but every channel the pipeline actually
  reads must bind under a formal gate, §v18-2.)
* ``panel_input_hash`` — SHA-256 aggregate (canonical JSON) of every input
  provenance component, so a change anywhere is a single key mismatch.

All three are ``_WARN_META_KEYS``: warn-and-proceed in explore mode, HARD-FAIL
in formal mode (``strict_external_meta=True``).  A legacy store without them is
refused in formal mode (recorded side missing), skipped in explore.
"""
import hashlib
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.production.train_panel_panel import (
    _asset_manifest_entries,
    _aux_asset_root_hash,
    _panel_store_meta,
)
from stoke_ml.models.panel.code_tree_hash import (
    canonical_json,
    feature_code_tree_hash,
    hash_json,
)
from stoke_ml.models.panel.panel_store import (
    load_panel_memmap,
    save_panel_memmap,
)


# ── local copies of the panel-store test helpers (tests/ has no __init__.py) ──

def _storeable_panel(n_stocks=10, n_days=100, seq_len=60, horizon=5, seed=0):
    """A complete panel dict every required store key expects (mirrors
    test_panel_store._storeable_panel — copied because tests are not a package)."""
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
        "fill_prob": np.full(n_days, np.nan, dtype=np.float64),
        "entry_fill_prob": np.full(n_days, np.nan, dtype=np.float64),
        "close_price": np.full((n_stocks, n_days), 10.0, dtype=np.float32),
        "open_price": np.full((n_stocks, n_days), 10.0, dtype=np.float32),
        "stock_codes": stocks,
        "past_known_cols": [f"pk_{i}" for i in range(12)],
        "past_observed_cols": [f"po_{i}" for i in range(6)],
    }


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


def _bound_meta(**over):
    """A meta carrying the §T6 provenance bindings."""
    m = _meta(
        feature_code_tree_hash="tree-abc",
        aux_asset_root_hash="aux-abc",
        panel_input_hash="agg-abc",
    )
    m.update(over)
    return m


def _panel_args(**over):
    """A minimal train_panel args namespace for the meta-fingerprint tests."""
    base = dict(
        universe="all", horizon=5, start="2024-01-01", end="2024-12-31",
        vintage_policy="allow-revised", minute=False,
        allow_fundamental_ablation=False, no_aux=False, prebuilt=None,
        feature_profile=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── code_tree_hash primitives ───────────────────────────────────────────

class TestCanonicalJson:
    """canonical_json / hash_json — the deterministic serialization the
    aggregate hashes are built on."""

    def test_key_order_independent(self):
        assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})

    def test_compact_no_whitespace(self):
        s = canonical_json({"a": [1, 2], "b": {"c": "x"}})
        assert " " not in s
        assert s == '{"a":[1,2],"b":{"c":"x"}}'

    def test_nested_sort(self):
        assert canonical_json({"z": {"b": 1, "a": 2}, "y": 0}) == \
            '{"y":0,"z":{"a":2,"b":1}}'

    def test_handles_none(self):
        assert canonical_json({"a": None, "b": 1}) == '{"a":null,"b":1}'

    def test_hash_json_is_sha256_of_canonical(self):
        obj = {"b": 2, "a": [1, None, "x"], "nested": {"z": True}}
        assert hash_json(obj) == hashlib.sha256(
            canonical_json(obj).encode("utf-8")).hexdigest()


class TestFeatureCodeTreeHash:
    """The content hash of the stoke_ml/ + scripts/production/ source trees."""

    def _dirs(self, tmp_path):
        (tmp_path / "stoke_ml" / "features").mkdir(parents=True)
        (tmp_path / "scripts" / "production").mkdir(parents=True)

    def test_deterministic(self, tmp_path):
        self._dirs(tmp_path)
        (tmp_path / "stoke_ml" / "features" / "a.py").write_text("x = 1\n")
        (tmp_path / "scripts" / "production" / "b.py").write_text("y = 2\n")
        assert feature_code_tree_hash(str(tmp_path)) == \
            feature_code_tree_hash(str(tmp_path))

    def test_edit_changes_hash(self, tmp_path):
        self._dirs(tmp_path)
        p = tmp_path / "stoke_ml" / "features" / "a.py"
        p.write_text("x = 1\n")
        h1 = feature_code_tree_hash(str(tmp_path))
        p.write_text("x = 2\n")
        assert feature_code_tree_hash(str(tmp_path)) != h1

    def test_ignores_non_py(self, tmp_path):
        self._dirs(tmp_path)
        p = tmp_path / "stoke_ml" / "features" / "a.py"
        p.write_text("x = 1\n")
        h1 = feature_code_tree_hash(str(tmp_path))
        (tmp_path / "stoke_ml" / "features" / "notes.txt").write_text("ignored")
        assert feature_code_tree_hash(str(tmp_path)) == h1

    def test_new_file_changes_hash(self, tmp_path):
        self._dirs(tmp_path)
        (tmp_path / "stoke_ml" / "features" / "a.py").write_text("x = 1\n")
        h1 = feature_code_tree_hash(str(tmp_path))
        (tmp_path / "stoke_ml" / "features" / "b.py").write_text("y = 2\n")
        assert feature_code_tree_hash(str(tmp_path)) != h1

    def test_rename_changes_hash(self, tmp_path):
        self._dirs(tmp_path)
        a = tmp_path / "stoke_ml" / "features" / "a.py"
        a.write_text("x = 1\n")
        h1 = feature_code_tree_hash(str(tmp_path))
        a.replace(tmp_path / "stoke_ml" / "features" / "c.py")
        assert feature_code_tree_hash(str(tmp_path)) != h1

    def test_both_dirs_are_hashed(self, tmp_path):
        self._dirs(tmp_path)
        (tmp_path / "stoke_ml" / "features" / "a.py").write_text("x = 1\n")
        h1 = feature_code_tree_hash(str(tmp_path))
        (tmp_path / "scripts" / "production" / "s.py").write_text("s = 1\n")
        assert feature_code_tree_hash(str(tmp_path)) != h1

    def test_empty_tree_returns_unknown(self, tmp_path):
        assert feature_code_tree_hash(str(tmp_path)) == "unknown"

    def test_only_scans_the_two_source_dirs(self, tmp_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        (other / "x.py").write_text("x = 1\n")
        assert feature_code_tree_hash(str(tmp_path)) == "unknown"


# ── aux asset-root hash ─────────────────────────────────────────────────

class TestAuxAssetRootHash:
    """The content hash of the REQUIRED channels' live asset-manifest roots."""

    def _data(self, tmp_path):
        return str(tmp_path / "data")

    def _sent_dir(self, tmp_path):
        root = os.path.join(self._data(tmp_path), "a_shares", "sentiment")
        os.makedirs(root, exist_ok=True)
        return root

    def _write(self, path, rows, written_at="2026-01-01T00:00:00+00:00", **extra):
        manifest = {"data_type": "sentiment", "rows": rows,
                    "schema_hash": "abc", "written_at": written_at}
        manifest.update(extra)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

    def test_empty_required_set_deterministic(self, tmp_path):
        h1 = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=True)
        h2 = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=True)
        assert h1 == h2

    def test_manifest_content_change_changes_hash(self, tmp_path):
        root = self._sent_dir(tmp_path)
        p = os.path.join(root, "000001.parquet.manifest.json")
        self._write(p, rows=100)
        h1 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        self._write(p, rows=200)
        h2 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        assert h1 != h2

    def test_written_at_excluded(self, tmp_path):
        """A content-identical rewrite of the same data (different written_at)
        must NOT change the hash — the sidecar's timestamp is write noise."""
        root = self._sent_dir(tmp_path)
        p = os.path.join(root, "000001.parquet.manifest.json")
        self._write(p, rows=100, written_at="2026-01-01T00:00:00+00:00")
        h1 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        self._write(p, rows=100, written_at="2026-01-02T00:00:00+00:00")
        h2 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        assert h1 == h2

    def test_updated_and_run_id_excluded(self, tmp_path):
        """Mirror of test_written_at_excluded for the OTHER per-write
        bookkeeping keys the daily store writes (updated / run_id): a
        content-identical rewrite that bumps only them must NOT change the hash
        — they are write-event identity, not data identity."""
        root = self._sent_dir(tmp_path)
        p = os.path.join(root, "000001.parquet.manifest.json")
        self._write(p, rows=100, updated="2026-01-01T00:00:00+00:00",
                    run_id="abc123")
        h1 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        self._write(p, rows=100, updated="2026-01-02T00:00:00+00:00",
                    run_id="def456")
        h2 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        assert h1 == h2

    def test_new_manifest_changes_hash(self, tmp_path):
        root = self._sent_dir(tmp_path)
        self._write(os.path.join(root, "000001.parquet.manifest.json"), rows=100)
        h1 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        self._write(os.path.join(root, "000002.parquet.manifest.json"), rows=100)
        h2 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        assert h1 != h2

    def test_manifest_removed_changes_hash(self, tmp_path):
        root = self._sent_dir(tmp_path)
        p = os.path.join(root, "000001.parquet.manifest.json")
        self._write(p, rows=100)
        h1 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        os.remove(p)
        h2 = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                  live_aux=True)
        assert h1 != h2

    def test_asset_manifest_entries_keyed_by_relpath(self, tmp_path):
        root = self._sent_dir(tmp_path)
        self._write(os.path.join(root, "000001.parquet.manifest.json"), rows=100)
        entries = _asset_manifest_entries(root)
        assert set(entries) == {"000001.parquet.manifest.json"}
        assert isinstance(entries["000001.parquet.manifest.json"], str)

    def test_live_aux_flag_distinguishes(self, tmp_path):
        h_live = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=True)
        h_not = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=False)
        assert h_live != h_not

    def test_required_channels_distinguish(self, tmp_path):
        root = self._sent_dir(tmp_path)
        self._write(os.path.join(root, "000001.parquet.manifest.json"), rows=100)
        h_empty = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=True)
        h_sent = _aux_asset_root_hash(self._data(tmp_path), {"sentiment"},
                                      live_aux=True)
        assert h_empty != h_sent

    def test_unknown_channel_marked_not_skipped(self, tmp_path):
        """A required channel with no CHANNEL_SOURCE entry is recorded as an
        explicit marker (fail-closed), never silently dropped."""
        h = _aux_asset_root_hash(self._data(tmp_path), {"no_such_channel"},
                                 live_aux=True)
        h_empty = _aux_asset_root_hash(self._data(tmp_path), set(), live_aux=True)
        assert h != h_empty

    def test_aux_asset_root_hash_binds_consumed_not_just_required(self, tmp_path):
        """§v18-2: the aux root binding must cover the channels the run CONSUMES,
        not only the required subset — a consumed-but-unrequired channel (e.g. the
        fundamental ablation) whose data changes must make the binding differ."""
        data = self._data(tmp_path)
        h_required_only = _aux_asset_root_hash(
            data, {"sentiment"}, live_aux=True)
        h_with_consumed = _aux_asset_root_hash(
            data, {"sentiment", "fundamental"}, live_aux=True)
        assert h_required_only != h_with_consumed

    def test_panel_store_meta_binds_consumed_channels(self, tmp_path):
        """§v18-2: _panel_store_meta's aux_asset_root_hash is computed over the
        CONSUMED channel set (derived from args + seq_len), so an ablation store
        differs from a non-ablation store even when required_set is empty."""
        base = _panel_store_meta(
            _panel_args(vintage_policy="revision-safe"),
            seq_len=60, stock_list=["000001"], data_dir=str(tmp_path),
            required_set=set())
        ablated = _panel_store_meta(
            _panel_args(vintage_policy="revision-safe",
                        allow_fundamental_ablation=True),
            seq_len=60, stock_list=["000001"], data_dir=str(tmp_path),
            required_set=set())
        assert base.get("aux_asset_root_hash") != ablated.get(
            "aux_asset_root_hash")


# ── strict validation of the provenance keys ────────────────────────────

class TestPanelStoreInputHashStrict:
    """§T6: the provenance keys hard-fail in formal mode, warn in explore.

    Three-state matrix per key (matching / wrong / missing on one side) ×
    mode (strict refuse vs explore warn-or-skip) — the same contract the
    existing _WARN_META_KEYS bindings follow."""

    def test_matching_provenance_loads_formal(self, tmp_path):
        panel = _storeable_panel()
        m = _bound_meta()
        save_panel_memmap(panel, tmp_path, meta=m)
        loaded = load_panel_memmap(tmp_path, expected_meta=_bound_meta(),
                                   strict_external_meta=True)
        assert loaded["stock_codes"] == panel["stock_codes"]

    def test_panel_input_hash_mismatch_refuses_formal(self, tmp_path):
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_bound_meta(panel_input_hash="agg-old"))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path,
                              expected_meta=_bound_meta(panel_input_hash="agg-new"),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "panel_input_hash" in msg
        assert "agg-old" in msg and "agg-new" in msg

    def test_feature_code_tree_mismatch_refuses_formal(self, tmp_path):
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_bound_meta(feature_code_tree_hash="tree-old"))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(
                tmp_path,
                expected_meta=_bound_meta(feature_code_tree_hash="tree-new"),
                strict_external_meta=True)
        msg = str(ei.value)
        assert "feature_code_tree_hash" in msg
        assert "tree-old" in msg and "tree-new" in msg

    def test_aux_root_mismatch_refuses_formal(self, tmp_path):
        save_panel_memmap(_storeable_panel(), tmp_path,
                          meta=_bound_meta(aux_asset_root_hash="aux-old"))
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path,
                              expected_meta=_bound_meta(aux_asset_root_hash="aux-new"),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "aux_asset_root_hash" in msg
        assert "aux-old" in msg and "aux-new" in msg

    def test_provenance_mismatch_warns_and_proceeds_explore(
            self, tmp_path, caplog):
        """Default (explore) mode keeps warn-and-proceed on the SAME drift."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path,
                          meta=_bound_meta(panel_input_hash="agg-old"))
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(
                tmp_path, expected_meta=_bound_meta(panel_input_hash="agg-new"))
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert any("panel_input_hash" in r.message for r in caplog.records)

    def test_legacy_store_missing_keys_refuses_formal(self, tmp_path):
        """A legacy store (no new keys) loaded by a run that carries them is
        refused in formal mode — the store cannot vouch for its provenance."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_bound_meta(),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "feature_code_tree_hash" in msg
        assert "side missing" in msg

    def test_legacy_store_missing_keys_skips_explore(self, tmp_path, caplog):
        """Explore mode keeps loading a legacy store — the recorded side is
        missing a key the run carries, which skips (mirrors config_hash)."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        with caplog.at_level(logging.WARNING):
            loaded = load_panel_memmap(tmp_path, expected_meta=_bound_meta())
        assert loaded["stock_codes"] == panel["stock_codes"]
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_expected_missing_keys_refuses_formal(self, tmp_path):
        """The reverse gap: recorded carries the new keys but the current run
        does not — formal mode refuses (requested side missing)."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_bound_meta())
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(tmp_path, expected_meta=_meta(),
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "panel_input_hash" in msg
        assert "requested" in msg

    def test_both_sides_absent_skips(self, tmp_path):
        """A hand-built expected_meta that (like a legacy store) carries no
        provenance keys skips even in strict mode — the both-absent skip."""
        panel = _storeable_panel()
        save_panel_memmap(panel, tmp_path, meta=_meta())
        loaded = load_panel_memmap(tmp_path, expected_meta=_meta(),
                                   strict_external_meta=True)
        assert loaded["stock_codes"] == panel["stock_codes"]


# ── end-to-end: _panel_store_meta provenance + formal load ──────────────

class TestPanelStoreInputHashEndToEnd:
    """§七: a live-mode store must bind today's aux asset roots + code tree;
    changed aux or code tomorrow makes the formal load refuse (never silently
    reused)."""

    def _sent_manifest(self, data_dir, rows, written_at="2026-01-01T00:00:00+00:00"):
        root = os.path.join(data_dir, "a_shares", "sentiment")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "000001.parquet.manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"data_type": "sentiment", "rows": rows,
                       "schema_hash": "abc", "written_at": written_at}, fh)

    def test_changed_live_channel_manifest_refuses_formal(self, tmp_path):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        self._sent_manifest(data_dir, rows=100)
        args = _panel_args()
        meta1 = _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                                  data_dir=data_dir, prebuilt_dir=None,
                                  required_set={"sentiment"})
        store = str(tmp_path / "store")
        save_panel_memmap(_storeable_panel(n_stocks=10), store, meta=meta1)
        # same-day formal load passes
        load_panel_memmap(store, expected_meta=meta1, strict_external_meta=True)
        # next day: the sentiment aux data changed (manifest content edited) →
        # the store's provenance no longer matches → formal load REFUSES.
        self._sent_manifest(data_dir, rows=200)
        meta2 = _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                                  data_dir=data_dir, prebuilt_dir=None,
                                  required_set={"sentiment"})
        assert meta2["aux_asset_root_hash"] != meta1["aux_asset_root_hash"]
        assert meta2["panel_input_hash"] != meta1["panel_input_hash"]
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(store, expected_meta=meta2,
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "panel_input_hash" in msg
        assert "aux_asset_root_hash" in msg
        # explore mode on the same drift still loads (warn-and-proceed)
        load_panel_memmap(store, expected_meta=meta2)

    def test_feature_code_change_refuses_formal(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        args = _panel_args()
        meta1 = _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                                  data_dir=data_dir, prebuilt_dir=None,
                                  required_set=set())
        store = str(tmp_path / "store")
        save_panel_memmap(_storeable_panel(n_stocks=10), store, meta=meta1)
        # A later run's feature code tree differs (an uncommitted edit) —
        # simulate by patching the tree hash at the panel-store-meta call site.
        monkeypatch.setattr(
            "scripts.production.train_panel_panel.feature_code_tree_hash",
            lambda: "changed-tree-hash")
        meta2 = _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                                  data_dir=data_dir, prebuilt_dir=None,
                                  required_set=set())
        assert meta2["feature_code_tree_hash"] == "changed-tree-hash"
        assert meta2["panel_input_hash"] != meta1["panel_input_hash"]
        with pytest.raises(RuntimeError) as ei:
            load_panel_memmap(store, expected_meta=meta2,
                              strict_external_meta=True)
        msg = str(ei.value)
        assert "feature_code_tree_hash" in msg
        assert "panel_input_hash" in msg

    def test_panel_store_meta_records_provenance_keys(self, tmp_path):
        """_panel_store_meta records the three provenance bindings: the code
        tree hash always; the aux asset-root hash for a live-aux build; and the
        panel_input_hash aggregate always."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        self._sent_manifest(data_dir, rows=100)
        args = _panel_args()
        meta = _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                                 data_dir=data_dir, prebuilt_dir=None,
                                 required_set={"sentiment"})
        assert "feature_code_tree_hash" in meta
        assert meta["feature_code_tree_hash"] != "unknown"
        assert "aux_asset_root_hash" in meta
        assert "panel_input_hash" in meta
        # a no-aux build records no aux root binding (nothing to bind)
        no_aux = _panel_store_meta(
            _panel_args(no_aux=True), seq_len=60, stock_list=["000001"],
            data_dir=data_dir, prebuilt_dir=None, required_set=set())
        assert "aux_asset_root_hash" not in no_aux
        assert "panel_input_hash" in no_aux

    def test_panel_input_hash_covers_profile_and_vintage(self, tmp_path):
        """vintage_policy + feature_profile are EXPLICIT components of the
        aggregate — two builds that differ only in policy/profile must not
        share a panel_input_hash even when the switch set is unchanged."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        base = _panel_store_meta(_panel_args(), seq_len=60,
                                 stock_list=["000001"], data_dir=data_dir,
                                 prebuilt_dir=None, required_set=set())
        profiled = _panel_store_meta(
            _panel_args(feature_profile="headline_v1"), seq_len=60,
            stock_list=["000001"], data_dir=data_dir, prebuilt_dir=None,
            required_set=set())
        assert base["panel_input_hash"] != profiled["panel_input_hash"]

    def test_prebuilt_vs_live_provenance_differs(self, tmp_path):
        """A prebuilt store (aux bound by prebuilt_feature_manifest_hash, no
        aux root binding) must not be interchangeable with a live store (aux
        root bound).  Even with the same required_set, the panel_input_hash
        differs because aux_asset_root_hash is present only for live."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        live = _panel_store_meta(_panel_args(), seq_len=60,
                                 stock_list=["000001"], data_dir=data_dir,
                                 prebuilt_dir=None, required_set={"sentiment"})
        prebuilt = _panel_store_meta(
            _panel_args(), seq_len=60, stock_list=["000001"],
            data_dir=data_dir, prebuilt_dir=str(tmp_path / "prebuilt"),
            required_set={"sentiment"})
        assert "aux_asset_root_hash" in live
        assert "aux_asset_root_hash" not in prebuilt
        assert live["panel_input_hash"] != prebuilt["panel_input_hash"]

    def test_required_set_is_required_keyword_param(self, tmp_path):
        """§T6 review follow-up: ``required_set`` is a REQUIRED keyword-only
        parameter — omitting it must raise TypeError at the call site.  A
        defaulted set would silently bind the store to an EMPTY aux asset root
        (``aux_asset_root_hash`` → ``channels={}``), making the §七 "changed aux
        tomorrow" guard vacuous — the exact fake-provenance hole T6 closes."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        args = _panel_args()
        with pytest.raises(TypeError):
            _panel_store_meta(args, seq_len=60, stock_list=["000001"],
                              data_dir=data_dir, prebuilt_dir=None)
