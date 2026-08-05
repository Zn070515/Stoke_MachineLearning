"""Tests for feature-cache sidecar manifests.

A prebuilt feature is only reusable when its manifest's recorded code version,
config signature, feature schema, and per-channel source fingerprints still
match the current inputs.  Existence + non-zero size alone is not enough.
"""
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stoke_ml.features import cache_manifest


def _write_parquet(path, rows=5, cols=("close", "volume")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows),
        **{c: [float(i) for i in range(rows)] for c in cols},
    })
    pq.write_table(pa.Table.from_pandas(df), path, compression="lz4")
    return path


def _make_data_dir(tmp_path, code="000001"):
    data_dir = str(tmp_path / "data")
    daily = os.path.join(data_dir, "a_shares", "daily", f"{code}.parquet")
    _write_parquet(daily)
    return data_dir


def _base_config(**over):
    cfg = {
        "seq_len": 60, "horizon": 1, "panel_mode": True,
        "use_technical": True, "use_scoring": True, "use_temporal": True,
        "use_sentiment": True, "use_guba": False, "use_comment": True,
        "use_limit_up": False, "use_pledge": True, "use_market_env": True,
        "use_market_env_refine": True, "use_index_membership": True,
        "start": "2000-01-01", "end": "2026-01-01",
    }
    cfg.update(over)
    return cfg


class TestFingerprints:
    def test_file_fingerprint_missing_is_none(self, tmp_path):
        assert cache_manifest.file_fingerprint(str(tmp_path / "nope.parquet")) is None

    def test_file_fingerprint_changes_on_content_change(self, tmp_path):
        p = str(tmp_path / "f.parquet")
        _write_parquet(p, rows=3)
        fp1 = cache_manifest.file_fingerprint(p)
        _write_parquet(p, rows=10)  # different size + mtime
        fp2 = cache_manifest.file_fingerprint(p)
        assert fp1 != fp2

    def test_preserved_mtime_same_size_detected(self, tmp_path):
        """§十一-2: a same-size, same-mtime byte replacement must be caught.

        The old size+mtime fingerprint was blind to this exact case; the
        content hash must not be.
        """
        p = tmp_path / "blob.bin"
        p.write_bytes(b"AAAA")
        fp1 = cache_manifest.file_fingerprint(str(p))
        st = p.stat()
        p.write_bytes(b"BBBB")  # same 4 bytes, different content
        os.utime(p, (st.st_atime, st.st_mtime))  # restore mtime -> old proxy blind
        fp2 = cache_manifest.file_fingerprint(str(p))
        assert fp1 != fp2

    def test_daily_sidecar_manifest_fingerprint(self, tmp_path):
        """daily K-line reuses the upstream storage content checksum."""
        daily = tmp_path / "daily"
        daily.mkdir()
        p = daily / "000001.parquet"
        _write_parquet(str(p))
        (daily / "000001.manifest.json").write_text(
            json.dumps({"schema_hash": "abc123"}), encoding="utf-8",
        )
        assert cache_manifest.file_fingerprint(str(p)) == \
            cache_manifest._sha1("upstream:abc123")

    def test_daily_sidecar_missing_falls_back_to_content(self, tmp_path):
        """No upstream sidecar -> content-hash the parquet bytes."""
        daily = tmp_path / "daily"
        daily.mkdir()
        p = daily / "000001.parquet"
        _write_parquet(str(p))
        fp = cache_manifest.file_fingerprint(str(p))
        assert fp is not None
        assert fp != cache_manifest._sha1("upstream:abc123")

    def test_config_hash_sensitive_to_build_inputs(self):
        a = cache_manifest.config_hash(_base_config())
        b = cache_manifest.config_hash(_base_config(horizon=5))
        c = cache_manifest.config_hash(_base_config(use_guba=True))
        assert a != b
        assert a != c
        assert a == cache_manifest.config_hash(_base_config())

    def test_schema_hash_differs_for_different_columns(self, tmp_path):
        p1 = str(tmp_path / "a.parquet")
        p2 = str(tmp_path / "b.parquet")
        _write_parquet(p1, cols=("close", "volume"))
        _write_parquet(p2, cols=("close", "volume", "extra"))
        assert cache_manifest.schema_hash(p1) != cache_manifest.schema_hash(p2)


class TestConfigSnapshot:
    """§十一-3: the manifest hash must cover the feature-affecting config."""

    @staticmethod
    def _snapshot():
        return {
            "features": {
                "seq_len": 60, "target_horizon": 1, "technical_indicators": True,
                "rule_based_scoring": True, "temporal_features": True,
                "use_sentiment": True, "flat_seq_len": 5,
            },
            "preprocessing": {
                "numeric": {
                    "missing": {"short_gap_max": 2, "medium_gap_max": 10},
                    "scaling": {"method": "robust", "window_days": 252,
                                "winsorize_sigma": 3.0},
                    "cross_section": {"enabled": True, "stages": ["sector"]},
                },
                "text": {"bipolar": {"threshold_positive": 0.2,
                                     "threshold_negative": -0.2}},
            },
            "universe": {"min_amount_60d": 5_000_000,
                         "long_suspension_days": 60},
            "fundamental": {"max_stale_days": 30, "interpolate": True},
        }

    def test_snapshot_hash_sensitive_to_section_values(self):
        base = cache_manifest.config_hash(self._snapshot())
        assert base == cache_manifest.config_hash(self._snapshot())
        # Technical-indicator switch
        snap = self._snapshot()
        snap["features"]["technical_indicators"] = False
        assert cache_manifest.config_hash(snap) != base
        # Missing-value handling
        snap = self._snapshot()
        snap["preprocessing"]["numeric"]["missing"]["short_gap_max"] = 5
        assert cache_manifest.config_hash(snap) != base
        # Bipolar threshold
        snap = self._snapshot()
        snap["preprocessing"]["text"]["bipolar"]["threshold_positive"] = 0.5
        assert cache_manifest.config_hash(snap) != base
        # Cross-sectional normalization param
        snap = self._snapshot()
        snap["preprocessing"]["numeric"]["scaling"]["winsorize_sigma"] = 2.0
        assert cache_manifest.config_hash(snap) != base
        # Universe gate
        snap = self._snapshot()
        snap["universe"]["min_amount_60d"] = 10_000_000
        assert cache_manifest.config_hash(snap) != base
        # Fundamental staleness policy
        snap = self._snapshot()
        snap["fundamental"]["max_stale_days"] = 60
        assert cache_manifest.config_hash(snap) != base

    def test_snapshot_and_flat_are_distinct_paths(self):
        flat = cache_manifest.config_hash(_base_config())
        snap = cache_manifest.config_hash(self._snapshot())
        assert flat != snap

    def test_current_config_hash_loads_from_real_config(self):
        h = cache_manifest.current_config_hash()
        assert h is not None and len(h) == 16
        # Deterministic across calls.
        assert h == cache_manifest.current_config_hash()

    def test_config_snapshot_accepts_flat_and_nested(self):
        snap = cache_manifest.config_snapshot({"features": {"seq_len": 60}})
        assert snap["features"] == {"seq_len": 60}
        assert snap["preprocessing"] == {}


class TestManifest:
    def _setup(self, tmp_path):
        code = "000001"
        data_dir = _make_data_dir(tmp_path, code)
        feature = os.path.join(str(tmp_path / "out"), f"{code}.parquet")
        _write_parquet(feature, cols=("date", "f1", "f2"))
        cfg = _base_config()
        cfg_hash = cache_manifest.config_hash(cfg)
        commit = cache_manifest.git_head()
        manifest_path = os.path.join(
            str(tmp_path / "out"), ".manifests", f"{code}.json"
        )
        cache_manifest.write_manifest(
            cache_manifest.make_manifest(code, cfg, feature, data_dir, commit, cfg_hash),
            manifest_path,
        )
        return data_dir, feature, manifest_path, cfg, cfg_hash, commit

    def test_roundtrip_matches(self, tmp_path):
        _data_dir, _feature, mpath, cfg, cfg_hash, commit = self._setup(tmp_path)
        assert cache_manifest.manifest_matches(
            mpath, "000001", cfg, _feature, _data_dir, commit, cfg_hash
        )

    def test_config_change_invalidates(self, tmp_path):
        _data_dir, _feature, mpath, cfg, _cfg_hash, commit = self._setup(tmp_path)
        new_cfg_hash = cache_manifest.config_hash(_base_config(horizon=5))
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, _feature, _data_dir, commit, new_cfg_hash
        )

    def test_commit_change_invalidates(self, tmp_path):
        _data_dir, _feature, mpath, cfg, cfg_hash, _commit = self._setup(tmp_path)
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, _feature, _data_dir, "other-commit", cfg_hash
        )

    def test_source_change_invalidates(self, tmp_path):
        data_dir, _feature, mpath, cfg, cfg_hash, commit = self._setup(tmp_path)
        daily = os.path.join(data_dir, "a_shares", "daily", "000001.parquet")
        _write_parquet(daily, rows=99)  # data updated -> fingerprint changed
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, _feature, data_dir, commit, cfg_hash
        )

    def test_source_deleted_invalidates(self, tmp_path):
        data_dir, _feature, mpath, cfg, cfg_hash, commit = self._setup(tmp_path)
        daily = os.path.join(data_dir, "a_shares", "daily", "000001.parquet")
        os.unlink(daily)
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, _feature, data_dir, commit, cfg_hash
        )

    def test_schema_change_invalidates(self, tmp_path):
        data_dir, feature, mpath, cfg, cfg_hash, commit = self._setup(tmp_path)
        _write_parquet(feature, cols=("date", "f1", "f2", "f3"))
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, commit, cfg_hash
        )

    def test_missing_manifest_no_match(self, tmp_path):
        data_dir, feature, _mpath, cfg, cfg_hash, commit = self._setup(tmp_path)
        gone = os.path.join(str(tmp_path / "out"), ".manifests", "000001.json")
        os.unlink(gone)
        assert not cache_manifest.manifest_matches(
            gone, "000001", cfg, feature, data_dir, commit, cfg_hash
        )

    def test_manifest_records_channels_and_range(self, tmp_path):
        code = "000001"
        data_dir = _make_data_dir(tmp_path, code)
        feature = os.path.join(str(tmp_path / "out"), f"{code}.parquet")
        _write_parquet(feature, cols=("date", "f1"))
        cfg = _base_config()
        payload = cache_manifest.make_manifest(
            code, cfg, feature, data_dir,
            "commit-1", cache_manifest.config_hash(cfg),
        )
        assert payload["stock_code"] == code
        assert payload["horizon"] == 1 and payload["seq_len"] == 60
        assert payload["channels"]["daily"] == "complete"
        # Only daily exists in the fake data dir -> every other channel optional-missing.
        assert payload["channels"]["sentiment"] == "missing_optional"
        assert payload["source_files"]["daily"]["range"] == ["2000-01-01", "2026-01-01"]
        assert payload["source_files"]["daily"]["hash"] is not None
        assert payload["source_files"]["margin"]["hash"] is None

    def test_manifest_records_shared_inputs_hash(self, tmp_path):
        """§十二.3/§P1-1: the aggregate over all market-wide shared inputs is
        recorded in every manifest, so a lineage-DEFINITION change (a new shared
        input added later) invalidates old manifests instead of trusting them."""
        code = "000001"
        data_dir = _make_data_dir(tmp_path, code)
        feature = os.path.join(str(tmp_path / "out"), f"{code}.parquet")
        _write_parquet(feature, cols=("date", "f1"))
        cfg = _base_config()
        payload = cache_manifest.make_manifest(
            code, cfg, feature, data_dir,
            "commit-1", cache_manifest.config_hash(cfg),
        )
        assert payload["shared_inputs_hash"] == cache_manifest.shared_inputs_hash(data_dir)
        # Changing any shared input's content flips the aggregate.  Shared
        # fingerprints are memoized per path (shared data is frozen during a
        # build), so the test clears the memo to simulate a fresh build.
        macro = os.path.join(data_dir, "a_shares", "macro", "macro_daily.parquet")
        _write_parquet(macro)
        cache_manifest._shared_fingerprint.cache_clear()
        assert cache_manifest.shared_inputs_hash(data_dir) != payload["shared_inputs_hash"]

    def test_manifest_without_shared_inputs_hash_is_stale(self, tmp_path):
        """§P1-1: a manifest written before the shared-inputs aggregate existed
        cannot vouch for the shared lineage — it is treated as stale so the
        cache rebuilds and records complete lineage."""
        data_dir, feature, mpath, cfg, cfg_hash, commit = \
            TestManifest()._setup(tmp_path)
        payload = cache_manifest.make_manifest(
            "000001", cfg, feature, data_dir, commit, cfg_hash,
        )
        payload.pop("shared_inputs_hash", None)  # emulate a pre-P1-1 manifest
        cache_manifest.write_manifest(payload, mpath)
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, commit, cfg_hash
        )


class TestCodeTreeHash:
    """§十-2: non-git fallback for code-version tracking in manifests."""

    def test_returns_deterministic_hash_in_repo(self):
        h1 = cache_manifest.feature_code_tree_hash()
        h2 = cache_manifest.feature_code_tree_hash()
        assert h1 != "unknown"
        assert h1 == h2
        # Recorded in every fresh manifest alongside git_commit.
        payload = cache_manifest.make_manifest(
            "000001", _base_config(), "/tmp/fake.parquet",
            "/tmp/fake", "unknown", "hash",
        )
        assert payload["git_commit"] == "unknown"
        assert payload["feature_code_tree_hash"] != "unknown"

    def test_non_git_roundtrip_matches(self, tmp_path, monkeypatch):
        # Both build and use are outside git (git_commit=unknown on both
        # sides) and the code tree is unchanged → cache still valid.
        tree_hash = cache_manifest.feature_code_tree_hash()
        monkeypatch.setattr(
            cache_manifest, "feature_code_tree_hash", lambda: tree_hash
        )
        data_dir, feature, mpath, cfg, cfg_hash, _commit = \
            TestManifest()._setup(tmp_path)
        payload = cache_manifest.make_manifest(
            "000001", cfg, feature, data_dir, "unknown", cfg_hash,
        )
        cache_manifest.write_manifest(payload, mpath)
        assert cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, "unknown", cfg_hash
        )

    def test_non_git_code_change_invalidates(self, tmp_path, monkeypatch):
        # Same non-git setup, but the feature code tree changed between build
        # and use → the manifest must now fail, where the git_commit=unknown
        # check alone would have let it through.
        data_dir, feature, mpath, cfg, cfg_hash, _commit = \
            TestManifest()._setup(tmp_path)
        payload = cache_manifest.make_manifest(
            "000001", cfg, feature, data_dir, "unknown", cfg_hash,
        )
        cache_manifest.write_manifest(payload, mpath)
        monkeypatch.setattr(
            cache_manifest, "feature_code_tree_hash", lambda: "cafebabe"
        )
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, "unknown", cfg_hash
        )

    def test_git_build_stale_tree_invalidates(self, tmp_path, monkeypatch):
        # §十二.2: git IS available but the feature code tree changed between
        # build and use (uncommitted edit under an unchanged commit SHA) — the
        # manifest must now FAIL.  The old behavior trusted git_commit alone
        # and let a stale cache survive this exact case.
        data_dir, feature, mpath, cfg, cfg_hash, commit = \
            TestManifest()._setup(tmp_path)
        assert commit != "unknown"
        monkeypatch.setattr(
            cache_manifest, "feature_code_tree_hash", lambda: "cafebabe"
        )
        assert not cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, commit, cfg_hash
        )

    def test_git_build_matching_tree_valid(self, tmp_path, monkeypatch):
        # §十二.2: same git setup with the code tree UNCHANGED → the manifest
        # stays valid.  The always-compare rule must not reject a genuine cache
        # hit whose recorded tree hash still matches the current tree.
        tree_hash = cache_manifest.feature_code_tree_hash()
        monkeypatch.setattr(
            cache_manifest, "feature_code_tree_hash", lambda: tree_hash
        )
        data_dir, feature, mpath, cfg, cfg_hash, commit = \
            TestManifest()._setup(tmp_path)
        assert commit != "unknown"
        assert cache_manifest.manifest_matches(
            mpath, "000001", cfg, feature, data_dir, commit, cfg_hash
        )


class TestPaths:
    def test_source_paths_shape(self, tmp_path):
        paths = cache_manifest.source_paths(str(tmp_path), "600519")
        assert paths["daily"] == os.path.join(
            str(tmp_path), "a_shares", "daily", "600519.parquet"
        )
        assert paths["margin"] == os.path.join(
            str(tmp_path), "a_shares", "margin", "600519.parquet"
        )
        assert paths["earnings_forecasts"] == os.path.join(
            str(tmp_path), "a_shares", "earnings", "forecasts.parquet"
        )
        # Shared market-wide inputs are also exposed (same for every code).
        assert paths["macro"] == os.path.join(
            str(tmp_path), "a_shares", "macro", "macro_daily.parquet"
        )
        assert paths["calendar"] == os.path.join(
            str(tmp_path), "exchange_calendar", "a_shares.parquet"
        )
        assert paths["etf_flow"] == os.path.join(
            str(tmp_path), "a_shares", "etf_flow"
        )


class TestSharedInputs:
    """§十-1: market-wide shared inputs (macro / market-env / industry /
    sector-mapper / calendar / ETF-flow dir) must be fingerprinted in every
    per-stock manifest so their change invalidates a stale feature cache."""

    def _write_shared(self, data_dir, rel, rows=10):
        p = os.path.join(data_dir, *rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=rows),
            "value": [float(i) for i in range(rows)],
        })
        pq.write_table(pa.Table.from_pandas(df), p, compression="lz4")
        return p

    def _built(self, tmp_path, code="000001"):
        data_dir = _make_data_dir(tmp_path, code)
        feature = os.path.join(str(tmp_path / "out"), f"{code}.parquet")
        _write_parquet(feature, cols=("date", "f1"))
        cfg = _base_config()
        mpath = os.path.join(str(tmp_path / "out"), ".manifests", f"{code}.json")
        cache_manifest.write_manifest(
            cache_manifest.make_manifest(
                code, cfg, feature, data_dir, "c1", cache_manifest.config_hash(cfg)),
            mpath,
        )
        return data_dir, feature, mpath, cfg

    def test_manifest_records_shared_inputs(self, tmp_path):
        data_dir = _make_data_dir(tmp_path, "000001")
        self._write_shared(data_dir, ("a_shares", "macro", "macro_daily.parquet"))
        self._write_shared(data_dir, ("exchange_calendar", "a_shares.parquet"))
        feature = os.path.join(str(tmp_path / "out"), "000001.parquet")
        _write_parquet(feature, cols=("date", "f1"))
        cfg = _base_config()
        payload = cache_manifest.make_manifest(
            "000001", cfg, feature, data_dir, "c1", cache_manifest.config_hash(cfg),
        )
        assert payload["source_files"]["macro"]["hash"] is not None
        assert payload["source_files"]["calendar"]["hash"] is not None
        assert payload["channels"]["macro"] == "complete"
        # Directory absent -> optional-missing, never "complete".
        assert payload["source_files"]["etf_flow"]["hash"] is None
        assert payload["channels"]["etf_flow"] == "missing_optional"

    def test_shared_input_missing_optional_when_absent(self, tmp_path):
        data_dir, feature, _mpath, cfg = self._built(tmp_path)
        payload = cache_manifest.make_manifest(
            "000001", cfg, feature, data_dir, "c1", cache_manifest.config_hash(cfg),
        )
        assert payload["channels"]["macro"] == "missing_optional"
        assert payload["source_files"]["calendar"]["hash"] is None

    def test_shared_input_change_invalidates(self, tmp_path):
        data_dir, feature, mpath, cfg = self._built(tmp_path)
        self._write_shared(data_dir, ("a_shares", "macro", "macro_daily.parquet"))
        cache_manifest._shared_fingerprint.cache_clear()
        cfg_hash = cache_manifest.config_hash(cfg)
        # Rebuild manifest WITH the macro file present.
        cache_manifest.write_manifest(
            cache_manifest.make_manifest(
                "000001", cfg, feature, data_dir, "c1", cfg_hash), mpath)
        assert cache_manifest.manifest_matches(mpath, "000001", cfg, feature, data_dir, "c1", cfg_hash)
        # Macro content changes -> same manifest must now FAIL (fresh process
        # semantics: the memo is per-process, so clear it to simulate a rerun).
        self._write_shared(data_dir, ("a_shares", "macro", "macro_daily.parquet"), rows=99)
        cache_manifest._shared_fingerprint.cache_clear()
        assert not cache_manifest.manifest_matches(mpath, "000001", cfg, feature, data_dir, "c1", cfg_hash)

    def test_shared_dir_change_invalidates(self, tmp_path):
        data_dir, feature, mpath, cfg = self._built(tmp_path)
        self._write_shared(data_dir, ("a_shares", "etf_flow", "sector_test.parquet"))
        cache_manifest._shared_fingerprint.cache_clear()
        cfg_hash = cache_manifest.config_hash(cfg)
        cache_manifest.write_manifest(
            cache_manifest.make_manifest(
                "000001", cfg, feature, data_dir, "c1", cfg_hash), mpath)
        assert cache_manifest.manifest_matches(mpath, "000001", cfg, feature, data_dir, "c1", cfg_hash)
        # ANY file under the shared dir changing invalidates the whole cache.
        self._write_shared(data_dir, ("a_shares", "etf_flow", "sector_test.parquet"), rows=7)
        cache_manifest._shared_fingerprint.cache_clear()
        assert not cache_manifest.manifest_matches(mpath, "000001", cfg, feature, data_dir, "c1", cfg_hash)

    def test_old_manifest_without_shared_inputs_invalidates(self, tmp_path):
        """A manifest written before §十-1 cannot vouch for the shared inputs —
        it must be rebuilt even if every other field still matches."""
        code = "000001"
        data_dir, feature, mpath, cfg, cfg_hash, commit = TestManifest()._setup(tmp_path)
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        for name in ("macro", "market_env", "industry", "sector_mapper",
                     "calendar", "etf_flow"):
            m["source_files"].pop(name, None)
            m["channels"].pop(name, None)
        cache_manifest.write_manifest(m, mpath)
        assert not cache_manifest.manifest_matches(
            mpath, code, cfg, feature, data_dir, commit, cfg_hash)
