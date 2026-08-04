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
