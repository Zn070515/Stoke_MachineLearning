"""§十一-1: prebuilt feature manifests must be REQUIRED for formal training.

``build_panel_features(require_feature_manifest=True)`` FAILS instead of
warning when a prebuilt feature parquet lacks a sidecar manifest, or its
manifest is stale (schema drift / built by a different git commit), or no
``.manifests/`` exists at all.  ``False`` keeps the legacy warn-only path so
un-manifested unit-test / legacy dirs still train.
"""
import json

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.features import cache_manifest
from stoke_ml.features.pipeline import FeaturePipeline

N_STOCKS = 8
N_DAYS = 100
SEQ_LEN = 10
HORIZON = 5


def _make_synthetic_panel(n_stocks=N_STOCKS, n_days=N_DAYS, seed=7):
    rng = np.random.RandomState(seed)
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    dates = pd.DatetimeIndex(
        TradingCalendar("a_shares").get_trading_days("2022-01-03", "2022-12-31")
    )[:n_days]
    rows = []
    for i, code in enumerate(codes):
        close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = np.abs(rng.normal(1e6, 2e5, n_days))
        for t in range(n_days):
            rows.append({
                "date": dates[t], "stock_code": code,
                "open": float(open_[t]), "high": float(high[t]),
                "low": float(low[t]), "close": float(close[t]),
                "volume": float(volume[t]), "amount": float(volume[t] * close[t]),
            })
    return pd.DataFrame(rows)


def _pipeline():
    return FeaturePipeline(
        seq_len=SEQ_LEN,
        minute_mode=False,
        use_board=False, use_sector=False, use_concept=False,
        min_history=SEQ_LEN,
    )


def _build_prebuilt_dir(tmp_path, panel):
    """Engineer + save panel-mode feature parquets for every stock."""
    codes = sorted(panel["stock_code"].unique())
    pdir = tmp_path / "features_panel"
    pdir.mkdir(exist_ok=True)
    pipe = _pipeline()
    for code in codes:
        df = panel[panel["stock_code"] == code].sort_values("date")
        pipe.save_features(
            str(pdir / f"{code}.parquet"),
            df,
            panel_mode=True,
        )
    return pdir, codes


def _write_manifest(pdir, code, commit, schema_hash, config_hash=None):
    mdir = pdir / ".manifests"
    mdir.mkdir(exist_ok=True)
    if config_hash is None:
        config_hash = cache_manifest.current_config_hash()
    (mdir / f"{code}.json").write_text(
        json.dumps({
            "git_commit": commit,
            "feature_schema_hash": schema_hash,
            "config_hash": config_hash,
        }),
        encoding="utf-8",
    )


class TestRequireFeatureManifest:
    def _call(self, pdir, panel, require):
        return _pipeline().build_panel_features(
            panel, horizon=HORIZON, prebuilt_dir=str(pdir),
            require_feature_manifest=require,
        )

    def test_no_manifests_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, _ = _build_prebuilt_dir(tmp_path, panel)
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_no_manifests_warn_only(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, _ = _build_prebuilt_dir(tmp_path, panel)
        # Legacy warn-only path still trains with an un-manifested dir.
        data = self._call(pdir, panel, require=False)
        assert data["static_features"].shape[0] == N_STOCKS

    def test_partial_manifests_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        _write_manifest(pdir, codes[0], commit,
                        cache_manifest.schema_hash(str(pdir / f"{codes[0]}.parquet")))
        # Only 1 of N_STOCKS has a manifest → still a hard failure when required.
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_valid_manifests_require_passes(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, commit,
                            cache_manifest.schema_hash(str(pdir / f"{c}.parquet")))
        data = self._call(pdir, panel, require=True)
        assert data["static_features"].shape[0] == N_STOCKS

    def test_stale_git_commit_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, commit,
                            cache_manifest.schema_hash(str(pdir / f"{c}.parquet")))
        # Corrupt ONE manifest's git_commit → whole run must fail when required.
        _write_manifest(pdir, codes[0], "deadbeef" * 5,
                        cache_manifest.schema_hash(str(pdir / f"{codes[0]}.parquet")))
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_stale_schema_hash_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, commit,
                            cache_manifest.schema_hash(str(pdir / f"{c}.parquet")))
        _write_manifest(pdir, codes[1], commit, "0" * 16)
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_stale_config_hash_require_raises(self, tmp_path):
        """§十一-3: a same-commit config.yaml change must fail a formal run.

        The manifest records the config snapshot hash; the training-time check
        recomputes it from the current config, so a config value that drifted
        after the build (without a schema or git change) is caught.
        """
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        good_hash = cache_manifest.current_config_hash()
        for c in codes:
            _write_manifest(pdir, c, commit,
                            cache_manifest.schema_hash(str(pdir / f"{c}.parquet")),
                            config_hash=good_hash)
        # Corrupt ONE manifest's config_hash → whole run fails when required.
        _write_manifest(pdir, codes[0], commit,
                        cache_manifest.schema_hash(str(pdir / f"{codes[0]}.parquet")),
                        config_hash="0" * 16)
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_missing_parquet_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        (pdir / f"{codes[0]}.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="feature parquets missing"):
            self._call(pdir, panel, require=True)

    def test_missing_parquet_warn_only_drops(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        (pdir / f"{codes[0]}.parquet").unlink()
        data = self._call(pdir, panel, require=False)
        # Dropped stock is excluded from the panel; the rest still train.
        assert data["static_features"].shape[0] == N_STOCKS - 1
