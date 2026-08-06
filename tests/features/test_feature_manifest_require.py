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
from stoke_ml.features.aux_cols import FUNDAMENTAL_COLS
from stoke_ml.features.pipeline import FeaturePipeline, _manifest_check_config

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


def _write_manifest(pdir, code, data_dir, commit, cfg_hash=None,
                    mconfig=None, **override):
    """Write a FULL sidecar manifest via ``make_manifest`` (not a hand-rolled
    subset).  §六 enforces the complete lineage check, so a partial 3-field
    manifest would fail for reasons unrelated to the test's intent — every
    field must be recorded exactly as ``build_panel_features`` validates it.
    ``**override`` lets a test corrupt one field to simulate staleness."""
    mdir = pdir / ".manifests"
    mdir.mkdir(exist_ok=True)
    if mconfig is None:
        mconfig = _manifest_check_config(SEQ_LEN, HORIZON)
    if cfg_hash is None:
        cfg_hash = cache_manifest.current_config_hash()
    m = cache_manifest.make_manifest(
        code, mconfig, str(pdir / f"{code}.parquet"), str(data_dir),
        commit, cfg_hash,
    )
    if override:
        m.update(override)
    (mdir / f"{code}.json").write_text(json.dumps(m), encoding="utf-8")


class TestRequireFeatureManifest:
    def _call(self, pdir, panel, require, tmp_path=None):
        return _pipeline().build_panel_features(
            panel, horizon=HORIZON, prebuilt_dir=str(pdir),
            require_feature_manifest=require,
            # §六: the full lineage check fingerprints source files + shared
            # inputs under data_dir — must be the SAME root the test used to
            # write the manifests, else every hash mismatches.
            data_dir=str(tmp_path or pdir.parent),
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
        _write_manifest(pdir, codes[0], str(tmp_path), commit)
        # Only 1 of N_STOCKS has a manifest → still a hard failure when required.
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_valid_manifests_require_passes(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, str(tmp_path), commit)
        data = self._call(pdir, panel, require=True)
        assert data["static_features"].shape[0] == N_STOCKS

    def test_stale_git_commit_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, str(tmp_path), commit)
        # Corrupt ONE manifest's git_commit → whole run must fail when required.
        _write_manifest(pdir, codes[0], str(tmp_path), commit,
                        git_commit="deadbeef" * 5)
        with pytest.raises(RuntimeError, match="feature-manifest check FAILED"):
            self._call(pdir, panel, require=True)

    def test_stale_schema_hash_require_raises(self, tmp_path):
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, str(tmp_path), commit)
        _write_manifest(pdir, codes[1], str(tmp_path), commit,
                        feature_schema_hash="0" * 16)
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
            _write_manifest(pdir, c, str(tmp_path), commit, cfg_hash=good_hash)
        # Corrupt ONE manifest's config_hash → whole run fails when required.
        _write_manifest(pdir, codes[0], str(tmp_path), commit, cfg_hash="0" * 16)
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

    def test_manifest_detailed_reasons_surface_in_raise(self, tmp_path):
        """§六: the full lineage check reports structured failure reasons, so a
        formal-run failure tells the user WHICH lineage entry went stale (not
        just "some mismatch")."""
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        commit = cache_manifest.git_head()
        for c in codes:
            _write_manifest(pdir, c, str(tmp_path), commit)
        # Corrupt the FIRST stock's config_hash and the SECOND's git_commit —
        # both must surface as structured reasons in the raise.
        _write_manifest(pdir, codes[0], str(tmp_path), commit, cfg_hash="0" * 16)
        _write_manifest(pdir, codes[1], str(tmp_path), commit,
                        git_commit="deadbeef" * 5)
        with pytest.raises(RuntimeError, match="config_changed") as exc:
            self._call(pdir, panel, require=True)
        assert "code_changed" in str(exc.value)
        assert "reason_counts" in str(exc.value)

    def test_all_stocks_cleaned_raises_with_stats(self):
        """§十四-1: every input stock dropped raises with drop stats, NOT the
        misleading 'Max timesteps (0)' (max_T collapses to 0 on an empty panel)."""
        # All-weekend dates → every stock's calendar cleaning returns None.
        weekend = pd.to_datetime(["2022-01-01", "2022-01-02"])  # Sat/Sun
        rows = []
        for code in ("600000", "600001"):
            for t in range(2):
                rows.append({
                    "date": weekend[t], "stock_code": code,
                    "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
                    "volume": 1e6, "amount": 1e7,
                })
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="every input stock was dropped") as exc:
            _pipeline().build_panel_features(panel, horizon=HORIZON)
        assert "drop_reason_counts" in str(exc.value)
        assert "calendar_clean_dropped" in str(exc.value)
        assert "2 input stock(s)" in str(exc.value)

    def test_use_topic_default_off_drops_topic_columns(self, tmp_path):
        """§七: topic_* columns (global_frozen topic model — non-PIT) are
        dropped on the PREBUILT read path by default."""
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        # Inject topic_* columns into one stock's prebuilt parquet.
        code = codes[0]
        df = pd.read_parquet(str(pdir / f"{code}.parquet"))
        df["topic_entropy"] = 0.5
        df["topic_dominant"] = 3
        df["topic_ipo_sent"] = 0.1
        df.to_parquet(str(pdir / f"{code}.parquet"), index=False, compression="lz4")
        data = self._call(pdir, panel, require=False)
        known = set(data["past_known_cols"]) | set(data["past_observed_cols"])
        assert not any(c.startswith("topic_") for c in known)

    def test_use_topic_true_keeps_topic_columns(self, tmp_path):
        """§七: explicitly enabling use_topic (ablation-only) keeps topic_*."""
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        code = codes[0]
        df = pd.read_parquet(str(pdir / f"{code}.parquet"))
        df["topic_entropy"] = 0.5
        df["topic_dominant"] = 3
        df.to_parquet(str(pdir / f"{code}.parquet"), index=False, compression="lz4")
        data = FeaturePipeline(
            seq_len=SEQ_LEN, minute_mode=False,
            use_board=False, use_sector=False, use_concept=False,
            min_history=SEQ_LEN, use_topic=True,
        ).build_panel_features(
            panel, horizon=HORIZON, prebuilt_dir=str(pdir),
            data_dir=str(tmp_path),
        )
        known = set(data["past_known_cols"]) | set(data["past_observed_cols"])
        assert any(c.startswith("topic_") for c in known)

    def _inject_fundamental_columns(self, pdir, code):
        """Inject all 8 FUNDAMENTAL_COLS into one stock's prebuilt parquet —
        the same shape a --panel-mode build_features.py run (all use_* True)
        would have baked in."""
        df = pd.read_parquet(str(pdir / f"{code}.parquet"))
        for i, col in enumerate(FUNDAMENTAL_COLS):
            df[col] = 0.01 * (i + 1)
        df.to_parquet(str(pdir / f"{code}.parquet"), index=False, compression="lz4")

    def test_use_fundamental_false_drops_fundamental_columns(self, tmp_path):
        """T3 decision #1: a safe-only run (use_fundamental=False) must NOT
        consume fundamental columns that a prebuilt parquet carries — the
        prebuilt read path drops FUNDAMENTAL_COLS exactly like topic_* columns
        (§七 pattern).  Without the fix, build_features.py's all-True build
        would silently leak the revised-aligned channel into a formal run."""
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        self._inject_fundamental_columns(pdir, codes[0])
        data = FeaturePipeline(
            seq_len=SEQ_LEN, minute_mode=False,
            use_board=False, use_sector=False, use_concept=False,
            min_history=SEQ_LEN, use_fundamental=False,
        ).build_panel_features(
            panel, horizon=HORIZON, prebuilt_dir=str(pdir),
            data_dir=str(tmp_path),
        )
        known = set(data["past_known_cols"]) | set(data["past_observed_cols"])
        assert not known & set(FUNDAMENTAL_COLS)

    def test_use_fundamental_true_keeps_fundamental_columns(self, tmp_path):
        """T3: the explicit ablation path (use_fundamental=True) KEEPS the
        FUNDAMENTAL_COLS — the flag is the ONLY way fundamental enters a
        safe-only run."""
        panel = _make_synthetic_panel()
        pdir, codes = _build_prebuilt_dir(tmp_path, panel)
        self._inject_fundamental_columns(pdir, codes[0])
        data = FeaturePipeline(
            seq_len=SEQ_LEN, minute_mode=False,
            use_board=False, use_sector=False, use_concept=False,
            min_history=SEQ_LEN, use_fundamental=True,
        ).build_panel_features(
            panel, horizon=HORIZON, prebuilt_dir=str(pdir),
            data_dir=str(tmp_path),
        )
        known = set(data["past_known_cols"]) | set(data["past_observed_cols"])
        assert set(FUNDAMENTAL_COLS) <= known
