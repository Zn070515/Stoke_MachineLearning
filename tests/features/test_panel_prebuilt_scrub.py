"""§T7/§十四: generic per-channel prebuilt scrub driven by CHANNEL_COLUMNS.

``build_features.py --panel-mode`` bakes EVERY channel in with all-True
defaults, so a restricted run (revision-safe vintage / ablation) would otherwise
silently consume channels its pipeline does not request.  The prebuilt load
loop drops the EXACT ``CHANNEL_COLUMNS`` set of every channel whose ``use_*``
switch is OFF — by set membership, never name-prefix matching (the
market_env-vs-macd / market_env_refine collision trap).

These tests inject denied-channel columns into synthetic prebuilt parquets and
assert they are scrubbed when the channels are OFF and KEPT when ON, plus the
fundamental_refine↔fundamental coupling that makes a revision-safe run drop the
refiner's columns too.
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.config.feature_profile import CHANNEL_COLUMNS
from stoke_ml.features.pipeline import FeaturePipeline

from features.test_static_feature_pit import _make_synthetic_panel

N_STOCKS = 4
N_DAYS = 60
SEQ_LEN = 10
HORIZON = 3

# Representative denied-channel columns picked from the real CHANNEL_COLUMNS
# sets, so the test keys on the channel→switch mapping rather than hand-rolled
# column names.
_DENIED_INJECT = {
    "fundamental": next(iter(CHANNEL_COLUMNS["fundamental"])),
    "fundamental_refine": next(iter(CHANNEL_COLUMNS["fundamental_refine"])),
    "market_env_refine": next(iter(CHANNEL_COLUMNS["market_env_refine"])),
    "earnings": next(iter(CHANNEL_COLUMNS["earnings"])),
    "macro": next(iter(CHANNEL_COLUMNS["macro"])),
    "pledge": next(iter(CHANNEL_COLUMNS["pledge"])),
}
# A market_env bare column (default-ON) that must SURVIVE a scrub that drops
# market_env_refine — the two channels' sets are disjoint.
_MENV_SURVIVOR = next(iter(CHANNEL_COLUMNS["market_env"]))
# A technical column that must survive any channel scrub.
_TECH_SURVIVOR = "rsi_12"


def _base_pipeline(**kw):
    base = dict(
        seq_len=SEQ_LEN, minute_mode=False,
        use_board=False, use_sector=False, use_concept=False,
        min_history=SEQ_LEN,
    )
    base.update(kw)
    return FeaturePipeline(**base)


def _build_prebuilt(tmp_path):
    panel = _make_synthetic_panel(n_stocks=N_STOCKS, n_days=N_DAYS)
    codes = sorted(panel["stock_code"].unique())
    pdir = tmp_path / "features_panel"
    pdir.mkdir(exist_ok=True)
    pipe = _base_pipeline()
    for code in codes:
        df = panel[panel["stock_code"] == code].sort_values("date")
        pipe.save_features(str(pdir / f"{code}.parquet"), df, panel_mode=True)
    return pdir, panel, codes


def _inject_columns(pdir, codes, cols):
    """Bake extra columns into every prebuilt parquet — the shape a full
    all-True build_features.py run would have produced."""
    for code in codes:
        df = pd.read_parquet(str(pdir / f"{code}.parquet"))
        for i, col in enumerate(cols):
            df[col] = float(i + 1) * 0.1
        df.to_parquet(str(pdir / f"{code}.parquet"), index=False,
                      compression="lz4")


def _build(pdir, panel, pipeline):
    data = pipeline.build_panel_features(
        panel, horizon=HORIZON, prebuilt_dir=str(pdir),
        data_dir=str(pdir.parent),
    )
    return set(data["past_known_cols"]) | set(data["past_observed_cols"])


def test_scrub_drops_denied_channel_columns_when_off(tmp_path):
    """A revision-safe-like switch set (fundamental/earnings/macro/pledge/
    market_env_refine OFF) must drop exactly those channels' injected columns —
    while keeping a market_env bare column and the technical survivor."""
    pdir, panel, codes = _build_prebuilt(tmp_path)
    inject = list(_DENIED_INJECT.values())
    _inject_columns(pdir, codes, inject + [_MENV_SURVIVOR, _TECH_SURVIVOR])
    pipe = _base_pipeline(
        use_fundamental=False,   # couples fundamental_refine off
        use_earnings=False,
        use_macro=False,
        use_pledge=False,
        use_market_env_refine=False,
    )
    known = _build(pdir, panel, pipe)
    for channel, col in _DENIED_INJECT.items():
        assert col not in known, f"{channel} column {col!r} leaked through scrub"
    assert _MENV_SURVIVOR in known, "market_env (default-ON) column was over-scrubbed"
    assert _TECH_SURVIVOR in known, "technical column was over-scrubbed"


def test_scrub_keeps_columns_when_channels_on(tmp_path):
    """An all-ON pipeline (the build-time switch set) must NOT scrub — the
    injected denied-channel columns are consumed, proving the scrub is driven
    by the pipeline's actual switches, not by a static blacklist."""
    pdir, panel, codes = _build_prebuilt(tmp_path)
    inject = list(_DENIED_INJECT.values())
    _inject_columns(pdir, codes, inject)
    known = _build(pdir, panel, _base_pipeline())
    for channel, col in _DENIED_INJECT.items():
        assert col in known, f"{channel} column {col!r} over-scrubbed when ON"


def test_scrub_uses_exact_set_membership_no_prefix_matching(tmp_path):
    """The core safety property: a column that merely SHARES a denied channel's
    prefix but is NOT in its CHANNEL_COLUMNS set survives the scrub.  A
    name-prefix scrub (the market_env-vs-macd / market_env_refine trap) would
    drop it; exact-set membership cannot."""
    pdir, panel, codes = _build_prebuilt(tmp_path)
    decoys = ["menv_fake_col", "shibor_fake_col", "has_forecast_fake"]
    _inject_columns(pdir, codes, decoys)
    pipe = _base_pipeline(
        use_fundamental=False,
        use_earnings=False,
        use_macro=False,
        use_pledge=False,
        use_market_env_refine=False,
    )
    known = _build(pdir, panel, pipe)
    for col in decoys:
        assert col in known, f"prefix decoy {col!r} was dropped (prefix scrub leak)"


def test_scrub_scoped_to_existing_columns_no_crash(tmp_path):
    """Scrubbing a channel whose columns are absent from the parquet (the usual
    synthetic case) must be a no-op — never a KeyError."""
    pdir, panel, codes = _build_prebuilt(tmp_path)
    # No injection: the parquets carry only the engineered + market_env/sentiment
    # columns the save produced; every denied channel is simply absent.
    pipe = _base_pipeline(
        use_fundamental=False,
        use_earnings=False,
        use_macro=False,
        use_pledge=False,
        use_market_env_refine=False,
    )
    known = _build(pdir, panel, pipe)  # must not raise
    assert _TECH_SURVIVOR in known


# ── fundamental_refine ↔ fundamental coupling (§T7 pipeline fix) ───────

def test_fundamental_refine_coupled_to_fundamental_off():
    """use_fundamental=False silently forces use_fundamental_refine off — the
    refiner must never run on columns a run that disabled the channel never
    requested (§T3 leak fix)."""
    pipe = _base_pipeline(use_fundamental=False)
    assert pipe.use_fundamental_refine is False
    assert pipe._fundamental_refiner is None


def test_fundamental_refine_kept_when_fundamental_on():
    pipe = _base_pipeline(use_fundamental=True)
    assert pipe.use_fundamental_refine is True


def test_coupling_makes_revision_safe_scrub_drop_refine_columns(tmp_path):
    """End-to-end: with fundamental OFF the pipeline ALSO reports
    use_fundamental_refine False, so the scrub drops the fundamental_refine
    columns of a full prebuilt — a revision-safe run never leaks them."""
    pdir, panel, codes = _build_prebuilt(tmp_path)
    _inject_columns(pdir, codes, ["f_score", "valuation_composite_z"])
    pipe = _base_pipeline(use_fundamental=False)
    assert pipe.use_fundamental_refine is False
    known = _build(pdir, panel, pipe)
    assert "f_score" not in known
    assert "valuation_composite_z" not in known
