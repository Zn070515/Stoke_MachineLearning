"""§十四 (T7): frozen feature profiles + per-channel column ownership.

``CHANNEL_COLUMNS`` is the single source of truth for the generic prebuilt
scrub (``panel_builder.py``): every channel's EXACT output-column set, with an
overlap invariant (a column belongs to ≤1 channel) that makes the scrub safe —
a column is dropped for exactly one channel's switch, never for the wrong one.
``FEATURE_PROFILES["headline_v1"]`` is the formal baseline: required channels
all safe-only-ALLOWED, with tunable minimum-coverage thresholds.

None of these read the 109GB feature store.
"""
import pytest

from stoke_ml.config.feature_profile import (
    CHANNEL_COLUMNS,
    FEATURE_PROFILES,
    FeatureProfile,
    minimum_coverage,
    profile_for,
    resolve_required_channels,
)
from stoke_ml.data.vintage_policy import VintagePolicy, channel_allowed

# The 27 channels CHANNEL_COLUMNS must own (docs table, §十四).
_EXPECTED_CHANNELS = frozenset({
    "sentiment", "guba", "comment", "announcement", "margin", "northbound",
    "dragon_tiger", "fundamental", "fundamental_refine", "earnings",
    "valuation", "etf_flow", "capital_flow", "block_trade", "shareholder",
    "lockup", "dividend", "board", "sector", "concept", "industry", "macro",
    "pledge", "index_membership", "market_env", "market_env_refine",
    "limit_up",
})


# ── CHANNEL_COLUMNS ownership + overlap invariant ─────────────────────

def test_channel_columns_owns_exactly_the_documented_channels():
    """Every documented data dimension is owned by exactly one CHANNEL_COLUMNS
    entry (plus nothing extra)."""
    assert set(CHANNEL_COLUMNS) == _EXPECTED_CHANNELS


def test_topic_not_a_channel_columns_owner():
    """topic_* columns are deliberately NOT in CHANNEL_COLUMNS — the frozen
    non-PIT topic model's columns are already dropped by the prefix-based
    FeaturePipeline._drop_topic_columns, so a generic scrub would be redundant
    (and a prefix drop there is the documented exception to exact-set scrub)."""
    assert "topic" not in CHANNEL_COLUMNS


def test_no_channel_column_set_is_empty():
    for channel, cols in CHANNEL_COLUMNS.items():
        assert len(cols) > 0, f"{channel} owns no columns"


def test_overlap_invariant_no_column_claimed_by_two_channels():
    """The scrub is only safe because ownership is exclusive: rebuild the
    column→channel owner map and prove no column is claimed twice."""
    owner = {}
    for channel, cols in CHANNEL_COLUMNS.items():
        for col in cols:
            assert col not in owner, (
                f"column {col!r} claimed by both {owner[col]!r} and "
                f"{channel!r} — CHANNEL_COLUMNS overlap invariant violated")
            owner[col] = channel
    assert len(owner) == sum(len(c) for c in CHANNEL_COLUMNS.values())


def test_market_env_bare_vs_refine_names_disjoint():
    """The exact trap the generic scrub avoids: market_env bare names (e.g.
    high_low_ratio) and market_env_refine menv_* names must be disjoint sets —
    a prefix-based scrub would conflate them; exact-set ownership cannot."""
    menv = CHANNEL_COLUMNS["market_env"]
    menv_refine = CHANNEL_COLUMNS["market_env_refine"]
    assert menv.isdisjoint(menv_refine)


# ── headline_v1 profile ───────────────────────────────────────────────

def _headline():
    return FEATURE_PROFILES["headline_v1"]


def test_headline_v1_is_a_frozen_profile():
    p = _headline()
    assert isinstance(p, FeatureProfile)
    with pytest.raises(Exception):
        p.required_channels = ("mutated",)  # frozen dataclass


def test_headline_v1_required_channels_all_safe_only_allowed():
    """Every required channel of the formal baseline must be admissible under
    the safe-only vintage policy — a required channel the policy denies would
    be self-contradictory (require something the run is forbidden to consume)."""
    for ch in _headline().required_channels:
        assert channel_allowed(ch, VintagePolicy.SAFE_ONLY), ch


def test_headline_v1_required_channels_exact_baseline():
    assert tuple(_headline().required_channels) == (
        "sentiment", "guba", "comment", "announcement", "margin",
        "northbound", "dragon_tiger", "capital_flow", "etf_flow",
        "block_trade", "lockup", "dividend", "industry", "market_env",
    )


def test_headline_v1_excludes_policy_denied_and_default_off_dims():
    """safe-only denies fundamental/macro/earnings/valuation/pledge/
    shareholder/index_membership/market_env_refine/sector/concept, and
    board/sector/concept/limit_up/topic default OFF — none may be required."""
    required = set(_headline().required_channels)
    for denied in ("fundamental", "fundamental_refine", "macro", "earnings",
                   "valuation", "pledge", "shareholder", "index_membership",
                   "market_env_refine", "sector", "concept", "board",
                   "limit_up", "topic"):
        assert denied not in required, denied


def test_headline_v1_minimum_coverage_within_unit_interval():
    for ch, m in _headline().minimum_coverage.items():
        assert 0.0 < m <= 1.0, f"{ch}: {m}"


def test_headline_v1_minimum_coverage_keys_subset_of_required():
    """A channel with a coverage threshold must also be required — a channel
    the run does not require should never be aborted on coverage."""
    p = _headline()
    assert set(p.minimum_coverage) <= set(p.required_channels)


def test_headline_v1_vintage_policy_is_safe_only():
    assert _headline().vintage_policy == "safe-only"


def test_headline_v1_is_the_only_shipped_profile():
    assert set(FEATURE_PROFILES) == {"headline_v1"}


# ── resolve_required_channels / minimum_coverage / profile_for ─────────

def test_resolve_required_channels_unions_profile_and_extra():
    rs = resolve_required_channels("headline_v1", {"sentiment", "extra_ch"})
    assert rs == set(_headline().required_channels) | {"extra_ch"}


def test_resolve_required_channels_no_profile_returns_only_extra():
    # None / "" / "none" (gate-off sentinel) / unknown → just the explicit set.
    for name in (None, "", "none", "bogus"):
        assert resolve_required_channels(name, {"x", "y"}) == {"x", "y"}, name


def test_resolve_required_channels_no_extra_returns_profile():
    assert resolve_required_channels("headline_v1", None) == set(
        _headline().required_channels)


def test_minimum_coverage_returns_profile_thresholds():
    assert minimum_coverage("headline_v1") == dict(
        _headline().minimum_coverage)


def test_minimum_coverage_empty_without_profile():
    for name in (None, "", "none", "bogus"):
        assert minimum_coverage(name) == {}, name


def test_profile_for_known_profile():
    assert profile_for("headline_v1") is _headline()


def test_profile_for_none_or_unknown_is_none():
    for name in (None, "", "none", "bogus"):
        assert profile_for(name) is None, name
