"""§十四 (T7): frozen feature profiles + per-channel column ownership.

``CHANNEL_COLUMNS`` is the single source of truth for the generic prebuilt
scrub (``panel_builder.py``): every channel's EXACT output-column set, with an
overlap invariant (a column belongs to ≤1 channel) that makes the scrub safe —
a column is dropped for exactly one channel's switch, never for the wrong one.
``FEATURE_PROFILES["headline_v1"]`` is the formal baseline: required channels
all revision-safe-ALLOWED, with tunable minimum-coverage thresholds.

None of these read the 109GB feature store.
"""
import pytest

from stoke_ml.config.feature_profile import (
    CHANNEL_COLUMNS,
    FEATURE_PROFILES,
    MARKET_ENV_ACCOUNT_COLS,
    MARKET_ENV_ACCOUNT_PIT,
    MARKET_ENV_PRICE_COLS,
    CoverageContract,
    FeatureProfile,
    _COVERAGE_METRICS,
    market_env_required_columns,
    minimum_coverage,
    profile_for,
    resolve_required_channels,
)
from stoke_ml.data.vintage_policy import VintagePolicy, channel_allowed

# The channels CHANNEL_COLUMNS must own (docs table, §十四 — the 27 documented
# dimensions, plus the §T5 market_env_account sub-channel for the PROXY-PIT
# ACCOUNT part of the market_env file).
_EXPECTED_CHANNELS = frozenset({
    "sentiment", "guba", "comment", "announcement", "margin", "northbound",
    "dragon_tiger", "fundamental", "fundamental_refine", "earnings",
    "valuation", "etf_flow", "capital_flow", "block_trade", "shareholder",
    "lockup", "dividend", "board", "sector", "concept", "industry", "macro",
    "pledge", "index_membership", "market_env", "market_env_account",
    "market_env_refine", "limit_up",
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


# ── §T5 market_env price/account split (enforcement) ───────────────────

def test_market_env_channel_owns_price_only_account_is_own_channel():
    """§T5: the market_env channel is the VERIFIED PRICE part; the PROXY ACCOUNT
    part is its own ablation-only channel so the generic scrub can drop it
    whenever use_market_env_account is OFF (the default)."""
    assert CHANNEL_COLUMNS["market_env"] == MARKET_ENV_PRICE_COLS
    assert CHANNEL_COLUMNS["market_env_account"] == MARKET_ENV_ACCOUNT_COLS
    # the two parts partition the 7-column consumer file exactly, no overlap
    assert MARKET_ENV_PRICE_COLS | MARKET_ENV_ACCOUNT_COLS == frozenset({
        "high_low_ratio", "mkt_cap_total_z", "avg_account_cap_z",
        "investor_new_num", "investor_new_z", "market_adv_ratio",
        "market_turnover_z",
    })
    assert MARKET_ENV_PRICE_COLS.isdisjoint(MARKET_ENV_ACCOUNT_COLS)


def test_market_env_required_columns_price_only_while_account_proxy():
    """§T5: while MARKET_ENV_ACCOUNT_PIT == 'proxy', the required sub-set is the
    PRICE part only — a required sub-set never includes an unverified part."""
    assert MARKET_ENV_ACCOUNT_PIT == "proxy"
    assert market_env_required_columns("headline_v1") == MARKET_ENV_PRICE_COLS


def test_market_env_required_columns_expands_when_account_verified(monkeypatch):
    """§T5: the moment the account part is declared verified (a real publish
    date is recorded by the builder), the account columns join the required
    sub-set automatically — no ablation flag needed."""
    import stoke_ml.config.feature_profile as _fp
    monkeypatch.setattr(_fp, "MARKET_ENV_ACCOUNT_PIT", "verified")
    assert _fp.market_env_account_is_verified() is True
    assert market_env_required_columns("headline_v1") == (
        MARKET_ENV_PRICE_COLS | MARKET_ENV_ACCOUNT_COLS
    )


def test_market_env_required_columns_empty_for_profile_without_market_env(monkeypatch):
    """§T5 quality: the ``profile_name`` argument is LOAD-BEARING — a profile
    that does not require the market_env channel requires none of its columns.
    (The default/None caller — the live merge — keeps the price part required.)"""
    import stoke_ml.config.feature_profile as _fp
    no_menv = FeatureProfile(
        name="no_menv", required_channels=("sentiment",),
        coverage_contracts={}, vintage_policy="revision-safe",
    )
    monkeypatch.setitem(_fp.FEATURE_PROFILES, "no_menv", no_menv)
    assert market_env_required_columns("no_menv") == frozenset()
    # the default / None caller is unchanged: price part still required
    assert market_env_required_columns(None) == MARKET_ENV_PRICE_COLS


# ── headline_v1 profile ───────────────────────────────────────────────

def _headline():
    return FEATURE_PROFILES["headline_v1"]


def test_headline_v1_is_a_frozen_profile():
    p = _headline()
    assert isinstance(p, FeatureProfile)
    with pytest.raises(Exception):
        p.required_channels = ("mutated",)  # frozen dataclass


def test_headline_v1_required_channels_all_revision_safe_allowed():
    """Every required channel of the formal baseline must be admissible under
    the revision-safe vintage policy — a required channel the policy denies
    would be self-contradictory (require something the run is forbidden to
    consume)."""
    for ch in _headline().required_channels:
        assert channel_allowed(ch, VintagePolicy.REVISION_SAFE), ch


def test_headline_v1_required_channels_exact_baseline():
    assert tuple(_headline().required_channels) == (
        "sentiment", "guba", "comment", "announcement", "margin",
        "northbound", "dragon_tiger", "capital_flow", "etf_flow",
        "block_trade", "lockup", "dividend", "industry", "market_env",
    )


def test_headline_v1_excludes_policy_denied_and_default_off_dims():
    """revision-safe denies fundamental/macro/earnings/valuation/pledge/
    shareholder/index_membership/market_env_refine/sector/concept, and
    board/sector/concept/limit_up/topic default OFF — none may be required."""
    required = set(_headline().required_channels)
    for denied in ("fundamental", "fundamental_refine", "macro", "earnings",
                   "valuation", "pledge", "shareholder", "index_membership",
                   "market_env_refine", "sector", "concept", "board",
                   "limit_up", "topic"):
        assert denied not in required, denied


def test_headline_v1_coverage_contracts_within_unit_interval():
    for ch, c in _headline().coverage_contracts.items():
        assert 0.0 < c.threshold <= 1.0, f"{ch}: {c.threshold}"
        assert c.metric in _COVERAGE_METRICS, f"{ch}: {c.metric}"


def test_coverage_metrics_include_era_metrics():
    """§T8: ``era_coverage`` (provider-era retrieval coverage, the metric behind
    the sentiment/guba contracts) and the reserved ``date_availability`` are
    valid declared metrics; ``era_coverage`` is ACTIVE (used by a contract),
    ``date_availability`` is reserved for a future date-presence metric."""
    assert "era_coverage" in _COVERAGE_METRICS
    assert "date_availability" in _COVERAGE_METRICS
    assert any(
        c.metric == "era_coverage"
        for c in _headline().coverage_contracts.values())
    assert all(
        c.metric != "date_availability"
        for c in _headline().coverage_contracts.values())


def test_headline_v1_coverage_contract_keys_subset_of_required():
    """A channel with a coverage contract must also be required — a channel
    the run does not require should never be aborted on coverage.
    dragon_tiger is REQUIRED but presence-only (no contract) — its presence
    convention is exactly what the absense from the contract map preserves."""
    p = _headline()
    assert set(p.coverage_contracts) <= set(p.required_channels)
    assert "dragon_tiger" in p.required_channels
    assert "dragon_tiger" not in p.coverage_contracts


def test_headline_v1_coverage_contracts_exact_map():
    """Pin the exact per-channel (metric, threshold) contract map (§T4).

    The SPARSE text event channels (sentiment / guba) use era_coverage (§T8) —
    the provider-era retrieval-coverage metric that distinguishes a stock with
    genuinely no events (no_event, legitimate) from an era we never observed
    (not_observed, a data gap).  The remaining per-stock channels use
    stock_coverage; the MARKET-WIDE broadcast channels (etf_flow / industry /
    market_env) use date_coverage — their value is the same for every stock per
    date, so stock coverage is vacuous (1.0 whenever the file exists) and date
    coverage is the meaningful metric.  dragon_tiger is presence-only (required,
    NO contract).  Thresholds are the historical minimum_coverage values — NOT
    re-tuned.
    """
    assert _headline().coverage_contracts == {
        "sentiment": CoverageContract("era_coverage", 0.90),
        "guba": CoverageContract("era_coverage", 0.90),
        "comment": CoverageContract("stock_coverage", 0.90),
        "announcement": CoverageContract("stock_coverage", 0.70),
        "margin": CoverageContract("stock_coverage", 0.95),
        "northbound": CoverageContract("stock_coverage", 0.90),
        "capital_flow": CoverageContract("stock_coverage", 0.90),
        "block_trade": CoverageContract("stock_coverage", 0.30),
        "lockup": CoverageContract("stock_coverage", 0.30),
        "dividend": CoverageContract("stock_coverage", 0.30),
        "etf_flow": CoverageContract("date_coverage", 0.80),
        "industry": CoverageContract("date_coverage", 0.95),
        "market_env": CoverageContract("date_coverage", 0.95),
    }


def test_headline_v1_vintage_policy_is_revision_safe():
    assert _headline().vintage_policy == "revision-safe"


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
    """The minimum_coverage() helper stays a dict[str, float] — it projects each
    contract's threshold, so callers that only need thresholds keep working."""
    assert minimum_coverage("headline_v1") == {
        ch: c.threshold for ch, c in _headline().coverage_contracts.items()}


def test_minimum_coverage_empty_without_profile():
    for name in (None, "", "none", "bogus"):
        assert minimum_coverage(name) == {}, name


def test_profile_for_known_profile():
    assert profile_for("headline_v1") is _headline()


def test_profile_for_none_or_unknown_is_none():
    for name in (None, "", "none", "bogus"):
        assert profile_for(name) is None, name
