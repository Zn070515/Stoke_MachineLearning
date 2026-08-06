"""Frozen feature profiles + per-channel column ownership (§十四).

Two responsibilities, both consumed by the v15 audit-fix plan:

1. ``CHANNEL_COLUMNS`` — the exact output-column set of every data channel,
   the single source of truth for the generic prebuilt scrub in
   ``stoke_ml.features.panel_builder``.  A prebuilt parquet built with
   all-True defaults may carry a channel whose ``use_*`` switch is OFF in the
   consuming run (safe-only vintage / ablation); the scrub drops exactly that
   channel's columns using ONLY these exact sets — never name-prefix matching,
   which is the market_env-vs-macd / market_env-refine collision trap.

2. ``FeatureProfile`` / ``FEATURE_PROFILES`` — frozen, named feature recipes
   declaring which channels a formal run REQUIRES and each required channel's
   minimum coverage, plus the vintage policy the profile is validated against.
   ``headline_v1`` is the formal baseline: the safe-only-ALLOWED, default-ON
   channels with tunable minimum-coverage thresholds.  The coverage gate in
   ``scripts/production/train_panel.py`` enforces these thresholds per run.

The values are imported from the existing channel column constants (the single
source of truth) rather than hand-copied, so a column rename in the feature
layer auto-propagates here.  ``topic`` is deliberately NOT in ``CHANNEL_COLUMNS``:
its ``topic_*`` columns are already dropped by the prefix-based
``FeaturePipeline._drop_topic_columns`` (a frozen global topic model's non-PIT
representation), and a generic prefix drop would be redundant there.
``date`` / ``stock_code`` and the OHLCV/technical columns belong to the
always-allowed ``daily_qfq`` price channel and are excluded by design.
"""
from __future__ import annotations

from dataclasses import dataclass

from stoke_ml.features import aux_cols as _aux
from stoke_ml.features import panel_helpers as _ph
from stoke_ml.features.market_env import (
    _COMPOSITE as _MENV_COMPOSITE,
    _FACTOR_RAW as _MENV_FACTOR_RAW,
    _FACTOR_Z as _MENV_FACTOR_Z,
)

# ── CHANNEL_COLUMNS ──────────────────────────────────────────────────────

# fundamental_refine: the exact output-column set written by
# stoke_ml/features/fundamental.py — the per-stock FundamentalRefiner outputs
# (f_score / quality_composite / earnings_quality / growth_quality /
# profitability_stability / margin_stability / pe_percentile_252d /
# pb_percentile_252d / pe_pb_divergence / deep_value / *_trend_4q / roe_accel /
# earnings_surprise) plus the cross-sectional add_cross_sectional outputs
# (*_sector_ratio / leverage_warning / valuation_composite_z).  Enumerated from
# the module's ``df["X"] = ...`` assignments (no reusable constant exists
# there), so it stays in sync with the refiner by construction of the scrub.
_FUNDAMENTAL_REFINE_COLS = frozenset({
    "f_score", "quality_composite", "earnings_quality", "growth_quality",
    "profitability_stability", "margin_stability",
    "pe_percentile_252d", "pb_percentile_252d",
    "pe_pb_divergence", "deep_value",
    "roe_trend_4q", "revenue_trend_4q", "margin_trend_4q",
    "roe_accel", "earnings_surprise",
    "pe_sector_ratio", "pb_sector_ratio", "ps_sector_ratio",
    "leverage_warning", "valuation_composite_z",
})

# market_env_refine: the exact ``menv_*`` output set of
# stoke_ml/features/market_env.py — derived from the module's own factor
# tables + the assembled regime score so a factor rename auto-propagates.
_MARKET_ENV_REFINE_COLS = frozenset(
    set(_MENV_FACTOR_Z) | set(_MENV_FACTOR_RAW) | set(_MENV_COMPOSITE)
    | {"menv_regime_z"}
)

CHANNEL_COLUMNS: dict[str, frozenset[str]] = {
    "sentiment": frozenset(_aux.SENTIMENT_COLS),
    "guba": frozenset(_aux.GUBA_COLS),
    "comment": frozenset(_aux.COMMENT_COLS),
    "announcement": frozenset(_ph.ANNOUNCEMENT_COLS),
    "margin": frozenset(_aux.MARGIN_COLS),
    "northbound": frozenset(_aux.NORTHBOUND_COLS),
    "dragon_tiger": (
        frozenset(_aux.DRAGON_TIGER_COLS) | frozenset(_ph.DRAGON_TIGER_SEAT_COLS)
    ),
    "fundamental": frozenset(_aux.FUNDAMENTAL_COLS),
    "fundamental_refine": _FUNDAMENTAL_REFINE_COLS,
    "earnings": frozenset(_aux.EARNINGS_COLS),
    "valuation": frozenset(_aux.VALUATION_COLS),
    "etf_flow": frozenset(_aux.ETF_FLOW_COLS),
    "capital_flow": frozenset(_ph.FLOW_COLS),
    "block_trade": frozenset(_ph.BLOCK_TRADE_COLS),
    "shareholder": frozenset(_ph.SHAREHOLDER_COLS),
    "lockup": frozenset(_ph.LOCKUP_COLS),
    "dividend": frozenset(_ph.DIVIDEND_COLS),
    "board": frozenset(_ph.BOARD_COLS) | frozenset(_ph._BOARD_ONEHOT_COLS),
    "sector": frozenset(_ph.SECTOR_COLS),
    "concept": frozenset(_ph.CONCEPT_COLS),
    "industry": frozenset(_aux.INDUSTRY_COLS),
    "macro": frozenset(_aux.MACRO_COLS),
    "pledge": frozenset(_ph.PLEDGE_COLS),
    "index_membership": frozenset(_ph.INDEX_MEMBER_COLS),
    "market_env": frozenset(_aux.MARKET_ENV_COLS),
    "market_env_refine": _MARKET_ENV_REFINE_COLS,
    "limit_up": frozenset(_ph.LIMIT_UP_COLS),
}

# Overlap invariant: every column appears in AT MOST ONE channel set.  This is
# what makes the generic scrub safe — a column is dropped for exactly one
# channel's switch, never for the wrong one.  The market_env bare names vs the
# menv_* refine names are disjoint by construction; this assert proves the
# whole map (build a column→channel owner and assert no duplicate claims).
_COLUMN_OWNER: dict[str, str] = {}
for _channel, _cols in CHANNEL_COLUMNS.items():
    for _col in _cols:
        assert _col not in _COLUMN_OWNER, (
            f"feature_profile: column {_col!r} is claimed by BOTH channel "
            f"{_COLUMN_OWNER[_col]!r} and {_channel!r} — the CHANNEL_COLUMNS "
            "overlap invariant is violated (a column must belong to ≤1 "
            "channel for the generic prebuilt scrub to be safe)"
        )
        _COLUMN_OWNER[_col] = _channel
del _channel, _cols, _col, _COLUMN_OWNER


# ── FeatureProfile ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureProfile:
    """A frozen feature recipe: required channels + minimum coverage.

    Attributes:
        name:            stable profile identifier (CLI ``--feature-profile``).
        required_channels: channels a formal run must not silently lose — a
            required channel with zero coverage aborts the run.
        minimum_coverage: channel → minimum fraction of loaded stocks/grid
            cells that must carry the channel; a PROBEABLE channel below its
            minimum aborts the run.  A channel absent from this dict is
            presence-only (must exceed zero).
        vintage_policy:   the VintagePolicy the profile is validated against
            (headline_v1 is the ``safe-only`` baseline).
    """

    name: str
    required_channels: tuple[str, ...]
    minimum_coverage: dict[str, float]
    vintage_policy: str


# The formal baseline profile (v15 §十四).  ``required_channels`` is exactly
# the safe-only-ALLOWED (raw_vintage_safe + derived_versioned) channels that
# default ON in FeaturePipeline — i.e. it excludes the 9
# latest_revised_aligned denied channels (fundamental/macro/earnings/valuation/
# pledge/shareholder/index_membership/market_env_refine/sector/concept) and the
# default-off board/sector/concept/limit_up/topic dimensions.  ``macro`` is
# denied under safe-only, so it is deliberately absent despite defaulting ON.
# ``minimum_coverage`` thresholds are the formal baseline; each is tunable per
# experiment and a channel absent from the dict is presence-only.
FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "headline_v1": FeatureProfile(
        name="headline_v1",
        required_channels=(
            "sentiment", "guba", "comment", "announcement", "margin",
            "northbound", "dragon_tiger", "capital_flow", "etf_flow",
            "block_trade", "lockup", "dividend", "industry", "market_env",
        ),
        minimum_coverage={
            "sentiment": 0.90, "guba": 0.90, "comment": 0.90,
            "announcement": 0.70, "margin": 0.95, "northbound": 0.90,
            "capital_flow": 0.90, "etf_flow": 0.80, "block_trade": 0.30,
            "lockup": 0.30, "dividend": 0.30, "industry": 0.95,
            "market_env": 0.95,
        },
        vintage_policy="safe-only",
    ),
}


def profile_for(profile_name: str | None) -> FeatureProfile | None:
    """The profile for ``profile_name``, or None when it is not configured.

    ``None`` / ``""`` / ``"none"`` (the gate-off sentinel) return None.
    """
    if profile_name in (None, "", "none"):
        return None
    return FEATURE_PROFILES.get(profile_name)


def resolve_required_channels(
    profile_name: str | None, extra: set[str] | None,
) -> set[str]:
    """The required-channel set a run must enforce (§十四).

    ``extra`` (the explicit ``--require-aux-channels`` set) always counts; the
    named profile's ``required_channels`` are UNIONED in when the profile is
    active (not ``None`` / ``""`` / ``"none"``).  An unknown profile name
    contributes nothing (the caller's gate decides whether that is an error).
    """
    extra = set(extra or ())
    prof = profile_for(profile_name)
    if prof is None:
        return extra
    return set(prof.required_channels) | extra


def minimum_coverage(profile_name: str | None) -> dict[str, float]:
    """The minimum-coverage thresholds for ``profile_name``; {} when none."""
    prof = profile_for(profile_name)
    return dict(prof.minimum_coverage) if prof is not None else {}
