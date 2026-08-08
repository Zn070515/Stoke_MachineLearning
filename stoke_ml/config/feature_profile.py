"""Frozen feature profiles + per-channel column ownership (§十四).

Two responsibilities, both consumed by the v15 audit-fix plan:

1. ``CHANNEL_COLUMNS`` — the exact output-column set of every data channel,
   the single source of truth for the generic prebuilt scrub in
   ``stoke_ml.features.panel_builder``.  A prebuilt parquet built with
   all-True defaults may carry a channel whose ``use_*`` switch is OFF in the
   consuming run (revision-safe vintage / ablation); the scrub drops exactly
   that channel's columns using ONLY these exact sets — never name-prefix
   matching,
   which is the market_env-vs-macd / market_env-refine collision trap.

2. ``FeatureProfile`` / ``FEATURE_PROFILES`` — frozen, named feature recipes
   declaring which channels a formal run REQUIRES and each required channel's
   coverage CONTRACT (the metric measured + the minimum), plus the vintage
   policy the profile is validated against.  ``headline_v1`` is the formal
   baseline: the revision-safe-ALLOWED, default-ON channels with tunable
   per-channel contracts.  The coverage gate in ``scripts/production/train_panel.py``
   enforces these contracts per run, measuring the declared metric (stock-level
   on the live path, cell-level on the prebuilt probe).

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


# ── market_env price/account part split (§八/T5) ──────────────────────────
# The broadcast ``a_shares/market_breadth/market_env_daily.parquet`` carries TWO
# time-attribution kinds in ONE consumer-facing file (backward compat — the
# feature layer / broadcast probe / formal manifest gate all read that single
# file).  The PRICE part (same-day trade data) is ``verified``-PIT and is the
# revision-safe REQUIRED sub-set.  The ACCOUNT part (monthly investor/mkt-cap
# stats) has a real publish date that the shipped account_stats.parquet does NOT
# record (only the data-month label), so its alignment is declared PROXY and it
# is CONSUMED ONLY via an explicit ablation opt-in (``use_market_env_account``,
# default OFF — mirroring ``use_topic``); the moment the builder upgrades the
# account part to ``verified`` (real publish date recorded), it joins the
# required/verified set automatically and the ablation flag stops mattering.
# The single file's manifest carries the STRICTER of the two labels
# (``vintage_pit="proxy"`` via channel_vintage); these constants are the
# per-part declaration that keeps the price part required while the account part
# stays honest.
#
# ENFORCEMENT wiring (these constants are NOT decorative — they are the single
# source of truth for the real consumption path):
#   * ``CHANNEL_COLUMNS["market_env"]`` is the PRICE-ONLY set → the generic
#     prebuilt scrub in panel_builder treats the market_env channel atomically
#     as the verified price subset (always kept, ``use_market_env`` default ON).
#   * ``CHANNEL_COLUMNS["market_env_account"]`` is the ACCOUNT set → the same
#     scrub drops it whenever ``use_market_env_account`` is OFF (the default),
#     and ``aux_aligner._merge_market_env`` merges it ONLY when the ablation
#     flag is ON or the account part is verified.
MARKET_ENV_PRICE_COLS: frozenset[str] = frozenset({
    "high_low_ratio", "market_adv_ratio", "market_turnover_z",
})
MARKET_ENV_ACCOUNT_COLS: frozenset[str] = frozenset({
    "mkt_cap_total_z", "avg_account_cap_z", "investor_new_num", "investor_new_z",
})
#: Honest PIT label for the account part: ``"proxy"`` while the raw source
#: records no real publish date (the shipped account_stats.parquet does not).
#: A builder that finds a real publish date upgrades to ``"verified"`` — the
#: channel_vintage ``market_env`` entry must follow the same upgrade (both stay
#: the STRICTER-of-the-two label).
MARKET_ENV_ACCOUNT_PIT: str = "proxy"


def market_env_account_is_verified() -> bool:
    """Whether the account part is declared ``verified`` (real publish date).

    Read as a FUNCTION (not a bare constant) so the builders/consumers that
    import it observe a future upgrade of ``MARKET_ENV_ACCOUNT_PIT`` at CALL
    time — a test (or a future builder upgrade) that flips the module global
    takes effect everywhere without re-importing.
    """
    return MARKET_ENV_ACCOUNT_PIT == "verified"


def market_env_required_columns(profile_name: str | None = None) -> frozenset[str]:
    """The market_env COLUMNS a run REQUIRES: the verified PRICE part, plus the
    ACCOUNT part once it is declared ``verified``.

    A required sub-set may never include a channel part whose PIT is unverified
    — while ``MARKET_ENV_ACCOUNT_PIT == "proxy"`` the account part is excluded
    (ablation-only, mirroring ``use_topic``); only a verified account part would
    be admitted back into the required sub-set.  ``profile_name`` is honored:
    a profile that does NOT require the ``market_env`` channel requires none of
    its columns (the price part is required only by a profile that declares the
    channel).  ``None`` (the live-merge default) means "the default/verified
    state" — the price part, plus the account part once verified.
    """
    prof = profile_for(profile_name)
    if prof is not None and "market_env" not in prof.required_channels:
        return frozenset()
    cols = set(MARKET_ENV_PRICE_COLS)
    if market_env_account_is_verified():
        cols |= set(MARKET_ENV_ACCOUNT_COLS)
    return frozenset(cols)


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
    # §T5: the market_env channel is the PRICE-ONLY (verified) part; the ACCOUNT
    # part is its own ablation-only channel so the generic scrub can drop it
    # whenever use_market_env_account is OFF (the default).  See the split block.
    "market_env": MARKET_ENV_PRICE_COLS,
    "market_env_account": MARKET_ENV_ACCOUNT_COLS,
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

# The coverage METRICS a channel contract can declare (§T4, the semantic
# unification fix).  STOCK coverage is the fraction of loaded stocks carrying
# the channel (the live path's per-stock load); CELL coverage is the fraction
# of (stock, day) grid cells with the channel's has_* flag set (the prebuilt
# probe); DATE coverage is the fraction of trading days in the window covered
# (the meaningful metric for MARKET-WIDE broadcast channels, whose value is the
# same for every stock per date); valid_state_cell_coverage is reserved for a
# state-channel forward-fill metric (not currently produced).
#
# §T8: ERA coverage is the provider-era retrieval coverage — the calendar-day
# fraction of a stock's provider-observable window that was actually retrieved,
# per the gold asset manifest's provider-era fields.  It is the metric for the
# SPARSE text event channels (sentiment / guba): it distinguishes a stock that
# genuinely had no events (no_event, legitimate) from an era we never observed
# (not_observed, a data gap).  §v18-3: ``era_observable_stock_fraction`` is the
# fraction of the REQUESTED universe that is era-observable (n_obs / len(stock_list))
# — the composite half of the era-coverage contract, so 10 era-observable stocks
# can never represent a 500-stock requested universe.  ``date_availability`` is
# reserved for a future date-presence metric (not currently produced).
_COVERAGE_METRICS: frozenset[str] = frozenset({
    "stock_coverage", "cell_coverage", "date_coverage",
    "valid_state_cell_coverage", "era_coverage", "era_observable_stock_fraction",
    "date_availability",
})


@dataclass(frozen=True)
class CoverageContract:
    """Per-channel coverage contract: which metric is measured and the minimum.

    The gate measures the DECLARED metric — stock-level on the live path,
    cell-level on the prebuilt probe, date-level for market-wide broadcasts —
    so the same threshold means the same thing regardless of path (§T4, §十一).

    ``requires`` is an OPTIONAL COMPOSITE requirement (§v18-3): a second metric
    that must ALSO clear its own threshold.  Used by the era-coverage contract
    — the provider-era retrieval coverage AND the fraction of the requested
    universe that is era-observable must BOTH pass, so 10/500 observable stocks
    can never represent the whole universe.
    """

    metric: str
    threshold: float
    requires: tuple[str, float] | None = None

    def __post_init__(self):
        if self.metric not in _COVERAGE_METRICS:
            raise ValueError(
                f"CoverageContract metric {self.metric!r} not in "
                f"{sorted(_COVERAGE_METRICS)}")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(
                f"CoverageContract threshold must be in (0,1], got "
                f"{self.threshold}")
        if self.requires is not None:
            req_metric, req_threshold = self.requires
            if req_metric not in _COVERAGE_METRICS:
                raise ValueError(
                    f"CoverageContract requires metric {req_metric!r} not in "
                    f"{sorted(_COVERAGE_METRICS)}")
            if not (0.0 < req_threshold <= 1.0):
                raise ValueError(
                    f"CoverageContract requires threshold must be in (0,1], "
                    f"got {req_threshold}")


@dataclass(frozen=True)
class FeatureProfile:
    """A frozen feature recipe: required channels + per-channel coverage contracts.

    Attributes:
        name:            stable profile identifier (CLI ``--feature-profile``).
        required_channels: channels a formal run must not silently lose — a
            required channel with zero coverage aborts the run.
        coverage_contracts: channel → CoverageContract (the metric measured and
            the minimum).  A PROBEABLE channel below its contract minimum aborts
            the run.  A channel absent from this dict is presence-only (must
            exceed zero) — dragon_tiger is the shipped profile's presence-only
            channel.
        vintage_policy:   the VintagePolicy the profile is validated against
            (headline_v1 is the ``revision-safe`` baseline).
    """

    name: str
    required_channels: tuple[str, ...]
    coverage_contracts: dict[str, CoverageContract]
    vintage_policy: str


# The formal baseline profile (v15 §十四, v16 §十二).  ``required_channels`` is
# exactly the revision-safe-ALLOWED (immutable_snapshot-sourced) channels that
# default ON in FeaturePipeline — i.e. it excludes the 10
# latest_revised-sourced denied channels (fundamental/macro/earnings/valuation/
# pledge/shareholder/index_membership/market_env_refine/sector/concept) and the
# default-off board/sector/concept/limit_up/topic dimensions.  ``macro`` is
# denied under revision-safe, so it is deliberately absent despite defaulting ON.
#
# ``coverage_contracts`` thresholds ARE the historical ``minimum_coverage``
# values — NOT re-tuned (§T4).  The metric split: the SPARSE text event channels
# (sentiment / guba) declare ``era_coverage`` (§T8) — the provider-era retrieval
# coverage that separates a stock with genuinely no events (no_event) from an
# era we never observed (not_observed), so a sparse-but-observed channel is not
# falsely gated as a data gap; every other per-stock channel declares
# ``stock_coverage``; the MARKET-WIDE broadcast channels
# (etf_flow / industry / market_env) declare ``date_coverage`` — their value is
# the same for every stock per date, so stock coverage is vacuous (1.0 whenever
# the file exists) and date coverage is the meaningful metric.  dragon_tiger is
# deliberately absent from the map (presence-only), preserving the convention
# that a required channel absent from the contract map is presence-only.
#
# §T5 market_env split: ``market_env`` stays required + ``date_coverage 0.95``
# because the file it gates carries the VERIFIED price part (high/low breadth,
# market turnover, industry advance) — see ``MARKET_ENV_PRICE_COLS``.  The
# ACCOUNT part (``MARKET_ENV_ACCOUNT_COLS``) is PROXY-PIT while
# ``MARKET_ENV_ACCOUNT_PIT == "proxy"`` and is excluded from the required
# sub-set (``market_env_required_columns`` returns the price part only).  It
# rides along in the same file as an ablation-only sub-part (consumed only when
# ``use_market_env_account`` is ON, or once the account part is verified),
# never required.
#
# §v18-9 (Lockbox declaration): ``headline_v1`` is the REVISION-SAFE formal
# baseline — it is NOT headline-strict.  ``industry`` remains required yet its
# historical sector classification is source_vintage=immutable_snapshot,
# transform=formula_versioned, pit_alignment=proxy (a reconstruction / proxy of
# today's sector map over history).  A formal headline_v1 conclusion therefore
# carries an industry PROXY in its feature set; the Lockbox final result may
# either adopt a future headline_strict_v1 (dropping industry) or keep
# headline_v1 and declare the proxy explicitly in the research writeup.
FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "headline_v1": FeatureProfile(
        name="headline_v1",
        required_channels=(
            "sentiment", "guba", "comment", "announcement", "margin",
            "northbound", "dragon_tiger", "capital_flow", "etf_flow",
            "block_trade", "lockup", "dividend", "industry", "market_env",
        ),
        coverage_contracts={
            "sentiment": CoverageContract(
                "era_coverage", 0.90,
                requires=("era_observable_stock_fraction", 0.90)),
            "guba": CoverageContract(
                "era_coverage", 0.90,
                requires=("era_observable_stock_fraction", 0.90)),
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
        },
        vintage_policy="revision-safe",
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
    """The minimum-coverage THRESHOLDS for ``profile_name``; {} when none.

    A projection of each channel's ``CoverageContract.threshold``, kept as a
    ``dict[str, float]`` so callers that only need the threshold (not the
    declared metric) keep working.  The gate itself reads the full contracts
    via ``FeatureProfile.coverage_contracts``.
    """
    prof = profile_for(profile_name)
    if prof is None:
        return {}
    return {ch: c.threshold for ch, c in prof.coverage_contracts.items()}
