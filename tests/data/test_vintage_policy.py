"""§T2 / §T7 / v15 §六/§十, v16 §十二 vintage-admission policy tests.

``stoke_ml.data.vintage_policy`` decides which channels a training run may
consume from their declared 3-dim vintage classification (source_vintage /
transform / pit_alignment).  Admission is SOURCE-based with a BOTH-LAYERS
check: an undeclared channel or one whose source/transform is the reserved
"unknown" fallback is denied under every policy.  §T3 adds a third tier:
``HEADLINE_STRICT`` additionally gates on ``pit_alignment == "verified"``,
with an explicit scale-invariant WAIVER whitelist for channels whose proxy
alignment does not bias the features a model consumes (QFQ returns /
market-breadth ratios).  The policy matrix is exercised over the REAL curated
declaration; the report's enforcement branches are proven with CRAFTED
declarations passed via the test-injection parameters (no module globals are
mutated).
"""
import dataclasses

import pytest

from stoke_ml.data.channel_vintage import (
    CHANNEL_VINTAGE,
    ChannelVintageStatus,
)
from stoke_ml.data.vintage_policy import (
    CSI_MONTHLY_RECONSTRUCTED,
    CSI_UNIVERSE_NAMES,
    HEADLINE_STRICT_WAIVER_CHANNELS,
    UniverseVintagePolicy,
    VintagePolicy,
    allowed_channels,
    channel_allowed,
    denied_channels,
    universe_membership_provenance,
    vintage_report,
)

REVISION_SAFE = VintagePolicy.REVISION_SAFE
ALLOW_REVISED = VintagePolicy.ALLOW_REVISED
HEADLINE_STRICT = VintagePolicy.HEADLINE_STRICT


def _declared_sets():
    immutable = {e.channel for e in CHANNEL_VINTAGE
                 if e.source_vintage == "immutable_snapshot"}
    revised = {e.channel for e in CHANNEL_VINTAGE
               if e.source_vintage == "latest_revised"}
    return immutable, revised


def _immutable_verified():
    """The immutable_snapshot-sourced channels whose pit_alignment is
    ``"verified"`` — the headline-strict base admission (before waivers)."""
    return {e.channel for e in CHANNEL_VINTAGE
            if e.source_vintage == "immutable_snapshot"
            and e.pit_alignment == "verified"}


def test_revision_safe_allows_immutable_denies_revised():
    """REVISION_SAFE: every immutable_snapshot-sourced channel is True, every
    latest_revised-sourced channel is False."""
    immutable, revised = _declared_sets()
    assert immutable and revised
    for e in CHANNEL_VINTAGE:
        if e.channel in immutable:
            assert channel_allowed(e.channel, REVISION_SAFE) is True, e.channel
        else:
            assert channel_allowed(e.channel, REVISION_SAFE) is False, e.channel


def test_allow_revised_allows_every_declared_status():
    """ALLOW_REVISED: every declared channel (any declared source_vintage) is
    admissible."""
    for e in CHANNEL_VINTAGE:
        assert channel_allowed(e.channel, ALLOW_REVISED) is True, e.channel


def test_undeclared_channel_denied_under_all_policies():
    """An undeclared channel (no declaration → the reserved "unknown" fallback)
    is the mandatory deny under EVERY policy — a consumer can never silently
    consume an uncurated channel."""
    assert channel_allowed("undeclared_channel_x", REVISION_SAFE) is False
    assert channel_allowed("undeclared_channel_x", ALLOW_REVISED) is False
    assert channel_allowed("undeclared_channel_x", HEADLINE_STRICT) is False


def test_allowed_channels_revision_safe_is_immutable_union():
    immutable, _ = _declared_sets()
    assert allowed_channels(REVISION_SAFE) == immutable


def test_allowed_channels_allow_revised_is_all_declared():
    assert allowed_channels(ALLOW_REVISED) == {e.channel for e in CHANNEL_VINTAGE}


def test_denied_channels_revision_safe_is_declared_revised_set():
    _, revised = _declared_sets()
    denied = denied_channels(REVISION_SAFE)
    assert denied == revised
    assert len(denied) == 10  # the 10 declared latest_revised-sourced channels


def test_denied_channels_allow_revised_is_empty():
    assert denied_channels(ALLOW_REVISED) == frozenset()


def test_price_channel_invariant_revision_safe():
    """daily_qfq must always be admissible under REVISION_SAFE
    (immutable_snapshot source) — a model cannot train without the price
    channel."""
    assert channel_allowed("daily_qfq", REVISION_SAFE) is True


def test_vintage_report_with_real_declaration():
    """The real declaration: no missing documented dims, price channel allowed,
    complete 3-dim declaration, every entry carries the 6 keys, and the policy
    string is the enum value."""
    report = vintage_report(REVISION_SAFE)
    assert report["vintage_policy"] == "revision-safe"
    assert report["missing_channels"] == []
    assert report["daily_qfq_allowed"] is True
    assert report["declaration_complete"] is True
    for entry in report["channels"]:
        assert set(entry) == {"channel", "source_vintage", "transform",
                              "pit_alignment", "rationale", "allowed"}
        assert isinstance(entry["allowed"], bool)


def test_vintage_report_crafted_partial_declaration_reports_missing():
    """A CRAFTED declaration that drops a documented channel makes
    missing_channels name it — the proof a formal run FAILs an incomplete
    declaration without touching module globals."""
    partial = tuple(e for e in CHANNEL_VINTAGE if e.channel != "fundamental")
    report = vintage_report(REVISION_SAFE, declaration=partial)
    assert "fundamental" in report["missing_channels"]
    assert len(report["channels"]) == len(CHANNEL_VINTAGE) - 1


def test_vintage_report_crafted_daily_qfq_denied_under_revision_safe():
    """A CRAFTED declaration labeling daily_qfq as latest_revised source makes
    daily_qfq_allowed False under REVISION_SAFE — proving the formal FAIL
    branch for a price-channel-denying policy."""
    crafted = tuple(
        dataclasses.replace(e, source_vintage="latest_revised")
        if e.channel == "daily_qfq" else e
        for e in CHANNEL_VINTAGE
    )
    report = vintage_report(REVISION_SAFE, declaration=crafted)
    assert report["daily_qfq_allowed"] is False


def test_vintage_report_declaration_complete_false_on_unknown_dim():
    """§T7: a crafted declaration where one channel declares transform="unknown"
    is NOT complete — and that channel is denied under EVERY policy by the
    both-layers admission check."""
    crafted = tuple(
        dataclasses.replace(e, transform="unknown")
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    report = vintage_report(REVISION_SAFE, declaration=crafted)
    assert report["declaration_complete"] is False
    assert channel_allowed("sentiment", REVISION_SAFE,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", ALLOW_REVISED,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", HEADLINE_STRICT,
                           vintage_by_name=crafted_by_name) is False


def test_vintage_report_declaration_complete_false_on_out_of_vocab_source():
    """§T7: a crafted declaration where one channel declares an OUT-OF-
    VOCABULARY source_vintage (e.g. a typo "immutable_snapshoot") is NOT
    complete — and that channel is denied under EVERY policy by the
    vocabulary-guarded admission check."""
    crafted = tuple(
        dataclasses.replace(e, source_vintage="immutable_snapshoot")
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    report = vintage_report(REVISION_SAFE, declaration=crafted)
    assert report["declaration_complete"] is False
    assert channel_allowed("sentiment", REVISION_SAFE,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", ALLOW_REVISED,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", HEADLINE_STRICT,
                           vintage_by_name=crafted_by_name) is False


# ── §T3: three-tier enum + headline-strict + waiver whitelist ───────────

def test_enum_has_exactly_three_tiers_with_canonical_values():
    """T3: the policy enum has exactly three tiers with the canonical
    serialized values — ``headline-strict`` is the NEW strictest tier."""
    assert [p.value for p in VintagePolicy] == [
        "revision-safe", "allow-revised", "headline-strict",
    ]
    assert VintagePolicy.REVISION_SAFE.value == "revision-safe"
    assert VintagePolicy.ALLOW_REVISED.value == "allow-revised"
    assert VintagePolicy.HEADLINE_STRICT.value == "headline-strict"


def test_legacy_safe_only_alias_coerces_to_revision_safe():
    """T3 backward compat: the old stored "safe-only" string parses to
    REVISION_SAFE (the pre-T3 name of the SAME tier), so legacy
    store/experiment records are not all rejected by strict reads."""
    assert VintagePolicy("safe-only") is VintagePolicy.REVISION_SAFE
    assert VintagePolicy("revision-safe") is VintagePolicy.REVISION_SAFE
    # The admission helpers coerce the alias too (same tier semantics).
    assert channel_allowed("fundamental", "safe-only") is False
    assert channel_allowed("daily_qfq", "safe-only") is True


def test_headline_strict_denies_proxy_unless_waived():
    """T3: a channel with source != latest_revised but pit_alignment="proxy"
    is denied under headline-strict UNLESS it is on the waiver whitelist.
    capital_flow (verified) passes; daily_qfq (proxy) passes ONLY via its
    waiver; industry (proxy, NOT waived) is denied under headline-strict but
    admitted under revision-safe (source layer only)."""
    assert channel_allowed("capital_flow", HEADLINE_STRICT) is True   # verified
    assert "daily_qfq" in HEADLINE_STRICT_WAIVER_CHANNELS
    assert channel_allowed("daily_qfq", HEADLINE_STRICT) is True      # proxy, waived
    assert "market_env" in HEADLINE_STRICT_WAIVER_CHANNELS
    assert channel_allowed("market_env", HEADLINE_STRICT) is True     # proxy, waived
    assert "industry" not in HEADLINE_STRICT_WAIVER_CHANNELS
    assert channel_allowed("industry", HEADLINE_STRICT) is False      # proxy, not waived
    assert channel_allowed("industry", REVISION_SAFE) is True


def test_headline_strict_admission_matrix_over_real_declaration():
    """T3: over the REAL curated declaration, headline-strict admits exactly
    the immutable_snapshot-sourced channels with pit_alignment="verified" PLUS
    the scale-invariant waiver whitelist; everything else (all
    latest_revised-sourced channels and any proxy channel not waived) is
    denied."""
    expected = _immutable_verified() | set(HEADLINE_STRICT_WAIVER_CHANNELS)
    assert expected <= {e.channel for e in CHANNEL_VINTAGE}
    for c in expected:
        assert channel_allowed(c, HEADLINE_STRICT) is True, c
    for e in CHANNEL_VINTAGE:
        if e.channel not in expected:
            assert channel_allowed(e.channel, HEADLINE_STRICT) is False, e.channel


def test_allowed_channels_headline_strict_is_verified_union_waiver():
    assert allowed_channels(HEADLINE_STRICT) == (
        _immutable_verified() | HEADLINE_STRICT_WAIVER_CHANNELS)


def test_denied_channels_headline_strict_exact_set():
    """T3: headline-strict denies the 10 latest_revised-sourced channels PLUS
    the non-waived proxy-aligned channels (industry, and the §T5 market_env
    ACCOUNT sub-part whose absolute-count features are NOT scale-invariant);
    daily_qfq / market_env are scale-invariant and waived (NOT denied)."""
    _, revised = _declared_sets()
    assert denied_channels(HEADLINE_STRICT) == (
        revised | {"industry", "market_env_account"})
    assert "daily_qfq" not in denied_channels(HEADLINE_STRICT)
    assert "market_env" not in denied_channels(HEADLINE_STRICT)


def test_headline_strict_denies_transform_unknown():
    """T3: a channel whose transform is the reserved "unknown" fallback is
    denied under headline-strict (mandatory deny-by-default) — and under
    revision-safe/allow-revised too, since the both-layers check is unchanged."""
    crafted = tuple(
        dataclasses.replace(e, transform="unknown")
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    assert channel_allowed("sentiment", HEADLINE_STRICT,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", REVISION_SAFE,
                           vintage_by_name=crafted_by_name) is False


def test_headline_strict_denies_latest_revised_even_when_verified():
    """T3: the source gate binds first — a latest_revised-sourced channel is
    denied under headline-strict EVEN if its pit_alignment were "verified"."""
    crafted = tuple(
        dataclasses.replace(e, pit_alignment="verified")
        if e.channel == "fundamental" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    assert channel_allowed("fundamental", HEADLINE_STRICT,
                           vintage_by_name=crafted_by_name) is False


def test_headline_strict_denies_out_of_vocab_pit_alignment():
    """T3: under headline-strict an OUT-OF-VOCABULARY pit_alignment (not
    "verified") is denied unless waived — deny-by-default on the alignment
    axis, mirroring the source/transform vocabulary guards."""
    crafted = tuple(
        dataclasses.replace(e, pit_alignment="verifiedd")  # typo
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    assert channel_allowed("sentiment", HEADLINE_STRICT,
                           vintage_by_name=crafted_by_name) is False


def test_headline_strict_waiver_whitelist_is_frozen_constant():
    """T3: the waiver whitelist is a frozen, declared-subset constant — a
    scale-invariant waiver is a deliberate reviewed decision, not a runtime
    tweak, and every waived channel must actually be declared."""
    assert isinstance(HEADLINE_STRICT_WAIVER_CHANNELS, frozenset)
    assert HEADLINE_STRICT_WAIVER_CHANNELS <= {e.channel for e in CHANNEL_VINTAGE}
    assert "industry" not in HEADLINE_STRICT_WAIVER_CHANNELS


def test_vintage_report_headline_strict_serializes_value_and_price_waiver():
    """T3: vintage_report serializes the headline-strict policy value and
    reports the price channel allowed via its waiver."""
    report = vintage_report(HEADLINE_STRICT)
    assert report["vintage_policy"] == "headline-strict"
    assert report["daily_qfq_allowed"] is True
    assert report["missing_channels"] == []
    assert report["declaration_complete"] is True


# ── §T6 / §十四: universe-membership provenance ──────────────────────────

_UM_DICT = {
    "source": "Baostock monthly reconstruction",
    "vintage": "latest-reconstructed",
    "resolution": "monthly",
}


def test_csi_monthly_reconstructed_provenance():
    """§T6 (§十四): the CSI universe-membership provenance is declared — a CSI
    universe gate consumes membership.parquet which is Baostock-MONTHLY-
    RECONSTRUCTED (NOT official effective-date data), so even a feature-vintage
    revision-safe run is NOT free of latest-reconstructed data in its universe
    gate; the provenance is declared EXPLICITLY, never implied."""
    assert CSI_MONTHLY_RECONSTRUCTED.provenance() == _UM_DICT


def test_csi_universe_names_is_the_strict_csi_set():
    assert CSI_UNIVERSE_NAMES == frozenset({"csi300", "csi500", "csi800"})


def test_universe_membership_provenance_csi_universes():
    """Every strict-CSI universe resolves to the same declared provenance."""
    assert universe_membership_provenance("csi300") == _UM_DICT
    assert universe_membership_provenance("csi500") == _UM_DICT
    assert universe_membership_provenance("csi800") == _UM_DICT


def test_universe_membership_provenance_non_csi_returns_none():
    """Non-CSI universes do not consume membership.parquet — no provenance."""
    for u in ("all", "random", "first", "stratified", "", None):
        assert universe_membership_provenance(u) is None, u


def test_universe_vintage_policy_is_frozen():
    """The provenance policy is immutable — a frozen dataclass, so no caller
    can mutate the declared membership vintage after construction."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CSI_MONTHLY_RECONSTRUCTED.source = "other"
