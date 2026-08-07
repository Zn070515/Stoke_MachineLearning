"""§T2 / §T7 / v15 §六/§十, v16 §十二 vintage-admission policy tests.

``stoke_ml.data.vintage_policy`` decides which channels a training run may
consume from their declared 3-dim vintage classification (source_vintage /
transform / pit_alignment).  Admission is SOURCE-based with a BOTH-LAYERS
check: an undeclared channel or one whose source/transform is the reserved
"unknown" fallback is denied under both policies.  The policy matrix is
exercised over the REAL curated declaration; the report's enforcement branches
are proven with CRAFTED declarations passed via the test-injection parameters
(no module globals are mutated).
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
    UniverseVintagePolicy,
    VintagePolicy,
    allowed_channels,
    channel_allowed,
    denied_channels,
    universe_membership_provenance,
    vintage_report,
)

SAFE_ONLY = VintagePolicy.SAFE_ONLY
ALLOW_REVISED = VintagePolicy.ALLOW_REVISED


def _declared_sets():
    immutable = {e.channel for e in CHANNEL_VINTAGE
                 if e.source_vintage == "immutable_snapshot"}
    revised = {e.channel for e in CHANNEL_VINTAGE
               if e.source_vintage == "latest_revised"}
    return immutable, revised


def test_safe_only_allows_raw_and_derived_denies_revised():
    """SAFE_ONLY: every immutable_snapshot-sourced channel is True, every
    latest_revised-sourced channel is False."""
    immutable, revised = _declared_sets()
    assert immutable and revised
    for e in CHANNEL_VINTAGE:
        if e.channel in immutable:
            assert channel_allowed(e.channel, SAFE_ONLY) is True, e.channel
        else:
            assert channel_allowed(e.channel, SAFE_ONLY) is False, e.channel


def test_allow_revised_allows_every_declared_status():
    """ALLOW_REVISED: every declared channel (any declared source_vintage) is
    admissible."""
    for e in CHANNEL_VINTAGE:
        assert channel_allowed(e.channel, ALLOW_REVISED) is True, e.channel


def test_undeclared_channel_denied_under_both_policies():
    """An undeclared channel (no declaration → the reserved "unknown" fallback)
    is the mandatory deny under BOTH policies — a consumer can never silently
    consume an uncurated channel."""
    assert channel_allowed("undeclared_channel_x", SAFE_ONLY) is False
    assert channel_allowed("undeclared_channel_x", ALLOW_REVISED) is False


def test_allowed_channels_safe_only_is_raw_union_derived():
    immutable, _ = _declared_sets()
    assert allowed_channels(SAFE_ONLY) == immutable


def test_allowed_channels_allow_revised_is_all_declared():
    assert allowed_channels(ALLOW_REVISED) == {e.channel for e in CHANNEL_VINTAGE}


def test_denied_channels_safe_only_is_declared_revised_set():
    _, revised = _declared_sets()
    denied = denied_channels(SAFE_ONLY)
    assert denied == revised
    assert len(denied) == 10  # the 10 declared latest_revised-sourced channels


def test_denied_channels_allow_revised_is_empty():
    assert denied_channels(ALLOW_REVISED) == frozenset()


def test_price_channel_invariant_safe_only():
    """daily_qfq must always be admissible under SAFE_ONLY (immutable_snapshot
    source) — a model cannot train without the price channel."""
    assert channel_allowed("daily_qfq", SAFE_ONLY) is True


def test_vintage_report_with_real_declaration():
    """The real declaration: no missing documented dims, price channel allowed,
    complete 3-dim declaration, every entry carries the 6 keys, and the policy
    string is the enum value."""
    report = vintage_report(SAFE_ONLY)
    assert report["vintage_policy"] == "safe-only"
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
    report = vintage_report(SAFE_ONLY, declaration=partial)
    assert "fundamental" in report["missing_channels"]
    assert len(report["channels"]) == len(CHANNEL_VINTAGE) - 1


def test_vintage_report_crafted_daily_qfq_denied_under_safe_only():
    """A CRAFTED declaration labeling daily_qfq as latest_revised source makes
    daily_qfq_allowed False under SAFE_ONLY — proving the formal FAIL branch for
    a price-channel-denying policy."""
    crafted = tuple(
        dataclasses.replace(e, source_vintage="latest_revised")
        if e.channel == "daily_qfq" else e
        for e in CHANNEL_VINTAGE
    )
    report = vintage_report(SAFE_ONLY, declaration=crafted)
    assert report["daily_qfq_allowed"] is False


def test_vintage_report_declaration_complete_false_on_unknown_dim():
    """§T7: a crafted declaration where one channel declares transform="unknown"
    is NOT complete — and that channel is denied under BOTH policies by the
    both-layers admission check."""
    crafted = tuple(
        dataclasses.replace(e, transform="unknown")
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    report = vintage_report(SAFE_ONLY, declaration=crafted)
    assert report["declaration_complete"] is False
    assert channel_allowed("sentiment", SAFE_ONLY,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", ALLOW_REVISED,
                           vintage_by_name=crafted_by_name) is False


def test_vintage_report_declaration_complete_false_on_out_of_vocab_source():
    """§T7: a crafted declaration where one channel declares an OUT-OF-
    VOCABULARY source_vintage (e.g. a typo "immutable_snapshoot") is NOT
    complete — and that channel is denied under BOTH policies by the
    vocabulary-guarded admission check."""
    crafted = tuple(
        dataclasses.replace(e, source_vintage="immutable_snapshoot")
        if e.channel == "sentiment" else e
        for e in CHANNEL_VINTAGE
    )
    crafted_by_name = {e.channel: e for e in crafted}
    report = vintage_report(SAFE_ONLY, declaration=crafted)
    assert report["declaration_complete"] is False
    assert channel_allowed("sentiment", SAFE_ONLY,
                           vintage_by_name=crafted_by_name) is False
    assert channel_allowed("sentiment", ALLOW_REVISED,
                           vintage_by_name=crafted_by_name) is False


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
    safe-only run is NOT free of latest-reconstructed data in its universe
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
