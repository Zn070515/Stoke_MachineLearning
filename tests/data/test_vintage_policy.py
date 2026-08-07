"""§T2 / v15 §六/§十 vintage-admission policy tests.

``stoke_ml.data.vintage_policy`` decides which channels a training run may
consume from their declared vintage status.  The policy matrix is exercised
over the REAL curated declaration; the report's enforcement branches are proven
with CRAFTED declarations passed via the test-injection parameters (no module
globals are mutated).
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
    raw = {e.channel for e in CHANNEL_VINTAGE if e.status == "raw_vintage_safe"}
    derived = {e.channel for e in CHANNEL_VINTAGE if e.status == "derived_versioned"}
    revised = {e.channel for e in CHANNEL_VINTAGE if e.status == "latest_revised_aligned"}
    return raw, derived, revised


def test_safe_only_allows_raw_and_derived_denies_revised():
    """SAFE_ONLY: raw_vintage_safe + derived_versioned are True, every
    latest_revised_aligned channel is False."""
    raw, derived, revised = _declared_sets()
    assert raw and derived and revised
    for e in CHANNEL_VINTAGE:
        if e.channel in (raw | derived):
            assert channel_allowed(e.channel, SAFE_ONLY) is True, e.channel
        else:
            assert channel_allowed(e.channel, SAFE_ONLY) is False, e.channel


def test_allow_revised_allows_every_declared_status():
    """ALLOW_REVISED: every declared channel (any of the 3 declared statuses) is
    admissible."""
    for e in CHANNEL_VINTAGE:
        assert channel_allowed(e.channel, ALLOW_REVISED) is True, e.channel


def test_undeclared_channel_denied_under_both_policies():
    """unknown_vintage (an undeclared channel) is the mandatory deny under BOTH
    policies — a consumer can never silently consume an uncurated channel."""
    assert channel_allowed("undeclared_channel_x", SAFE_ONLY) is False
    assert channel_allowed("undeclared_channel_x", ALLOW_REVISED) is False


def test_allowed_channels_safe_only_is_raw_union_derived():
    raw, derived, _ = _declared_sets()
    assert allowed_channels(SAFE_ONLY) == (raw | derived)


def test_allowed_channels_allow_revised_is_all_declared():
    assert allowed_channels(ALLOW_REVISED) == {e.channel for e in CHANNEL_VINTAGE}


def test_denied_channels_safe_only_is_declared_revised_set():
    _, _, revised = _declared_sets()
    denied = denied_channels(SAFE_ONLY)
    assert denied == revised
    assert len(denied) == 10  # the 10 declared latest_revised_aligned channels


def test_denied_channels_allow_revised_is_empty():
    assert denied_channels(ALLOW_REVISED) == frozenset()


def test_price_channel_invariant_safe_only():
    """daily_qfq must always be admissible under SAFE_ONLY (derived_versioned) —
    a model cannot train without the price channel."""
    assert channel_allowed("daily_qfq", SAFE_ONLY) is True


def test_vintage_report_with_real_declaration():
    """The real declaration: no missing documented dims, price channel allowed,
    every entry carries the 4 keys, and the policy string is the enum value."""
    report = vintage_report(SAFE_ONLY)
    assert report["vintage_policy"] == "safe-only"
    assert report["missing_channels"] == []
    assert report["daily_qfq_allowed"] is True
    for entry in report["channels"]:
        assert set(entry) == {"channel", "status", "rationale", "allowed"}
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
    """A CRAFTED declaration labeling daily_qfq as latest_revised_aligned makes
    daily_qfq_allowed False under SAFE_ONLY — proving the formal FAIL branch for
    a price-channel-denying policy."""
    crafted = tuple(
        dataclasses.replace(e, status="latest_revised_aligned")
        if e.channel == "daily_qfq" else e
        for e in CHANNEL_VINTAGE
    )
    report = vintage_report(SAFE_ONLY, declaration=crafted)
    assert report["daily_qfq_allowed"] is False


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
