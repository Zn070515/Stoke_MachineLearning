"""v15 §六/§十 channel-vintage declaration tests.

The module ``stoke_ml.data.channel_vintage`` is the curated governance
declaration that the formal gate report surfaces.  Every declared channel must
carry one of the three DECLARED statuses (``raw_vintage_safe`` /
``derived_versioned`` / ``latest_revised_aligned``) and a non-empty rationale;
``unknown_vintage`` is the reserved deny-by-default fallback of ``status_of()``
and is NEVER a declared label.  The documented revision-leakage sources
(fundamental, macro) must be locked as ``latest_revised_aligned``; the derived
+ versioned price channel (daily_qfq) must be ``derived_versioned``; the
declaration must stay deterministic and in sync with the CLAUDE.md ``use_*``
dimension list.
"""
import importlib

import stoke_ml.data.channel_vintage as cv
from stoke_ml.data.channel_vintage import CHANNEL_VINTAGE

VALID_STATUSES = {
    "raw_vintage_safe",
    "derived_versioned",
    "latest_revised_aligned",
}


def test_all_entries_have_valid_status_and_nonempty_rationale():
    """Every channel has a legal declared status and a concrete, non-empty
    rationale."""
    assert len(CHANNEL_VINTAGE) > 0
    for e in CHANNEL_VINTAGE:
        assert e.status in VALID_STATUSES, f"{e.channel}: bad status {e.status!r}"
        assert e.status != "unknown_vintage", (
            f"{e.channel}: unknown_vintage is never a declared status"
        )
        assert isinstance(e.rationale, str) and e.rationale.strip(), (
            f"{e.channel}: empty rationale"
        )


def test_channels_are_unique():
    """No duplicate channel names in the declaration."""
    names = [e.channel for e in CHANNEL_VINTAGE]
    assert len(names) == len(set(names)), "duplicate channel names present"


def test_core_revision_leakage_sources_are_latest_revised_aligned():
    """§十五: fundamental and macro are the documented revision-leakage source
    — locked as latest_revised_aligned, exactly."""
    by_name = cv.CHANNEL_VINTAGE_BY_NAME
    assert by_name["fundamental"].status == "latest_revised_aligned"
    assert by_name["macro"].status == "latest_revised_aligned"


def test_derived_price_channel_is_derived_versioned():
    """§T2: the qfq price channel is derived + versioned (not an immutable raw
    snapshot), and the immutable-event sentiment channel is raw_vintage_safe."""
    by_name = cv.CHANNEL_VINTAGE_BY_NAME
    assert by_name["daily_qfq"].status == "derived_versioned"
    assert by_name["sentiment"].status == "raw_vintage_safe"


def test_known_statuses_is_exactly_all_four_states():
    """§T2: KNOWN_STATUSES is EXACTLY the 3 declared states plus the reserved
    unknown_vintage deny-by-default fallback — a stray 5th status added to the
    vocabulary must fail this equality, not slip through a superset check."""
    assert set(cv.KNOWN_STATUSES) == {
        "raw_vintage_safe", "derived_versioned", "latest_revised_aligned",
        "unknown_vintage",
    }


def test_status_of_defaults_to_unknown_vintage():
    """§T2: status_of() returns unknown_vintage for any undeclared channel (the
    deny-by-default fallback) and the declared status for curated channels."""
    assert cv.status_of("definitely_not_a_channel") == "unknown_vintage"
    assert cv.status_of("sentiment") == "raw_vintage_safe"


def test_declaration_is_deterministic_across_imports():
    """Two (re)imports yield byte-identical content — frozen dataclass + stable
    ordering make the report serialization deterministic."""
    first = [(e.channel, e.status, e.rationale) for e in cv.CHANNEL_VINTAGE]
    reloaded = importlib.reload(cv)
    second = [(e.channel, e.status, e.rationale) for e in reloaded.CHANNEL_VINTAGE]
    assert first == second


def test_channels_cover_documented_use_dims():
    """Canary: the declaration covers every documented CLAUDE.md ``use_*``
    dimension name, so the declaration cannot silently drift from the docs."""
    declared = {e.channel for e in CHANNEL_VINTAGE}
    assert cv.DOCUMENTED_USE_DIMS <= declared, (
        f"missing channels: {sorted(cv.DOCUMENTED_USE_DIMS - declared)}"
    )


def test_by_name_index_matches_declaration():
    """The convenience index agrees with the canonical tuple (same module state
    — the reload test may have swapped in fresh frozen instances)."""
    assert set(cv.CHANNEL_VINTAGE_BY_NAME) == {e.channel for e in cv.CHANNEL_VINTAGE}
    for e in cv.CHANNEL_VINTAGE:
        assert cv.CHANNEL_VINTAGE_BY_NAME[e.channel] is e
