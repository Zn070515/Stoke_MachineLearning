"""v14 §十五 channel-vintage declaration tests.

The module ``stoke_ml.data.channel_vintage`` is the curated governance
declaration that the formal gate report surfaces.  Every declared channel must
carry a valid two-status label and a non-empty rationale; the documented
revision-leakage sources (fundamental, macro) must be locked as
``latest_revised_aligned``; the declaration must stay deterministic and in sync
with the CLAUDE.md ``use_*`` dimension list.
"""
import importlib

import stoke_ml.data.channel_vintage as cv
from stoke_ml.data.channel_vintage import CHANNEL_VINTAGE

VALID_STATUSES = {"vintage_safe", "latest_revised_aligned"}


def test_all_entries_have_valid_status_and_nonempty_rationale():
    """Every channel has a legal status and a concrete, non-empty rationale."""
    assert len(CHANNEL_VINTAGE) > 0
    for e in CHANNEL_VINTAGE:
        assert e.status in VALID_STATUSES, f"{e.channel}: bad status {e.status!r}"
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
