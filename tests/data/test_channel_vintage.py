"""v15 §六/§十 / v16 §十二 channel-vintage declaration tests.

The module ``stoke_ml.data.channel_vintage`` is the curated governance
declaration that the formal gate report surfaces.  Every declared channel must
carry the three DECLARED 3-dim labels (``source_vintage`` / ``transform`` /
``pit_alignment`` — none ``"unknown"``) and a non-empty rationale; the
``"unknown"`` values are the reserved deny-by-default fallbacks of the
accessors and are NEVER declared labels.  The documented revision-leakage
sources (fundamental, macro) must be locked as ``latest_revised`` source; the
price channel (daily_qfq) must be ``immutable_snapshot`` source with a
``formula_versioned`` transform and ``proxy`` alignment; the declaration must
stay deterministic and in sync with the CLAUDE.md ``use_*`` dimension list.
"""
import importlib

import stoke_ml.data.channel_vintage as cv
from stoke_ml.data.channel_vintage import CHANNEL_VINTAGE

DECLARED_SOURCE_VINTAGES = {"immutable_snapshot", "latest_revised"}
DECLARED_TRANSFORMS = {"raw", "model_versioned", "formula_versioned"}
DECLARED_PIT_ALIGNMENTS = {"verified", "proxy"}


def test_all_entries_have_valid_status_and_nonempty_rationale():
    """Every channel carries the three DECLARED 3-dim labels (none "unknown")
    and a concrete, non-empty rationale."""
    assert len(CHANNEL_VINTAGE) > 0
    for e in CHANNEL_VINTAGE:
        assert e.source_vintage in DECLARED_SOURCE_VINTAGES, (
            f"{e.channel}: bad source_vintage {e.source_vintage!r}"
        )
        assert e.transform in DECLARED_TRANSFORMS, (
            f"{e.channel}: bad transform {e.transform!r}"
        )
        assert e.pit_alignment in DECLARED_PIT_ALIGNMENTS, (
            f"{e.channel}: bad pit_alignment {e.pit_alignment!r}"
        )
        for dim, value in (("source_vintage", e.source_vintage),
                           ("transform", e.transform),
                           ("pit_alignment", e.pit_alignment)):
            assert value != "unknown", (
                f"{e.channel}: 'unknown' is never a declared {dim}"
            )
        assert isinstance(e.rationale, str) and e.rationale.strip(), (
            f"{e.channel}: empty rationale"
        )


def test_channels_are_unique():
    """No duplicate channel names in the declaration."""
    names = [e.channel for e in CHANNEL_VINTAGE]
    assert len(names) == len(set(names)), "duplicate channel names present"


def test_core_revision_leakage_sources_are_latest_revised_sourced():
    """§十五: fundamental and macro are the documented revision-leakage source
    — locked as latest_revised-sourced, exactly."""
    by_name = cv.CHANNEL_VINTAGE_BY_NAME
    assert by_name["fundamental"].source_vintage == "latest_revised"
    assert by_name["macro"].source_vintage == "latest_revised"


def test_derived_price_channel_is_formula_versioned():
    """§T2/§T7: the qfq price channel is immutable-sourced (raw OHLC +
    adjustment-factor history) with a formula_versioned transform (re-anchoring)
    and proxy alignment — not an immutable raw-value snapshot."""
    by_name = cv.CHANNEL_VINTAGE_BY_NAME
    assert by_name["daily_qfq"].source_vintage == "immutable_snapshot"
    assert by_name["daily_qfq"].transform == "formula_versioned"
    assert by_name["daily_qfq"].pit_alignment == "proxy"


def test_known_axis_vocabularies_are_exact():
    """§T2/§T7: each axis vocabulary is EXACTLY its declared labels plus the
    reserved "unknown" deny-by-default fallback — a stray extra label added to
    a vocabulary must fail this equality, not slip through a superset check."""
    assert set(cv.KNOWN_SOURCE_VINTAGES) == {
        "immutable_snapshot", "latest_revised", "unknown",
    }
    assert set(cv.KNOWN_TRANSFORMS) == {
        "raw", "model_versioned", "formula_versioned", "unknown",
    }
    assert set(cv.KNOWN_PIT_ALIGNMENTS) == {"verified", "proxy", "unknown"}


def test_accessors_default_to_unknown():
    """§T2/§T7: the 3-dim accessors return "unknown" for any undeclared channel
    (the deny-by-default fallback) and the declared labels for curated channels;
    declaration_of() returns None for the undeclared one."""
    assert cv.declaration_of("definitely_not_a_channel") is None
    assert cv.source_vintage_of("definitely_not_a_channel") == "unknown"
    assert cv.transform_of("definitely_not_a_channel") == "unknown"
    assert cv.pit_alignment_of("definitely_not_a_channel") == "unknown"
    assert cv.source_vintage_of("sentiment") == "immutable_snapshot"
    assert cv.transform_of("sentiment") == "model_versioned"
    assert cv.pit_alignment_of("sentiment") == "verified"


def test_declaration_is_deterministic_across_imports():
    """Two (re)imports yield byte-identical content — frozen dataclass + stable
    ordering make the report serialization deterministic."""
    first = [(e.channel, e.source_vintage, e.transform, e.pit_alignment,
              e.rationale) for e in cv.CHANNEL_VINTAGE]
    reloaded = importlib.reload(cv)
    second = [(e.channel, e.source_vintage, e.transform, e.pit_alignment,
               e.rationale) for e in reloaded.CHANNEL_VINTAGE]
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


def test_news_three_dim_classification():
    """§T7 (plan's named example): news 原文 is immutable_snapshot, but the
    sentiment score is a model_versioned transform (version must be recorded,
    not the channel denied); alignment is verified."""
    e = cv.CHANNEL_VINTAGE_BY_NAME["sentiment"]
    assert e.source_vintage == "immutable_snapshot"
    assert e.transform == "model_versioned"
    assert e.pit_alignment == "verified"


def test_capital_flow_three_dim_classification():
    """§T7 (plan's named example): capital_flow stays ALLOWED but is HONESTLY
    labeled — immutable_snapshot source with a formula_versioned transform
    (vendor-computed) and verified alignment."""
    e = cv.CHANNEL_VINTAGE_BY_NAME["capital_flow"]
    assert e.source_vintage == "immutable_snapshot"
    assert e.transform == "formula_versioned"
    assert e.pit_alignment == "verified"
