"""Single source of truth for feature-channel → data-directory paths (§T2).

The v17 audit (§六) found the LIVE and PREBUILT feature paths for the same
channel hard-coded as string literals in five consumer modules, so a rename
could drift silently (e.g. live ``capital_flow`` at ``a_shares/capital_flow/``
vs prebuilt ``capital_flow_processed`` at ``a_shares/capital_flow_processed/``):

  - ``scripts/production/train_panel_panel.py`` — the live ``load_aux_data``
    MarketWideStorage loop and the industry/market_env broadcast probe;
  - ``stoke_ml/features/cache_manifest.py`` — prebuilt lineage fingerprints the
    per-stock source file under the channel dir;
  - ``scripts/production/build_features.py`` — the prebuilt builder reads the
    ``*_processed`` dirs directly;
  - ``stoke_ml/data/market_wide_storage.py`` — the ``*_processed`` whitelist
    entries;
  - ``scripts/production/data_quality_gate.py`` — the ``AUX_CLOSE_DIRS`` /
    ``AUX_PCT_DIRS`` processed dirs.

:data:`CHANNEL_SOURCE` is now the single registry every consumer derives its
paths from.  All paths are relative to the project ``data_dir`` and start with
``a_shares/`` so consumers join them onto their own root.

``live_dir``      — directory the LIVE feature path reads (per-stock files),
                    i.e. the ``a_shares`` subdir ``MarketWideStorage(data_dir,
                    <last segment>)`` or the storage class reads.
``processed_dir`` — optional ``*_processed`` variant the PREBUILT path reads.
                    ``None`` means the channel has no separate processed dir —
                    prebuilt reads the same ``live_dir``.  For channels whose
                    only dir IS the ``*_processed`` dir (board / sector /
                    concept / pledge / index_membership) live and processed
                    coincide.
``storage_kind``  — which reader serves the channel: a storage class name,
                    ``"MarketWideStorage"``, ``"flat_parquet"`` (direct
                    per-stock parquet read), or ``"shared_parquet"`` (one
                    market-wide file, broadcast to every stock).

All values were derived from the exact string literals the consumers used
before this change — byte-identical, never "one of the two" — and the
rewire tests pin that (see ``tests/data/test_channel_sources.py``).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSourceSpec:
    """Where a feature channel's live / processed per-stock data lives."""

    channel: str
    live_dir: str             # data_dir-rooted; e.g. "a_shares/capital_flow"
    storage_kind: str         # e.g. "MarketWideStorage", "NewsStorage", ...
    processed_dir: str | None = None  # e.g. "a_shares/capital_flow_processed"


CHANNEL_SOURCE: dict[str, ChannelSourceSpec] = {
    # ── DataStorage: canonical daily K-line (not a feature channel per se) ──
    "daily": ChannelSourceSpec(
        "daily", "a_shares/daily", "DataStorage",
    ),
    # ── Text / event / state channels served by their storage class; the
    #    prebuilt builder reads the SAME dir, so no *_processed variant. ──
    "sentiment": ChannelSourceSpec(
        "sentiment", "a_shares/sentiment", "NewsStorage",
    ),
    "guba": ChannelSourceSpec(
        "guba", "a_shares/guba_sentiment", "GubaStorage",
    ),
    "comment": ChannelSourceSpec(
        "comment", "a_shares/comment_sentiment", "CommentStorage",
    ),
    "announcement": ChannelSourceSpec(
        "announcement", "a_shares/announcements/sentiment", "AnnouncementStorage",
    ),
    "fundamental": ChannelSourceSpec(
        "fundamental", "a_shares/fundamentals", "FundamentalStorage",
    ),
    "etf_flow": ChannelSourceSpec(
        "etf_flow", "a_shares/etf_flow", "ETFStorage",
    ),
    # ── MarketWideStorage live channels with NO processed variant ──
    "margin": ChannelSourceSpec(
        "margin", "a_shares/margin", "MarketWideStorage",
    ),
    "northbound": ChannelSourceSpec(
        "northbound", "a_shares/northbound", "MarketWideStorage",
    ),
    "dragon_tiger": ChannelSourceSpec(
        "dragon_tiger", "a_shares/dragon_tiger", "MarketWideStorage",
    ),
    "valuation": ChannelSourceSpec(
        "valuation", "a_shares/valuation", "MarketWideStorage",
    ),
    # ── MarketWideStorage channels with a DISTINCT *_processed prebuilt
    #    variant (the live-vs-prebuilt split the audit flagged). ──
    "capital_flow": ChannelSourceSpec(
        "capital_flow", "a_shares/capital_flow",
        "MarketWideStorage", "a_shares/capital_flow_processed",
    ),
    "block_trade": ChannelSourceSpec(
        "block_trade", "a_shares/block_trade",
        "MarketWideStorage", "a_shares/block_trade_processed",
    ),
    "shareholder": ChannelSourceSpec(
        "shareholder", "a_shares/shareholder",
        "MarketWideStorage", "a_shares/shareholder_processed",
    ),
    "lockup": ChannelSourceSpec(
        "lockup", "a_shares/lockup",
        "MarketWideStorage", "a_shares/lockup_processed",
    ),
    "dividend": ChannelSourceSpec(
        "dividend", "a_shares/dividend",
        "MarketWideStorage", "a_shares/dividend_processed",
    ),
    # ── MarketWideStorage channels whose ONLY dir is the *_processed dir —
    #    live and prebuilt coincide (no separate un-processed variant). ──
    "board": ChannelSourceSpec(
        "board", "a_shares/board_processed",
        "MarketWideStorage", "a_shares/board_processed",
    ),
    "sector": ChannelSourceSpec(
        "sector", "a_shares/industry_ranking_processed",
        "MarketWideStorage", "a_shares/industry_ranking_processed",
    ),
    "concept": ChannelSourceSpec(
        "concept", "a_shares/concept_blocks_processed",
        "MarketWideStorage", "a_shares/concept_blocks_processed",
    ),
    # ── Processed-only channels served by a direct flat parquet read (no
    #    MarketWideStorage whitelist entry) — live == processed == the
    #    *_processed dir. ──
    "pledge": ChannelSourceSpec(
        "pledge", "a_shares/pledge_processed",
        "flat_parquet", "a_shares/pledge_processed",
    ),
    "index_membership": ChannelSourceSpec(
        "index_membership", "a_shares/index_membership_processed",
        "flat_parquet", "a_shares/index_membership_processed",
    ),
    # ── Shared market-wide parquet files (one file, broadcast to every
    #    stock, loaded by aux_aligner from the config root): live_dir is the
    #    file's DIRECTORY; the filename is a channel-specific artifact. ──
    "industry": ChannelSourceSpec(
        "industry", "a_shares/industry", "shared_parquet",
    ),
    "market_env": ChannelSourceSpec(
        "market_env", "a_shares/market_breadth", "shared_parquet",
    ),
}


def _last_segment(path: str) -> str:
    """The final path segment — the data-type name under ``a_shares/``."""
    return path.rstrip("/").rsplit("/", 1)[-1]


def live_data_type(spec: ChannelSourceSpec) -> str:
    """MarketWideStorage-style data type for a channel's LIVE dir.

    Meaningful for ``MarketWideStorage``-kind channels (the last ``a_shares``
    segment IS the data type); for nested live dirs of other storage kinds the
    last segment is not used by MarketWideStorage consumers.
    """
    return _last_segment(spec.live_dir)


def processed_data_type(spec: ChannelSourceSpec) -> str | None:
    """MarketWideStorage-style data type for a channel's PROCESSED dir.

    ``None`` when the channel has no processed variant.
    """
    if spec.processed_dir is None:
        return None
    return _last_segment(spec.processed_dir)


def source_dir(spec: ChannelSourceSpec) -> str:
    """The directory the PREBUILT / live feature reads (data_dir-rooted).

    The processed variant when one exists, else the live dir — a channel with
    no ``*_processed`` dir reads its live files directly.
    """
    return spec.processed_dir or spec.live_dir


def source_subdirs(spec: ChannelSourceSpec) -> tuple[str, ...]:
    """Path segments under ``a_shares/`` for the prebuilt source dir.

    Matches the historical ``cache_manifest.SOURCE_SUBDIRS`` tuple form
    (e.g. ``("announcements", "sentiment")``), so ``source_paths`` joins them
    unchanged.
    """
    return tuple(source_dir(spec).split("/")[1:])
