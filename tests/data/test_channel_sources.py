"""§T2: the CHANNEL_SOURCE registry is the single source of truth for the
live-vs-prebuilt feature path strings.

The v17 audit found the same channel's LIVE and PREBUILT paths hard-coded in
five consumer modules (``train_panel_panel.py``, ``cache_manifest.py``,
``build_features.py``, ``market_wide_storage.py``, ``data_quality_gate.py``), so
a rename could drift silently.  Every consumer now derives its paths from
:data:`stoke_ml.data.channel_sources.CHANNEL_SOURCE`.  These tests pin that the
derived values are byte-identical to the string literals the consumers used
before the change, and that the registry covers every headline_v1 + processed
channel.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from stoke_ml.data.channel_sources import (
    CHANNEL_SOURCE,
    live_data_type,
    processed_data_type,
    source_dir,
    source_subdirs,
)

# Every headline_v1 channel (§五-1) plus the processed-only channels whose dirs
# also appear as literals in the consumers.  ``daily`` is the canonical K-line
# (not a feature channel) but is the registry's DataStorage anchor.
HEADLINE_V1_CHANNELS = [
    "sentiment", "guba", "comment", "announcement", "margin", "northbound",
    "dragon_tiger", "capital_flow", "etf_flow", "block_trade", "lockup",
    "dividend", "industry", "market_env",
]
PROCESSED_ONLY_CHANNELS = ["board", "sector", "concept", "pledge",
                           "index_membership"]
OTHER_CHANNELS = ["daily", "fundamental", "valuation", "shareholder"]

# ── Historical string literals the consumers used BEFORE §T2 (byte-identical
#    targets — never "one of the two"). ─────────────────────────────────────
HIST_SOURCE_SUBDIRS = {
    "daily": ("daily",),
    "sentiment": ("sentiment",),
    "guba_sentiment": ("guba_sentiment",),
    "comment_sentiment": ("comment_sentiment",),
    "announcements_sentiment": ("announcements", "sentiment"),
    "fundamentals": ("fundamentals",),
    "margin": ("margin",),
    "northbound": ("northbound",),
    "dragon_tiger": ("dragon_tiger",),
    "valuation": ("valuation",),
    "capital_flow": ("capital_flow_processed",),
    "board": ("board_processed",),
    "sector": ("industry_ranking_processed",),
    "block_trade": ("block_trade_processed",),
    "dividend": ("dividend_processed",),
    "lockup": ("lockup_processed",),
    "shareholder": ("shareholder_processed",),
    "concept": ("concept_blocks_processed",),
    "pledge": ("pledge_processed",),
    "index_membership": ("index_membership_processed",),
}
HIST_MARKET_DATA_TYPES = {
    "dragon_tiger", "margin", "northbound", "capital_flow", "limit_up_zt",
    "limit_up_zb", "limit_up_dt", "limit_up_yzt", "limit_up_sentiment",
    "block_trade", "shareholder", "lockup", "lockup_upcoming", "dividend",
    "industry_ranking", "concept_blocks", "sina_fund_flow",
    "capital_flow_processed", "block_trade_processed", "shareholder_processed",
    "lockup_processed", "dividend_processed", "industry_ranking_processed",
    "concept_blocks_processed", "board_processed", "valuation",
}
HIST_AUX_CLOSE_DIRS = [
    "block_trade_processed", "board_processed", "dividend_processed",
    "industry_ranking_processed", "lockup_processed", "shareholder_processed",
]
HIST_AUX_PCT_DIRS = ["board_processed", "industry_ranking_processed"]


# ── Script consumers are loaded via importlib (same pattern as
#    tests/scripts/test_build_features_t18.py) so the test stays hermetic. ──
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "production"


def _load_script(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_features_mod():
    return _load_script("build_features")


@pytest.fixture(scope="module")
def data_quality_gate_mod():
    return _load_script("data_quality_gate")


@pytest.fixture(scope="module")
def train_panel_panel_mod():
    return _load_script("train_panel_panel")


class TestRegistry:
    def test_covers_all_headline_v1_channels(self):
        for ch in HEADLINE_V1_CHANNELS + PROCESSED_ONLY_CHANNELS + OTHER_CHANNELS:
            assert ch in CHANNEL_SOURCE, f"missing channel: {ch}"

    def test_spec_fields_consistent(self):
        for key, spec in CHANNEL_SOURCE.items():
            assert spec.channel == key
            assert spec.live_dir.startswith("a_shares/")
            assert spec.storage_kind
            if spec.processed_dir is not None:
                assert spec.processed_dir.startswith("a_shares/")

    def test_processed_channels_have_processed_dir(self):
        for ch in PROCESSED_ONLY_CHANNELS:
            assert CHANNEL_SOURCE[ch].processed_dir is not None

    def test_live_dir_last_segment_is_live_data_type(self):
        # MarketWideStorage(data_dir, data_type) — the data type IS the last
        # a_shares segment of the live dir.
        for ch in ("margin", "northbound", "dragon_tiger", "capital_flow",
                   "block_trade", "shareholder", "lockup", "dividend",
                   "valuation"):
            spec = CHANNEL_SOURCE[ch]
            assert spec.storage_kind == "MarketWideStorage"
            assert live_data_type(spec) == spec.live_dir.split("/")[-1]
            assert live_data_type(spec) == ch or \
                live_data_type(spec).startswith(ch)


class TestHelpers:
    def test_source_dir_prefers_processed(self):
        # capital_flow has a distinct *_processed prebuilt variant → source_dir
        # is that variant (what build_features/cache_manifest read).
        spec = CHANNEL_SOURCE["capital_flow"]
        assert source_dir(spec) == "a_shares/capital_flow_processed"

    def test_source_dir_falls_back_to_live(self):
        # margin has no processed variant → prebuilt reads the live dir.
        spec = CHANNEL_SOURCE["margin"]
        assert source_dir(spec) == "a_shares/margin"

    def test_source_subdirs(self):
        assert source_subdirs(CHANNEL_SOURCE["announcement"]) == \
            ("announcements", "sentiment")
        assert source_subdirs(CHANNEL_SOURCE["sector"]) == \
            ("industry_ranking_processed",)
        assert source_subdirs(CHANNEL_SOURCE["margin"]) == ("margin",)

    def test_processed_data_type_none_for_no_variant(self):
        assert processed_data_type(CHANNEL_SOURCE["margin"]) is None
        assert processed_data_type(CHANNEL_SOURCE["capital_flow"]) == \
            "capital_flow_processed"


class TestConsumerDerivations:
    """The rewired consumers must reproduce the historical string literals
    byte-identically (§T2 — backward-compat, never "one of the two")."""

    def test_cache_manifest_source_subdirs_identical(self):
        from stoke_ml.features import cache_manifest
        assert cache_manifest.SOURCE_SUBDIRS == HIST_SOURCE_SUBDIRS
        assert len(cache_manifest.SOURCE_SUBDIRS) == 20

    def test_market_wide_storage_data_types_identical(self):
        from stoke_ml.data.market_wide_storage import MARKET_DATA_TYPES
        assert set(MARKET_DATA_TYPES) == HIST_MARKET_DATA_TYPES
        assert len(MARKET_DATA_TYPES) == len(HIST_MARKET_DATA_TYPES)

    def test_data_quality_gate_aux_dirs_identical(self, data_quality_gate_mod):
        assert data_quality_gate_mod.AUX_CLOSE_DIRS == HIST_AUX_CLOSE_DIRS
        assert data_quality_gate_mod.AUX_PCT_DIRS == HIST_AUX_PCT_DIRS

    def test_build_features_channel_stock_dir(self, build_features_mod,
                                              tmp_path):
        data_dir = str(tmp_path / "data")
        assert build_features_mod._channel_stock_dir(
            data_dir, "pledge") == os.path.join(data_dir, "a_shares",
                                                "pledge_processed")
        assert build_features_mod._channel_stock_dir(
            data_dir, "capital_flow") == os.path.join(data_dir, "a_shares",
                                                     "capital_flow_processed")
        assert build_features_mod._channel_stock_dir(
            data_dir, "valuation") == os.path.join(data_dir, "a_shares",
                                                   "valuation")

    def test_train_panel_panel_live_data_type(self, train_panel_panel_mod):
        # The MarketWideStorage loop reads the LIVE dir, not the *_processed
        # one — live_data_type must resolve capital_flow → "capital_flow".
        spec = CHANNEL_SOURCE["capital_flow"]
        assert live_data_type(spec) == "capital_flow"
        assert train_panel_panel_mod._MARKET_WIDE_CHANNELS == (
            "margin", "northbound", "dragon_tiger", "capital_flow",
            "block_trade", "shareholder", "lockup", "dividend", "valuation",
        )
        # Every channel in the loop resolves to a valid MarketWideStorage type.
        for ch in train_panel_panel_mod._MARKET_WIDE_CHANNELS:
            assert live_data_type(CHANNEL_SOURCE[ch]) in \
                {d for d in __import__(
                    "stoke_ml.data.market_wide_storage",
                    fromlist=["MARKET_DATA_TYPES"]).MARKET_DATA_TYPES}
