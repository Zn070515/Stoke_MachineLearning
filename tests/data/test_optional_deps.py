"""Optional-dependency boundaries.

The online A-share providers (Efinance etc.) carry optional crawler deps
(``curl_cffi``).  Importing ``stoke_ml.data.sources.a_shares`` — and even
constructing ``AShareDownloader`` for an offline/mock-sourced test — must not
require those deps.  This simulates a box without curl_cffi and asserts the
module chain imports, the downloader constructs, the missing source reports
itself unavailable, and an offline fetch degrades to empty instead of raising.
"""

import sys

import pytest

from stoke_ml.data.sources.a_shares.failover import AShareDownloader

# Modules in the a_shares import chain (sources are imported lazily by
# AShareDownloader.__init__).  We force a fresh re-import under the blocked
# state to prove the boundary, rather than trusting an already-imported cache.
_CHAIN = [
    "stoke_ml.data.sources.a_shares",
    "stoke_ml.data.sources.a_shares.failover",
    "stoke_ml.data.sources.a_shares.efinance_source",
    "stoke_ml.data.sources.a_shares.akshare_source",
    "stoke_ml.data.sources.a_shares.tushare_source",
    "stoke_ml.data.sources.a_shares.baostock_source",
]


@pytest.fixture
def block_curl_cffi(monkeypatch):
    """Make ``import curl_cffi`` raise ImportError for the test duration."""
    # None-in-sys.modules makes any `import curl_cffi` raise ImportError.
    monkeypatch.setitem(sys.modules, "curl_cffi", None)
    for name in _CHAIN:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_package_and_downloader_import_without_curl_cffi(block_curl_cffi):
    import importlib
    pkg = importlib.import_module("stoke_ml.data.sources.a_shares")
    assert hasattr(pkg, "AShareDownloader")


def test_downloader_constructs_without_curl_cffi(block_curl_cffi):
    dl = AShareDownloader()
    assert dl._sources[0].SOURCE_NAME == "efinance"


def test_efinance_reports_unavailable_without_curl_cffi(block_curl_cffi):
    dl = AShareDownloader()
    assert dl._sources[0].is_available() is False
    # Other sources (no crawler dep at import time) still construct fine.
    assert all(s.is_available() is False or s.SOURCE_NAME != "efinance"
               for s in dl._sources)


def test_efinance_fetch_degrades_to_empty_without_curl_cffi(block_curl_cffi):
    dl = AShareDownloader()
    out = dl._sources[0].fetch_daily("000001", "2024-01-01", "2024-01-02")
    assert out.empty


def test_stitch_helpers_still_work_without_curl_cffi(block_curl_cffi):
    # _stitch_segments / _repair_pct_change are pure offline logic used by the
    # source-normalization tests; they must not require the crawler dep either.
    import pandas as pd
    back = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [10.0, 10.5], "high": [11.0, 11.0], "low": [9.5, 10.0],
        "close": [10.0, 10.5], "volume": [1e5, 1.1e5],
    })
    primary = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [21.0, 21.5], "high": [22.0, 22.0], "low": [20.5, 21.0],
        "close": [21.0, 21.5], "volume": [2e5, 2.1e5],
    })
    rebased, ratio = AShareDownloader._stitch_segments(back, primary)
    assert ratio is not None and 1.9 <= ratio <= 2.1
    assert list(rebased["date"]) == ["2024-01-01"]  # overlap row removed
