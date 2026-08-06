"""§T18: build_features.py per-load error handling.

The NARROW site (_load_stock_parquet) must only swallow (OSError, ValueError);
the KEEP_BROAD sites (_load_opt / _load_etf) must keep their broad catch but
record every failure into the module-level ``_load_summary`` so the skip is
reported, not silently dropped.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "production" / "build_features.py"
_spec = importlib.util.spec_from_file_location("build_features_mod", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _baseline() -> int:
    return mod._load_summary.total()


class _RaiserChannel:
    def load_daily_sentiment(self, code, start, end):
        raise RuntimeError("channel boom")


class _RaiserSectorMapper:
    def get_sector(self, code):
        raise RuntimeError("sector boom")


class _RaiserETF:
    def load_sector_flow(self, sector, start, end):
        raise RuntimeError("etf boom")


class TestLoadStockParquetNarrow:
    def test_corrupt_parquet_returns_empty_and_records(self, tmp_path):
        """A corrupt aux parquet degrades to an empty frame AND is counted."""
        (tmp_path / "000001.parquet").write_bytes(b"not a parquet")
        before = _baseline()
        out = mod._load_stock_parquet(str(tmp_path), "000001")
        assert out.empty
        assert mod._load_summary.total() == before + 1

    def test_missing_file_returns_empty_no_record(self, tmp_path):
        """A missing file is a normal no-data state, not an error."""
        before = _baseline()
        out = mod._load_stock_parquet(str(tmp_path), "999999")
        assert out.empty
        assert mod._load_summary.total() == before

    def test_non_oserror_valueerror_propagates(self, tmp_path, monkeypatch):
        """§T18: only (OSError, ValueError) are swallowed — anything else must
        propagate instead of being silently converted to an empty frame."""
        (tmp_path / "000001.parquet").write_bytes(b"not a parquet")

        def _raise(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mod.pd, "read_parquet", _raise)
        with pytest.raises(RuntimeError, match="unexpected"):
            mod._load_stock_parquet(str(tmp_path), "000001")


class TestLoadOptETFAggregation:
    def test_load_opt_records_channel_failure(self):
        """§T18: a channel load failure keeps its broad catch but is counted."""
        args = {"news_storage": _RaiserChannel(),
                "start": "2024-01-01", "end": "2024-12-31"}
        before = _baseline()
        out = mod._load_opt(args, "news_storage", "load_daily_sentiment", "000001")
        assert out is None
        assert mod._load_summary.total() == before + 1

    def test_load_etf_records_failure(self):
        """§T18: an ETF-flow load failure keeps its broad catch but is counted."""
        args = {"sector_mapper": _RaiserSectorMapper(),
                "etf_storage": _RaiserETF(),
                "start": "2024-01-01", "end": "2024-12-31"}
        before = _baseline()
        out = mod._load_etf(args, "000001")
        assert out is None
        assert mod._load_summary.total() == before + 1
