"""Unit tests for the channel-coverage manifest (v7 §六.2).

train_panel.py must be able to tell "this aux channel is absent from disk"
(MISSING) from "the channel broke while loading" (FAILED), so a storage schema
change or missing dir can no longer silently zero out a data dimension.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "train_panel.py")


@pytest.fixture(scope="module")
def tp():
    spec = importlib.util.spec_from_file_location("train_panel_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _finalize_channel status matrix ───────────────────────────────────

def test_new_channel_entry_defaults(tp):
    e = tp._new_channel_entry(True, False)
    assert e["status"] == "MISSING"
    assert e["loaded_stocks"] == 0 and e["errors"] == 0
    assert e["coverage"] == 0.0
    assert e["requested"] is True and e["required"] is False


def test_finalize_missing(tp):
    e = tp._new_channel_entry(True, False)
    tp._finalize_channel(e, "x", loaded=0, errors=0, n=10)
    assert e["status"] == "MISSING"
    assert e["coverage"] == 0.0


def test_finalize_failed(tp):
    e = tp._new_channel_entry(True, False)
    tp._finalize_channel(e, "x", loaded=0, errors=10, n=10)
    assert e["status"] == "FAILED"
    assert e["coverage"] == 0.0


def test_finalize_partial(tp):
    e = tp._new_channel_entry(True, False)
    tp._finalize_channel(e, "x", loaded=4, errors=6, n=10)
    assert e["status"] == "PARTIAL"
    assert e["coverage"] == 0.4


def test_finalize_ok(tp):
    e = tp._new_channel_entry(True, False)
    tp._finalize_channel(e, "x", loaded=10, errors=0, n=10)
    assert e["status"] == "OK"
    assert e["coverage"] == 1.0


def test_finalize_zero_n_no_divzero(tp):
    e = tp._new_channel_entry(True, False)
    tp._finalize_channel(e, "x", loaded=0, errors=0, n=0)
    assert e["status"] == "MISSING"
    assert e["coverage"] == 0.0


# ── _load_channel_aux with fake storage ───────────────────────────────

def _df():
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "v": [1.0, 2.0, 3.0]})


def test_load_channel_storage_construction_fails(tp):
    result = {"000001": {}, "000002": {}}
    manifest = {}

    def boom():
        raise RuntimeError("schema broke")

    tp._load_channel_aux("margin", ["000001", "000002"], result, manifest,
                         make_storage=boom, load_one=lambda st, c: _df())
    e = manifest["margin"]
    assert e["status"] == "FAILED"
    assert e["errors"] == 2
    assert "schema broke" in e["note"]
    assert result["000001"] == {} and result["000002"] == {}


def test_load_channel_missing_from_disk(tp):
    result = {"000001": {}, "000002": {}}
    manifest = {}
    tp._load_channel_aux("guba", ["000001", "000002"], result, manifest,
                         make_storage=lambda: object(),
                         load_one=lambda st, c: None)
    e = manifest["guba"]
    assert e["status"] == "MISSING"
    assert e["loaded_stocks"] == 0 and e["errors"] == 0


def test_load_channel_all_reads_fail(tp):
    result = {"000001": {}, "000002": {}}
    manifest = {}

    def fail(st, c):
        raise ValueError("bad parquet")

    tp._load_channel_aux("sentiment", ["000001", "000002"], result, manifest,
                         make_storage=lambda: object(), load_one=fail)
    e = manifest["sentiment"]
    assert e["status"] == "FAILED"
    assert e["loaded_stocks"] == 0 and e["errors"] == 2


def test_load_channel_partial(tp):
    result = {"000001": {}, "000002": {}, "000003": {}}
    manifest = {}

    def load_one(st, c):
        if c == "000002":
            raise ValueError("bad")
        return _df()

    tp._load_channel_aux("comment", ["000001", "000002", "000003"], result,
                         manifest, make_storage=lambda: object(), load_one=load_one)
    e = manifest["comment"]
    assert e["status"] == "PARTIAL"
    assert e["loaded_stocks"] == 2 and e["errors"] == 1
    assert e["coverage"] == 0.6667
    assert "comment" in result["000001"] and "comment" in result["000003"]
    assert "comment" not in result["000002"]


def test_load_channel_ok(tp):
    result = {"000001": {}, "000002": {}}
    manifest = {}
    tp._load_channel_aux("announcement", ["000001", "000002"], result, manifest,
                         make_storage=lambda: object(), load_one=lambda st, c: _df())
    e = manifest["announcement"]
    assert e["status"] == "OK"
    assert e["loaded_stocks"] == 2 and e["errors"] == 0
    assert e["coverage"] == 1.0


def test_load_channel_empty_df_counts_as_not_loaded(tp):
    result = {"000001": {}}
    manifest = {}
    tp._load_channel_aux("guba", ["000001"], result, manifest,
                         make_storage=lambda: object(),
                         load_one=lambda st, c: pd.DataFrame())
    assert manifest["guba"]["status"] == "MISSING"
    assert "guba" not in result["000001"]


def test_load_channel_required_flag_propagates(tp):
    result = {"000001": {}}
    manifest = {}
    tp._load_channel_aux("sentiment", ["000001"], result, manifest,
                         make_storage=lambda: object(), load_one=lambda st, c: _df(),
                         required=True)
    assert manifest["sentiment"]["required"] is True


# ── _prebuilt_channel_coverage (has_* flag probe) ─────────────────────

_FLAGS = ["has_news", "has_guba_post", "has_comment",
          "has_announce", "has_forecast", "has_pledge", "has_hot_board"]


def _panel(flags, grid):
    # grid: list of (stock, day) rows; each row is a list of flag values.
    po = np.asarray(grid, dtype=np.float32).reshape(len(grid), 1, len(flags))
    return {"past_observed": po, "past_observed_cols": list(flags)}


def test_prebuilt_probes_has_flags(tp):
    grid = [
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ]
    ch = tp._prebuilt_channel_coverage(_panel(_FLAGS, grid))
    assert ch["sentiment"]["status"] == "OK"
    assert ch["sentiment"]["coverage"] == 1.0
    assert ch["guba"]["status"] == "MISSING"
    assert ch["guba"]["coverage"] == 0.0
    assert ch["comment"]["status"] == "OK"
    assert ch["comment"]["coverage"] == 0.25
    assert ch["comment"]["cells"] == 4
    assert ch["concept"]["status"] == "MISSING"


def test_prebuilt_flag_not_in_cols_skipped(tp):
    # has_hot_board absent from the column union -> concept channel omitted.
    flags = ["has_news", "has_guba_post"]
    grid = [[1, 1]]
    ch = tp._prebuilt_channel_coverage(_panel(flags, grid))
    assert "sentiment" in ch and "guba" in ch
    assert "concept" not in ch


def test_prebuilt_no_past_observed_unknown(tp):
    ch = tp._prebuilt_channel_coverage({"past_observed_cols": list(_FLAGS)})
    assert ch["_note"]["status"] == "UNKNOWN"


# ── load_aux_data end-to-end (empty dir -> all channels MISSING) ──────

def test_load_aux_data_returns_manifest(tp, tmp_path):
    data_dir = str(tmp_path / "data")
    stocks = ["000001", "600519"]
    result, manifest = tp.load_aux_data(
        stocks, data_dir, "2024-01-01", "2024-12-31",
        required_channels={"sentiment"},
    )
    assert set(result) == set(stocks)
    for code in stocks:
        assert "etf_flow" not in result[code] or len(result[code]["etf_flow"]) >= 0

    expected = {"sentiment", "announcement", "guba", "comment", "fundamental",
                "margin", "northbound", "dragon_tiger", "capital_flow",
                "block_trade", "shareholder", "lockup", "dividend", "valuation",
                "etf_flow"}
    assert set(manifest) == expected

    assert manifest["sentiment"]["required"] is True
    for name, e in manifest.items():
        assert e["status"] in {"MISSING", "FAILED", "PARTIAL", "OK"}, name
        if name != "etf_flow":
            assert e["required"] is (name == "sentiment")
