"""Unit tests for the channel-coverage manifest.

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
SCRIPT = os.path.join(ROOT, "scripts", "production", "train_panel.py")


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


def _panel_nt(flags, flag_grids):
    """A (N, T, D) past_observed grid; ``flag_grids`` maps a flag name to an
    (N, T) 0/1 presence grid.  Unlisted flags stay all-zero."""
    grids = list(flag_grids.values()) or [np.zeros((1, 1), dtype=np.float32)]
    n = max(g.shape[0] for g in grids)
    t = max(g.shape[1] for g in grids)
    po = np.zeros((n, t, len(flags)), dtype=np.float32)
    for i, flag in enumerate(flags):
        if flag in flag_grids:
            po[:, :, i] = flag_grids[flag]
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
    assert ch["sentiment"]["stock_coverage"] == 1.0
    assert ch["guba"]["status"] == "MISSING"
    assert ch["guba"]["coverage"] == 0.0
    assert ch["comment"]["status"] == "OK"
    assert ch["comment"]["coverage"] == 0.25
    # comment: 1 present cell of 4 -> cell_coverage 0.25, stock_coverage 0.25
    # (1 of 4 stocks), date_coverage 1.0 (the single date has >=1 present).
    assert ch["comment"]["cell_coverage"] == 0.25
    assert ch["comment"]["stock_coverage"] == 0.25
    assert ch["comment"]["date_coverage"] == 1.0
    assert ch["comment"]["cells"] == 4
    assert ch["concept"]["status"] == "MISSING"


def test_prebuilt_metrics_stock_vs_cell_vs_date(tp):
    """§T4 contract matrix: the three named metrics must DIFFER when presence
    spans stocks × dates unevenly.  stock_coverage = fraction of stocks with >=1
    present cell (mask.any(axis=1).mean()), cell_coverage = fraction of cells
    (mask.mean()), date_coverage = fraction of dates with >=1 present cell
    (mask.any(axis=0).mean())."""
    M = np.zeros((4, 3), dtype=np.float32)
    M[0, 0] = 1.0   # stock 0, day 0
    M[1, 1] = 1.0   # stock 1, day 1
    ch = tp._prebuilt_channel_coverage(
        _panel_nt(_FLAGS, {"has_news": M}))
    s = ch["sentiment"]
    assert s["stock_coverage"] == 0.5      # 2 of 4 stocks have >=1 cell
    assert s["cell_coverage"] == 0.1667    # 2 of 12 cells
    assert s["date_coverage"] == 0.6667    # days 0 and 1 of 3 have >=1 cell
    assert s["coverage"] == s["cell_coverage"]  # legacy alias preserved
    assert s["cells"] == 12


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
                "etf_flow", "industry", "market_env"}
    assert set(manifest) == expected

    assert manifest["sentiment"]["required"] is True
    for name, e in manifest.items():
        assert e["status"] in {"MISSING", "FAILED", "PARTIAL", "OK"}, name
        if name != "etf_flow":
            assert e["required"] is (name == "sentiment")


# ── live broadcast probes (industry / market_env date coverage) ────────

def test_load_aux_data_broadcast_date_coverage(tp, tmp_path):
    """industry / market_env are MARKET-WIDE broadcasts (same value for every
    stock per date) — stock coverage is vacuous; DATE coverage (distinct dates
    in [start,end] over trading days in range) is the meaningful metric."""
    from stoke_ml.data.calendar import get_research_calendar

    data_dir = str(tmp_path / "data")
    cal = get_research_calendar(strict=True, data_dir=data_dir)
    lo, hi = pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date()
    tdays = cal.get_trading_days(lo, hi)
    assert tdays  # 2024 is well inside the verified calendar window

    # industry: every trading day present -> date_coverage == 1.0 (DatetimeIndex)
    ind_dir = os.path.join(data_dir, "a_shares", "industry")
    os.makedirs(ind_dir, exist_ok=True)
    pd.DataFrame(
        np.random.RandomState(0).randn(len(tdays), 3),
        index=pd.DatetimeIndex(tdays),
    ).to_parquet(os.path.join(ind_dir, "industry_returns.parquet"))

    # market_env: first half of trading days -> date_coverage == 0.5 (date col)
    half = len(tdays) // 2
    env_dir = os.path.join(data_dir, "a_shares", "market_breadth")
    os.makedirs(env_dir, exist_ok=True)
    pd.DataFrame({"date": tdays[:half], "dummy": 1.0}).to_parquet(
        os.path.join(env_dir, "market_env_daily.parquet"))

    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"industry", "market_env"})
    assert manifest["industry"]["status"] == "OK"
    assert manifest["industry"]["date_coverage"] == 1.0
    assert manifest["industry"]["stock_coverage"] == 1.0
    assert manifest["industry"]["loaded_stocks"] is None  # broadcast, not per-stock
    assert manifest["market_env"]["status"] == "OK"
    assert manifest["market_env"]["date_coverage"] == round(half / len(tdays), 4)
    assert manifest["market_env"]["stock_coverage"] == 1.0
    assert manifest["market_env"]["required"] is True
    assert manifest["industry"]["required"] is True


def test_load_aux_data_broadcast_missing_files(tp, tmp_path):
    """A broadcast channel with no file on disk is MISSING with 0.0 date/stock
    coverage (so a required broadcast channel with no data aborts the gate)."""
    data_dir = str(tmp_path / "data")
    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"industry", "market_env"})
    assert manifest["industry"]["status"] == "MISSING"
    assert manifest["industry"]["date_coverage"] == 0.0
    assert manifest["industry"]["stock_coverage"] == 0.0
    assert manifest["market_env"]["status"] == "MISSING"
    assert manifest["market_env"]["date_coverage"] == 0.0
    assert manifest["market_env"]["stock_coverage"] == 0.0
