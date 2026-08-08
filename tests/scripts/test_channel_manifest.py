"""Unit tests for the channel-coverage manifest.

train_panel.py must be able to tell "this aux channel is absent from disk"
(MISSING) from "the channel broke while loading" (FAILED), so a storage schema
change or missing dir can no longer silently zero out a data dimension.
"""
import importlib.util
import json
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


# ── broadcast probe FAILED degradation (whole-body try, §T4 review) ────

def test_probe_broadcast_dates_misnamed_datetimeindex_returns_failed(tp, tmp_path):
    """A broadcast parquet whose DatetimeIndex is named other than 'date'/None
    must degrade to FAILED, not raise a KeyError out of load_aux_data — the
    reset_index().rename(columns={"index": "date"}) only renames an UNNAMED
    index; a named index ('idx') survives as a misnamed column and the probe
    used to raise on df['date']."""
    data_dir = str(tmp_path / "data")
    env_dir = os.path.join(data_dir, "a_shares", "market_breadth")
    os.makedirs(env_dir, exist_ok=True)
    df = pd.DataFrame(
        np.random.RandomState(0).randn(3, 2),
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"],
                               name="idx"),
        columns=["a", "b"])
    df.to_parquet(os.path.join(env_dir, "market_env_daily.parquet"))
    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"market_env"})
    assert manifest["market_env"]["status"] == "FAILED"
    assert manifest["market_env"]["date_coverage"] == 0.0
    assert manifest["market_env"]["stock_coverage"] == 0.0


def test_probe_broadcast_dates_unparseable_dates_returns_failed(tp, tmp_path):
    """A broadcast parquet whose date column cannot be parsed must degrade to
    FAILED, not raise out of load_aux_data — pd.to_datetime on garbage raises
    DateParseError, which the whole-body try now converts to FAILED."""
    data_dir = str(tmp_path / "data")
    env_dir = os.path.join(data_dir, "a_shares", "market_breadth")
    os.makedirs(env_dir, exist_ok=True)
    pd.DataFrame({"date": ["not-a-date", "also-bad"], "dummy": [1.0, 2.0]}).to_parquet(
        os.path.join(env_dir, "market_env_daily.parquet"))
    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"market_env"})
    assert manifest["market_env"]["status"] == "FAILED"
    assert manifest["market_env"]["date_coverage"] == 0.0


# ── etf_flow date_coverage is WINDOW-FILTERED (§T4 review) ────────────

def test_load_aux_data_etf_flow_date_coverage_window_filtered(tp, tmp_path):
    """Regression: etf_flow date_coverage must count dates INSIDE [start, end]
    only.  Pre-fix the numerator counted every distinct date across all sector
    files (full history), so data entirely outside the window still reported
    date_coverage ~1.0 — making the 0.80 contract vacuous (could falsely pass
    with zero in-window data)."""
    from stoke_ml.data.calendar import get_research_calendar

    data_dir = str(tmp_path / "data")
    cal = get_research_calendar(strict=True, data_dir=data_dir)
    lo, hi = pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date()
    tdays = cal.get_trading_days(lo, hi)
    etf_dir = os.path.join(data_dir, "a_shares", "etf_flow")
    os.makedirs(etf_dir, exist_ok=True)
    # Sector data entirely OUTSIDE the probe window (2022) — in-window count 0.
    outside = pd.date_range("2022-01-03", periods=len(tdays), freq="B")
    pd.DataFrame({
        "date": outside,
        "etf_flow_sum": [1.0] * len(outside),
        "etf_amount_sum": [1.0] * len(outside),
    }).to_parquet(os.path.join(etf_dir, "sector_banks.parquet"))

    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"etf_flow"})
    assert manifest["etf_flow"]["status"] == "OK"
    assert manifest["etf_flow"]["date_coverage"] == 0.0
    assert manifest["etf_flow"]["stock_coverage"] == 1.0  # vacuous broadcast flag


def test_load_aux_data_etf_flow_date_coverage_in_window(tp, tmp_path):
    """With sector data on exactly the in-window trading days, etf_flow
    date_coverage is the in-window distinct-date fraction (half the trading
    days here) — NOT 1.0 just because the aggregated broadcast exists."""
    from stoke_ml.data.calendar import get_research_calendar

    data_dir = str(tmp_path / "data")
    cal = get_research_calendar(strict=True, data_dir=data_dir)
    lo, hi = pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date()
    tdays = cal.get_trading_days(lo, hi)
    half = len(tdays) // 2
    etf_dir = os.path.join(data_dir, "a_shares", "etf_flow")
    os.makedirs(etf_dir, exist_ok=True)
    pd.DataFrame({
        "date": tdays[:half],
        "etf_flow_sum": [1.0] * half,
        "etf_amount_sum": [1.0] * half,
    }).to_parquet(os.path.join(etf_dir, "sector_banks.parquet"))

    _, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"etf_flow"})
    assert manifest["etf_flow"]["status"] == "OK"
    assert manifest["etf_flow"]["date_coverage"] == round(half / len(tdays), 4)


# ── _trading_day_count fallback + _date_coverage_fraction clip (§T4) ───

def test_trading_day_count_falls_back_on_strict_calendar_overflow(tp, monkeypatch, caplog):
    """A strict-calendar range overflow past verified_until must fall back to the
    ~weekday estimate (no crash) instead of propagating — and the fallback is
    logged at debug so a corrupted-artifact error is not silently masked."""
    import logging

    class _StrictCal:
        def get_trading_days(self, lo, hi):
            raise ValueError("2027-01-01 is past verified_until 2026-12-31")

    monkeypatch.setattr(
        "scripts.production.train_panel_panel.get_research_calendar",
        lambda *a, **k: _StrictCal())
    # 2027-01-01..2027-03-31 = 89 days -> round(89 * 5/7) = 64.
    with caplog.at_level(logging.DEBUG):
        n = tp._trading_day_count("data", "2027-01-01", "2027-03-31")
    assert n == max(1, int(round(89 * 5 / 7)))
    assert any("weekday estimate" in m for m in caplog.messages)


def test_date_coverage_fraction_clips_at_1_0(tp, tmp_path):
    """A broadcast file written with resample('D') counts weekend dates in the
    numerator while the denominator is trading days only, so the raw ratio can
    exceed 1.0 — the fraction must clip at 1.0, never report >1.0 coverage."""
    from stoke_ml.data.calendar import get_research_calendar

    data_dir = str(tmp_path / "data")
    cal = get_research_calendar(strict=True, data_dir=data_dir)
    n_trading = len(cal.get_trading_days(
        pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date()))
    assert n_trading < 300  # 2024 has ~365 calendar days, ~242 trading days
    assert tp._date_coverage_fraction(
        300, data_dir, "2024-01-01", "2024-12-31") == 1.0


# ── §T8 provider-era coverage probe (no_event vs not_observed) ─────────

def _gold_manifest(data_dir, channel, code, win, retrieved, gaps=()):
    """Write a synthetic gold asset manifest for a stock; return its path."""
    base = os.path.join(data_dir, "a_shares")
    if channel == "sentiment":
        d = os.path.join(base, "sentiment", "2024", "01")
    elif channel == "guba":
        d = os.path.join(base, "guba_sentiment")
    elif channel == "comment":
        d = os.path.join(base, "comment_sentiment")
    else:
        d = os.path.join(base, "announcements", "sentiment")
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, f"{code}.parquet.manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({
            "provider_available_start": win[0],
            "provider_available_end": win[1],
            "retrieved_ranges": list(retrieved),
            "known_gaps": list(gaps),
        }, f)
    return mp


def test_stock_era_coverage_reads_gold_manifest(tp, tmp_path):
    """§T8 acceptance (1): a provider window FULLY covered by retrieved_ranges
    → era_coverage 1.0 — a day inside a retrieved range is observed whether or
    not the stock had an event that day."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "sentiment", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])
    cov, no = tp._stock_era_coverage(data_dir, "sentiment", "000001")
    assert no is False
    assert cov == 1.0


def test_stock_era_coverage_partial_retrieval_fraction(tp, tmp_path):
    """§T8 acceptance (3): a window 60% retrieved → era_coverage 0.6."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "guba", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-06"]])
    cov, no = tp._stock_era_coverage(data_dir, "guba", "000001")
    assert no is False
    assert cov == 0.6


def test_stock_era_coverage_no_gold_manifest_not_observed(tp, tmp_path):
    """No gold manifest → not_observed (era_coverage None), never counted as 0."""
    data_dir = str(tmp_path / "data")
    cov, no = tp._stock_era_coverage(data_dir, "sentiment", "000001")
    assert no is True
    assert cov is None


def test_stock_era_coverage_finds_partitioned_sentiment(tp, tmp_path):
    """sentiment gold lives in year/month partitions — the probe globs them."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "sentiment", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])
    assert len(tp._gold_manifest_paths(data_dir, "sentiment", "000001")) == 1
    cov, no = tp._stock_era_coverage(data_dir, "sentiment", "000001")
    assert no is False
    assert cov == 1.0


def test_probe_era_coverage_excludes_not_observed_from_numerator(tp, tmp_path):
    """§T8 acceptance (2): a not_observed stock is EXCLUDED from the era
    numerator and counted separately — never pulled down as a zero."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "sentiment", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])   # 1.0
    _gold_manifest(data_dir, "sentiment", "000002",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-06"]])   # 0.6
    # 000003 has no gold manifest -> not_observed
    mean, n_obs, n_not = tp._probe_era_coverage(
        data_dir, "sentiment", ["000001", "000002", "000003"])
    assert mean == pytest.approx(0.8)
    assert n_obs == 2
    assert n_not == 1


def test_probe_era_coverage_none_when_zero_observable(tp, tmp_path):
    """Zero era-observable stocks → mean None (the gate treats the channel as
    UNPROBEABLE — a formal run must not proceed on an unobserved channel)."""
    data_dir = str(tmp_path / "data")
    mean, n_obs, n_not = tp._probe_era_coverage(
        data_dir, "sentiment", ["000001", "000002"])
    assert mean is None
    assert n_obs == 0
    assert n_not == 2


def test_merge_era_coverage_adds_fields_only_to_present_channels(tp, tmp_path):
    """The merge stamps era fields on the era-capable channels ALREADY present
    and leaves non-era channels and absent channels untouched."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "sentiment", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])
    _gold_manifest(data_dir, "sentiment", "000002",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-06"]])
    _gold_manifest(data_dir, "guba", "000002",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])
    manifest = {
        "sentiment": {"status": "OK", "coverage": 1.0},
        "guba": {"status": "OK", "coverage": 1.0},
        "margin": {"status": "OK", "coverage": 1.0},
    }
    tp._merge_era_coverage(manifest, data_dir, ["000001", "000002", "000003"])
    assert manifest["sentiment"]["era_coverage"] == pytest.approx(0.8)
    assert manifest["sentiment"]["era_observable_stocks"] == 2
    assert manifest["sentiment"]["era_not_observed_stocks"] == 1
    assert manifest["guba"]["era_coverage"] == 1.0
    assert manifest["guba"]["era_observable_stocks"] == 1
    assert "era_coverage" not in manifest["margin"]  # not era-capable
    assert "announcement" not in manifest  # absent channel not created


def test_merge_era_coverage_empty_manifest_stays_empty(tp, tmp_path):
    """A --no-aux build's empty channel manifest stays empty — a channel not
    actually in the panel is not probed."""
    data_dir = str(tmp_path / "data")
    _gold_manifest(data_dir, "sentiment", "000001",
                   ("2024-01-01", "2024-01-10"), [["2024-01-01", "2024-01-10"]])
    manifest = {}
    tp._merge_era_coverage(manifest, data_dir, ["000001"])
    assert manifest == {}


def test_merge_era_coverage_force_false_migrates_pre_t3_store(tp, monkeypatch):
    """§v18-3 (Important 2): a pre-T3 store persisted ``era_coverage`` but NOT
    ``era_observable_stock_fraction`` — on the force=False store-LOAD path that
    entry is INCOMPLETE for the composite gate, so the merge must RE-PROBE it
    and fill the fraction (denominator = the requested universe, not just the
    era-observable stocks)."""
    calls = []

    def _spy(data_dir, ch, stock_list):
        calls.append((data_dir, ch, list(stock_list)))
        return 0.8, 2, 1  # (mean, n_obs, n_not)

    monkeypatch.setattr(
        "scripts.production.train_panel_panel._probe_era_coverage", _spy)
    manifest = {"sentiment": {"status": "OK", "era_coverage": 1.0}}
    out = tp._merge_era_coverage(
        manifest, "data", ["000001", "000002", "000003"], force=False)
    assert len(calls) == 1                      # the probe RAN
    assert calls[0][1] == "sentiment"
    assert calls[0][2] == ["000001", "000002", "000003"]
    assert out["sentiment"]["era_coverage"] == pytest.approx(0.8)
    assert out["sentiment"]["era_observable_stocks"] == 2
    assert out["sentiment"]["era_not_observed_stocks"] == 1
    assert out["sentiment"]["era_observable_stock_fraction"] == pytest.approx(2 / 3)


def test_merge_era_coverage_force_false_skips_complete_entry(tp, monkeypatch):
    """§v18-3 (Important 2): a store that persisted BOTH ``era_coverage`` AND
    ``era_observable_stock_fraction`` is complete for the composite gate — the
    force=False store-LOAD path skips re-probing entirely (the probe must NOT
    run) and leaves the persisted values intact."""
    def _boom(*a, **k):
        raise AssertionError(
            "_probe_era_coverage must not run when the composite entry is "
            "already present")

    monkeypatch.setattr(
        "scripts.production.train_panel_panel._probe_era_coverage", _boom)
    manifest = {"sentiment": {"status": "OK", "era_coverage": 0.95,
                              "era_observable_stock_fraction": 0.90}}
    out = tp._merge_era_coverage(manifest, "data", ["000001", "000002"],
                                 force=False)
    assert out["sentiment"]["era_coverage"] == 0.95
    assert out["sentiment"]["era_observable_stock_fraction"] == 0.90
    assert "era_observable_stocks" not in out["sentiment"]  # untouched


def test_era_capable_channels_derives_from_profile_contracts(tp, monkeypatch):
    """§T8 review (Important 2): the era-probe channel set is DERIVED from the
    feature profiles' era_coverage contracts — adding an era contract to a
    profile automatically adds that channel to the probe, so there is NO
    hard-coded list to keep in sync.  A stock_coverage-contracted channel
    (comment) is not probed until it declares era_coverage; the union over all
    profiles keeps the live-default (profile_name=None) probe non-empty."""
    import dataclasses
    from stoke_ml.config.feature_profile import (
        FEATURE_PROFILES,
        CoverageContract,
    )

    base = tp._era_capable_channels()
    assert "sentiment" in base and "guba" in base
    assert base == frozenset({"sentiment", "guba"})   # only era-contracted
    assert "comment" not in base                      # stock_coverage, not era

    # Give comment an era_coverage contract → the probe must now cover it.
    updated = dataclasses.replace(
        FEATURE_PROFILES["headline_v1"],
        coverage_contracts={
            **FEATURE_PROFILES["headline_v1"].coverage_contracts,
            "comment": CoverageContract("era_coverage", 0.90),
        },
    )
    monkeypatch.setattr(
        "stoke_ml.config.feature_profile.FEATURE_PROFILES",
        {"headline_v1": updated})
    assert "comment" in tp._era_capable_channels()
    assert "sentiment" in tp._era_capable_channels()


def test_stock_era_coverage_first_existing_partition_wins(tp, tmp_path):
    """§T8 review (Minor 5): a stock with gold manifests in MULTIPLE year/month
    partitions is probed from the FIRST existing candidate (sorted glob order) —
    every partition was stamped from the SAME downloader manifest, so the
    "first existing path wins" invariant holds (never a merge / full scan)."""
    data_dir = str(tmp_path / "data")
    base = os.path.join(data_dir, "a_shares", "sentiment")
    # 2024/01 is fully retrieved (1.0); 2024/02 half-retrieved (0.6).  Both are
    # legitimate candidates; the sorted glob picks 2024/01 first.
    for y, m, retrieved in (
        ("2024", "01", [["2024-01-01", "2024-01-10"]]),
        ("2024", "02", [["2024-01-01", "2024-01-06"]]),
    ):
        d = os.path.join(base, y, m)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "000001.parquet.manifest.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "provider_available_start": "2024-01-01",
                "provider_available_end": "2024-01-10",
                "retrieved_ranges": retrieved,
                "known_gaps": [],
            }, f)
    paths = tp._gold_manifest_paths(data_dir, "sentiment", "000001")
    assert len(paths) == 2
    assert os.path.basename(os.path.dirname(paths[0])) == "01"  # first is 2024/01
    cov, no = tp._stock_era_coverage(data_dir, "sentiment", "000001")
    assert no is False
    assert cov == 1.0   # 2024/01's full retrieval — NOT a merge with 2024/02
