"""§八/T5: formal ``build_market_env.py`` — the market_env price/account split.

The formal production builder replaces the archived
``scripts/maintenance/legacy/_preprocess_market_env.py``.  It writes the SAME
consumer-facing parquet (``a_shares/market_breadth/market_env_daily.parquet``,
7 columns on a DatetimeIndex) but declares the price/account split honestly:

  - PRICE part (high_low_ratio / market_adv_ratio / market_turnover_z):
    same-day trade data → ``pit_alignment="verified"``.
  - ACCOUNT part (mkt_cap_total_z / avg_account_cap_z / investor_new_num /
    investor_new_z): monthly account statistics → ``"proxy"`` when the raw
    source records no real publish date (the shipped account_stats.parquet
    does not), ``"verified"`` when a real publish-date column is present.

The single file's manifest carries the STRICTER of the two labels
(``vintage_pit="proxy"`` via channel_vintage); the per-part labels are recorded
in the manifest ``parts`` field and in feature_profile's
``MARKET_ENV_PRICE_COLS`` / ``MARKET_ENV_ACCOUNT_COLS`` split.  headline_v1
requires the PRICE part (the ``market_env`` channel) and excludes the account
part while it is proxy (ablation-only, mirroring ``use_topic``).

Synthetic data on ``tmp_path``, no network, no real market data.
"""
import importlib.util
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "production", "build_market_env.py")


@pytest.fixture(scope="module")
def bme():
    spec = importlib.util.spec_from_file_location("build_market_env_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── synthetic inputs ──────────────────────────────────────────────────────

def _write_account_stats(tmp_path, *, publish_col=None):
    """monthly account_stats.parquet; optional real publish-date column."""
    br = tmp_path / "a_shares" / "market_breadth"
    br.mkdir(parents=True, exist_ok=True)
    data = {
        "数据日期": ["2015-04", "2015-05", "2015-06"],
        "新增投资者-数量": [497.53, 415.87, 464.22],
        "沪深总市值": [563491.335, 627465.456, 584573.654],
        "沪深户均市值": [69.4956, 73.6062, 65.0300],
    }
    if publish_col:
        data[publish_col] = ["2015-05-10", "2015-06-12", "2015-07-12"]
    pd.DataFrame(data).to_parquet(br / "account_stats.parquet", index=False)


def _write_highs_lows(tmp_path):
    br = tmp_path / "a_shares" / "market_breadth"
    br.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": ["2024-07-03", "2024-07-04", "2024-07-05"],
        "close": [2982.38, 2957.57, 2949.93],
        "high20": [202, 74, 135],
        "low20": [886, 2145, 579],
    }).to_parquet(br / "highs_lows.parquet", index=False)


def _write_industry(tmp_path):
    """§v18-5: the REAL market_adv_ratio upstream is download_industry_ranking.py's
    a_shares/industry_ranking.parquet (date/sector_code/change_pct), not the
    legacy industry/industry_ranking_computed.parquet (date/ind_return)."""
    daily = tmp_path / "a_shares"
    daily.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "sector_code": ["SEC0000", "SEC0001", "SEC0000", "SEC0001"],
        "change_pct": [0.01, -0.02, 0.00, 0.03],
    })
    rows.to_parquet(daily / "industry_ranking.parquet", index=False)
    return daily


def _write_daily(tmp_path):
    daily = tmp_path / "a_shares" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for code in ("000001", "000002"):
        pd.DataFrame({
            "date": pd.to_datetime(["2024-07-03", "2024-07-04", "2024-07-05"]),
            "amount": [1e9, 1.1e9, 1.2e9],
        }).to_parquet(daily / f"{code}.parquet", index=False)


def _build_full(tmp_path, bme, *, publish_col=None):
    _write_account_stats(tmp_path, publish_col=publish_col)
    _write_highs_lows(tmp_path)
    _write_industry(tmp_path)
    _write_daily(tmp_path)
    return bme.build_market_env(str(tmp_path))


# ── (a) price part: verified PIT declaration ──────────────────────────────

def test_price_part_declared_verified(bme, tmp_path):
    """The price part (same-day trade data) is declared ``verified`` and its
    columns land in the output."""
    df, parts = _build_full(tmp_path, bme)
    assert parts["price"]["pit_alignment"] == "verified"
    assert set(parts["price"]["columns"]) == {
        "high_low_ratio", "market_adv_ratio", "market_turnover_z"}
    for c in ("high_low_ratio", "market_adv_ratio", "market_turnover_z"):
        assert c in df.columns


def test_build_market_env_asserts_full_price_cols(bme, tmp_path):
    """§v18-5: a builder that cannot produce the full MARKET_ENV_PRICE_COLS set
    must FAIL, not write a partial asset with a valid manifest."""
    _write_industry(tmp_path)
    with pytest.raises(ValueError):
        bme.build_market_env(str(tmp_path))


def test_build_market_env_price_cols_from_real_upstream(bme, tmp_path):
    """§v18-5: with all three price sources present, market_adv_ratio is computed
    from a_shares/industry_ranking.parquet (fraction of sectors with change_pct>0
    per date) and the full PRICE column set is asserted present."""
    daily = _write_industry(tmp_path)
    (daily / "market_breadth").mkdir(exist_ok=True)
    hl = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                       "high20": [10.0, 12.0], "low20": [8.0, 9.0]})
    hl.to_parquet(daily / "market_breadth" / "highs_lows.parquet", index=False)
    d = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                      "amount": [1e6, 2e6], "open": [10.0, 10.5],
                      "high": [11.0, 11.5], "low": [9.0, 9.5],
                      "close": [10.5, 11.0], "volume": [1e5, 1.2e5]})
    (daily / "daily").mkdir(exist_ok=True)
    d.to_parquet(daily / "daily" / "000001.parquet", index=False)
    df, parts = bme.build_market_env(str(tmp_path))
    assert set(bme.MARKET_ENV_PRICE_COLS) <= set(df.columns)
    # 2024-01-02: sector0 +0.01 (adv), sector1 -0.02 (decl) → 0.5
    assert abs(df.loc["2024-01-02", "market_adv_ratio"] - 0.5) < 1e-9


# ── (b) account part: proxy when no real publish date ─────────────────────

def test_account_part_proxy_without_publish_date(bme, tmp_path):
    """The shipped account_stats.parquet records only the data-MONTH label, so
    the real publish date is not determinable → the account part is PROXY."""
    df, parts = _build_full(tmp_path, bme)
    assert parts["account"]["pit_alignment"] == "proxy"
    # the proxy fallback is the legacy month-end approximation: "2015-04"→04-28
    assert pd.Timestamp("2015-04-28") in df.index


# ── (c) account part: real publish date when available ────────────────────

def test_account_part_verified_with_real_publish_date(bme, tmp_path):
    """When the raw source records a real publish-date column, the account part
    uses THOSE dates and is declared ``verified`` — never the proxy fallback."""
    df, parts = _build_full(tmp_path, bme, publish_col="发布日期")
    assert parts["account"]["pit_alignment"] == "verified"
    # the account part's first effective date follows the publish date, and the
    # month-end proxy date is NOT in the index at all
    assert pd.Timestamp("2015-05-10") in df.index
    assert pd.Timestamp("2015-04-28") not in df.index


# ── (d) manifest write + re-validation ────────────────────────────────────

def test_manifest_written_and_formal_read_passes(bme, tmp_path):
    """The builder writes the parquet + a valid MARKET_ENV_ASSET manifest; the
    formal read (require_valid_manifest=True) passes; the manifest vintage_pit
    carries the STRICTER label (proxy) while the ``parts`` record the finer
    price=verified / account=proxy split."""
    from stoke_ml.data.asset_contract import (
        check_asset_read, validate_asset_manifest)
    from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET

    _write_account_stats(tmp_path)
    _write_highs_lows(tmp_path)
    _write_industry(tmp_path)
    _write_daily(tmp_path)
    data_dir = str(tmp_path)
    df, parts = bme.build_market_env(data_dir)
    path = bme.write_market_env(data_dir, df, parts)

    assert os.path.isfile(path)
    report = validate_asset_manifest(path, MARKET_ENV_ASSET)
    assert report["ok"], report["mismatches"]
    # channel-level vintage = the STRICTER of the two parts (proxy)
    assert report["manifest"]["vintage_pit"] == "proxy"
    assert report["manifest"]["parts"]["price"]["pit_alignment"] == "verified"
    assert report["manifest"]["parts"]["account"]["pit_alignment"] == "proxy"
    # formal read of the re-read file passes (schema_hash survives round-trip)
    reread = pd.read_parquet(path)
    check_asset_read(path, MARKET_ENV_ASSET, reread, require_valid_manifest=True)


# ── backward-compat consumer schema ───────────────────────────────────────

def test_output_schema_backward_compatible(bme, tmp_path):
    """The consumer-facing file is unchanged: DatetimeIndex named ``date`` with
    exactly the 7 MARKET_ENV_COLS the feature layer merges."""
    from stoke_ml.features.aux_cols import MARKET_ENV_COLS
    df, _ = _build_full(tmp_path, bme)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert set(df.columns) == set(MARKET_ENV_COLS)


# ── (e) headline_v1 required-set reflects the split ───────────────────────

def test_headline_v1_required_set_reflects_split():
    """headline_v1 under revision-safe keeps the PRICE part required (the
    ``market_env`` channel = the file that carries it) and excludes the ACCOUNT
    part while it is proxy (ablation-only, its own channel so the generic scrub
    drops it whenever use_market_env_account is OFF)."""
    from stoke_ml.config.feature_profile import (
        CHANNEL_COLUMNS,
        FEATURE_PROFILES,
        MARKET_ENV_ACCOUNT_COLS,
        MARKET_ENV_ACCOUNT_PIT,
        MARKET_ENV_PRICE_COLS,
        market_env_required_columns,
    )
    p = FEATURE_PROFILES["headline_v1"]
    assert "market_env" in p.required_channels          # price part required
    assert "market_env_account" not in p.required_channels  # account part ablation-only
    assert MARKET_ENV_ACCOUNT_PIT == "proxy"            # account part proxy
    # the account part is excluded from the required sub-set while proxy
    assert market_env_required_columns("headline_v1") == MARKET_ENV_PRICE_COLS
    # the split partitions the 7-column file exactly, no overlap, and the
    # market_env channel is the PRICE-ONLY part (verified default consumption)
    assert CHANNEL_COLUMNS["market_env"] == MARKET_ENV_PRICE_COLS
    assert CHANNEL_COLUMNS["market_env_account"] == MARKET_ENV_ACCOUNT_COLS
    assert MARKET_ENV_PRICE_COLS | MARKET_ENV_ACCOUNT_COLS == frozenset({
        "high_low_ratio", "mkt_cap_total_z", "avg_account_cap_z",
        "investor_new_num", "investor_new_z", "market_adv_ratio",
        "market_turnover_z",
    })
    assert MARKET_ENV_PRICE_COLS.isdisjoint(MARKET_ENV_ACCOUNT_COLS)


def test_empty_input_dir_fails_loudly(bme, tmp_path, monkeypatch):
    """A formal builder must not silently write an empty rows:0 manifest: with
    no usable inputs the CLI path raises instead of producing an empty asset.
    §v18-5: an empty dir now trips the PRICE-atomicity ValueError (all three
    MARKET_ENV_PRICE_COLS missing) BEFORE main()'s empty-panel SystemExit, so the
    ValueError propagates out of the CLI — still an honest loud failure."""
    import sys
    empty = tmp_path / "empty_data"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "build_market_env.py", "--data-dir", str(empty)])
    with pytest.raises(ValueError) as ei:
        bme.main()
    assert "PRICE columns missing" in str(ei.value)
    assert not os.path.exists(os.path.join(
        str(empty), "a_shares", "market_breadth", "market_env_daily.parquet"))


# ── legacy script superseded marker ───────────────────────────────────────

# ── quality: unreadable source files are warned, never silently skipped ─────

def test_turnover_scan_logs_unreadable_file(bme, tmp_path, caplog):
    """A corrupt daily parquet is skipped with a WARNING (not silently) so an
    incomplete turnover series leaves a diagnosable trace."""
    import logging
    daily = tmp_path / "a_shares" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    # one valid file (the glob finds something) + one corrupt file
    pd.DataFrame({
        "date": pd.to_datetime(["2024-07-03", "2024-07-04"]),
        "amount": [1e9, 1.1e9],
    }).to_parquet(daily / "000001.parquet", index=False)
    (daily / "000002.parquet").write_bytes(b"NOT A PARQUET FILE")
    with caplog.at_level(logging.WARNING):
        turn = bme.build_turnover_daily(str(tmp_path / "a_shares"))
    assert any("000002.parquet" in r.message for r in caplog.records), (
        "the unreadable file must be named in a warning log record")
    # the valid file's turnover still contributes to the series
    assert not turn.empty


# ── quality: account_stats with an unexpected schema degrades, never crashes ─

def test_account_stats_missing_month_label_degrades_gracefully(bme, tmp_path):
    """When account_stats exists but has NEITHER a real publish-date column NOR
    the 数据日期 month-label column, the account part degrades to an empty frame
    (proxy) — NOT a KeyError.  The PRICE part still builds."""
    br = tmp_path / "a_shares" / "market_breadth"
    br.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "新增投资者-数量": [497.53],
        "沪深总市值": [563491.335],
        "沪深户均市值": [69.4956],
    }).to_parquet(br / "account_stats.parquet", index=False)
    _write_highs_lows(tmp_path)
    _write_industry(tmp_path)
    _write_daily(tmp_path)
    df, parts = bme.build_market_env(str(tmp_path))
    assert parts["account"]["pit_alignment"] == "proxy"
    for c in bme.MARKET_ENV_ACCOUNT_COLS:
        assert c not in df.columns, f"account col {c!r} should be absent"
    # the price part is unaffected
    for c in bme.MARKET_ENV_PRICE_COLS:
        assert c in df.columns, f"price col {c!r} missing"


# ── quality: state-channel gaps forward-fill, never zero-fill ──────────────

def test_account_part_forward_filled_not_zero_filled(bme, tmp_path):
    """The ACCOUNT part is a state channel: a missing monthly value means
    "unchanged", so a date AFTER the last monthly observation must carry the
    last known value forward — NOT a spurious 0.0.  The panel extends into the
    PRICE part's modern dates (2024) beyond the account part's 2015 months."""
    _write_account_stats(tmp_path)   # monthly 2015-04..2015-06
    _write_highs_lows(tmp_path)      # 2024-07
    _write_industry(tmp_path)
    _write_daily(tmp_path)
    df, _ = bme.build_market_env(str(tmp_path))
    assert df.index.max() >= pd.Timestamp("2024-07-03")
    # investor_new_num on a 2024 date = the last known monthly value, not 0.0
    last_2015 = df.loc[df.index <= "2015-12-31", "investor_new_num"].dropna().iloc[-1]
    val_2024 = df.loc["2024-07-03", "investor_new_num"]
    assert val_2024 == last_2015
    assert val_2024 != 0.0


def test_price_part_gaps_are_nan_not_zero(bme, tmp_path):
    """A date one PRICE source lacks but another covers stays NaN in the file so
    the consumer's _batch_fill_shift (policy='ffill') carries the last breadth —
    a hard 0.0 would bypass that state-channel forward-fill."""
    _write_account_stats(tmp_path)
    _write_industry(tmp_path)   # 2024-07-03..05
    _write_daily(tmp_path)
    # highs_lows covers ONLY 2024-07-03 (missing 07-04 / 07-05)
    br = tmp_path / "a_shares" / "market_breadth"
    pd.DataFrame({
        "date": ["2024-07-03"], "close": [2982.38],
        "high20": [202], "low20": [886],
    }).to_parquet(br / "highs_lows.parquet", index=False)
    df, _ = bme.build_market_env(str(tmp_path))
    # the union index includes 07-04 (industry + daily cover it); high_low_ratio
    # is a genuine gap there → NaN, not 0.0
    assert "2024-07-04" in df.index
    assert pd.isna(df.loc["2024-07-04", "high_low_ratio"]), (
        "a price gap must stay NaN for the consumer's ffill, not be zero-filled")


def test_legacy_script_marked_superseded():
    """The archived script keeps its behavior but is marked superseded so a
    future agent does not treat it as the canonical builder."""
    legacy = os.path.join(ROOT, "scripts", "maintenance", "legacy",
                          "_preprocess_market_env.py")
    with open(legacy, encoding="utf-8") as f:
        head = f.read(400)
    assert "build_market_env.py" in head
    assert "SUPERSEDED" in head
