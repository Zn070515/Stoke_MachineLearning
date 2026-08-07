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
    ind = tmp_path / "a_shares" / "industry"
    ind.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": ["2024-07-03", "2024-07-04", "2024-07-05"],
        "ind_return": [0.01, -0.02, 0.03],
    }).to_parquet(ind / "industry_ranking_computed.parquet", index=False)


def _write_daily(tmp_path):
    daily = tmp_path / "a_shares" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for code in ("000001", "000002"):
        pd.DataFrame({
            "date": pd.to_datetime(["2024-07-03", "2024-07-04", "2024-07-05"]),
            "amount": [1e9, 1.1e9, 1.2e9],
        }).to_parquet(daily / f"{code}.parquet", index=False)


def _build_full(tmp_path, bme, *, skip_turnover=False, publish_col=None):
    _write_account_stats(tmp_path, publish_col=publish_col)
    _write_highs_lows(tmp_path)
    _write_industry(tmp_path)
    _write_daily(tmp_path)
    return bme.build_market_env(str(tmp_path), skip_turnover=skip_turnover)


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
    no usable inputs the CLI path raises instead of producing an empty asset."""
    import sys
    empty = tmp_path / "empty_data"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "build_market_env.py", "--data-dir", str(empty), "--skip-turnover"])
    with pytest.raises(SystemExit) as ei:
        bme.main()
    assert "EMPTY panel" in str(ei.value)
    assert not os.path.exists(os.path.join(
        str(empty), "a_shares", "market_breadth", "market_env_daily.parquet"))


# ── legacy script superseded marker ───────────────────────────────────────

def test_legacy_script_marked_superseded():
    """The archived script keeps its behavior but is marked superseded so a
    future agent does not treat it as the canonical builder."""
    legacy = os.path.join(ROOT, "scripts", "maintenance", "legacy",
                          "_preprocess_market_env.py")
    with open(legacy, encoding="utf-8") as f:
        head = f.read(400)
    assert "build_market_env.py" in head
    assert "SUPERSEDED" in head
