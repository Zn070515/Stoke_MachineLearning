"""T1 (§十一): the 3 production downloaders define data_dir and treat a
run-manifest write loss as a HARD failure.

Regression: download_market_breadth / download_index_constituent /
download_earnings all called ``write_run_manifest(data_dir, ...)`` with an
UNDEFINED name (only ``cfg``/``out_dir`` exist in scope).  The NameError is
raised while evaluating the call's argument — inside a ``try/except Exception``
block — so it was swallowed and each script exited 0 with the run manifest
silently never written (the parquet data itself still landed).

Fix under test: every script defines ``data_dir = cfg.project.data_dir`` and
routes the manifest write through ``write_run_manifest_or_exit`` (a shared
helper in stoke_ml.data.download_manifest) — on success it returns normally; on
failure it logs at error and exits non-zero.  download_index_constituent keeps
its existing empty-fetch "no data" return semantics, but a manifest-write
failure still exits non-zero there too.

No network, no real downloads: source classes are monkeypatched to synthetic
fixtures; ``load_config`` is monkeypatched to a temp data_dir.
"""
import importlib
import os
import sys

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.production import download_market_breadth as dmb
from scripts.production import download_index_constituent as dic
from scripts.production import download_earnings as de
from scripts.production import download_industry as dind
from scripts.production import download_industry_ranking as dindr
from stoke_ml.data import download_manifest as dm

# dataset subdir under data_dir where each script's manifest must land.
DATASET_DIR = {
    dmb: "a_shares/market_breadth",
    dic: "a_shares/index_constituents",
    de: "a_shares/earnings",
    dind: "a_shares/industry",
    dindr: "a_shares/industry_ranking",
}


def _fake_cfg(tmp_path):
    return OmegaConf.create({
        "project": {
            "data_dir": str(tmp_path / "data"),
            "model_dir": str(tmp_path / "models"),
        },
    })


def _run(monkeypatch, tmp_path, mod, argv=(), **module_patches):
    """Run a downloader's main() with load_config + sys.argv patched and the
    given module attributes patched onto ``mod``.  Returns exit_code (None when
    main() returned normally)."""
    monkeypatch.setattr(mod, "load_config", lambda *a, **k: _fake_cfg(tmp_path))
    monkeypatch.setattr(sys, "argv", ["prog.py", *argv])
    for name, val in module_patches.items():
        monkeypatch.setattr(mod, name, val)
    try:
        mod.main()
        return None
    except SystemExit as exc:
        return exc.code


def _manifest_path(tmp_path, mod):
    return tmp_path / "data" / DATASET_DIR[mod] / "download_manifest.json"


def _boom(*a, **k):
    raise RuntimeError("disk on fire")


# ── download_market_breadth ──────────────────────────────────────────

class _BreadthSource:
    def fetch_all(self):
        return {"breadth": pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-05"]), "new_high": [5.0]})}


def test_market_breadth_success_writes_manifest(tmp_path, monkeypatch):
    rc = _run(monkeypatch, tmp_path, dmb,
              MarketBreadthSource=_BreadthSource)
    assert rc is None
    assert _manifest_path(tmp_path, dmb).is_file()


def test_market_breadth_manifest_failure_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, dmb,
              MarketBreadthSource=_BreadthSource)
    assert rc == 1
    assert not _manifest_path(tmp_path, dmb).is_file()


def test_market_breadth_writes_valid_high_lows_asset_manifest(tmp_path, monkeypatch):
    """§v19: writing market_breadth/highs_lows.parquet ALSO writes a valid
    HIGH_LOWS_ASSET manifest (per-file content manifest) — the same
    write_asset_manifest pattern sector_membership / industry_ranking use — so
    build_market_env's formal read can verify the file.  The manifest honestly
    declares the channel's vintage: same-day trade breadth → verified PIT."""
    from stoke_ml.data.asset_contract import validate_asset_manifest
    from scripts.production.download_market_breadth import HIGH_LOWS_ASSET

    class _BreadthHL:
        def fetch_all(self):
            return {
                "highs_lows": pd.DataFrame({
                    "date": pd.to_datetime(["2026-08-05", "2026-08-06"]),
                    "close": [2982.0, 2957.0],
                    "high20": [202, 74],
                    "low20": [886, 2145],
                }),
                "account_stats": pd.DataFrame({
                    "数据日期": ["2026-08"],
                    "新增投资者-数量": [1.0],
                    "沪深总市值": [1.0],
                    "沪深户均市值": [1.0],
                }),
            }

    rc = _run(monkeypatch, tmp_path, dmb, MarketBreadthSource=_BreadthHL)
    assert rc is None
    assert _manifest_path(tmp_path, dmb).is_file()
    hl = os.path.join(str(tmp_path / "data"), "a_shares", "market_breadth",
                      "highs_lows.parquet")
    assert os.path.isfile(hl)
    report = validate_asset_manifest(hl, HIGH_LOWS_ASSET)
    assert report["ok"], report["mismatches"]
    # honest vintage: same-day trade breadth is verified-by-nature PIT
    assert report["manifest"]["vintage_source"] == "immutable_snapshot"
    assert report["manifest"]["vintage_pit"] == "verified"


# ── download_index_constituent ───────────────────────────────────────

class _ConstituentSource:
    def __init__(self, empty):
        self._empty = empty

    def fetch_all_indices(self):
        if self._empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "index_code": ["000300"],
            "index_name": ["沪深300"],
            "stock_code": ["600000"],
            "weight": [1.0],
        })


def test_index_constituent_success_writes_manifest(tmp_path, monkeypatch):
    rc = _run(monkeypatch, tmp_path, dic,
              IndexConstituentSource=lambda: _ConstituentSource(empty=False))
    assert rc is None
    m = _manifest_path(tmp_path, dic)
    assert m.is_file()
    assert m.read_text(encoding="utf-8")


def test_index_constituent_empty_fetch_keeps_return_semantics_but_writes_manifest(
    tmp_path, monkeypatch
):
    rc = _run(monkeypatch, tmp_path, dic,
              IndexConstituentSource=lambda: _ConstituentSource(empty=True))
    assert rc is None  # empty fetch still returns cleanly (no data, no failure)
    m = _manifest_path(tmp_path, dic)
    assert m.is_file(), "empty fetch must still record the failed run"


def test_index_constituent_success_manifest_failure_exits_nonzero(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, dic,
              IndexConstituentSource=lambda: _ConstituentSource(empty=False))
    assert rc == 1


def test_index_constituent_empty_fetch_manifest_failure_exits_nonzero(
    tmp_path, monkeypatch
):
    """A manifest-write failure in the empty-fetch branch must NOT masquerade
    as a clean empty run — it exits non-zero even though no data was fetched."""
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, dic,
              IndexConstituentSource=lambda: _ConstituentSource(empty=True))
    assert rc == 1
    assert not _manifest_path(tmp_path, dic).is_file()


# ── download_earnings ────────────────────────────────────────────────

class _EarningsSource:
    def fetch_forecasts(self, date):
        return pd.DataFrame({
            "stock_code": ["600519"],
            "announce_date": pd.to_datetime(["2026-08-05"]),
            "forecast_metric": ["净利润"],
            "value": [1.0],
        })

    def fetch_express(self, date):
        return pd.DataFrame({
            "stock_code": ["600519"],
            "announce_date": pd.to_datetime(["2026-08-05"]),
            "value": [2.0],
        })


def test_earnings_success_writes_manifest(tmp_path, monkeypatch):
    rc = _run(monkeypatch, tmp_path, de,
              ("--report-dates", "20260331"),
              EarningsSource=_EarningsSource)
    assert rc is None
    assert _manifest_path(tmp_path, de).is_file()


def test_earnings_manifest_failure_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, de,
              ("--report-dates", "20260331"),
              EarningsSource=_EarningsSource)
    assert rc == 1
    assert not _manifest_path(tmp_path, de).is_file()


# ── download_industry ─────────────────────────────────────────────────

class _FakeIndustry:
    """A fetch-all that raises: main() must still reach the unconditional
    run-manifest write (fetch failures only populate ``failed``, never abort)."""

    def fetch_all_returns(self, start_date=None, end_date=None):
        raise RuntimeError("synthetic industry fetch failure")

    def fetch_stock_industry_map(self):
        return pd.DataFrame()


def test_industry_success_writes_manifest(tmp_path, monkeypatch):
    rc = _run(monkeypatch, tmp_path, dind, ("--no-mapping",),
              IndustrySource=_FakeIndustry)
    assert rc is None
    assert _manifest_path(tmp_path, dind).is_file()


def test_industry_manifest_failure_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, dind, ("--no-mapping",),
              IndustrySource=_FakeIndustry)
    assert rc == 1
    assert not _manifest_path(tmp_path, dind).is_file()


# ── download_industry_ranking ─────────────────────────────────────────

def _qfq_frame(code, dates, amounts):
    """A well-formed qfq daily batch that survives the RESEARCH_QFQ_DAILY
    formal contract (full OHLC + volume + amount + stock_code + pct_change)."""
    dates = list(pd.to_datetime(dates))
    n = len(dates)
    closes = [10.0 + 0.1 * i for i in range(n)]
    df = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1e6] * n,
        "amount": list(amounts),
        "stock_code": code,
    })
    pct = pd.Series([float("nan")] * n)
    if n > 1:
        closes_s = pd.Series(closes, dtype="float64")
        pct.iloc[1:] = 100.0 * closes_s.pct_change().iloc[1:].to_numpy()
    df["pct_change"] = pct
    df.attrs["source"] = "test"
    df.attrs["adjustment_mode"] = "qfq"
    return df


def _industry_ranking_fixtures(tmp_path):
    """stock_sector_cache.csv + one canonical daily file under the _fake_cfg
    data_dir.  The daily file must be canonical (parquet + valid manifest):
    the §v19 P0#4 refactor reads daily via require_valid_manifest=True, so a
    bare parquet without a manifest would abort the build instead of feeding it."""
    data_dir = tmp_path / "data"
    daily = data_dir / "a_shares" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stock_code": ["600519"],
        "sector": ["白酒"],
    }).to_csv(data_dir / "a_shares" / "stock_sector_cache.csv", index=False)
    from stoke_ml.data.storage import DataStorage
    DataStorage(str(data_dir)).save_daily(_qfq_frame(
        "600519", ["2024-01-02", "2024-01-03"], [1e8, 2e8]))


def test_industry_ranking_success_writes_manifest(tmp_path, monkeypatch):
    # §v19 P0.2: snapshot fallback is OPT-IN — the fixture has no
    # sector_membership.parquet, so main() needs the CLI flag to build over the
    # legacy current-snapshot cache instead of failing closed.
    _industry_ranking_fixtures(tmp_path)
    rc = _run(monkeypatch, tmp_path, dindr,
              argv=("--allow-snapshot-sector-fallback",))
    assert rc is None
    assert _manifest_path(tmp_path, dindr).is_file()


def test_industry_ranking_manifest_failure_exits_nonzero(tmp_path, monkeypatch):
    _industry_ranking_fixtures(tmp_path)
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    rc = _run(monkeypatch, tmp_path, dindr,
              argv=("--allow-snapshot-sector-fallback",))
    assert rc == 1
    assert not _manifest_path(tmp_path, dindr).is_file()


# ── P1#7: every remaining production downloader is fail-closed ─────────

MIGRATED = [
    "download_analyst", "download_cninfo_announcements", "download_datacenter",
    "download_etf_flow", "download_fundamentals", "download_index_hist",
    "download_macro", "download_minute", "download_ipo_st", "download_pledge",
    "download_shareholder", "download_valuation",
]


def test_all_production_downloaders_use_fail_closed_run_manifest():
    import inspect

    for mod in MIGRATED:
        m = importlib.import_module(f"scripts.production.{mod}")
        src = inspect.getsource(m)
        assert "write_run_manifest_or_exit" in src, f"{mod} not migrated"
        assert "write_run_manifest(" not in src, \
            f"{mod} still calls legacy write_run_manifest"
