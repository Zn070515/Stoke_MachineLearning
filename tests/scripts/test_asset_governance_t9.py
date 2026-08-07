"""§T9: DataAssetContract adoption for the remaining headline channels.

Covers the two write ends T9 closes:
- ``download_industry.py`` writes ``industry_returns.parquet`` ATOMICALLY (temp +
  os.replace) with an INDUSTRY_ASSET manifest; a failed write leaves no orphan
  parquet and no ``.tmp`` residue.
- ``download_earnings.py`` ``_accumulate`` writes an EARNINGS_ASSET manifest for
  BOTH snapshots (forecasts.parquet / express.parquet), HONESTLY declaring the
  channel's vintage (latest_revised / raw / proxy — governance only, never a
  formal admission claim).

market_env (build_market_env.py) is already covered by
``tests/scripts/test_build_market_env.py::test_manifest_written_and_formal_read_passes``
and the cninfo formal-rejection by
``tests/scripts/test_train_panel_formal_manifest.py::test_formal_cninfo_announcement_fails``
— no new tests needed there.

None of these touch the network: the industry source is faked, earnings
``_accumulate`` is called directly with synthetic frames.
"""
import importlib
import json
import os
import sys
import types

import pandas as pd
import pytest

from scripts.production.download_earnings import (
    EXPRESS_DEDUP,
    FORECAST_DEDUP,
    _accumulate,
)
from stoke_ml.data.asset_contract import (
    check_asset_read,
    validate_asset_manifest,
)
from stoke_ml.data.broadcast_assets import INDUSTRY_ASSET
from stoke_ml.data.earnings_storage import EARNINGS_ASSET

DI = importlib.import_module("scripts.production.download_industry")


# ── helpers ───────────────────────────────────────────────────────────────

def _manifest_of(parquet_path: str) -> dict:
    with open(parquet_path + ".manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _rewrite_manifest(path: str, mutator) -> None:
    m = _manifest_of(path)
    mutator(m)
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f)


def _tmp_files(root: str) -> list[str]:
    return [
        os.path.join(d, f)
        for d, _dirs, files in os.walk(root)
        for f in files if ".tmp" in f
    ]


def _industry_returns() -> pd.DataFrame:
    """A synthetic fetch_all_returns frame: DatetimeIndex, industries as cols."""
    df = pd.DataFrame(
        {"半导体": [0.1, -0.2, 0.3], "白酒": [0.2, 0.1, -0.1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    df.index.name = "date"
    return df


def _fake_config(tmp_path):
    return types.SimpleNamespace(
        project=types.SimpleNamespace(data_dir=str(tmp_path)))


def _run_industry(tmp_path, monkeypatch):
    """Run download_industry.main() with a faked source + config (--no-mapping)."""
    monkeypatch.setattr(
        DI.IndustrySource, "fetch_all_returns",
        lambda self, **kw: _industry_returns())
    monkeypatch.setattr(DI, "load_config", lambda *a, **k: _fake_config(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download_industry.py", "--no-mapping"])
    DI.main()
    return os.path.join(str(tmp_path), "a_shares", "industry",
                        "industry_returns.parquet")


# ── industry: atomic write + valid manifest ───────────────────────────────

def test_industry_main_writes_atomic_parquet_and_manifest(tmp_path, monkeypatch):
    """Success path: atomic parquet + valid INDUSTRY_ASSET manifest, no .tmp
    residue, mapping skipped under --no-mapping."""
    path = _run_industry(tmp_path, monkeypatch)
    assert os.path.isfile(path)
    report = validate_asset_manifest(path, INDUSTRY_ASSET)
    assert report["ok"], report["mismatches"]
    # the manifest carries the channel_vintage labels (industry, formula-derived)
    assert report["manifest"]["vintage_source"] == "immutable_snapshot"
    assert report["manifest"]["vintage_transform"] == "formula_versioned"
    # formal read of the re-read file passes (schema_hash survives round-trip)
    reread = pd.read_parquet(path)
    check_asset_read(path, INDUSTRY_ASSET, reread, require_valid_manifest=True)
    assert _tmp_files(str(tmp_path)) == []
    assert not os.path.exists(os.path.join(
        str(tmp_path), "a_shares", "industry", "stock_industry_map.parquet"))


def test_industry_manifest_tamper_rejected(tmp_path, monkeypatch):
    """A tampered industry manifest is rejected by validate + formal read."""
    path = _run_industry(tmp_path, monkeypatch)
    _rewrite_manifest(path, lambda m: m.update(schema_hash="deadbeef"))
    report = validate_asset_manifest(path, INDUSTRY_ASSET)
    assert not report["ok"]
    assert any("schema_hash" in s for s in report["mismatches"])
    reread = pd.read_parquet(path)
    with pytest.raises(ValueError, match="require_valid_manifest"):
        check_asset_read(path, INDUSTRY_ASSET, reread,
                         require_valid_manifest=True)


def test_industry_failed_write_leaves_no_orphan(tmp_path, monkeypatch):
    """Atomicity: a crash mid-write leaves NO parquet, NO manifest and no .tmp —
    a torn industry_returns.parquet can never pair with a manifest claiming it
    is valid."""
    monkeypatch.setattr(
        DI.IndustrySource, "fetch_all_returns",
        lambda self, **kw: _industry_returns())
    monkeypatch.setattr(DI, "load_config", lambda *a, **k: _fake_config(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download_industry.py", "--no-mapping"])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    DI.main()  # failure is recorded in the run manifest's failed list, no raise
    path = os.path.join(str(tmp_path), "a_shares", "industry",
                        "industry_returns.parquet")
    assert not os.path.exists(path)
    assert not os.path.exists(path + ".manifest.json")
    assert _tmp_files(str(tmp_path)) == []


# ── earnings: EARNINGS_ASSET manifest for both snapshots ──────────────────

def _forecast_rows():
    return pd.DataFrame({
        "stock_code": ["000001", "000002"],
        "announce_date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
        "forecast_metric": ["净利润", "净利润"],
        "net_profit_yoy": [12.5, -3.0],
        "net_profit": [1000.0, 500.0],
    })


def _express_rows():
    return pd.DataFrame({
        "stock_code": ["000001"],
        "announce_date": pd.to_datetime(["2024-01-05"]),
        "net_profit": [1000.0],
    })


def test_accumulate_forecasts_writes_valid_manifest(tmp_path):
    """forecasts.parquet: EARNINGS_ASSET manifest validates, honestly declares
    latest_revised / raw / proxy vintage, extent bounds announce_date, formal
    read of the re-read file passes, no .tmp residue."""
    out = tmp_path / "earnings"
    out.mkdir()
    path = _accumulate(str(out), "forecasts.parquet", _forecast_rows(),
                       FORECAST_DEDUP)
    assert os.path.isfile(path + ".manifest.json")
    report = validate_asset_manifest(path, EARNINGS_ASSET)
    assert report["ok"], report["mismatches"]
    # honest vintage from channel_vintage (earnings = latest_revised / raw / proxy)
    assert report["manifest"]["vintage_source"] == "latest_revised"
    assert report["manifest"]["vintage_transform"] == "raw"
    assert report["manifest"]["vintage_pit"] == "proxy"
    assert report["manifest"]["start"] == "2024-01-05"
    assert report["manifest"]["end"] == "2024-01-06"
    reread = pd.read_parquet(path)
    check_asset_read(path, EARNINGS_ASSET, reread, require_valid_manifest=True)
    assert _tmp_files(str(out)) == []


def test_accumulate_express_merge_manifest_reflects_merged(tmp_path):
    """express.parquet: a second accumulate merges + dedups and the manifest
    describes the MERGED file on disk (rows + extent), not just the new batch."""
    out = tmp_path / "earnings"
    out.mkdir()
    path = _accumulate(str(out), "express.parquet", _express_rows(), EXPRESS_DEDUP)
    # same stock+announce_date with a NEW value, plus a new stock
    _accumulate(str(out), "express.parquet", pd.DataFrame({
        "stock_code": ["000001", "000003"],
        "announce_date": pd.to_datetime(["2024-01-05", "2024-01-07"]),
        "net_profit": [9999.0, 2000.0],
    }), EXPRESS_DEDUP)

    report = validate_asset_manifest(path, EARNINGS_ASSET)
    assert report["ok"], report["mismatches"]
    on_disk = pd.read_parquet(path)
    assert len(on_disk) == 2          # 000001 deduped to the latest value
    assert report["actual"]["rows"] == 2
    assert "000001" in set(on_disk["stock_code"])
    assert "000003" in set(on_disk["stock_code"])
    assert report["manifest"]["end"] == "2024-01-07"


def test_accumulate_tampered_manifest_rejected(tmp_path):
    """A tampered earnings manifest is rejected by validate + formal read."""
    out = tmp_path / "earnings"
    out.mkdir()
    path = _accumulate(str(out), "express.parquet", _express_rows(), EXPRESS_DEDUP)
    _rewrite_manifest(path, lambda m: m.update(rows=999))
    report = validate_asset_manifest(path, EARNINGS_ASSET)
    assert not report["ok"]
    assert any("rows" in s for s in report["mismatches"])
    reread = pd.read_parquet(path)
    with pytest.raises(ValueError, match="require_valid_manifest"):
        check_asset_read(path, EARNINGS_ASSET, reread,
                         require_valid_manifest=True)
