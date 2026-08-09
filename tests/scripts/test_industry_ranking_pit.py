"""§v19 P0#1: ``industry_ranking`` derives sectors from PIT sector membership.

The genuine-PIT contract under test: when ``sector_membership.parquet`` (per-date
``[date, stock_code, sector_code, sector_name]``) is present,
``build_industry_ranking`` joins every daily row to it on ``(date, stock_code)``
— a stock with NO asserted CSRC gate that date is EXCLUDED from the sector
aggregates (honest unclassified, never present-backfilled with today's
classification).

§v19 P0.2 (fail-closed): snapshot fallback is OPT-IN — without
``allow_snapshot_fallback=True`` a missing ``sector_membership.parquet`` aborts
the build (never a silent proxy-PIT ranking).  §v19 P0.3: a PRESENT membership
parquet must first pass its own ``SECTOR_MEMBERSHIP_ASSET`` manifest check before
it can feed the derivation.

The daily files must be canonical (parquet + valid manifest) because the
builder reads them via ``require_valid_manifest=True`` (§v19 P0#4) — a bare
parquet without a manifest would abort the build, not feed it.
"""
import pytest
import pandas as pd

from scripts.production import download_industry_ranking as dir_mod
from scripts.production.download_sector_membership import (
    SECTOR_MEMBERSHIP_ASSET,
)
from stoke_ml.data.asset_contract import write_asset_manifest


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


def _write_daily(base, code, dates):
    from stoke_ml.data.storage import DataStorage
    # data_dir = parent of the a_shares dir
    amounts = [1e8 * (i + 1) for i in range(len(dates))]
    DataStorage(str(base.parent)).save_daily(_qfq_frame(
        code, dates, amounts))


def _write_membership(base, mem):
    """Write ``sector_membership.parquet`` the way download_sector_membership.py
    does: parquet + SECTOR_MEMBERSHIP_ASSET manifest (the builder now validates,
    §v19 P0.3)."""
    mem_path = base / "sector_membership.parquet"
    mem.attrs["source"] = "test"
    mem.to_parquet(mem_path, index=False)
    write_asset_manifest(str(mem_path), SECTOR_MEMBERSHIP_ASSET, mem)
    return mem_path


def test_build_industry_ranking_uses_pit_membership(tmp_path):
    """industry_ranking must derive sectors from sector_membership.parquet
    (per-date) rather than the current-snapshot cache."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    for code in ("000001", "600519"):
        _write_daily(base, code, ["2024-01-02", "2024-01-03", "2024-01-04"])
    mem = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
        "stock_code": ["000001", "000001", "600519", "600519"],
        "sector_code": ["J", "J", "C", "C"],
        "sector_name": ["金融业", "金融业", "制造业", "制造业"],
    })
    _write_membership(base, mem)
    df, prov = dir_mod.build_industry_ranking(str(base))
    assert prov["pit_alignment"] == "verified"
    assert prov["membership_source"] == "pit"
    assert set(df["sector_code"]) == {"J", "C"}
    assert set(df["sector_name"]) == {"金融业", "制造业"}
    # daily data exists on 2024-01-04 but the membership asserts no gate there →
    # that date is excluded from the sector aggregates (honest unclassified,
    # never present-backfilled).  §v19 P0#1 review #4c.
    assert (df["date"] == pd.Timestamp("2024-01-04")).sum() == 0


def test_build_industry_ranking_excludes_unclassified_stocks(tmp_path):
    """Genuine-PIT: a stock whose daily history has NO asserted gate that date
    is excluded from the sector aggregates — today's classification is never
    present-backfilled onto its earlier rows."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    # 000001 has an asserted gate; 000002 has daily data but NO membership row
    _write_daily(base, "000001", ["2024-01-02", "2024-01-03"])
    _write_daily(base, "000002", ["2024-01-02", "2024-01-03"])
    mem = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "stock_code": ["000001", "000001"],
        "sector_code": ["J", "J"],
        "sector_name": ["金融业", "金融业"],
    })
    _write_membership(base, mem)
    df, prov = dir_mod.build_industry_ranking(str(base))
    assert prov["pit_alignment"] == "verified"
    assert prov["membership_source"] == "pit"
    assert set(df["sector_code"]) == {"J"}
    # the unclassified stock must never surface as a constituent or leader
    assert not (df["leader"] == "000002").any()
    assert df["n_stocks"].max() == 1


def test_build_industry_ranking_falls_back_to_snapshot_cache(tmp_path):
    """Snapshot fallback is OPT-IN (§v19 P0.2): with allow_snapshot_fallback=True
    a missing sector_membership.parquet uses the legacy current-snapshot cache
    (SEC#### short codes) and reports pit_alignment='proxy'."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    pd.DataFrame({
        "stock_code": ["600519"],
        "sector": ["白酒"],
    }).to_csv(base / "stock_sector_cache.csv", index=False)
    df, prov = dir_mod.build_industry_ranking(
        str(base), allow_snapshot_fallback=True)
    assert prov["pit_alignment"] == "proxy"
    assert prov["membership_source"] == "snapshot_fallback"
    assert set(df["sector_code"]) == {"SEC0000"}
    assert set(df["sector_name"]) == {"白酒"}


def test_build_industry_ranking_fails_closed_without_membership(tmp_path):
    """§v19 P0.2: default (no flag) + missing sector_membership.parquet → the
    build FAILS closed — never a silent proxy-PIT snapshot ranking."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    # no sector_membership.parquet and no stock_sector_cache.csv
    with pytest.raises(SystemExit) as ei:
        dir_mod.build_industry_ranking(str(base))
    assert "sector_membership.parquet missing" in str(ei.value)


def test_build_industry_ranking_rejects_bare_membership(tmp_path):
    """§v19 P0.3: a PRESENT sector_membership.parquet with NO manifest (a bare /
    pre-manifest file) fails the SECTOR_MEMBERSHIP_ASSET check → SystemExit."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    mem = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "stock_code": ["600519", "600519"],
        "sector_code": ["C", "C"],
        "sector_name": ["制造业", "制造业"],
    })
    # bare parquet — NO write_asset_manifest sidecar
    mem.to_parquet(base / "sector_membership.parquet", index=False)
    with pytest.raises(SystemExit) as ei:
        dir_mod.build_industry_ranking(str(base))
    assert "sector_membership.parquet FAILED its asset manifest check" in str(ei.value)
    # the actionable reason (a bare / pre-manifest file) is surfaced, not swallowed
    assert "manifest missing" in str(ei.value)


def test_compute_lineage_tracks_upstream_roots(tmp_path):
    """compute_lineage returns the three §v19 P0#2 keys; changing the
    sector_membership.parquet bytes flips upstream_roots['sector_membership']
    while the unchanged daily upstream stays stable."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    mem_path = base / "sector_membership.parquet"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "stock_code": ["600519", "600519"],
        "sector_code": ["C", "C"],
        "sector_name": ["制造业", "制造业"],
    }).to_parquet(mem_path, index=False)
    prov = {"membership_source": "pit", "pit_alignment": "verified"}
    lineage = dir_mod.compute_lineage(str(base.parent), prov, ["date"])
    assert set(lineage) == {"upstream_roots", "transform_code_hash",
                            "transform_config_hash"}
    assert set(lineage["upstream_roots"]) == {"daily", "sector_membership"}
    first = lineage["upstream_roots"]["sector_membership"]
    # rewrite the membership parquet with DIFFERENT bytes → upstream root flips
    pd.DataFrame({
        "date": ["2024-01-02"],
        "stock_code": ["600519"],
        "sector_code": ["J"],
        "sector_name": ["金融业"],
    }).to_parquet(mem_path, index=False)
    lineage2 = dir_mod.compute_lineage(str(base.parent), prov, ["date"])
    assert lineage2["upstream_roots"]["sector_membership"] != first
    # the daily upstream (untouched) is stable across the membership rewrite
    assert lineage2["upstream_roots"]["daily"] == lineage["upstream_roots"]["daily"]


def test_compute_lineage_two_arg_derives_columns_from_disk(tmp_path):
    """The two-arg form (the Task 4 formal-gate contract, mirroring
    build_market_env.compute_lineage(data_dir, parts)) derives output_columns
    from the on-disk industry_ranking.parquet, so its lineage dict equals the
    write-time three-arg form exactly."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    mem = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "stock_code": ["600519", "600519"],
        "sector_code": ["C", "C"],
        "sector_name": ["制造业", "制造业"],
    })
    _write_membership(base, mem)
    result, prov = dir_mod.build_industry_ranking(str(base))
    # persist the ranking exactly as main() does after its AtomicCommit block
    result.to_parquet(base / "industry_ranking.parquet", index=False)

    two_arg = dir_mod.compute_lineage(str(base.parent), prov)
    three_arg = dir_mod.compute_lineage(
        str(base.parent), prov, list(result.columns))
    assert two_arg == three_arg
    assert set(two_arg["upstream_roots"]) == {"daily", "sector_membership"}
