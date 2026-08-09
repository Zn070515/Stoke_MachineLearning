"""§v19 P0#1: ``industry_ranking`` derives sectors from PIT sector membership.

The genuine-PIT contract under test: when ``sector_membership.parquet`` (per-date
``[date, stock_code, sector_code, sector_name]``) is present,
``build_industry_ranking`` joins every daily row to it on ``(date, stock_code)``
— a stock with NO asserted CSRC gate that date is EXCLUDED from the sector
aggregates (honest unclassified, never present-backfilled with today's
classification).  With the PIT artifact absent it falls back to the legacy
snapshot cache (SEC#### codes) so the historical behavior is preserved.

The daily files must be canonical (parquet + valid manifest) because the
builder reads them via ``require_valid_manifest=True`` (§v19 P0#4) — a bare
parquet without a manifest would abort the build, not feed it.
"""
import pandas as pd

from scripts.production import download_industry_ranking as dir_mod


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
    mem.to_parquet(base / "sector_membership.parquet", index=False)
    df = dir_mod.build_industry_ranking(str(base))
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
    mem.to_parquet(base / "sector_membership.parquet", index=False)
    df = dir_mod.build_industry_ranking(str(base))
    assert set(df["sector_code"]) == {"J"}
    # the unclassified stock must never surface as a constituent or leader
    assert not (df["leader"] == "000002").any()
    assert df["n_stocks"].max() == 1


def test_build_industry_ranking_falls_back_to_snapshot_cache(tmp_path):
    """Without sector_membership.parquet the builder falls back to the legacy
    snapshot cache (SEC#### short codes) — the pre-P0#1 behavior is preserved."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    _write_daily(base, "600519", ["2024-01-02", "2024-01-03"])
    pd.DataFrame({
        "stock_code": ["600519"],
        "sector": ["白酒"],
    }).to_csv(base / "stock_sector_cache.csv", index=False)
    df = dir_mod.build_industry_ranking(str(base))
    assert set(df["sector_code"]) == {"SEC0000"}
    assert set(df["sector_name"]) == {"白酒"}
