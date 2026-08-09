"""§v19 P0#1: CNINFO industry-change events → per-date PIT sector membership.

The parse is pure (pandas only, no network): CNINFO records CHANGE events only,
so a stock whose CSRC gate never changed yields ONE event (the most recent
standard rename).  The honest-PIT rule under test: a stock's gate is asserted
only from its FIRST CSRC event's ``变更日期`` forward — earlier dates stay
unclassified.  All three 证监会 standard labels merge at 门类 level because the
A–S gate letters are stable across the 2001 / 2012 / 中国上市公司协会 renames.
"""
import os

import pandas as pd

from scripts.production.download_sector_membership import (
    _fetch_stock,
    parse_cninfo_events,
)


def test_parse_cninfo_events_merges_csrc_labels_and_expands():
    # 002594-style events: one 2012-standard change + one 中国上市公司协会 rename
    events = pd.DataFrame({
        "证券代码": ["002594", "002594"],
        "行业门类": ["制造业", "制造业"],
        "分类标准": ["证监会行业分类标准（2012）",
                      "中国上市公司协会上市公司行业分类标准"],
        "行业编码": ["C36", "C36"],
        "变更日期": ["2011-06-30", "2024-02-08"],
    })
    df = parse_cninfo_events("002594", events)
    # only the CSRC gate letter C survives; intervals: [2011-06-30, 2024-02-07]
    # then [2024-02-08, end-of-time], both with sector_code C
    assert {"date", "stock_code", "sector_code", "sector_name"} <= set(df.columns)
    assert set(df["sector_code"]) == {"C"}
    assert (df["date"] >= "2011-06-30").all()


def test_parse_cninfo_events_empty_and_non_csrc_return_empty():
    """A stock with no CSRC events (or only non-CSRC standards) has no gate —
    an empty frame, so it contributes no membership rows (honest unclassified)."""
    empty = pd.DataFrame()
    out = parse_cninfo_events("000001", empty)
    assert out.empty
    assert {"date", "stock_code", "sector_code", "sector_name"} <= set(out.columns)

    only_other = pd.DataFrame({
        "分类标准": ["申万行业分类标准"],
        "行业门类": ["白酒"],
        "变更日期": ["2024-01-01"],
    })
    out2 = parse_cninfo_events("000001", only_other)
    assert out2.empty


def test_parse_cninfo_events_boundary_on_gate_change():
    """A gate CHANGE mid-history (C → J at d2) yields two intervals with the
    boundary exactly at d2-1 — the C gate is not asserted into the J era."""
    events = pd.DataFrame({
        "证券代码": ["000001", "000001"],
        "行业门类": ["制造业", "金融业"],
        "分类标准": ["证监会行业分类标准（2012）",
                      "证监会行业分类标准（2012）"],
        "行业编码": ["C", "J"],
        "变更日期": ["2011-06-30", "2024-02-08"],
    })
    df = parse_cninfo_events("000001", events)
    assert set(df["sector_code"]) == {"C", "J"}
    c_int = df[df["sector_code"] == "C"]
    j_int = df[df["sector_code"] == "J"]
    assert c_int["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert c_int["_end"].iloc[0] == pd.Timestamp("2024-02-07")  # d2 - 1
    assert j_int["date"].iloc[0] == pd.Timestamp("2024-02-08")
    assert j_int["_end"].iloc[0] == pd.Timestamp("2099-12-31")


def test_parse_cninfo_events_unknown_gate_does_not_extend_prior_interval():
    """reverse-PIT: a change to an UNRECOGNIZED 门类 at d2 must END the C
    interval at d2-1 and exclude the unknown era — the previous gate is never
    asserted PAST the event that disproves it."""
    events = pd.DataFrame({
        "证券代码": ["000001", "000001"],
        "行业门类": ["制造业", "新行业门类"],  # second gate NOT in CSRC_GATE_CODES
        "分类标准": ["证监会行业分类标准（2012）",
                      "证监会行业分类标准（2012）"],
        "行业编码": ["C36", "X99"],
        "变更日期": ["2011-06-30", "2024-02-08"],
    })
    df = parse_cninfo_events("000001", events)
    # only the recognized C interval survives; it must NOT extend past d2
    assert set(df["sector_code"]) == {"C"}
    assert df["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert df["_end"].iloc[0] == pd.Timestamp("2024-02-07")
    assert not (df["date"] > pd.Timestamp("2024-02-07")).any()


def test_fetch_stock_normalizes_true_empty_to_cached_empty(monkeypatch, tmp_path):
    """§v19 P0#1 review #1: akshare's stock_industry_change_cninfo raises
    KeyError('变更日期') when a stock has ZERO records (pd.DataFrame([]) has no
    columns).  _fetch_stock must normalize that to a legit-empty result that is
    CACHED (a re-run skips the network), NOT burn retry/backoff and mark the
    stock failed every run."""
    import akshare as ak

    calls = {"n": 0}

    def _true_empty_keyerror(*a, **k):
        calls["n"] += 1
        # replicate akshare's exact empty-records failure: index a column-less frame
        pd.DataFrame([])["变更日期"]  # noqa: B018
        raise AssertionError("the KeyError above must fire first")

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _true_empty_keyerror)
    cache_dir = str(tmp_path / "cache")
    intervals = _fetch_stock("000001", cache_dir, "19900101", "20260809")
    assert intervals.empty
    # normalized at the single-attempt layer: no retry on the deterministic error
    assert calls["n"] == 1
    # cached as empty -> a re-run hits the cache, never the network
    assert os.path.isfile(os.path.join(cache_dir, "000001.json"))
    monkeypatch.setattr(
        ak, "stock_industry_change_cninfo",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("network must not be hit on a cache hit")),
    )
    again = _fetch_stock("000001", cache_dir, "19900101", "20260809")
    assert again.empty
    assert calls["n"] == 1
