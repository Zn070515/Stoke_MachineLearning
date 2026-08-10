"""§v19 P0#1: CNINFO industry-change events → per-date PIT sector membership.

The parse is pure (pandas only, no network): CNINFO records CHANGE events only,
so a stock whose CSRC gate never changed yields ONE event (the most recent
standard rename).  The honest-PIT rule under test: a stock's gate is asserted
only from its FIRST CSRC event's ``变更日期`` forward — earlier dates stay
unclassified.  All three 证监会 standard labels merge at 门类 level because the
A–S gate letters are stable across the 2001 / 2012 / 中国上市公司协会 renames.

Also covered here are the hardening rules that make the downloader fail-closed:
§v19 §十八 (a KeyError for a non-变更日期 column is a schema drift, re-raised),
§v19 §十九 (the interval cache is versioned + parser-hashed; a mismatch is
refetched), §v19 §十五 (completion requires the full pipeline; a daily failure
is a failure), and §v19 §十六 (the coverage denominator is the stocks active
that year).
"""
import json
import os

import pandas as pd
import pytest

from scripts.production.download_sector_membership import (
    _CACHE_VERSION,
    _FETCH_ATTEMPTS,
    _coverage_by_year,
    _expand_stock,
    _expansion_bucket,
    _fetch_stock,
    _load_intervals_cache,
    _parser_hash,
    _traded_counts_by_day,
    _valid_cached_end,
    _write_intervals_cache,
    _write_membership_asset,
    parse_cninfo_events,
)
from stoke_ml.data.storage import DataStorage


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


# ── §v19 §十八: narrow the CNINFO KeyError catch ────────────────────────────

def test_fetch_stock_re_raises_unrecognized_keyerror(monkeypatch, tmp_path):
    """§v19 §十八: a KeyError for a DIFFERENT column (schema drift) must NOT be
    normalized into a legit no-gate stock — it re-raises so the retry loop marks
    the stock failed (and nothing is cached)."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    calls = {"n": 0}

    def _drift_keyerror(*a, **k):
        calls["n"] += 1
        pd.DataFrame([])["最新记录标识"]  # noqa: B018  a schema-drift column index
        raise AssertionError("the KeyError above must fire first")

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _drift_keyerror)
    cache_dir = str(tmp_path / "cache")
    with pytest.raises(KeyError):
        _fetch_stock("000001", cache_dir, "19900101", "20260809")
    # retried as a real failure (not normalized-empty at attempt 1)
    assert calls["n"] == _FETCH_ATTEMPTS
    # a failed stock is never cached as empty
    assert not os.path.isfile(os.path.join(cache_dir, "000001.json"))


# ── §v19 §十九: versioned + parser-hashed intervals cache ───────────────────

def test_load_intervals_cache_returns_intervals_and_meta(tmp_path):
    """The cache payload carries the cache/parser identity; meta excludes the
    intervals themselves."""
    path = str(tmp_path / "000001.json")
    _write_intervals_cache(path, _one_interval(), "19900101", "20260809")
    iv, meta = _load_intervals_cache(path)
    assert not iv.empty
    assert meta["cache_version"] == _CACHE_VERSION
    assert meta["parser_hash"] == _parser_hash()
    assert meta["source"] == "cninfo"
    assert meta["start_date"] == "19900101"
    assert meta["end_date"] == "20260809"
    assert "intervals" not in meta


def test_fetch_stock_reuses_matching_versioned_cache(monkeypatch, tmp_path):
    """§v19 §十九: a cache whose cache_version/parser_hash/source/range all match
    the current run is reused — the network is never hit."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")
    monkeypatch.setattr(
        ak, "stock_industry_change_cninfo",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("network must not be hit on a cache hit")),
    )
    got = _fetch_stock("000001", cache_dir, "19900101", "20260809")
    assert not got.empty
    assert got["sector_code"].iloc[0] == "C"


@pytest.mark.parametrize("override, drop", [
    ({"cache_version": "v1"}, None),               # legacy/older schema version
    ({"parser_hash": "0" * 12}, None),             # parser logic changed
    ({"source": "other"}, None),                   # not the cninfo writer
    ({"start_date": "19000101", "end_date": "19010101"}, None),  # stale range
    ({}, "cache_version"),                         # un-versioned legacy cache
])
def test_fetch_stock_refetches_mismatched_cache(monkeypatch, tmp_path,
                                                override, drop):
    """§v19 §十九: a cache that fails ANY identity check must NOT be trusted —
    it is removed and refetched once (fail closed), and rewritten under the
    current schema."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "000001.json")
    base = {
        "cache_version": _CACHE_VERSION,
        "parser_hash": _parser_hash(),
        "source": "cninfo",
        "start_date": "19900101",
        "end_date": "20260809",
        "intervals": [{
            "date": "2011-06-30", "end": "2099-12-31", "stock_code": "000001",
            "sector_code": "C", "sector_name": "制造业",
        }],
    }
    base.update(override)
    if drop:
        base.pop(drop, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False)

    calls = {"n": 0}

    def _events(*a, **k):
        calls["n"] += 1
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["C36"],
            "变更日期": ["2011-06-30"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260809")
    assert calls["n"] == 1  # refetched: the stale cache was NOT trusted
    assert not got.empty
    with open(path, "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["cache_version"] == _CACHE_VERSION
    assert rewritten["parser_hash"] == _parser_hash()


def test_fetch_stock_recovers_from_unreadable_cache(monkeypatch, tmp_path):
    """§v19 §十九: a corrupt/unreadable cache JSON is NOT a permanent failure —
    it is removed and refetched once (fail closed on the FILE, not the stock),
    then rewritten as a valid current-schema cache."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "000001.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ this is not valid json !!!")

    calls = {"n": 0}

    def _events(*a, **k):
        calls["n"] += 1
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["C36"],
            "变更日期": ["2011-06-30"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260809")
    assert calls["n"] == 1  # the corrupt cache was NOT trusted → refetched
    assert not got.empty
    with open(path, "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["cache_version"] == _CACHE_VERSION
    assert rewritten["parser_hash"] == _parser_hash()


# ── V14 §八: range-extension — daily runs fetch only the tail ───────────────

def test_fetch_stock_extends_cache_fetches_only_tail(monkeypatch, tmp_path):
    """V14 §八: a same-start, later-end cache with matching provenance is
    EXTENDED — the network is hit ONCE for ONLY the tail window (start one day
    past the cached end, NOT the full 36y range), the new change event merges
    into the intervals, and the cache is rewritten with the new end_date."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    calls = []

    def _events(*a, **k):
        calls.append((a, k))
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["金融业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["J66"],
            "变更日期": ["2026-08-10"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    # fetch called once, for the TAIL window only: a one-day boundary overlap at
    # the cached end (catches late-published same-day records) + the new day —
    # NOT the full 36y range start
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "20260809"
    assert calls[0][1]["end_date"] == "20260810"
    # merged: [2011-06-30, 2026-08-09] C then [2026-08-10, 2099-12-31] J
    assert len(got) == 2
    assert got["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert got["_end"].iloc[0] == pd.Timestamp("2026-08-09")
    assert got["sector_code"].iloc[0] == "C"
    assert got["date"].iloc[1] == pd.Timestamp("2026-08-10")
    assert got["_end"].iloc[1] == pd.Timestamp("2099-12-31")
    assert got["sector_code"].iloc[1] == "J"
    # cache rewritten with the new end_date
    with open(os.path.join(cache_dir, "000001.json"), "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["start_date"] == "19900101"
    assert rewritten["end_date"] == "20260810"


def test_fetch_stock_extends_cache_with_no_new_events_advances_meta(
        monkeypatch, tmp_path):
    """V14 §八: the common daily case — the tail window has NO new change events,
    so the cached intervals are unchanged (the last interval still runs to
    end-of-time) and only the cache meta end_date advances."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    calls = []

    def _empty_events(*a, **k):
        calls.append((a, k))
        return pd.DataFrame()

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _empty_events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "20260809"  # one-day boundary overlap
    assert calls[0][1]["end_date"] == "20260810"
    assert len(got) == 1
    assert got["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert got["_end"].iloc[0] == pd.Timestamp("2099-12-31")
    with open(os.path.join(cache_dir, "000001.json"), "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["end_date"] == "20260810"


def test_fetch_stock_extends_cache_same_gate_is_contiguous_not_double_counted(
        monkeypatch, tmp_path):
    """V14 §八: a rename under the SAME gate (no code change) yields two
    contiguous same-code intervals with a clean boundary at the new event date —
    disjoint day coverage, no double counting."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    def _events(*a, **k):
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],   # same gate C — standard rename only
            "分类标准": ["中国上市公司协会上市公司行业分类标准"],
            "行业编码": ["C36"],
            "变更日期": ["2026-08-10"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    assert len(got) == 2
    assert got["sector_code"].iloc[0] == "C"
    assert got["_end"].iloc[0] == pd.Timestamp("2026-08-09")
    assert got["sector_code"].iloc[1] == "C"
    assert got["date"].iloc[1] == pd.Timestamp("2026-08-10")


def test_fetch_stock_extends_cache_unrecognized_gate_caps_prior_interval(
        monkeypatch, tmp_path):
    """V14 §八: reverse-PIT is preserved across the extension — a tail change to
    an UNRECOGNIZED 门类 ends the cached last interval at the event date − 1 and
    contributes no rows (its own interval is excluded), never asserting the old
    gate past the event that disproves it."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    def _events(*a, **k):
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["新行业门类"],   # not in CSRC_GATE_CODES
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["X99"],
            "变更日期": ["2026-08-10"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    assert len(got) == 1
    assert got["sector_code"].iloc[0] == "C"
    assert got["_end"].iloc[0] == pd.Timestamp("2026-08-09")
    assert not (got["date"] > pd.Timestamp("2026-08-09")).any()


def test_fetch_stock_shrunk_end_refetches_full_range(monkeypatch, tmp_path):
    """V14 §八: a requested end EARLIER than the cached end is NOT silently
    truncated — the cache is discarded and the stock is refetched over the full
    requested range (fail-closed), exactly as before."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    calls = []

    def _events(*a, **k):
        calls.append((a, k))
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["C36"],
            "变更日期": ["2011-06-30"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    _fetch_stock("000001", cache_dir, "19900101", "20260808")
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "19900101"   # full range, NOT the tail
    assert calls[0][1]["end_date"] == "20260808"


def test_fetch_stock_extends_only_when_provenance_matches(monkeypatch, tmp_path):
    """V14 §八: range-extension is ONLY valid when the cache provenance matches —
    a parser_hash mismatch on an EXTENDED request is still a fail-closed FULL
    refetch (never a trusted tail), and the cache is rewritten under the current
    schema."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "000001.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "cache_version": _CACHE_VERSION,
            "parser_hash": "0" * 12,   # parser logic changed → NOT trusted
            "source": "cninfo",
            "start_date": "19900101",
            "end_date": "20260809",
            "intervals": [{
                "date": "2011-06-30", "end": "2099-12-31", "stock_code": "000001",
                "sector_code": "C", "sector_name": "制造业",
            }],
        }, f, ensure_ascii=False)

    calls = []

    def _events(*a, **k):
        calls.append((a, k))
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["C36"],
            "变更日期": ["2011-06-30"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "19900101"   # full range, NOT the tail
    assert not got.empty


def test_fetch_stock_extends_cached_empty_without_network(monkeypatch, tmp_path):
    """V14 §八: a no-gate stock cached EMPTY extends by advancing the cache meta —
    the network is never hit (a stock with no CSRC records never gains them)."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           pd.DataFrame(), "19900101", "20260809")
    monkeypatch.setattr(
        ak, "stock_industry_change_cninfo",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("network must not be hit for a no-gate stock")),
    )
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    assert got.empty
    with open(os.path.join(cache_dir, "000001.json"), "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["start_date"] == "19900101"
    assert rewritten["end_date"] == "20260810"


def test_fetch_stock_extends_cache_picks_up_boundary_day_late_record(
        monkeypatch, tmp_path):
    """V14 §八 (the 1-day boundary overlap): a CSRC-standard change record whose
    变更日期 == the cached end date — published a day late, so the previous run
    missed it — is re-read by the one-day overlap and merged as a NEW interval,
    capping the cached end-of-time interval at its date − 1.  This is the whole
    point of the boundary overlap: the merge's ``d > d_last`` filter keeps the
    boundary-day record (d_last is the cached last interval's START, far earlier
    than the cached END the record sits on)."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           _one_interval(), "19900101", "20260809")

    calls = []

    def _events(*a, **k):
        calls.append((a, k))
        # 变更日期 == the cached end date 2026-08-09 (published late; the
        # previous run, which fetched only through 2026-08-09, missed it)
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["金融业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["J66"],
            "变更日期": ["2026-08-09"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    # tail fetch over the boundary-overlap window ONLY (not the full 36y range)
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "20260809"
    assert calls[0][1]["end_date"] == "20260810"
    # boundary-day interval AND the capped interval:
    # [2011-06-30, 2026-08-08] C then [2026-08-09, 2099-12-31] J
    assert len(got) == 2
    assert got["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert got["_end"].iloc[0] == pd.Timestamp("2026-08-08")  # capped at 08-09 − 1
    assert got["sector_code"].iloc[0] == "C"
    assert got["date"].iloc[1] == pd.Timestamp("2026-08-09")  # the boundary day
    assert got["_end"].iloc[1] == pd.Timestamp("2099-12-31")
    assert got["sector_code"].iloc[1] == "J"
    with open(os.path.join(cache_dir, "000001.json"), "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["end_date"] == "20260810"


def test_fetch_stock_extends_cache_does_not_rewrite_finite_end_last_interval(
        monkeypatch, tmp_path):
    """V14 §八 (the ``_end < 2099`` branch): a cached LAST interval that already
    ends BEFORE end-of-time (a trailing unclassified boundary encoded in a finite
    ``_end``) is left UNTOUCHED by the merge — never rewritten to first_new − 1,
    never resurrected — while recognized tail intervals are appended."""
    import akshare as ak

    cache_dir = str(tmp_path / "cache")
    cached = pd.DataFrame({
        "date": [pd.Timestamp("2011-06-30")],
        "_end": [pd.Timestamp("2024-02-07")],   # trailing unclassified boundary
        "stock_code": ["000001"],
        "sector_code": ["C"],
        "sector_name": ["制造业"],
    })
    _write_intervals_cache(os.path.join(cache_dir, "000001.json"),
                           cached, "19900101", "20260809")

    def _events(*a, **k):
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["金融业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["J66"],
            "变更日期": ["2026-08-10"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    # the cached finite-_end interval is byte-identical; the recognized tail
    # interval is appended after it
    assert len(got) == 2
    assert got["date"].iloc[0] == pd.Timestamp("2011-06-30")
    assert got["_end"].iloc[0] == pd.Timestamp("2024-02-07")  # UNTOUCHED
    assert got["sector_code"].iloc[0] == "C"
    assert got["date"].iloc[1] == pd.Timestamp("2026-08-10")
    assert got["_end"].iloc[1] == pd.Timestamp("2099-12-31")
    assert got["sector_code"].iloc[1] == "J"
    # the rewritten cache's first interval is byte-identical to the cached one
    with open(os.path.join(cache_dir, "000001.json"), "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["intervals"][0] == {
        "date": "2011-06-30", "end": "2024-02-07", "stock_code": "000001",
        "sector_code": "C", "sector_name": "制造业",
    }


def test_fetch_stock_tail_fetch_failure_preserves_cache(monkeypatch, tmp_path):
    """V14 §八: a tail-fetch FAILURE raises WITHOUT touching the cache — a bad
    daily run never destroys a good 36y cache.  The exception propagates (which
    is exactly what main() catches to land the stock in ``fetch_failed`` this
    run), and the still-valid cached range survives for a later retry."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    path = os.path.join(cache_dir, "000001.json")
    _write_intervals_cache(path, _one_interval(), "19900101", "20260809")
    with open(path, "rb") as f:
        before = f.read()

    def _explode(*a, **k):
        raise RuntimeError("CNINFO tail fetch failed")

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _explode)
    with pytest.raises(RuntimeError):
        _fetch_stock("000001", cache_dir, "19900101", "20260810")
    # fail-closed: the cache file on disk is byte-identical to before
    with open(path, "rb") as f:
        after = f.read()
    assert after == before


@pytest.mark.parametrize("bad_end", [
    20260809,        # non-string (hand-edited/corrupt meta) — an int
    "not-a-date",    # a well-formed string but a garbage end_date
])
def test_fetch_stock_malformed_end_date_meta_refetches_full_range(
        monkeypatch, tmp_path, bad_end):
    """V14 §八: a malformed/non-string ``end_date`` cache meta must NOT raise or
    drive a range-extension (which would feed a garbage tail window into the
    fetch) — it fails closed into a FULL refetch over the requested range, and
    the cache is rewritten under the current schema.  ``_valid_cached_end``
    rejects the non-string BEFORE the ``<`` range comparison, so an int
    end_date is a cache mismatch, never a TypeError that would leave the stock
    stuck in ``fetch_failed``."""
    import akshare as ak

    monkeypatch.setattr(
        "scripts.production.download_sector_membership.time.sleep",
        lambda *_: None,
    )
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "000001.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "cache_version": _CACHE_VERSION,
            "parser_hash": _parser_hash(),
            "source": "cninfo",
            "start_date": "19900101",
            "end_date": bad_end,
            "intervals": [{
                "date": "2011-06-30", "end": "2099-12-31", "stock_code": "000001",
                "sector_code": "C", "sector_name": "制造业",
            }],
        }, f, ensure_ascii=False)

    calls = []

    def _events(*a, **k):
        calls.append((a, k))
        return pd.DataFrame({
            "证券代码": ["000001"],
            "行业门类": ["制造业"],
            "分类标准": ["证监会行业分类标准（2012）"],
            "行业编码": ["C36"],
            "变更日期": ["2011-06-30"],
        })

    monkeypatch.setattr(ak, "stock_industry_change_cninfo", _events)
    got = _fetch_stock("000001", cache_dir, "19900101", "20260810")
    # refetched over the FULL range — NOT a tail extension, no exception, so the
    # stock is NOT left stuck in fetch_failed
    assert len(calls) == 1
    assert calls[0][1]["start_date"] == "19900101"
    assert calls[0][1]["end_date"] == "20260810"
    assert not got.empty
    with open(path, "r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["cache_version"] == _CACHE_VERSION
    assert rewritten["end_date"] == "20260810"


def test_valid_cached_end_rejects_non_string_and_garbage():
    """A non-string ``end_date`` (int/None) or a garbage string is INVALID — it
    must fail the extension guard (never raise), so it refetches fail-closed."""
    assert _valid_cached_end("20260809") is True
    assert _valid_cached_end(20260809) is False     # non-string: no TypeError
    assert _valid_cached_end(None) is False
    assert _valid_cached_end("not-a-date") is False
    assert _valid_cached_end("") is False


def _one_interval() -> pd.DataFrame:
    return pd.DataFrame({
        "date": [pd.Timestamp("2011-06-30")],
        "_end": [pd.Timestamp("2099-12-31")],
        "stock_code": ["000001"],
        "sector_code": ["C"],
        "sector_name": ["制造业"],
    })


# ── §v19 §十五: completion requires the full pipeline ───────────────────────

def test_expand_stock_no_gate_returns_empty_without_touching_daily():
    """A legit no-gate stock (no intervals) expands to an empty frame and never
    reads the daily store."""
    class _NoopStorage:
        def load_daily(self, *a, **k):
            raise AssertionError("daily must not be read for a no-gate stock")

    out = _expand_stock(pd.DataFrame(), _NoopStorage(), "000001")
    assert out.empty


def test_expand_stock_raises_on_daily_read_failure():
    """§v19 §十五: a canonical daily read failure (corrupt/missing manifest) is
    a FAILURE — _expand_stock raises (via the formal require_valid_manifest
    read), it never silently skips."""
    class _BrokenStorage:
        def load_daily(self, stock_code, start_date, end_date,
                       market="a_shares", require_valid_manifest=False):
            assert require_valid_manifest is True
            raise ValueError("corrupt daily: manifest mismatch")

    with pytest.raises(ValueError):
        _expand_stock(_one_interval(), _BrokenStorage(), "000001")


def test_expand_stock_empty_daily_is_a_defect():
    """A stock with CSRC intervals but NO daily history is a defect, not a
    silent skip — it must land in failed."""
    class _EmptyDailyStorage:
        def load_daily(self, stock_code, start_date, end_date,
                       market="a_shares", require_valid_manifest=False):
            return pd.DataFrame()

    with pytest.raises(ValueError):
        _expand_stock(_one_interval(), _EmptyDailyStorage(), "000001")


def test_expand_stock_expands_over_own_daily_days():
    """The happy path: intervals expand to one row per OWN trading day, through
    the formal require_valid_manifest daily read."""
    class _DailyStorage:
        def load_daily(self, stock_code, start_date, end_date,
                       market="a_shares", require_valid_manifest=False):
            assert require_valid_manifest is True
            return pd.DataFrame({"date": ["2011-06-30", "2011-07-01",
                                          "2011-07-04"]})

    out = _expand_stock(_one_interval(), _DailyStorage(), "000001")
    assert len(out) == 3
    assert set(out["date"].astype(str)) == {"2011-06-30", "2011-07-01",
                                            "2011-07-04"}


def test_expansion_bucket_never_upgrades_fetch_failed_to_complete():
    """§v19 §十五: a code ABSENT from intervals_by_code (its CNINFO fetch never
    succeeded — it is already in fetch_failed) must be classified
    ``fetch_failed``, NEVER the ``no_gate`` bucket that would mark it
    complete.  Only an EMPTY frame (a real no-CSRC-records result) is no_gate."""
    # fetch never succeeded → None → fetch_failed (loop `continue`s; NOT complete)
    assert _expansion_bucket(None) == "fetch_failed"
    # legit no-gate → empty frame → no_gate (loop adds to complete)
    assert _expansion_bucket(pd.DataFrame()) == "no_gate"
    # non-empty intervals → expand
    assert _expansion_bucket(_one_interval()) == "expand"


# ── §v19 §十六 (V14 §五/§六): bar-based daily active-stock coverage ──────────

def _qfq_frame(code, dates):
    """A well-formed qfq daily batch that survives the RESEARCH_QFQ_DAILY formal
    contract (full OHLC + volume + amount + stock_code + pct_change)."""
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
        "amount": [1e8 * (i + 1) for i in range(n)],
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


def _write_daily(tmp_path, code, dates):
    DataStorage(str(tmp_path)).save_daily(_qfq_frame(code, dates))


def test_coverage_by_year_is_bar_based_daily_not_year_unique():
    """§五 counterexample: 100 stocks each with a membership row ONLY on the last
    trading day of 2020 must NOT report year coverage 1.0 (the old year-unique
    metric counted a stock as covered for a whole year on ANY membership day).
    The bar-based per-day metric reports ~1 classified day of ~261 → p05 ≈ 0."""
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    traded = {pd.Timestamp(d): 100 for d in dates}
    mem = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31"] * 100),
        "stock_code": [f"{i:06d}" for i in range(100)],
        "sector_code": ["C"] * 100,
        "sector_name": ["制造业"] * 100,
    })
    cov, daily = _coverage_by_year(mem, traded)
    assert cov["2020"] < 0.01                      # ≈ 1/261, NOT 1.0
    assert daily["2020"]["days"] == len(dates)
    assert daily["2020"]["days_below_0_80"] == len(dates) - 1
    assert daily["2020"]["days_below_0_95"] == len(dates) - 1


def test_coverage_by_year_full_daily_coverage_reports_one():
    """Positive control: all traded stocks classified on every day → per-day
    coverage 1.0 → year p05 1.0."""
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    traded = {pd.Timestamp(d): 2 for d in dates}
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "stock_code": "000001",
                     "sector_code": "C", "sector_name": "制造业"})
        rows.append({"date": d, "stock_code": "000002",
                     "sector_code": "C", "sector_name": "制造业"})
    mem = pd.DataFrame(rows)
    cov, daily = _coverage_by_year(mem, traded)
    assert cov["2020"] == 1.0
    assert daily["2020"]["days_below_0_80"] == 0
    assert daily["2020"]["days_below_0_95"] == 0


def test_coverage_by_year_empty_membership_is_zero():
    """A year with trading days but NO membership reads 0.0 (fail-closed), and
    the richer audit still reports the day count."""
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    traded = {pd.Timestamp(d): 100 for d in dates}
    cov, daily = _coverage_by_year(pd.DataFrame(), traded)
    assert cov["2020"] == 0.0
    assert daily["2020"]["days"] == len(dates)
    assert daily["2020"]["days_below_0_80"] == len(dates)


def test_traded_counts_by_day_is_bar_based_denominator(tmp_path):
    """§五: the active-stock denominator is BAR-BASED — a stock whose manifest
    SPAN covers a day but that has NO bar that day is NOT active that day and
    must not depress coverage.  ``000002`` has no 2020-01-02 bar, so only
    ``000001`` is active then (coverage 1.0, not 1/2)."""
    _write_daily(tmp_path, "000001", ["2020-01-02", "2020-01-03"])
    _write_daily(tmp_path, "000002", ["2020-01-03"])   # no 2020-01-02 bar
    storage = DataStorage(str(tmp_path))
    traded = _traded_counts_by_day(storage, storage.list_stocks("a_shares"))
    assert traded[pd.Timestamp("2020-01-02")] == 1    # only 000001
    assert traded[pd.Timestamp("2020-01-03")] == 2
    # 000001 is classified both days; 000002 has no bar on 01-02 and no gate.
    mem = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "stock_code": ["000001", "000001"],
        "sector_code": ["C", "C"], "sector_name": ["制造业", "制造业"],
    })
    cov, daily = _coverage_by_year(mem, traded)
    # 2020-01-02: only 000001 trades → coverage 1.0 (a manifest-span denominator
    # would count 000002 as active → 0.5).  Only 2020-01-03 (000002 trades but
    # is unclassified → 0.5) falls below the floor.
    assert daily["2020"]["days_below_0_80"] == 1
    assert daily["2020"]["days_below_0_95"] == 1


def test_traded_counts_fail_closed_on_missing_manifest(tmp_path):
    """§六: a requested stock whose daily MANIFEST is missing/unreadable must
    RAISE the coverage audit — never silently drop the stock (which would shrink
    the denominator and inflate coverage)."""
    _write_daily(tmp_path, "000001", ["2020-01-02", "2020-01-03"])
    _write_daily(tmp_path, "000002", ["2020-01-03"])
    os.remove(os.path.join(str(tmp_path), "a_shares", "daily",
                           "000002.manifest.json"))
    storage = DataStorage(str(tmp_path))
    with pytest.raises(ValueError):
        _traded_counts_by_day(storage, storage.list_stocks("a_shares"))


def test_manifest_writes_coverage_by_year_daily(tmp_path):
    """The asset manifest carries BOTH the gate-facing ``coverage_by_year``
    (per-year p05) AND the richer ``coverage_by_year_daily`` audit."""
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    traded = {pd.Timestamp(d): 100 for d in dates}
    mem = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31"] * 100),
        "stock_code": [f"{i:06d}" for i in range(100)],
        "sector_code": ["C"] * 100,
        "sector_name": ["制造业"] * 100,
    })
    cov, daily = _coverage_by_year(mem, traded)
    out = str(tmp_path / "sector_membership.parquet")
    _write_membership_asset(out, mem, cov, daily)
    with open(out + ".manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["coverage_by_year"]["2020"] == cov["2020"]
    assert m["coverage_by_year_daily"]["2020"]["days"] == len(dates)
    assert m["coverage_by_year_daily"]["2020"]["days_below_0_80"] == \
        len(dates) - 1
