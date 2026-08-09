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
    _active_stocks_by_year,
    _coverage_by_year,
    _expand_stock,
    _expansion_bucket,
    _fetch_stock,
    _load_intervals_cache,
    _parser_hash,
    _write_intervals_cache,
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


# ── §v19 §十六: active-stock coverage denominator ───────────────────────────

def test_coverage_by_year_uses_active_denominator():
    """§v19 §十六: the coverage denominator is the stocks ACTIVE that year, not
    the whole universe — a stock with no membership rows that year (pre-gate or
    no-gate) lowers the fraction; a year with no rows reports 0.0."""
    membership = pd.DataFrame({
        "date": pd.to_datetime(["2010-03-15", "2010-06-01", "2010-09-01",
                                "2010-11-20", "2010-12-05"]),
        "stock_code": ["000001", "000002", "000003", "000004", "000005"],
        "sector_code": ["C", "C", "C", "C", "C"],
        "sector_name": ["制造业"] * 5,
    })
    out = _coverage_by_year(membership, {2010: 10, 2011: 10})
    assert out == {"2010": 0.5, "2011": 0.0}


def test_coverage_by_year_empty_membership_is_zero_for_all_active_years():
    out = _coverage_by_year(pd.DataFrame(), {2010: 10, 2011: 10})
    assert out == {"2010": 0.0, "2011": 0.0}


def test_active_stocks_by_year_counts_manifest_span(tmp_path):
    """A stock is active every year in its daily manifest [start, end]; a
    missing/unreadable manifest is ignored (that stock counts nowhere)."""
    daily_dir = str(tmp_path / "daily")
    os.makedirs(daily_dir, exist_ok=True)
    for code, start, end in [
        ("000001", "2005-01-01", "2026-08-09"),
        ("000002", "2010-06-01", "2020-12-31"),
        ("000003", "2015-03-01", "2018-08-09"),
    ]:
        with open(os.path.join(daily_dir, f"{code}.manifest.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"start": start, "end": end}, f)
    out = _active_stocks_by_year(
        daily_dir, ["000001", "000002", "000003", "000999"])
    # 000999 has no manifest → never counted (conservative)
    assert out[2005] == 1   # only 000001
    assert out[2010] == 2   # 000001 + 000002
    assert out[2015] == 3   # all three active in 2015
    assert out[2018] == 3   # 000003 active through 2018
    assert out[2021] == 1   # only 000001
    assert 2006 in out      # gap-free span years are counted
    assert 2004 not in out  # pre-span years are not
