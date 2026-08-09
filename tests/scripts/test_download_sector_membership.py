"""§v19 P0#1: CNINFO industry-change events → per-date PIT sector membership.

The parse is pure (pandas only, no network): CNINFO records CHANGE events only,
so a stock whose CSRC gate never changed yields ONE event (the most recent
standard rename).  The honest-PIT rule under test: a stock's gate is asserted
only from its FIRST CSRC event's ``变更日期`` forward — earlier dates stay
unclassified.  All three 证监会 standard labels merge at 门类 level because the
A–S gate letters are stable across the 2001 / 2012 / 中国上市公司协会 renames.
"""
import pandas as pd

from scripts.production.download_sector_membership import parse_cninfo_events


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
