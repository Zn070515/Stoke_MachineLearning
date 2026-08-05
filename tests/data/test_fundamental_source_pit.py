"""Fundamental data PIT audit — statutory disclosure-date mapping (§四).

A-share quarterly financial reports carry a legal disclosure deadline
(法定披露截止日): Q1→04-30, H1→08-31, Q3→10-31, Annual→04-30 next year.  The
upstream API (``ak.stock_financial_abstract``) exposes no announcement-date
field, so the source must NOT reuse the report-period end as the disclosure
day — that would leak figures published weeks later (often late April for a
March 31 quarter) into the model.  These tests pin:

1. the statutory-deadline mapping itself;
2. the source emitting that deadline (never the report-period end);
3. the storage warning for rows that carry no disclosure date at all, which
   previously were dropped silently.
"""
import logging

import pandas as pd
import pytest

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.fundamental_storage import FundamentalStorage
from stoke_ml.data.sources.a_shares.fundamental_source import (
    FundamentalSource,
    _statutory_disclosure_date,
)


@pytest.mark.parametrize(
    "report_date, expected",
    [
        ("2025-03-31", "2025-04-30"),  # Q1 → same-year 04-30
        ("2025-06-30", "2025-08-31"),  # H1 → same-year 08-31
        ("2025-09-30", "2025-10-31"),  # Q3 → same-year 10-31
        ("2025-12-31", "2026-04-30"),  # Annual → next-year 04-30
    ],
)
def test_statutory_disclosure_date_mapping(report_date, expected):
    assert _statutory_disclosure_date(report_date) == pd.Timestamp(expected)


def test_statutory_disclosure_date_never_before_report_date():
    """Even for non-canonical report dates the deadline must be >= the end."""
    for rd in ("2025-01-15", "2025-05-20", "2025-08-01", "2025-11-11"):
        assert _statutory_disclosure_date(rd) >= pd.Timestamp(rd)


class TestFundamentalSourceDiscloseDate:
    """The source must emit a statutory deadline, never the report-period end."""

    def _normalize(self):
        # Mimic the wide-format ak.stock_financial_abstract layout:
        # 8-digit YYYYMMDD report-date columns × indicator rows.
        raw = pd.DataFrame({
            "选项": ["常用指标", "常用指标", "常用指标"],
            "指标": ["净资产收益率(ROE)", "营业总收入", "归母净利润"],
            "20250331": [10.5, 100.0, 20.0],
            "20250630": [11.0, 120.0, 25.0],
            "20250930": [12.0, 140.0, 30.0],
            "20251231": [13.0, 160.0, 35.0],
        })
        return FundamentalSource()._normalize(raw, "000001")

    def test_disclose_date_is_statutory_deadline(self):
        res = self._normalize()
        expected = {
            pd.Timestamp("2025-03-31"): pd.Timestamp("2025-04-30"),
            pd.Timestamp("2025-06-30"): pd.Timestamp("2025-08-31"),
            pd.Timestamp("2025-09-30"): pd.Timestamp("2025-10-31"),
            pd.Timestamp("2025-12-31"): pd.Timestamp("2026-04-30"),
        }
        assert set(res["report_date"]) == set(expected)
        for rd, dd in zip(res["report_date"], res["disclose_date"]):
            assert dd == expected[rd], f"{rd} → {dd}, want {expected[rd]}"
            assert dd >= rd, f"disclose_date {dd} must not precede {rd}"

    def test_disclose_date_not_equal_to_report_date(self):
        """The report-period end must not be reused as the disclosure day."""
        res = self._normalize()
        assert (res["disclose_date"] != res["report_date"]).all()


class TestFundamentalStorageNanDiscloseDate:
    """forward_fill_to_daily must warn (not silently drop) a NaN disclose_date."""

    def _storage(self, tmp_path) -> FundamentalStorage:
        return FundamentalStorage(
            str(tmp_path), calendar=TradingCalendar("a_shares")
        )

    def test_nan_disclose_date_warns(self, tmp_path, caplog):
        storage = self._storage(tmp_path)
        df = pd.DataFrame({
            "stock_code": ["000001", "000001"],
            "report_date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "disclose_date": pd.to_datetime([pd.NaT, "2025-08-31"]),
            "roe": [10.0, 11.0],
        })
        storage.save(df)

        with caplog.at_level(
            logging.WARNING, logger="stoke_ml.data.fundamental_storage"
        ):
            storage.forward_fill_to_daily("000001", "2025-04-01", "2025-09-30")

        msgs = [r.getMessage() for r in caplog.records]
        assert msgs, "expected a warning for the NaN disclose_date row"
        assert any(
            "000001" in m and "2025-03-31" in m for m in msgs
        ), f"warning must name the stock & report period, got {msgs}"

    def test_valid_disclose_date_does_not_warn(self, tmp_path, caplog):
        """All disclose dates present → no warning, values forward-fill."""
        storage = self._storage(tmp_path)
        df = pd.DataFrame({
            "stock_code": ["000001"],
            "report_date": pd.to_datetime(["2025-06-30"]),
            "disclose_date": pd.to_datetime(["2025-08-31"]),
            "roe": [11.0],
        })
        storage.save(df)

        with caplog.at_level(
            logging.WARNING, logger="stoke_ml.data.fundamental_storage"
        ):
            out = storage.forward_fill_to_daily(
                "000001", "2025-08-01", "2025-09-30"
            )

        msgs = [r.getMessage() for r in caplog.records]
        assert not any("no valid disclose_date" in m for m in msgs)
        # roe anchored at the disclosure date reaches daily rows
        assert out["roe"].notna().any()
