"""TradingCalendar correctness tests (review v6 §五).

A-shares NEVER trade on weekends — not even on 调休 makeup workdays.  The
calendar is weekdays MINUS official SSE/SZSE holiday closures, verified to
match the SSE official trading calendar exactly for 2001-2026.
"""
import datetime as dt

import pytest

from stoke_ml.data.calendar import TradingCalendar


@pytest.fixture
def cal():
    return TradingCalendar("a_shares")


class TestNoWeekendTrading:
    """Weekends are never trading days, even 调休 makeup workdays."""

    @pytest.mark.parametrize("d", [
        dt.date(2026, 2, 14),   # 调休 makeup Sat (春节 2026)
        dt.date(2026, 2, 22),   # 调休 makeup Sun
        dt.date(2026, 5, 9),    # 调休 makeup Sat (劳动节 2026)
        dt.date(2026, 10, 10),  # 调休 makeup Sat (国庆 2026)
        dt.date(2015, 10, 10),  # 调休 makeup Sat (国庆 2015)
        dt.date(2024, 2, 18),   # 调休 makeup Sun (春节 2024)
    ])
    def test_makeup_weekend_not_trading(self, cal, d):
        assert d.weekday() >= 5
        assert not cal.is_trading_day(d)

    def test_plain_weekend_not_trading(self, cal):
        assert not cal.is_trading_day(dt.date(2026, 8, 2))   # Sunday
        assert not cal.is_trading_day(dt.date(2026, 8, 1))   # Saturday


class Test2026Holidays:
    """2026 closures per 上证公告〔2025〕45号."""

    @pytest.mark.parametrize("d", [
        dt.date(2026, 1, 1),    # 元旦
        dt.date(2026, 1, 2),    # 元旦 (was missing before v6)
        dt.date(2026, 2, 16),   # 春节 (was missing before v6)
        dt.date(2026, 2, 17), dt.date(2026, 2, 18), dt.date(2026, 2, 19),
        dt.date(2026, 2, 20), dt.date(2026, 2, 23),
        dt.date(2026, 4, 6),    # 清明
        dt.date(2026, 5, 1), dt.date(2026, 5, 4), dt.date(2026, 5, 5),  # 劳动节
        dt.date(2026, 6, 19),   # 端午 (was 6/22 wrongly, 6/19 missing)
        dt.date(2026, 9, 25),   # 中秋 (was 9/28 wrongly)
        dt.date(2026, 10, 1), dt.date(2026, 10, 2), dt.date(2026, 10, 5),
        dt.date(2026, 10, 6), dt.date(2026, 10, 7),  # 国庆
    ])
    def test_closure(self, cal, d):
        assert not cal.is_trading_day(d)

    @pytest.mark.parametrize("d", [
        dt.date(2026, 1, 5),    # 元旦 resume
        dt.date(2026, 2, 24),   # 春节 resume
        dt.date(2026, 6, 22),   # 端午 resume (real trading day, per data)
        dt.date(2026, 9, 28),   # 中秋 resume
        dt.date(2026, 10, 8),   # 国庆 resume
        dt.date(2026, 7, 15),   # normal Wednesday
    ])
    def test_trading(self, cal, d):
        assert cal.is_trading_day(d)


class TestHistoricalClosures:
    """Pre-2015 closures the calendar used to fabricate as trading days."""

    @pytest.mark.parametrize("d", [
        dt.date(2001, 10, 1),   # 国庆 2001
        dt.date(2002, 2, 20),   # 春节 2002
        dt.date(2004, 5, 3),    # 五一 2004
        dt.date(2008, 6, 9),    # 端午 2008 (first 端午 holiday)
        dt.date(2008, 4, 4),    # 清明 2008 (first 清明 holiday)
        dt.date(2009, 10, 8),   # 国庆 2009
        dt.date(2010, 2, 16),   # 春节 2010
        dt.date(2013, 9, 19),   # 中秋 2013
        dt.date(2014, 10, 1),   # 国庆 2014
        dt.date(2018, 12, 31),  # 2019 元旦 arrangement (was missing)
    ])
    def test_closure(self, cal, d):
        assert not cal.is_trading_day(d)

    @pytest.mark.parametrize("d", [
        dt.date(2001, 7, 6),    # normal Friday 2001
        dt.date(2008, 6, 18),   # normal Wednesday 2008
        dt.date(2014, 9, 11),   # normal Thursday 2014
    ])
    def test_normal_trading(self, cal, d):
        assert cal.is_trading_day(d)


class TestSequences:
    def test_get_trading_days_excludes_holidays_and_weekends(self, cal):
        days = cal.get_trading_days("2010-02-01", "2010-02-28")
        assert days == sorted(set(days))
        # 春节 2010 closures (2/15-2/19) excluded; all returned are weekdays
        # outside those dates
        for d in days:
            assert d.weekday() < 5
        assert dt.date(2010, 2, 15) not in days
        assert dt.date(2010, 2, 16) not in days
        assert dt.date(2010, 2, 24) in days  # normal Wed after holiday

    def test_get_trading_days_2001_golden_week(self, cal):
        # 2001 国庆 7-day break: 10/1-10/7 closed, resumes 10/8
        days = cal.get_trading_days("2001-09-28", "2001-10-10")
        assert days == [
            dt.date(2001, 9, 28),
            dt.date(2001, 10, 8), dt.date(2001, 10, 9), dt.date(2001, 10, 10),
        ]

    def test_next_trading_day_across_spring_festival(self, cal):
        # 2026-02-13 (Fri) is the last day before 春节; next is 2/24 (Tue)
        assert cal.next_trading_day(dt.date(2026, 2, 13)) == dt.date(2026, 2, 24)

    def test_next_trading_day_after_weekend(self, cal):
        assert cal.next_trading_day(dt.date(2026, 7, 31)) == dt.date(2026, 8, 3)


class TestSeededCounts:
    """A few year-end snapshots to guard against wholesale regression."""

    @pytest.mark.parametrize("year,expected_days", [
        (2002, 237), (2008, 246), (2013, 238), (2019, 244),
    ])
    def test_year_trading_day_count(self, cal, year, expected_days):
        days = cal.get_trading_days(f"{year}-01-01", f"{year}-12-31")
        assert len(days) == expected_days, (
            f"{year}: {len(days)} != {expected_days}")
