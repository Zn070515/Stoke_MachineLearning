"""Trading calendar for A-shares and US markets.

A-shares NEVER trade on weekends — not even on 调休 makeup workdays
(State Council 调休 applies to banks/government offices; the SSE/SZSE/BSE
do not follow it).  Verified against stored market data (all makeup weekend
dates are absent from every stock's daily bars) and against the official
exchange holiday notices.  The calendar is therefore just: weekdays MINUS
official holiday closures.

Only dates through ``VERIFIED_UNTIL[market]`` are verified
exchange fact — 2027+ A-share closures are forward estimates and 2029-2030 have
no published holiday data at all.  The artifact records the verified window
(``verified_until`` / ``generated_at`` / ``status_after_verified_until``) and
strict-mode calendars (used by formal OOS flows) FAIL on any query beyond the
verified range instead of silently guessing.
"""
import datetime as dt
import hashlib
import pathlib

import pandas as pd


class TradingCalendar:
    """Trading day calendar for a specific market."""

    # Version stamp — experiments freeze the calendar version they were
    # scheduled on.  Bump whenever the holiday set or its derivation changes.
    CALENDAR_VERSION = "2026-08-04"

    # Holiday closures (weekday dates only).  Weekends are never trading days
    # and are excluded unconditionally by `is_trading_day`.
    # - 2001-2014: derived from the union of all-stock daily bars (a weekday
    #   with zero bars market-wide but active neighbors = closure) and verified
    #   to match the SSE official trading calendar EXACTLY (265 closures, 0
    #   false +/-, via akshare tool_trade_date_hist_sina).
    # - 2015-2026: SSE/SZSE published notices (上证公告〔2025〕45号 etc.),
    #   verified against stored daily bars (incl. 2018-12-31).
    # - 2027-2028: forward estimates, pending official publication (not verified
    #   fact — see VERIFIED_UNTIL / strict mode).
    A_SHARES_HOLIDAYS = {
        # 2001
        dt.date(2001, 1, 1), dt.date(2001, 1, 22), dt.date(2001, 1, 23), dt.date(2001, 1, 24), dt.date(2001, 1, 25),
        dt.date(2001, 1, 26), dt.date(2001, 1, 29), dt.date(2001, 1, 30), dt.date(2001, 1, 31), dt.date(2001, 2, 1),
        dt.date(2001, 2, 2), dt.date(2001, 5, 1), dt.date(2001, 5, 2), dt.date(2001, 5, 3), dt.date(2001, 5, 4),
        dt.date(2001, 5, 7), dt.date(2001, 10, 1), dt.date(2001, 10, 2), dt.date(2001, 10, 3), dt.date(2001, 10, 4),
        dt.date(2001, 10, 5),
        # 2002
        dt.date(2002, 1, 1), dt.date(2002, 1, 2), dt.date(2002, 1, 3), dt.date(2002, 2, 11), dt.date(2002, 2, 12),
        dt.date(2002, 2, 13), dt.date(2002, 2, 14), dt.date(2002, 2, 15), dt.date(2002, 2, 18), dt.date(2002, 2, 19),
        dt.date(2002, 2, 20), dt.date(2002, 2, 21), dt.date(2002, 2, 22), dt.date(2002, 5, 1), dt.date(2002, 5, 2),
        dt.date(2002, 5, 3), dt.date(2002, 5, 6), dt.date(2002, 5, 7), dt.date(2002, 9, 30), dt.date(2002, 10, 1),
        dt.date(2002, 10, 2), dt.date(2002, 10, 3), dt.date(2002, 10, 4), dt.date(2002, 10, 7),
        # 2003
        dt.date(2003, 1, 1), dt.date(2003, 1, 30), dt.date(2003, 1, 31), dt.date(2003, 2, 3), dt.date(2003, 2, 4),
        dt.date(2003, 2, 5), dt.date(2003, 2, 6), dt.date(2003, 2, 7), dt.date(2003, 5, 1), dt.date(2003, 5, 2),
        dt.date(2003, 5, 5), dt.date(2003, 5, 6), dt.date(2003, 5, 7), dt.date(2003, 5, 8), dt.date(2003, 5, 9),
        dt.date(2003, 10, 1), dt.date(2003, 10, 2), dt.date(2003, 10, 3), dt.date(2003, 10, 6), dt.date(2003, 10, 7),
        # 2004
        dt.date(2004, 1, 1), dt.date(2004, 1, 19), dt.date(2004, 1, 20), dt.date(2004, 1, 21), dt.date(2004, 1, 22),
        dt.date(2004, 1, 23), dt.date(2004, 1, 26), dt.date(2004, 1, 27), dt.date(2004, 1, 28), dt.date(2004, 5, 3),
        dt.date(2004, 5, 4), dt.date(2004, 5, 5), dt.date(2004, 5, 6), dt.date(2004, 5, 7), dt.date(2004, 10, 1),
        dt.date(2004, 10, 4), dt.date(2004, 10, 5), dt.date(2004, 10, 6), dt.date(2004, 10, 7),
        # 2005
        dt.date(2005, 1, 3), dt.date(2005, 2, 7), dt.date(2005, 2, 8), dt.date(2005, 2, 9), dt.date(2005, 2, 10),
        dt.date(2005, 2, 11), dt.date(2005, 2, 14), dt.date(2005, 2, 15), dt.date(2005, 5, 2), dt.date(2005, 5, 3),
        dt.date(2005, 5, 4), dt.date(2005, 5, 5), dt.date(2005, 5, 6), dt.date(2005, 10, 3), dt.date(2005, 10, 4),
        dt.date(2005, 10, 5), dt.date(2005, 10, 6), dt.date(2005, 10, 7),
        # 2006
        dt.date(2006, 1, 2), dt.date(2006, 1, 3), dt.date(2006, 1, 26), dt.date(2006, 1, 27), dt.date(2006, 1, 30),
        dt.date(2006, 1, 31), dt.date(2006, 2, 1), dt.date(2006, 2, 2), dt.date(2006, 2, 3), dt.date(2006, 5, 1),
        dt.date(2006, 5, 2), dt.date(2006, 5, 3), dt.date(2006, 5, 4), dt.date(2006, 5, 5), dt.date(2006, 10, 2),
        dt.date(2006, 10, 3), dt.date(2006, 10, 4), dt.date(2006, 10, 5), dt.date(2006, 10, 6),
        # 2007
        dt.date(2007, 1, 1), dt.date(2007, 1, 2), dt.date(2007, 1, 3), dt.date(2007, 2, 19), dt.date(2007, 2, 20),
        dt.date(2007, 2, 21), dt.date(2007, 2, 22), dt.date(2007, 2, 23), dt.date(2007, 5, 1), dt.date(2007, 5, 2),
        dt.date(2007, 5, 3), dt.date(2007, 5, 4), dt.date(2007, 5, 7), dt.date(2007, 10, 1), dt.date(2007, 10, 2),
        dt.date(2007, 10, 3), dt.date(2007, 10, 4), dt.date(2007, 10, 5), dt.date(2007, 12, 31),
        # 2008
        dt.date(2008, 1, 1), dt.date(2008, 2, 6), dt.date(2008, 2, 7), dt.date(2008, 2, 8), dt.date(2008, 2, 11),
        dt.date(2008, 2, 12), dt.date(2008, 4, 4), dt.date(2008, 5, 1), dt.date(2008, 5, 2), dt.date(2008, 6, 9),
        dt.date(2008, 9, 15), dt.date(2008, 9, 29), dt.date(2008, 9, 30), dt.date(2008, 10, 1), dt.date(2008, 10, 2),
        dt.date(2008, 10, 3),
        # 2009
        dt.date(2009, 1, 1), dt.date(2009, 1, 2), dt.date(2009, 1, 26), dt.date(2009, 1, 27), dt.date(2009, 1, 28),
        dt.date(2009, 1, 29), dt.date(2009, 1, 30), dt.date(2009, 4, 6), dt.date(2009, 5, 1), dt.date(2009, 5, 28),
        dt.date(2009, 5, 29), dt.date(2009, 10, 1), dt.date(2009, 10, 2), dt.date(2009, 10, 5), dt.date(2009, 10, 6),
        dt.date(2009, 10, 7), dt.date(2009, 10, 8),
        # 2010
        dt.date(2010, 1, 1), dt.date(2010, 2, 15), dt.date(2010, 2, 16), dt.date(2010, 2, 17), dt.date(2010, 2, 18),
        dt.date(2010, 2, 19), dt.date(2010, 4, 5), dt.date(2010, 5, 3), dt.date(2010, 6, 14), dt.date(2010, 6, 15),
        dt.date(2010, 6, 16), dt.date(2010, 9, 22), dt.date(2010, 9, 23), dt.date(2010, 9, 24), dt.date(2010, 10, 1),
        dt.date(2010, 10, 4), dt.date(2010, 10, 5), dt.date(2010, 10, 6), dt.date(2010, 10, 7),
        # 2011
        dt.date(2011, 1, 3), dt.date(2011, 2, 2), dt.date(2011, 2, 3), dt.date(2011, 2, 4), dt.date(2011, 2, 7),
        dt.date(2011, 2, 8), dt.date(2011, 4, 4), dt.date(2011, 4, 5), dt.date(2011, 5, 2), dt.date(2011, 6, 6),
        dt.date(2011, 9, 12), dt.date(2011, 10, 3), dt.date(2011, 10, 4), dt.date(2011, 10, 5), dt.date(2011, 10, 6),
        dt.date(2011, 10, 7),
        # 2012
        dt.date(2012, 1, 2), dt.date(2012, 1, 3), dt.date(2012, 1, 23), dt.date(2012, 1, 24), dt.date(2012, 1, 25),
        dt.date(2012, 1, 26), dt.date(2012, 1, 27), dt.date(2012, 4, 2), dt.date(2012, 4, 3), dt.date(2012, 4, 4),
        dt.date(2012, 4, 30), dt.date(2012, 5, 1), dt.date(2012, 6, 22), dt.date(2012, 10, 1), dt.date(2012, 10, 2),
        dt.date(2012, 10, 3), dt.date(2012, 10, 4), dt.date(2012, 10, 5),
        # 2013
        dt.date(2013, 1, 1), dt.date(2013, 1, 2), dt.date(2013, 1, 3), dt.date(2013, 2, 11), dt.date(2013, 2, 12),
        dt.date(2013, 2, 13), dt.date(2013, 2, 14), dt.date(2013, 2, 15), dt.date(2013, 4, 4), dt.date(2013, 4, 5),
        dt.date(2013, 4, 29), dt.date(2013, 4, 30), dt.date(2013, 5, 1), dt.date(2013, 6, 10), dt.date(2013, 6, 11),
        dt.date(2013, 6, 12), dt.date(2013, 9, 19), dt.date(2013, 9, 20), dt.date(2013, 10, 1), dt.date(2013, 10, 2),
        dt.date(2013, 10, 3), dt.date(2013, 10, 4), dt.date(2013, 10, 7),
        # 2014
        dt.date(2014, 1, 1), dt.date(2014, 1, 31), dt.date(2014, 2, 3), dt.date(2014, 2, 4), dt.date(2014, 2, 5),
        dt.date(2014, 2, 6), dt.date(2014, 4, 7), dt.date(2014, 5, 1), dt.date(2014, 5, 2), dt.date(2014, 6, 2),
        dt.date(2014, 9, 8), dt.date(2014, 10, 1), dt.date(2014, 10, 2), dt.date(2014, 10, 3), dt.date(2014, 10, 6),
        dt.date(2014, 10, 7),
        # 2015
        dt.date(2015, 1, 1), dt.date(2015, 1, 2),
        dt.date(2015, 2, 18), dt.date(2015, 2, 19), dt.date(2015, 2, 20),
        dt.date(2015, 2, 23), dt.date(2015, 2, 24),
        dt.date(2015, 4, 6),
        dt.date(2015, 5, 1),
        dt.date(2015, 6, 22),
        dt.date(2015, 9, 3), dt.date(2015, 9, 4),
        dt.date(2015, 10, 1), dt.date(2015, 10, 2), dt.date(2015, 10, 5),
        dt.date(2015, 10, 6), dt.date(2015, 10, 7),
        # 2016
        dt.date(2016, 1, 1),
        dt.date(2016, 2, 8), dt.date(2016, 2, 9), dt.date(2016, 2, 10),
        dt.date(2016, 2, 11), dt.date(2016, 2, 12),
        dt.date(2016, 4, 4),
        dt.date(2016, 5, 2),
        dt.date(2016, 6, 9), dt.date(2016, 6, 10),
        dt.date(2016, 9, 15), dt.date(2016, 9, 16),
        dt.date(2016, 10, 3), dt.date(2016, 10, 4), dt.date(2016, 10, 5),
        dt.date(2016, 10, 6), dt.date(2016, 10, 7),
        # 2017
        dt.date(2017, 1, 2),
        dt.date(2017, 1, 27), dt.date(2017, 1, 30), dt.date(2017, 1, 31),
        dt.date(2017, 2, 1), dt.date(2017, 2, 2),
        dt.date(2017, 4, 3), dt.date(2017, 4, 4),
        dt.date(2017, 5, 1),
        dt.date(2017, 5, 29), dt.date(2017, 5, 30),
        dt.date(2017, 10, 2), dt.date(2017, 10, 3), dt.date(2017, 10, 4),
        dt.date(2017, 10, 5), dt.date(2017, 10, 6),
        # 2018
        dt.date(2018, 1, 1),
        dt.date(2018, 2, 15), dt.date(2018, 2, 16), dt.date(2018, 2, 19),
        dt.date(2018, 2, 20), dt.date(2018, 2, 21),
        dt.date(2018, 4, 5), dt.date(2018, 4, 6),
        dt.date(2018, 4, 30), dt.date(2018, 5, 1),
        dt.date(2018, 6, 18),
        dt.date(2018, 9, 24),
        dt.date(2018, 10, 1), dt.date(2018, 10, 2), dt.date(2018, 10, 3),
        dt.date(2018, 10, 4), dt.date(2018, 10, 5),
        dt.date(2018, 12, 31),  # 2019 元旦 arrangement: 12/30-1/1 closed
        # 2019
        dt.date(2019, 1, 1),
        dt.date(2019, 2, 4), dt.date(2019, 2, 5), dt.date(2019, 2, 6),
        dt.date(2019, 2, 7), dt.date(2019, 2, 8),
        dt.date(2019, 4, 5),
        dt.date(2019, 5, 1), dt.date(2019, 5, 2), dt.date(2019, 5, 3),
        dt.date(2019, 6, 7),
        dt.date(2019, 9, 13),
        dt.date(2019, 10, 1), dt.date(2019, 10, 2), dt.date(2019, 10, 3),
        dt.date(2019, 10, 4), dt.date(2019, 10, 7),
        # 2020
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 24), dt.date(2020, 1, 27), dt.date(2020, 1, 28),
        dt.date(2020, 1, 29), dt.date(2020, 1, 30), dt.date(2020, 1, 31),
        dt.date(2020, 4, 6),
        dt.date(2020, 5, 1), dt.date(2020, 5, 4), dt.date(2020, 5, 5),
        dt.date(2020, 6, 25), dt.date(2020, 6, 26),
        dt.date(2020, 10, 1), dt.date(2020, 10, 2), dt.date(2020, 10, 5),
        dt.date(2020, 10, 6), dt.date(2020, 10, 7), dt.date(2020, 10, 8),
        # 2021
        dt.date(2021, 1, 1),
        dt.date(2021, 2, 11), dt.date(2021, 2, 12), dt.date(2021, 2, 15),
        dt.date(2021, 2, 16), dt.date(2021, 2, 17),
        dt.date(2021, 4, 5),
        dt.date(2021, 5, 3), dt.date(2021, 5, 4), dt.date(2021, 5, 5),
        dt.date(2021, 6, 14),
        dt.date(2021, 9, 20), dt.date(2021, 9, 21),
        dt.date(2021, 10, 1), dt.date(2021, 10, 4), dt.date(2021, 10, 5),
        dt.date(2021, 10, 6), dt.date(2021, 10, 7),
        # 2022
        dt.date(2022, 1, 3),
        dt.date(2022, 1, 31), dt.date(2022, 2, 1), dt.date(2022, 2, 2),
        dt.date(2022, 2, 3), dt.date(2022, 2, 4),
        dt.date(2022, 4, 4), dt.date(2022, 4, 5),
        dt.date(2022, 5, 2), dt.date(2022, 5, 3), dt.date(2022, 5, 4),
        dt.date(2022, 6, 3),
        dt.date(2022, 9, 12),
        dt.date(2022, 10, 3), dt.date(2022, 10, 4), dt.date(2022, 10, 5),
        dt.date(2022, 10, 6), dt.date(2022, 10, 7),
        # 2023
        dt.date(2023, 1, 2),
        dt.date(2023, 1, 23), dt.date(2023, 1, 24), dt.date(2023, 1, 25),
        dt.date(2023, 1, 26), dt.date(2023, 1, 27),
        dt.date(2023, 4, 5),
        dt.date(2023, 5, 1), dt.date(2023, 5, 2), dt.date(2023, 5, 3),
        dt.date(2023, 6, 22), dt.date(2023, 6, 23),
        dt.date(2023, 9, 29),
        dt.date(2023, 10, 2), dt.date(2023, 10, 3), dt.date(2023, 10, 4),
        dt.date(2023, 10, 5), dt.date(2023, 10, 6),
        # 2024
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 9), dt.date(2024, 2, 10), dt.date(2024, 2, 11),
        dt.date(2024, 2, 12), dt.date(2024, 2, 13), dt.date(2024, 2, 14),
        dt.date(2024, 2, 15), dt.date(2024, 2, 16),
        dt.date(2024, 4, 4), dt.date(2024, 4, 5),
        dt.date(2024, 5, 1), dt.date(2024, 5, 2), dt.date(2024, 5, 3),
        dt.date(2024, 6, 10),
        dt.date(2024, 9, 16), dt.date(2024, 9, 17),
        dt.date(2024, 10, 1), dt.date(2024, 10, 2), dt.date(2024, 10, 3),
        dt.date(2024, 10, 4), dt.date(2024, 10, 7),
        # 2025
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 28), dt.date(2025, 1, 29), dt.date(2025, 1, 30),
        dt.date(2025, 1, 31), dt.date(2025, 2, 3), dt.date(2025, 2, 4),
        dt.date(2025, 4, 4),
        dt.date(2025, 5, 1), dt.date(2025, 5, 2), dt.date(2025, 5, 5),
        dt.date(2025, 6, 2),
        dt.date(2025, 10, 1), dt.date(2025, 10, 2), dt.date(2025, 10, 3),
        dt.date(2025, 10, 6), dt.date(2025, 10, 7), dt.date(2025, 10, 8),
        # 2026 — 上证公告〔2025〕45号 (2025-12-22 沪深北联合发布):
        # 元旦 1/1-1/3 | 春节 2/15-2/23 | 清明 4/4-4/6 | 劳动节 5/1-5/5 |
        # 端午 6/19-6/21 | 中秋 9/25-9/27 | 国庆 10/1-10/7.
        # Weekday closures only; resume days (1/5, 2/24, 6/22, 9/28, 10/8)
        # are trading days and are NOT listed.  Verified: 6/19 closed /
        # 6/22 open in stored daily bars.
        dt.date(2026, 1, 1), dt.date(2026, 1, 2),
        dt.date(2026, 2, 16), dt.date(2026, 2, 17), dt.date(2026, 2, 18),
        dt.date(2026, 2, 19), dt.date(2026, 2, 20), dt.date(2026, 2, 23),
        dt.date(2026, 4, 6),
        dt.date(2026, 5, 1), dt.date(2026, 5, 4), dt.date(2026, 5, 5),
        dt.date(2026, 6, 19),
        dt.date(2026, 9, 25),
        dt.date(2026, 10, 1), dt.date(2026, 10, 2), dt.date(2026, 10, 5),
        dt.date(2026, 10, 6), dt.date(2026, 10, 7),
        # 2027
        dt.date(2027, 1, 1),
        dt.date(2027, 2, 8), dt.date(2027, 2, 9), dt.date(2027, 2, 10),
        dt.date(2027, 2, 11), dt.date(2027, 2, 12),
        dt.date(2027, 4, 5),
        dt.date(2027, 5, 3), dt.date(2027, 5, 4), dt.date(2027, 5, 5),
        dt.date(2027, 6, 10), dt.date(2027, 6, 11),
        dt.date(2027, 9, 24), dt.date(2027, 9, 27),
        dt.date(2027, 10, 1), dt.date(2027, 10, 4), dt.date(2027, 10, 5),
        dt.date(2027, 10, 6), dt.date(2027, 10, 7),
        # 2028
        dt.date(2028, 1, 3),
        dt.date(2028, 1, 26), dt.date(2028, 1, 27), dt.date(2028, 1, 28),
        dt.date(2028, 1, 31),
        dt.date(2028, 4, 4), dt.date(2028, 4, 5),
        dt.date(2028, 5, 1), dt.date(2028, 5, 2),
        dt.date(2028, 6, 19),
        dt.date(2028, 9, 18),
        dt.date(2028, 10, 2), dt.date(2028, 10, 3), dt.date(2028, 10, 4),
        dt.date(2028, 10, 5), dt.date(2028, 10, 6),
    }

    US_HOLIDAYS_2024 = {
        dt.date(2024, 1, 1), dt.date(2024, 1, 15), dt.date(2024, 2, 19),
        dt.date(2024, 3, 29), dt.date(2024, 5, 27), dt.date(2024, 6, 19),
        dt.date(2024, 7, 4), dt.date(2024, 9, 2), dt.date(2024, 11, 28),
        dt.date(2024, 12, 25),
    }

    HOLIDAYS = {"a_shares": A_SHARES_HOLIDAYS, "us": US_HOLIDAYS_2024}

    def __init__(
        self,
        market: str = "a_shares",
        calendar_dir: str | pathlib.Path | None = None,
        strict: bool = False,
    ):
        if market not in self.HOLIDAYS:
            raise ValueError(f"Unknown market: {market}. Choose: a_shares, us")
        self.market = market
        # Strict calendars (formal OOS flows) FAIL on any query beyond the
        # verified range instead of guessing from forward estimates.  The
        # non-strict default preserves downloader/scheduling behaviour.
        self.strict = strict
        self._holidays = self.HOLIDAYS[market]
        # The A-share calendar is the EXCHANGE-published calendar, not "workdays
        # minus holidays".  When an externally-published
        # calendar artifact (exchange_calendar/{market}.parquet) is supplied it
        # becomes the authoritative source of truth; the hardcoded holiday set
        # is the fallback (and the generator for the artifact).  Every consumer
        # goes through this one calendar, so a data-driven correction never
        # requires touching code.
        self._external = None
        self.verified_until = VERIFIED_UNTIL[market]
        if calendar_dir is not None:
            self._external = load_calendar(calendar_dir, market)
            if self._external is not None:
                self.verified_until = pd.Timestamp(
                    self._external["verified_until"].iloc[0]).date()

    def get_trading_days(
        self, start: str | dt.date, end: str | dt.date
    ) -> list[dt.date]:
        if isinstance(start, str):
            start = dt.date.fromisoformat(start)
        if isinstance(end, str):
            end = dt.date.fromisoformat(end)
        if self.strict and end > self.verified_until:
            raise ValueError(
                f"strict calendar {self.market}: range up to {end} extends past "
                f"verified_until {self.verified_until} — forward dates are "
                f"estimates, not verified exchange fact")
        if self._external is not None:
            f = self._external
            lo, hi = f["date"].min().date(), f["date"].max().date()
            if start >= lo and end <= hi:
                m = ((f["date"].dt.date >= start) & (f["date"].dt.date <= end)
                     & f["is_open"])
                return f.loc[m, "date"].dt.date.tolist()
        # Fallback (also covers ranges outside the external window): weekdays
        # only (pd.bdate_range drops weekends — A-shares never trade on
        # weekends, even on 调休 makeup workdays) minus holiday closures.
        dates = pd.bdate_range(start=start, end=end).date
        return [d for d in dates if d not in self._holidays]

    def is_trading_day(self, date: dt.date) -> bool:
        if self.strict and date > self.verified_until:
            raise ValueError(
                f"strict calendar {self.market}: {date} is past verified_until "
                f"{self.verified_until} — forward dates are estimates, not "
                f"verified exchange fact")
        if self._external is not None:
            f = self._external
            if f["date"].min().date() <= date <= f["date"].max().date():
                hit = f.loc[f["date"].dt.date == date, "is_open"]
                if not hit.empty:
                    return bool(hit.iloc[0])
        if date.weekday() >= 5:
            return False
        if date in self._holidays:
            return False
        return True

    def next_trading_day(self, date: dt.date) -> dt.date:
        candidate = date + dt.timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += dt.timedelta(days=1)
        return candidate

    def first_trading_day(self) -> dt.date:
        """Earliest known completed trading session for this market.

        The freshness clamp bound for ``most_recent_completed_trading_day``: a
        ref_date before the market's first session must return this instead of
        fabricating a pre-market weekday (an unbounded backward walk).  Uses the
        earliest ``is_open`` row of the external artifact when present, else
        walks forward from the materialized window's start to the first
        non-holiday weekday.
        """
        if self._external is not None:
            open_days = self._external.loc[self._external["is_open"], "date"]
            if not open_days.empty:
                return open_days.min().date()
            return self._external["date"].min().date()
        lo, _ = CALENDAR_WINDOW
        d = lo
        while not self.is_trading_day(d):
            d += dt.timedelta(days=1)
        return d


def get_research_calendar(
    market: str = "a_shares",
    strict: bool = False,
    data_dir: str | pathlib.Path | None = None,
) -> TradingCalendar:
    """Construct a TradingCalendar backed by the external calendar artifact.

    Single source of truth for research flows (§八): every formal consumer —
    feature date cleaning, preprocessing, topic cutoff mapping, OOS train —
    goes through this factory so one frozen ``exchange_calendar`` artifact is
    used repo-wide instead of each module silently building a hardcoded
    default calendar.  ``data_dir`` is the project data root containing
    ``exchange_calendar/{market}.parquet``; when omitted it is resolved lazily
    from ``stoke_ml.config.load_config()["project"]["data_dir"]`` (imported
    inside the function to avoid module-level circular imports and import cost
    at calendar-import time).

    ``strict=True`` makes the calendar FAIL on any query past the artifact's
    ``verified_until`` (forward estimates are not verified exchange fact) —
    use for formal research/OOS paths.  Non-strict (default) preserves
    downloader/scheduling behaviour and lets PIT mapping walk into the future.

    When the artifact is absent the calendar transparently falls back to the
    code-derived holiday set (the artifact is generated from that same code,
    so semantics are identical) — the factory does not raise.  ``TradingCalendar``
    itself remains directly constructible for tests/downloaders that want the
    artifact-free calendar.
    """
    if data_dir is None:
        from stoke_ml.config import load_config  # lazy: avoid circular import
        data_dir = load_config()["project"]["data_dir"]
    return TradingCalendar(market, calendar_dir=data_dir, strict=strict)


# ── External calendar artifact ───────────────────────────────────────────────
# The calendar is published as a self-describing parquet so consumers never
# parse holiday rules themselves.  `build_calendar_frame` materializes it from
# the verified holiday set; `save_calendar`/`load_calendar` persist and read
# it; `validate_calendar` cross-checks an on-disk artifact against the
# generator so a drifted calendar surfaces loudly (mirrors the repo's
# docs-vs-code drift guard).

# Window covered by the materialized artifact.  Wide enough to stay
# authoritative for every range real data can ask (2000-2026) plus the forward
# estimates; queries outside it transparently fall back to the code formula.
CALENDAR_WINDOW = (dt.date(2000, 1, 1), dt.date(2030, 12, 31))

# The last date each market's holiday set is VERIFIED exchange fact.
# A-share 2027-2028 are forward estimates and 2029-2030 have no published data;
# US holidays are only maintained through 2024.  Anything past this date is a
# guess, and strict calendars fail rather than answer it.
VERIFIED_UNTIL = {
    "a_shares": dt.date(2026, 12, 31),
    "us": dt.date(2024, 12, 31),
}

_CALENDAR_SCHEMA = {
    "date", "is_open", "exchange", "source", "version",
    "verified_until", "generated_at", "status_after_verified_until",
}
_EXCHANGE_NAMES = {"a_shares": "SSE/SZSE/BSE", "us": "NYSE/NASDAQ"}
_CALENDAR_SOURCES = {
    "a_shares": "sse/szse/bse_notices+verified_stored_bars",
    "us": "nyse_nasdaq_published_schedule",
}


def _calendar_path(data_dir: str | pathlib.Path, market: str = "a_shares") -> pathlib.Path:
    return pathlib.Path(data_dir) / "exchange_calendar" / f"{market}.parquet"


def build_calendar_frame(
    market: str = "a_shares",
    generated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Materialize the full trading calendar as a self-describing frame.

    One row per weekday in CALENDAR_WINDOW with the exact schema an external
    exchange-calendar feed would carry: date / is_open / exchange / source /
    version / verified_until / generated_at / status_after_verified_until.  The
    code holiday set (verified against official exchange notices and stored
    bars) is the generator; the persisted parquet is the artifact all modules
    read.  ``verified_until`` marks where the dates stop being verified
    fact; rows beyond it are forward estimates flagged ``UNKNOWN``.
    """
    if market not in TradingCalendar.HOLIDAYS:
        raise ValueError(f"Unknown market: {market}. Choose: a_shares, us")
    lo, hi = CALENDAR_WINDOW
    days = pd.bdate_range(start=lo, end=hi).date
    closed = TradingCalendar.HOLIDAYS[market]
    verified_until = VERIFIED_UNTIL[market]
    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame({
        "date": pd.Series(days, dtype="datetime64[ns]"),
        "is_open": [d not in closed for d in days],
        "exchange": _EXCHANGE_NAMES[market],
        "source": _CALENDAR_SOURCES[market],
        "version": TradingCalendar.CALENDAR_VERSION,
        "verified_until": pd.Timestamp(verified_until),
        "generated_at": generated_at,
        # Forward-estimate rows exist past verified_until, so the state of the
        # post-verified tail is unknown, not "no estimates beyond this point".
        "status_after_verified_until": "UNKNOWN" if hi > verified_until else "NONE",
    })


def save_calendar(data_dir: str | pathlib.Path, market: str = "a_shares") -> pathlib.Path:
    """Persist the calendar artifact; returns the written path."""
    path = _calendar_path(data_dir, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_calendar_frame(market).to_parquet(path, index=False)
    return path


def load_calendar(
    data_dir: str | pathlib.Path, market: str = "a_shares"
) -> pd.DataFrame | None:
    """Read the external calendar artifact, or None if it is absent.

    A present-but-malformed artifact raises — silently trusting a corrupted
    calendar is worse than failing loudly, since every downstream module would
    inherit the wrong trading days.
    """
    path = _calendar_path(data_dir, market)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"calendar artifact {path} is empty")
    missing = _CALENDAR_SCHEMA - set(frame.columns)
    if missing:
        raise ValueError(
            f"calendar artifact {path} missing columns {sorted(missing)}")
    if frame["date"].duplicated().any():
        raise ValueError(f"calendar artifact {path} has duplicate dates")
    # A gap INSIDE the artifact's own window hides corruption — a
    # missing weekday must surface at load, not silently fall back to formula
    # (which is exactly how a torn/partial artifact would get papered over).
    dates = pd.to_datetime(frame["date"]).dt.date
    lo, hi = dates.min(), dates.max()
    expected = set(pd.bdate_range(start=lo, end=hi).date)
    gaps = sorted(expected - set(dates))
    if gaps:
        raise ValueError(
            f"calendar artifact {path} is incomplete: missing weekday(s) inside "
            f"its window, e.g. {gaps[:5]}")
    return frame


def validate_calendar(
    data_dir: str | pathlib.Path, market: str = "a_shares"
) -> dict:
    """Cross-check a persisted calendar artifact against the code-derived frame.

    Returns a report dict (never raises on mismatch).  Validation is a FULL
    OUTER JOIN on ``date`` — a missing or extra date cannot silently
    vanish the way an inner merge lets it.  Checks six dimensions: missing
    dates, extra dates, status (is_open) disagreement, version, source, and
    verified-through.  A malformed artifact is reported (not raised) via
    ``reason``.
    """
    path = _calendar_path(data_dir, market)
    if not path.exists():
        return {"path": str(path), "exists": False, "trading_days": 0,
                "mismatches": 0, "ok": False, "reason": "artifact not present"}
    try:
        on_disk = load_calendar(data_dir, market)
    except ValueError as exc:
        return {"path": str(path), "exists": True, "trading_days": 0,
                "mismatches": 1, "ok": False, "reason": str(exc),
                "problems": {"load_error": str(exc)}}
    derived = build_calendar_frame(market)
    merged = on_disk.merge(derived, on="date", how="outer",
                           suffixes=("_disk", "_derived"))
    disk_open = merged["is_open_disk"].notna()
    derived_open = merged["is_open_derived"].notna()
    missing = merged[derived_open & ~disk_open]
    extra = merged[disk_open & ~derived_open]
    both = merged[disk_open & derived_open]
    problems = {
        "missing_dates": int(len(missing)),
        "extra_dates": int(len(extra)),
        "status_mismatches": int((both["is_open_disk"] != both["is_open_derived"]).sum()),
        "version_mismatch": int((both["version_disk"].fillna("") != both["version_derived"].fillna("")).sum()),
        "source_mismatch": int((both["source_disk"].fillna("") != both["source_derived"].fillna("")).sum()),
        "verified_until_mismatch": bool(
            pd.Timestamp(on_disk["verified_until"].iloc[0]).date()
            != pd.Timestamp(derived["verified_until"].iloc[0]).date()),
    }
    total = int(sum(problems.values()))
    parts = []
    if problems["missing_dates"]:
        parts.append(f"{problems['missing_dates']} missing date(s)")
    if problems["extra_dates"]:
        parts.append(f"{problems['extra_dates']} extra date(s)")
    if problems["status_mismatches"]:
        parts.append(f"{problems['status_mismatches']} status disagreement(s)")
    if problems["version_mismatch"]:
        parts.append("version mismatch")
    if problems["source_mismatch"]:
        parts.append("source mismatch")
    if problems["verified_until_mismatch"]:
        parts.append("verified_until mismatch")
    return {
        "path": str(path),
        "exists": True,
        "trading_days": int(on_disk["is_open"].sum()),
        "mismatches": total,
        "ok": bool(total == 0),
        "reason": "" if total == 0 else "artifact disagrees with the generator: " + ", ".join(parts),
        "problems": problems,
    }


def calendar_artifact_hash(
    data_dir: str | pathlib.Path, market: str = "a_shares"
) -> str:
    """Deterministic content hash (sha1, 16-hex) of the trading calendar the
    given data root resolves to.

    Text-canonical, not parquet bytes: ``generated_at`` and parquet-engine
    metadata are dropped so a fresh ``save_calendar`` round-trip hashes
    identically across pandas/parquet versions.  When the frozen artifact is
    absent the code-derived frame is hashed instead — that is the calendar a
    lenient consumer would transparently fall back to.  The digest matches the
    one ``train_panel`` records as experiment identity (``calendar_artifact_hash``),
    so a data-quality-gate report can be cross-checked against it (§九).
    """
    try:
        frame = load_calendar(data_dir, market)
    except Exception:
        frame = None
    if frame is None:
        frame = build_calendar_frame(market)
    return _calendar_frame_hash(frame)


def _calendar_frame_hash(frame: pd.DataFrame) -> str:
    canonical = (
        frame.drop(columns=["generated_at"])
        if "generated_at" in frame.columns else frame
    )
    cols = sorted(canonical.columns)
    canon_sorted = canonical.sort_values("date").reset_index(drop=True)
    lines = ["|".join(str(row[c]) for c in cols)
             for _, row in canon_sorted.iterrows()]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def most_recent_completed_trading_day(
    calendar: TradingCalendar, ref_date: dt.date
) -> dt.date:
    """Most recent trading day whose session is complete as of ``ref_date``.

    A trading day's data is published after its close, so the current day (when
    it is itself a trading day) is not yet complete — the most recently
    completed session is the last trading day strictly before ``ref_date``.
    Across 春节/国庆 7-8 day closures this walks back over the holiday weekdays
    to the last real session, so a fully-current dataset is never judged stale
    by natural-day age (§九).
    """
    d = ref_date - dt.timedelta(days=1)
    # Clamp degenerate input: a ref_date before the market's first session must
    # not fabricate a pre-market weekday (an unbounded backward walk).  Return
    # the calendar's earliest known session instead.
    earliest = calendar.first_trading_day()
    if d < earliest:
        return earliest
    while not calendar.is_trading_day(d):
        d -= dt.timedelta(days=1)
        if d < earliest:
            return earliest
    return d
