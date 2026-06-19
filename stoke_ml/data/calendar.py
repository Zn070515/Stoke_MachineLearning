"""Trading calendar for A-shares and US markets.

Generates trading day lists with weekend and holiday exclusion.
"""
import datetime as dt
import pandas as pd


class TradingCalendar:
    """Trading day calendar for a specific market."""

    A_SHARES_HOLIDAYS = {
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
        # 2026
        dt.date(2026, 1, 1),
        dt.date(2026, 2, 17), dt.date(2026, 2, 18), dt.date(2026, 2, 19),
        dt.date(2026, 2, 20), dt.date(2026, 2, 23),
        dt.date(2026, 4, 6),
        dt.date(2026, 5, 1), dt.date(2026, 5, 4), dt.date(2026, 5, 5),
        dt.date(2026, 6, 22),
        dt.date(2026, 9, 28),
        dt.date(2026, 10, 1), dt.date(2026, 10, 2), dt.date(2026, 10, 5),
        dt.date(2026, 10, 6), dt.date(2026, 10, 7),
    }

    US_HOLIDAYS_2024 = {
        dt.date(2024, 1, 1), dt.date(2024, 1, 15), dt.date(2024, 2, 19),
        dt.date(2024, 3, 29), dt.date(2024, 5, 27), dt.date(2024, 6, 19),
        dt.date(2024, 7, 4), dt.date(2024, 9, 2), dt.date(2024, 11, 28),
        dt.date(2024, 12, 25),
    }

    HOLIDAYS = {"a_shares": A_SHARES_HOLIDAYS, "us": US_HOLIDAYS_2024}

    def __init__(self, market: str = "a_shares"):
        if market not in self.HOLIDAYS:
            raise ValueError(f"Unknown market: {market}. Choose: a_shares, us")
        self.market = market
        self._holidays = self.HOLIDAYS[market]

    def get_trading_days(
        self, start: str | dt.date, end: str | dt.date
    ) -> list[dt.date]:
        if isinstance(start, str):
            start = dt.date.fromisoformat(start)
        if isinstance(end, str):
            end = dt.date.fromisoformat(end)
        dates = pd.bdate_range(start=start, end=end).date
        return [d for d in dates if d not in self._holidays]

    def is_trading_day(self, date: dt.date) -> bool:
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
