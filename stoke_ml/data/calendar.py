"""Trading calendar for A-shares and US markets.

Generates trading day lists with weekend and holiday exclusion.
"""
import datetime as dt
import pandas as pd


class TradingCalendar:
    """Trading day calendar for a specific market."""

    A_SHARES_HOLIDAYS_2024 = {
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
    }

    US_HOLIDAYS_2024 = {
        dt.date(2024, 1, 1), dt.date(2024, 1, 15), dt.date(2024, 2, 19),
        dt.date(2024, 3, 29), dt.date(2024, 5, 27), dt.date(2024, 6, 19),
        dt.date(2024, 7, 4), dt.date(2024, 9, 2), dt.date(2024, 11, 28),
        dt.date(2024, 12, 25),
    }

    HOLIDAYS = {"a_shares": A_SHARES_HOLIDAYS_2024, "us": US_HOLIDAYS_2024}

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
