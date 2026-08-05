"""Base interface for A-share data sources."""
from abc import ABC, abstractmethod
import pandas as pd


class AShareSourceBase(ABC):
    """Abstract base for A-share market data fetchers."""

    SOURCE_NAME: str = "base"

    @abstractmethod
    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data and return normalized DataFrame."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is currently accessible."""
        ...

    def supported_markets(self) -> frozenset[str]:
        """Markets this source can fetch — subset of ``{"SH", "SZ", "BJ"}``.

        The failover loop consults this BEFORE calling :meth:`fetch_daily` and
        skips a source that cannot reach the requested market WITHOUT counting
        it as a failure (an unsupported market is not a fetch error, so it must
        not trip the circuit breaker).  Base default covers all three A-share
        markets; a provider that cannot serve a market overrides with a
        narrower set.
        """
        return frozenset({"SH", "SZ", "BJ"})

    def supports_market(self, market: str) -> bool:
        """True iff this source can fetch data for ``market`` (SH/SZ/BJ)."""
        return market in self.supported_markets()
