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
