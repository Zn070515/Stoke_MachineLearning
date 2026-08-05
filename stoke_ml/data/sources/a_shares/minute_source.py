"""Minute K-line data source via AKShare Sina Finance.

Provides 5/15/30/60-minute K-line data for A-shares. Sina returns
up to ~1970 bars per call, covering approximately:
  - 5min:  ~2 months
  - 15min: ~6 months
  - 30min: ~1 year
  - 60min: ~2 years

Data is retrieved with forward-adjusted (qfq) prices by default.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional

import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code

logger = logging.getLogger(__name__)

_RATE_LIMIT = 0.6  # seconds between calls (Sina is lenient but be polite)
_API_TIMEOUT = 30.0  # per-call timeout to prevent py_mini_racer hangs


class MinuteSource:
    """Minute K-line fetcher via AKShare -> Sina Finance."""

    SOURCE_NAME = "akshare-sina-minute"

    @staticmethod
    def _to_sina_symbol(stock_code: str) -> str:
        """Map stock code to Sina symbol (sh/sz prefix)."""
        # §六: route through the single sanitizer — a bare str().zfill(6)
        # turns the float 600001.0 into the illegal "600001.0".
        code = normalize_stock_code(stock_code)
        if code is None:
            raise ValueError(f"Unusable stock code for minute fetch: {stock_code!r}")
        if code.startswith(("6", "9")):
            return f"sh{code}"
        return f"sz{code}"

    def fetch(
        self,
        stock_code: str,
        period: str = "60",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Fetch minute K-line data for a single stock.

        Args:
            stock_code: 6-digit A-share code, e.g. '000001', '600519'.
            period: bar frequency — '5', '15', '30', or '60' minutes.
            adjust: 'qfq' (forward, default), 'hfq' (backward), or '' (none).

        Returns:
            DataFrame with columns: [datetime, open, high, low, close,
            volume, amount, stock_code, bar_period].
            Empty DataFrame if no data available.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.error("AKShare not installed")
            return pd.DataFrame()

        symbol = self._to_sina_symbol(stock_code)

        time.sleep(_RATE_LIMIT)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                ak.stock_zh_a_minute,
                symbol=symbol, period=period, adjust=adjust,
            )
            raw = future.result(timeout=_API_TIMEOUT)
        except FutureTimeoutError:
            executor.shutdown(wait=False)
            logger.warning("Minute fetch timeout for %s (%s) after %.0fs",
                           stock_code, symbol, _API_TIMEOUT)
            return pd.DataFrame()
        except Exception as e:
            executor.shutdown(wait=False)
            logger.warning("Minute fetch failed for %s (%s): %s",
                           stock_code, symbol, str(e)[:100])
            return pd.DataFrame()
        else:
            executor.shutdown(wait=True)

        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        return self._normalize(raw, stock_code, period)

    def _normalize(
        self,
        df: pd.DataFrame,
        stock_code: str,
        period: str,
    ) -> pd.DataFrame:
        """Normalize Sina minute columns to standard schema."""
        df = df.copy()
        # Sina returns columns: [day, open, high, low, close, volume, amount]
        df.rename(columns={"day": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["stock_code"] = stock_code

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["bar_period"] = period

        # Drop rows with NaN OHLC (shouldn't happen but be safe)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # Basic sanity: drop zero/negative prices
        df = df[(df["close"] > 0) & (df["open"] > 0)]

        keep = ["datetime", "open", "high", "low", "close",
                "volume", "amount", "stock_code", "bar_period"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
