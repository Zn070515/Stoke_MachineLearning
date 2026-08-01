"""Minute K-line data source via Tencent Finance mkline HTTP API.

Pure HTTP JSON — independent server from Sina. Returns up to ~320 bars:
  - 5min:  ~2 weeks
  - 15min: ~1 month
  - 30min: ~2 months
  - 60min: ~4 months

Best used for recent data backfill or as a cross-validation source.
Column order is [time, open, close, high, low, volume] per Tencent's format.
"""
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TENCENT_MKLINE_URL = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
_MAX_COUNT = 320
_RATE_LIMIT = 0.5  # seconds between calls
_REQUEST_TIMEOUT = 15.0


class TencentMinuteSource:
    """Minute K-line fetcher via Tencent Finance mkline HTTP API."""

    SOURCE_NAME = "tencent-minute"

    @staticmethod
    def _to_tencent_symbol(stock_code: str) -> str:
        code = str(stock_code).zfill(6)
        if code.startswith(("6", "9")):
            return f"sh{code}"
        return f"sz{code}"

    def fetch(
        self,
        stock_code: str,
        period: str = "60",
        adjust: str = "",
    ) -> pd.DataFrame:
        """Fetch minute K-line data for a single stock.

        Args:
            stock_code: 6-digit A-share code, e.g. '000001', '600519'.
            period: bar frequency — '5', '15', '30', or '60' minutes.
            adjust: ignored (Tencent always returns unfadjusted close).

        Returns:
            DataFrame with [datetime, open, high, low, close, volume,
            stock_code, bar_period].
        """
        symbol = self._to_tencent_symbol(stock_code)
        ts = int(period)

        time.sleep(_RATE_LIMIT)

        try:
            resp = requests.get(
                _TENCENT_MKLINE_URL,
                params={"param": f"{symbol},m{ts},,{_MAX_COUNT}"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Tencent mkline fetch failed for %s (%s): %s",
                           stock_code, symbol, str(e)[:100])
            return pd.DataFrame()

        try:
            data = resp.json()
        except Exception:
            logger.warning("Tencent mkline JSON parse failed for %s (%s)",
                           stock_code, symbol)
            return pd.DataFrame()

        mkey = f"m{ts}"
        stock_data = data.get("data", {}).get(symbol, {})
        bars = stock_data.get(mkey, [])

        if not bars:
            return pd.DataFrame()

        return self._normalize(bars, stock_code, period)

    def _normalize(
        self,
        raw: list[list],
        stock_code: str,
        period: str,
    ) -> pd.DataFrame:
        """Convert Tencent mkline format to standard schema.

        Tencent returns: [time, open, close, high, low, volume]
        Note: column order differs from standard (close before high/low).
        """
        columns = ["time_str", "open", "close", "high", "low", "volume"]
        df = pd.DataFrame(raw, columns=columns)

        # Parse time: YYYYMMDDHHMM → datetime
        df["datetime"] = pd.to_datetime(df["time_str"], format="%Y%m%d%H%M")
        df.drop(columns=["time_str"], inplace=True)

        df["stock_code"] = stock_code

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["bar_period"] = period

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df = df[(df["close"] > 0) & (df["open"] > 0)]

        keep = ["datetime", "open", "high", "low", "close",
                "volume", "stock_code", "bar_period"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
