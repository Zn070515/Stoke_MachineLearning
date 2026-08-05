"""Minute K-line data source via Sina Finance direct HTTP API.

Pure HTTP JSON — no py_mini_racer V8 engine, no akshare dependency.
Returns up to ~1970 bars per call:
  - 5min:  ~2 months
  - 15min: ~6 months
  - 30min: ~1 year
  - 60min: ~2 years (recommended for maximum history)
"""
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from stoke_ml.data.codes import normalize_stock_code

logger = logging.getLogger(__name__)

_SINA_MINUTE_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_MAX_DATALEN = 1970
_RATE_LIMIT = 0.3  # seconds between calls (direct HTTP, no V8 overhead)
_REQUEST_TIMEOUT = 15.0


class SinaDirectMinuteSource:
    """Minute K-line fetcher via Sina Finance direct HTTP API."""

    SOURCE_NAME = "sina-direct-minute"

    @staticmethod
    def _to_sina_symbol(stock_code: str) -> str:
        # §六: single sanitizer, never bare str().zfill(6).
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
        adjust: str = "",
    ) -> pd.DataFrame:
        """Fetch minute K-line data for a single stock.

        Args:
            stock_code: 6-digit A-share code, e.g. '000001', '600519'.
            period: bar frequency — '5', '15', '30', or '60' minutes.
            adjust: ignored (Sina direct API always returns qfq data).

        Returns:
            DataFrame with [datetime, open, high, low, close, volume,
            stock_code, bar_period]. No 'amount' column (Sina direct
            does not provide turnover amount).
            Empty DataFrame if no data available.
        """
        symbol = self._to_sina_symbol(stock_code)
        scale = int(period)

        time.sleep(_RATE_LIMIT)

        try:
            resp = requests.get(
                _SINA_MINUTE_URL,
                params={
                    "symbol": symbol,
                    "scale": scale,
                    "ma": "no",
                    "datalen": _MAX_DATALEN,
                },
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Sina direct fetch failed for %s (%s): %s",
                           stock_code, symbol, str(e)[:100])
            return pd.DataFrame()

        text = resp.text.strip()
        if not text:
            return pd.DataFrame()

        try:
            raw = resp.json()
        except Exception:
            logger.warning("Sina direct JSON parse failed for %s (%s)",
                           stock_code, symbol)
            return pd.DataFrame()

        if not isinstance(raw, list) or len(raw) == 0:
            return pd.DataFrame()

        return self._normalize(raw, stock_code, period)

    def _normalize(
        self,
        raw: list[dict],
        stock_code: str,
        period: str,
    ) -> pd.DataFrame:
        """Convert Sina JSON to standard schema."""
        df = pd.DataFrame(raw)
        df.rename(columns={"day": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["stock_code"] = stock_code

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["bar_period"] = period

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df = df[(df["close"] > 0) & (df["open"] > 0)]

        keep = ["datetime", "open", "high", "low", "close",
                "volume", "stock_code", "bar_period"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
