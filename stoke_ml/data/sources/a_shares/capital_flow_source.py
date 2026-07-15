"""Capital flow data source (资金流向) via EastMoney push2 / push2his.

Provides per-stock daily and minute-level capital flow:
- Main force net flow (主力净流入)
- Super-large order net flow (超大单)
- Large order net flow (大单)
- Medium order net flow (中单)
- Small order net flow (小单)

All amounts in CNY (元).

API endpoints:
- Minute: push2.eastmoney.com/api/qt/stock/fflow/kline/get
- Daily 120d: push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
"""

import logging
from typing import Optional

import pandas as pd

from stoke_ml.crawler.eastmoney import EastMoneyClient

logger = logging.getLogger(__name__)

PUSH2_FFLOW_URL = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
PUSH2HIS_FFLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"

EASTMONEY_HEADERS = {
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
}

DAILY_NET_COLS = [
    "date", "stock_code",
    "main_net", "small_net", "mid_net", "large_net", "super_net",
]

MINUTE_NET_COLS = [
    "time", "stock_code",
    "main_net", "small_net", "mid_net", "large_net", "super_net",
]


def _market_code(stock_code: str) -> str:
    """EastMoney market prefix: 1 for SH (6xxxxx), 0 for SZ."""
    return "1" if stock_code.startswith("6") else "0"


class CapitalFlowSource:
    """Fetch per-stock capital flow from EastMoney."""

    SOURCE_NAME = "eastmoney_capital_flow"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch_daily(self, code: str) -> pd.DataFrame:
        """Fetch 120 trading days of daily capital flow for a stock.

        Returns DataFrame with columns:
            date, stock_code, main_net, small_net, mid_net,
            large_net, super_net
        All amounts in CNY.
        """
        secid = f"{_market_code(code)}.{code}"
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,"
                       "f60,f61,f62,f63,f64,f65",
            "lmt": "120",
        }
        try:
            r = self._client.get(
                PUSH2HIS_FFLOW_URL, params=params,
                headers=EASTMONEY_HEADERS, timeout=15,
            )
            r.raise_for_status()
            d = r.json()
        except Exception:
            logger.warning("Capital flow daily fetch failed for %s", code)
            return pd.DataFrame(columns=DAILY_NET_COLS)

        klines = d.get("data", {}).get("klines", [])
        if not klines:
            return pd.DataFrame(columns=DAILY_NET_COLS)

        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append({
                "date": parts[0],
                "stock_code": code,
                "main_net": _safe_float(parts[1]),
                "small_net": _safe_float(parts[2]),
                "mid_net": _safe_float(parts[3]),
                "large_net": _safe_float(parts[4]),
                "super_net": _safe_float(parts[5]),
            })

        df = pd.DataFrame(rows, columns=DAILY_NET_COLS)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_minute(self, code: str) -> pd.DataFrame:
        """Fetch today's minute-level capital flow for a stock.

        Returns DataFrame with columns:
            time, stock_code, main_net, small_net, mid_net,
            large_net, super_net
        All amounts in CNY.
        """
        secid = f"{_market_code(code)}.{code}"
        params = {
            "secid": secid,
            "klt": 1,  # 1-minute bars
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
        }
        try:
            r = self._client.get(
                PUSH2_FFLOW_URL, params=params,
                headers=EASTMONEY_HEADERS, timeout=10,
            )
            r.raise_for_status()
            d = r.json()
        except Exception:
            logger.warning("Capital flow minute fetch failed for %s", code)
            return pd.DataFrame(columns=MINUTE_NET_COLS)

        rows = []
        for line in d.get("data", {}).get("klines", []):
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append({
                "time": parts[0],
                "stock_code": code,
                "main_net": _safe_float(parts[1]),
                "small_net": _safe_float(parts[2]),
                "mid_net": _safe_float(parts[3]),
                "large_net": _safe_float(parts[4]),
                "super_net": _safe_float(parts[5]),
            })

        df = pd.DataFrame(rows, columns=MINUTE_NET_COLS)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df

    def fetch_batch(
        self, codes: list[str], start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily capital flow for multiple stocks.

        Date filtering is applied post-fetch (API always returns 120d).
        """
        frames = []
        for code in codes:
            df = self.fetch_daily(code)
            if df.empty:
                continue
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=DAILY_NET_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._client.close()


def _safe_float(val: str) -> float:
    """Parse float, return 0.0 for '-' or invalid values."""
    if val == "-" or val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
