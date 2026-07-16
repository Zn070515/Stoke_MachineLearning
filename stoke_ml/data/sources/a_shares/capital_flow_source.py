"""Capital flow data source (资金流向) via Sina Finance.

Provides per-stock daily capital flow:
- Main force net flow (主力净流入)

EastMoney push2his daily endpoint went offline 2026-07; switched to Sina
Finance which returns net_amount (total net flow). Tiered breakdown
(super/large/mid/small) is not available from Sina.

API endpoint:
- Sina: vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/
  MoneyFlow.ssl_qsfx_zjlrqs
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SINA_FFLOW_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_zjlrqs"
)

SINA_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DAILY_NET_COLS = [
    "date", "stock_code",
    "main_net", "small_net", "mid_net", "large_net", "super_net",
]


def _sina_market_code(code: str) -> str:
    """Sina market prefix: sh for 6xxxxx/9xxxxx, sz for 0xxxxx/3xxxxx, bj for 8xxxxx."""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


class CapitalFlowSource:
    """Fetch per-stock capital flow from Sina Finance.

    Sina only provides total net_amount (no tier breakdown). We map
    net_amount → main_net and leave tier columns as 0 so that
    FlowDecomposer's L2-L4 layers still work.
    """

    SOURCE_NAME = "sina_capital_flow"

    def __init__(self, min_interval: float = 1.2):
        self._min_interval = min_interval
        self._last_call: float = 0.0

    def _throttle(self):
        """Sleep to maintain min_interval between API calls."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def fetch_daily(self, code: str, days: int = 3000) -> pd.DataFrame:
        """Fetch daily capital flow from Sina Finance.

        Returns DataFrame with columns:
            date, stock_code, main_net, small_net, mid_net,
            large_net, super_net
        """
        self._throttle()
        prefix = _sina_market_code(code)
        url = (
            f"{SINA_FFLOW_URL}?page=1&num={days}&sort=opendate&asc=0"
            f"&daima={prefix}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": SINA_UA,
                "Referer": "https://finance.sina.com.cn/",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Sina fund flow request failed for %s: %s", code, e)
            return pd.DataFrame(columns=DAILY_NET_COLS)

        # Try UTF-8 first, fall back to GBK for legacy Sina responses
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="replace")

        if "[" not in text or "]" not in text:
            logger.warning("Sina fund flow empty response for %s", code)
            return pd.DataFrame(columns=DAILY_NET_COLS)

        try:
            arr = json.loads(text[text.index("[") : text.rindex("]") + 1])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Sina fund flow JSON parse failed for %s", code)
            return pd.DataFrame(columns=DAILY_NET_COLS)

        rows = []
        for x in arr:
            net = float(x.get("netamount") or 0)
            rows.append({
                "date": x.get("opendate", ""),
                "stock_code": code,
                "main_net": net,
                "small_net": 0.0,
                "mid_net": 0.0,
                "large_net": 0.0,
                "super_net": 0.0,
            })

        if not rows:
            return pd.DataFrame(columns=DAILY_NET_COLS)
        df = pd.DataFrame(rows, columns=DAILY_NET_COLS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, codes: list[str], start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily capital flow for multiple stocks."""
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
        pass
