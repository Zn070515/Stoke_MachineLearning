"""Backup & alternative data sources on non-EastMoney domains.

These provide WAF diversity — when EastMoney blocks your IP, these sources
on sina.com.cn / gtimg.cn continue to work since they're on completely
different CDNs and have different (or no) rate limits.

Sources:
- SinaFundFlowSource: day-level fund flow via Sina Finance (新浪资金流)
- TencentQuoteSource: real-time quote + valuation via Tencent (腾讯行情)
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SINA_FFLOW_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_zjlrqs"
)
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

SINA_FFLOW_COLS = [
    "date", "stock_code", "close", "net_amount", "turnover",
]

TENCENT_QUOTE_COLS = [
    "date", "stock_code", "stock_name",
    "price", "last_close", "open", "high", "low",
    "change_amt", "change_pct", "amount_wan", "turnover_pct",
    "pe_ttm", "pe_static", "pb", "mcap_yi", "float_mcap_yi",
    "amplitude_pct", "vol_ratio", "limit_up", "limit_down",
]


def _sina_market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


def _tencent_market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


# ── Sina fund flow (新浪资金流备用源) ─────────────────────────────────

class SinaFundFlowSource:
    """Sina Finance daily fund flow — WAF-diverse backup for EastMoney.

    Returns per-stock daily net flow + turnover. Not as granular as
    EastMoney's 4-tier breakdown, but runs on sina.com.cn so it's
    immune to EastMoney IP bans. Zero auth, zero rate limit.
    """

    SOURCE_NAME = "sina_fund_flow"

    def fetch(self, code: str, days: int = 3000) -> pd.DataFrame:
        """Fetch daily fund flow for a stock from Sina.

        Args:
            code: Stock code (e.g. "600519").
            days: Number of trading days to fetch (default 60).

        Returns DataFrame with: date, close, net_amount(净额),
            turnover(成交额).
        """
        prefix = _sina_market_prefix(code)
        url = (
            f"{SINA_FFLOW_URL}?page=1&num={days}&sort=opendate&asc=0"
            f"&daima={prefix}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": UA,
                "Referer": "https://finance.sina.com.cn/",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                text = r.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Sina fund flow request failed for %s: %s", code, e)
            return pd.DataFrame(columns=SINA_FFLOW_COLS)

        if "[" not in text or "]" not in text:
            return pd.DataFrame(columns=SINA_FFLOW_COLS)

        try:
            arr = json.loads(text[text.index("[") : text.rindex("]") + 1])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Sina fund flow JSON parse failed for %s", code)
            return pd.DataFrame(columns=SINA_FFLOW_COLS)

        rows = []
        for x in arr:
            rows.append({
                "date": x.get("opendate", ""),
                "stock_code": code,
                "close": float(x.get("trade") or 0),
                "net_amount": float(x.get("netamount") or 0),
                "turnover": float(x.get("turnover") or 0),
            })

        if not rows:
            return pd.DataFrame(columns=SINA_FFLOW_COLS)
        df = pd.DataFrame(rows, columns=SINA_FFLOW_COLS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, codes: list[str], days: int = 60,
    ) -> pd.DataFrame:
        """Fetch Sina fund flow for multiple stocks."""
        frames = []
        for code in codes:
            df = self.fetch(code, days=days)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=SINA_FFLOW_COLS)
        return pd.concat(frames, ignore_index=True)


# ── Tencent quote (腾讯行情估值) ──────────────────────────────────────

class TencentQuoteSource:
    """Tencent Finance real-time quote + valuation.

    Batch HTTP GET, GBK encoding, ~ separated 53+ fields. No rate
    limit — Tencent is a CDN, not an API server. Returns PE/PB/
    market cap/limit prices that complement OHLCV data.

    Works for stocks, indices (000001=上证指数), and ETFs.
    """

    SOURCE_NAME = "tencent_quote"

    def fetch(self, codes: list[str]) -> pd.DataFrame:
        """Fetch real-time quotes for multiple codes.

        Args:
            codes: List of stock/index/ETF codes.

        Returns DataFrame with: price, pe_ttm, pe_static, pb,
            mcap_yi(总市值亿), float_mcap_yi(流通市值亿), change_pct,
            turnover_pct, amplitude_pct, vol_ratio, limit_up, limit_down.
        """
        prefixed = [_tencent_market_prefix(c) for c in codes]
        url = TENCENT_QUOTE_URL + ",".join(prefixed)

        req = urllib.request.Request(url)
        req.add_header("User-Agent", UA)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode("gbk", errors="ignore")
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Tencent quote request failed: %s", e)
            return pd.DataFrame(columns=TENCENT_QUOTE_COLS)

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        rows = []
        for line in data.strip().split(";"):
            if not line.strip() or "=" not in line or '"' not in line:
                continue
            key = line.split("=")[0].split("_")[-1]
            vals = line.split('"')[1].split("~")
            if len(vals) < 53:
                continue
            code = key[2:]
            rows.append({
                "date": today,
                "stock_code": code,
                "stock_name": vals[1],
                "price": float(vals[3]) if vals[3] else 0.0,
                "last_close": float(vals[4]) if vals[4] else 0.0,
                "open": float(vals[5]) if vals[5] else 0.0,
                "high": float(vals[33]) if vals[33] else 0.0,
                "low": float(vals[34]) if vals[34] else 0.0,
                "change_amt": float(vals[31]) if vals[31] else 0.0,
                "change_pct": float(vals[32]) if vals[32] else 0.0,
                "amount_wan": float(vals[37]) if vals[37] else 0.0,
                "turnover_pct": float(vals[38]) if vals[38] else 0.0,
                "pe_ttm": float(vals[39]) if vals[39] else 0.0,
                "amplitude_pct": float(vals[43]) if vals[43] else 0.0,
                "mcap_yi": float(vals[44]) if vals[44] else 0.0,
                "float_mcap_yi": float(vals[45]) if vals[45] else 0.0,
                "pb": float(vals[46]) if vals[46] else 0.0,
                "limit_up": float(vals[47]) if vals[47] else 0.0,
                "limit_down": float(vals[48]) if vals[48] else 0.0,
                "vol_ratio": float(vals[49]) if vals[49] else 0.0,
                "pe_static": float(vals[52]) if vals[52] else 0.0,
            })

        if not rows:
            return pd.DataFrame(columns=TENCENT_QUOTE_COLS)
        df = pd.DataFrame(rows, columns=TENCENT_QUOTE_COLS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_one(self, code: str) -> pd.DataFrame:
        """Fetch quote for a single code."""
        return self.fetch([code])

    def fetch_batch(
        self, codes: list[str], batch_size: int = 50,
    ) -> pd.DataFrame:
        """Fetch quotes in batches (腾讯 recommends batches ≤50)."""
        frames = []
        for i in range(0, len(codes), batch_size):
            batch = codes[i : i + batch_size]
            df = self.fetch(batch)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=TENCENT_QUOTE_COLS)
        return pd.concat(frames, ignore_index=True)
