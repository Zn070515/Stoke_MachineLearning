"""Limit-up board data source (打板数据) via EastMoney push2ex + 同花顺.

Provides daily limit-up/limit-down/busted pools and board sentiment:
- ZT pool (涨停池): all stocks that hit limit-up today
- ZB pool (炸板池): stocks that hit limit-up then opened
- DT pool (跌停池): all stocks that hit limit-down today
- YZT pool (昨日涨停池): yesterday's ZT stocks, today's performance
- Sentiment summary: break rate, ladder, max height, advance rate

API endpoints:
- push2ex.eastmoney.com/getTopicZTPool (涨停池)
- push2ex.eastmoney.com/getTopicZBPool (炸板池)
- push2ex.eastmoney.com/getTopicDTPool (跌停池)
- push2ex.eastmoney.com/getTopicYesterdayZTPool (昨日涨停池)
- data.10jqka.com.cn/dataapi/limit_up/limit_up_pool (同花顺涨停揭秘)
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from stoke_ml.crawler.eastmoney import EastMoneyClient
from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

PUSH2EX_BASE = "https://push2ex.eastmoney.com"
THS_LIMIT_UP_URL = "https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool"

ZTB_UT = "7eea3edcaed734bea9cbfc24409ed989"

EASTMONEY_HEADERS = {
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
}

ZT_POOL_COLS = [
    "date", "stock_code", "stock_name", "price", "pct",
    "amount", "float_cap", "turnover", "limit_days",
    "first_seal", "last_seal", "seal_fund", "break_times",
    "industry", "zt_stat",
]

ZB_POOL_COLS = [
    "date", "stock_code", "stock_name", "price", "limit_price", "pct",
    "turnover", "first_seal", "break_times", "amplitude", "speed",
    "industry", "zt_stat",
]

DT_POOL_COLS = [
    "date", "stock_code", "stock_name", "price", "pct",
    "turnover", "pe", "seal_fund", "last_seal",
    "board_amount", "dt_days", "open_times", "industry",
]

YZT_POOL_COLS = [
    "date", "stock_code", "stock_name", "price", "pct",
    "turnover", "amplitude", "speed",
    "y_first_seal", "y_limit_days", "industry", "zt_stat",
]

THS_LIMIT_UP_COLS = [
    "date", "stock_code", "stock_name", "price", "pct",
    "reason", "board_type", "seal_rate", "break_times",
    "seal_amount", "high_days", "first_time", "is_again",
]

SENTIMENT_COLS = [
    "date", "zt_count", "zb_count", "dt_count", "yzt_count",
    "break_rate", "max_height", "advance_rate",
    "ladder_2", "ladder_3", "ladder_4", "ladder_5",
    "ladder_6plus",
]


def _fmt_zt_time(t) -> str:
    """Format limit-up time integer or string -> HH:MM:SS (92500 -> 09:25:00)."""
    if t is None or t == "":
        return ""
    s = str(t).strip()
    if ":" in s:
        return s  # already formatted like "09:25:00"
    s = s.zfill(6)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"


def _safe_float(val, default=0.0):
    if val is None or val == "-":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None or val == "-":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _date8(date_str: str) -> str:
    """Normalize date string to YYYYMMDD."""
    return date_str.replace("-", "")


class LimitUpSource:
    """Fetch limit-up board data from EastMoney push2ex + 同花顺 enrichment."""

    SOURCE_NAME = "eastmoney_limit_up"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    # ── EastMoney push2ex pools ──────────────────────────────────────────

    def _em_zt_api(self, endpoint: str, sort: str, date: str) -> list[dict]:
        """EastMoney limit-up board center unified request.

        endpoint: getTopicZTPool / getTopicZBPool / getTopicDTPool / getYesterdayZTPool
        Returns data.pool raw list (data is null on non-trading days).
        """
        url = f"{PUSH2EX_BASE}/{endpoint}"
        params = {
            "ut": ZTB_UT,
            "dpt": "wz.ztzt",
            "Pageindex": 0,
            "pagesize": 10000,
            "sort": sort,
            "date": _date8(date),
        }
        try:
            r = self._client.get(
                url, params=params, headers=EASTMONEY_HEADERS, timeout=10,
            )
            r.raise_for_status()
            return (r.json().get("data") or {}).get("pool") or []
        except Exception:
            logger.warning("Limit-up pool %s failed for %s", endpoint, date)
            return []

    def _parse_zt_stat(self, p: dict) -> str:
        """Parse zttj field into 'N天M板' string."""
        zttj = p.get("zttj") or {}
        days = zttj.get("days", "?")
        ct = zttj.get("ct", "?")
        return f"{days}天{ct}板"

    def fetch_zt_pool(self, date: str) -> pd.DataFrame:
        """Fetch limit-up pool (涨停池) for a trading day.

        Returns columns: code, name, price(actual), pct, amount, float_cap,
        turnover, limit_days(连板数), first_seal, last_seal, seal_fund(封板资金),
        break_times(炸板次数), industry, zt_stat(N天M板)
        """
        rows = []
        for p in self._em_zt_api("getTopicZTPool", "fbt:asc", date):
            rows.append({
                "date": date,
                "stock_code": str(p.get("c", "")).zfill(6),
                "stock_name": p.get("n", ""),
                "price": _safe_float(p.get("p")) / 1000,
                "pct": round(_safe_float(p.get("zdp")), 2),
                "amount": _safe_float(p.get("amount")),
                "float_cap": _safe_float(p.get("ltsz")),
                "turnover": round(_safe_float(p.get("hs")), 2),
                "limit_days": _safe_int(p.get("lbc")),
                "first_seal": _fmt_zt_time(p.get("fbt")),
                "last_seal": _fmt_zt_time(p.get("lbt")),
                "seal_fund": _safe_float(p.get("fund")),
                "break_times": _safe_int(p.get("zbc")),
                "industry": p.get("hybk", ""),
                "zt_stat": self._parse_zt_stat(p),
            })
        if not rows:
            return pd.DataFrame(columns=ZT_POOL_COLS)
        df = pd.DataFrame(rows, columns=ZT_POOL_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_zb_pool(self, date: str) -> pd.DataFrame:
        """Fetch busted limit-up pool (炸板池) for a trading day."""
        rows = []
        for p in self._em_zt_api("getTopicZBPool", "fbt:asc", date):
            rows.append({
                "date": date,
                "stock_code": str(p.get("c", "")).zfill(6),
                "stock_name": p.get("n", ""),
                "price": _safe_float(p.get("p")) / 1000,
                "limit_price": _safe_float(p.get("ztp")) / 1000,
                "pct": round(_safe_float(p.get("zdp")), 2),
                "turnover": round(_safe_float(p.get("hs")), 2),
                "first_seal": _fmt_zt_time(p.get("fbt")),
                "break_times": _safe_int(p.get("zbc")),
                "amplitude": round(_safe_float(p.get("zf")), 2),
                "speed": round(_safe_float(p.get("zs")), 2),
                "industry": p.get("hybk", ""),
                "zt_stat": self._parse_zt_stat(p),
            })
        if not rows:
            return pd.DataFrame(columns=ZB_POOL_COLS)
        df = pd.DataFrame(rows, columns=ZB_POOL_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_dt_pool(self, date: str) -> pd.DataFrame:
        """Fetch limit-down pool (跌停池) for a trading day."""
        rows = []
        for p in self._em_zt_api("getTopicDTPool", "fund:asc", date):
            rows.append({
                "date": date,
                "stock_code": str(p.get("c", "")).zfill(6),
                "stock_name": p.get("n", ""),
                "price": _safe_float(p.get("p")) / 1000,
                "pct": round(_safe_float(p.get("zdp")), 2),
                "turnover": round(_safe_float(p.get("hs")), 2),
                "pe": _safe_float(p.get("pe")),
                "seal_fund": _safe_float(p.get("fund")),
                "last_seal": _fmt_zt_time(p.get("lbt")),
                "board_amount": _safe_float(p.get("fba")),
                "dt_days": _safe_int(p.get("days")),
                "open_times": _safe_int(p.get("oc")),
                "industry": p.get("hybk", ""),
            })
        if not rows:
            return pd.DataFrame(columns=DT_POOL_COLS)
        df = pd.DataFrame(rows, columns=DT_POOL_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_yzt_pool(self, date: str) -> pd.DataFrame:
        """Fetch yesterday's ZT pool performance (昨日涨停池) for a trading day."""
        rows = []
        for p in self._em_zt_api("getYesterdayZTPool", "zs:desc", date):
            rows.append({
                "date": date,
                "stock_code": str(p.get("c", "")).zfill(6),
                "stock_name": p.get("n", ""),
                "price": _safe_float(p.get("p")) / 1000,
                "pct": round(_safe_float(p.get("zdp")), 2),
                "turnover": round(_safe_float(p.get("hs")), 2),
                "amplitude": round(_safe_float(p.get("zf")), 2),
                "speed": round(_safe_float(p.get("zs")), 2),
                "y_first_seal": _fmt_zt_time(p.get("yfbt")),
                "y_limit_days": _safe_int(p.get("ylbc")),
                "industry": p.get("hybk", ""),
                "zt_stat": self._parse_zt_stat(p),
            })
        if not rows:
            return pd.DataFrame(columns=YZT_POOL_COLS)
        df = pd.DataFrame(rows, columns=YZT_POOL_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── Sentiment summary ────────────────────────────────────────────────

    def fetch_sentiment(self, date: str) -> dict:
        """Compute board sentiment: break rate, ladder, max height, advance rate.

        Returns dict with: zt_count, zb_count, dt_count, yzt_count,
        break_rate(%), max_height, advance_rate(%), ladder_2..ladder_6plus.

        NOTE: This makes 4 EastMoney API calls per date (zt/zb/dt/yzt pools).
        For batch use, prefer fetch_sentiment_batch() which iterates day-by-day
        with serial throttling — avoid calling this in a tight loop without
        the EastMoneyClient rate limiter (default 1.2s between calls).
        """
        zt = self.fetch_zt_pool(date)
        zb = self.fetch_zb_pool(date)
        dt = self.fetch_dt_pool(date)
        yzt = self.fetch_yzt_pool(date)
        return self.compute_sentiment(date, zt, zb, dt, yzt)

    @staticmethod
    def compute_sentiment(date: str, zt: pd.DataFrame, zb: pd.DataFrame,
                          dt: pd.DataFrame, yzt: pd.DataFrame) -> dict:
        """Compute board sentiment from pre-fetched pool DataFrames.

        Pure function — no API calls. Callers that already have pool data
        on disk can load it and pass it here to avoid re-fetching.
        """
        zt_n, zb_n, dt_n, yzt_n = len(zt), len(zb), len(dt), len(yzt)

        # Break rate: busted / (busted + sealed) * 100
        total_attempts = zt_n + zb_n
        break_rate = round(zb_n / total_attempts * 100, 1) if total_attempts else 0.0

        # Max consecutive limit-up days
        max_height = int(zt["limit_days"].max()) if zt_n else 0

        # Advance rate: yesterday's ZT that stayed ZT today (pct >= 9.8)
        if yzt_n:
            advanced = int((yzt["pct"] >= 9.8).sum())
            advance_rate = round(advanced / yzt_n * 100, 1)
        else:
            advance_rate = 0.0

        # Ladder: count by consecutive board count
        ladder: dict[int, int] = {}
        if zt_n:
            for days in zt["limit_days"]:
                ladder[days] = ladder.get(days, 0) + 1

        return {
            "date": date,
            "zt_count": zt_n,
            "zb_count": zb_n,
            "dt_count": dt_n,
            "yzt_count": yzt_n,
            "break_rate": break_rate,
            "max_height": max_height,
            "advance_rate": advance_rate,
            "ladder_2": ladder.get(2, 0),
            "ladder_3": ladder.get(3, 0),
            "ladder_4": ladder.get(4, 0),
            "ladder_5": ladder.get(5, 0),
            "ladder_6plus": sum(v for k, v in ladder.items() if k >= 6),
        }

    # ── Batch ────────────────────────────────────────────────────────────

    def fetch_batch(
        self, start_date: str, end_date: str,
        pools: tuple[str, ...] = ("zt", "zb", "dt", "yzt"),
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple pool types over a date range.

        Returns dict mapping pool name -> concatenated DataFrame.
        """
        calendar = TradingCalendar("a_shares")
        dates = calendar.get_trading_days(start_date, end_date)
        results: dict[str, list[pd.DataFrame]] = {p: [] for p in pools}

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            if "zt" in pools:
                results["zt"].append(self.fetch_zt_pool(date_str))
            if "zb" in pools:
                results["zb"].append(self.fetch_zb_pool(date_str))
            if "dt" in pools:
                results["dt"].append(self.fetch_dt_pool(date_str))
            if "yzt" in pools:
                results["yzt"].append(self.fetch_yzt_pool(date_str))

        return {
            k: pd.concat(v, ignore_index=True) if v
            else pd.DataFrame()
            for k, v in results.items()
        }

    def fetch_sentiment_batch(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily sentiment summary over a date range."""
        calendar = TradingCalendar("a_shares")
        dates = calendar.get_trading_days(start_date, end_date)
        rows = []
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            rows.append(self.fetch_sentiment(date_str))

        if not rows:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 同花顺 enrichment ───────────────────────────────────────────────

    def fetch_ths_limit_up(self, date: str) -> pd.DataFrame:
        """Fetch limit-up reason + board type + seal quality from 同花顺.

        Enriches ZT pool with: reason(题材), board_type(换手板/一字板/T字板),
        seal_rate(封板成功率), seal_amount(封单额), high_days(几天几板),
        first_time(首次涨停时间), is_again(是否回封).
        """
        url = THS_LIMIT_UP_URL
        params = {
            "page": 1,
            "limit": 200,
            "field": "199112,10,9001,330323,330324,330325,9002,330329,"
                     "133971,133970,1968584,3475914,9003,9004",
            "filter": "HS,GEM2STAR",
            "order_field": "330324",
            "order_type": "0",
            "date": _date8(date),
        }
        try:
            qs = urllib.parse.urlencode(params)
            req = urllib.request.Request(
                f"{url}?{qs}",
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://data.10jqka.com.cn/",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                info = (json.loads(r.read()).get("data") or {}).get("info", [])
        except Exception:
            logger.warning("同花顺 limit-up pool failed for %s", date)
            return pd.DataFrame(columns=THS_LIMIT_UP_COLS)

        rows = []
        for it in info:
            ft = it.get("first_limit_up_time")
            rows.append({
                "date": date,
                "stock_code": str(it.get("code", "")).zfill(6),
                "stock_name": it.get("name", ""),
                "price": _safe_float(it.get("latest")),
                "pct": _safe_float(it.get("change_rate")),
                "reason": it.get("reason_type", ""),
                "board_type": it.get("limit_up_type", ""),
                "seal_rate": _safe_float(it.get("limit_up_suc_rate")),
                "break_times": _safe_int(it.get("open_num")),
                "seal_amount": _safe_float(it.get("order_amount")),
                "high_days": it.get("high_days", ""),
                "first_time": (
                    datetime.fromtimestamp(
                        int(ft) // 1000 if int(ft) > 1e10 else int(ft),
                        tz=timezone(timedelta(hours=8)),
                    ).strftime("%H:%M:%S")
                    if ft else ""
                ),
                "is_again": _safe_int(it.get("is_again_limit")),
            })

        if not rows:
            return pd.DataFrame(columns=THS_LIMIT_UP_COLS)
        df = pd.DataFrame(rows, columns=THS_LIMIT_UP_COLS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def close(self):
        self._client.close()
