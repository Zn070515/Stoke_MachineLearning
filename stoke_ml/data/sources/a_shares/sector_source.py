"""Sector & industry data sources via EastMoney push2.

Two sources:
- IndustryRankingSource: daily industry sector ranking (全行业板块排名)
- ConceptBlockSource: per-stock concept/industry/region board membership (概念板块归属)

Note: push2 APIs don't need curl-cffi impersonation — plain requests works
and is more reliable. EastMoneyClient's TLS fingerprinting is overkill here.
"""

import json
import logging
import random
import time
from typing import Optional

import pandas as pd
import requests

from stoke_ml.crawler.eastmoney import EastMoneyClient

logger = logging.getLogger(__name__)

PUSH2_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
PUSH2_SLIST_URL = "https://push2.eastmoney.com/api/qt/slist/get"

EASTMONEY_HEADERS = {
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
}

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

INDUSTRY_RANKING_COLS = [
    "date", "rank", "name", "code", "change_pct",
    "up_count", "down_count", "leader", "leader_change",
]

CONCEPT_BLOCK_COLS = [
    "date", "stock_code",
    "board_name", "board_code", "board_change_pct", "lead_stock",
]


def _market_code(stock_code: str) -> str:
    return "1" if stock_code.startswith("6") else "0"


# ── Industry ranking (行业板块排名) ─────────────────────────────────────

class IndustryRankingSource:
    """Fetch daily industry sector ranking from EastMoney push2.

    One call returns ~100 industries ranked by change %. Uses
    EastMoney's internal sector classification (m:90+t:2).

    Uses plain urllib instead of EastMoneyClient because the push2
    API doesn't require TLS fingerprint impersonation.
    """

    SOURCE_NAME = "eastmoney_industry_ranking"

    def __init__(self, min_interval: float = 2.5):
        self._min_interval = min_interval
        self._last_call: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _UA, **EASTMONEY_HEADERS,
        })

    def _throttle(self):
        """Sleep until min_interval has passed since last call, plus jitter."""
        elapsed = time.time() - self._last_call
        wait = max(0.0, self._min_interval - elapsed) + random.uniform(0.3, 1.5)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _get_json(self, url: str, params: dict, timeout: int = 15) -> dict:
        """GET a push2 API and return parsed JSON dict.

        Uses a persistent Session for connection pooling.  Retries up
        to 5 times with jittered exponential backoff — push2delay CDN
        is aggressive with rate-limiting on bursty traffic.
        """
        last_err = None
        for attempt in range(5):
            try:
                r = self._session.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_err = e
                if attempt < 4:
                    backoff = 3.0 * (2 ** attempt) + random.uniform(0.5, 2.0)
                    time.sleep(backoff)
        raise last_err  # type: ignore[misc]

    def fetch(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch industry sector ranking for a trading day.

        Returns DataFrame sorted by rank with: rank, name, code,
        change_pct, up_count, down_count, leader, leader_change.
        """
        self._throttle()
        params = {
            "pn": "1", "pz": "200", "po": "1", "np": "1",
            "fltt": "2", "invt": "2", "fid": "f3",
            "fs": "m:90+t:2",
            "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141,f207",
        }
        try:
            d = self._get_json(PUSH2_CLIST_URL, params)
        except Exception as e:
            logger.warning("Industry ranking fetch failed: %s", e)
            return pd.DataFrame(columns=INDUSTRY_RANKING_COLS)

        items = d.get("data", {}).get("diff", [])
        if not items:
            return pd.DataFrame(columns=INDUSTRY_RANKING_COLS)

        today = date or pd.Timestamp.now().strftime("%Y-%m-%d")
        rows = []
        for i, item in enumerate(items):
            rows.append({
                "date": today,
                "rank": i + 1,
                "name": item.get("f14", ""),
                "code": item.get("f12", ""),
                "change_pct": float(item.get("f3") or 0),
                "up_count": int(item.get("f104") or 0),
                "down_count": int(item.get("f105") or 0),
                "leader": item.get("f140", ""),
                "leader_change": float(item.get("f136") or 0),
            })

        df = pd.DataFrame(rows, columns=INDUSTRY_RANKING_COLS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch industry rankings over a date range.

        Uses A-share trading calendar (not US freq='B') to avoid
        hammering the API on non-trading days.
        """
        from stoke_ml.data.calendar import TradingCalendar
        calendar = TradingCalendar("a_shares")
        dates = calendar.get_trading_days(start_date, end_date)

        frames = []
        for i, d in enumerate(dates):
            date_str = d.strftime("%Y-%m-%d")
            df = self.fetch(date=date_str)
            if not df.empty:
                frames.append(df)
            if (i + 1) % 60 == 0:
                logger.info(
                    "  industry_ranking: %d/%d days (%.0f%%)",
                    i + 1, len(dates), (i + 1) / len(dates) * 100,
                )
        if not frames:
            return pd.DataFrame(columns=INDUSTRY_RANKING_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._session.close()


# ── Concept block membership (概念板块归属) ─────────────────────────────

class ConceptBlockSource:
    """Fetch per-stock concept/industry/region board membership.

    One call returns ALL boards a stock belongs to: industry (行业),
    concept (概念), and region (地域) — mixed in one list. Board names
    are self-explanatory (e.g., '食品饮料'=industry, '贵州板块'=region,
    '酿酒概念'=concept).

    Core value: theme attribution (题材归因) and sector linkage analysis.
    """

    SOURCE_NAME = "eastmoney_concept_blocks"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch(self, code: str) -> pd.DataFrame:
        """Fetch all boards a stock belongs to.

        Returns DataFrame with: board_name, board_code (BK码),
        board_change_pct (板块当日涨跌幅), lead_stock (板块龙头).
        """
        params = {
            "fltt": "2", "invt": "2",
            "secid": f"{_market_code(code)}.{code}",
            "spt": "3", "pi": "0", "pz": "200", "po": "1",
            "fields": "f12,f14,f3,f128",
        }
        try:
            r = self._client.get(
                PUSH2_SLIST_URL, params=params,
                headers=EASTMONEY_HEADERS, timeout=15,
            )
            r.raise_for_status()
            d = r.json()
        except Exception:
            logger.warning("Concept block fetch failed for %s", code)
            return pd.DataFrame(columns=CONCEPT_BLOCK_COLS)

        diff = (d.get("data") or {}).get("diff") or {}
        items = diff.values() if isinstance(diff, dict) else diff

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        rows = []
        for it in items:
            if not isinstance(it, dict):
                continue
            rows.append({
                "date": today,
                "stock_code": code,
                "board_name": str(it.get("f14", "")),
                "board_code": str(it.get("f12", "")),
                "board_change_pct": float(it.get("f3") or 0),
                "lead_stock": str(it.get("f128", "")),
            })

        if not rows:
            return pd.DataFrame(columns=CONCEPT_BLOCK_COLS)
        df = pd.DataFrame(rows, columns=CONCEPT_BLOCK_COLS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_tags(self, code: str) -> list[str]:
        """Fetch just the concept tag names for a stock.

        Convenience method — returns plain list of board names.
        """
        df = self.fetch(code)
        if df.empty:
            return []
        return df["board_name"].tolist()

    def fetch_batch(self, codes: list[str]) -> pd.DataFrame:
        """Fetch concept blocks for multiple stocks."""
        frames = []
        for code in codes:
            df = self.fetch(code)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=CONCEPT_BLOCK_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._client.close()
