"""Xueqiu (雪球) news source for A-share stocks."""
import json
import logging
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

XUEQIU_SEARCH_URL = "https://xueqiu.com/statuses/search.json"
XUEQIU_STOCK_URL = "https://xueqiu.com/S/{code}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://xueqiu.com/",
    "Accept": "application/json, text/plain, */*",
}


class XueqiuNewsSource:
    """Fetch stock-related news and discussion from Xueqiu."""

    @staticmethod
    def _to_xq_code(stock_code: str) -> str:
        if stock_code.startswith("6"):
            return f"SH{stock_code}"
        return f"SZ{stock_code}"

    def _init_session(self) -> requests.Session:
        """Visit homepage to get cookies, then return a session with cookies set."""
        session = requests.Session()
        try:
            resp = session.get(
                "https://xueqiu.com/",
                headers=HEADERS,
                impersonate="chrome120",
                timeout=15,
            )
            if resp.status_code == 200:
                logger.debug("Xueqiu session cookies acquired")
        except Exception:
            pass
        return session

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Fetch news/discussion from Xueqiu search API."""
        xq_code = self._to_xq_code(stock_code)
        session = self._init_session()
        all_items = []

        for page in range(1, max_pages + 1):
            try:
                resp = session.get(
                    XUEQIU_SEARCH_URL,
                    params={
                        "count": 20,
                        "page": page,
                        "symbol": xq_code,
                        "source": "all",
                        "sort": "time",
                    },
                    headers=HEADERS,
                    impersonate="chrome120",
                    timeout=15,
                )
                if resp.status_code != 200:
                    if page == 1:
                        logger.debug(
                            "Xueqiu search API returned %d for %s",
                            resp.status_code, stock_code,
                        )
                    break

                data = resp.json()
            except (json.JSONDecodeError, Exception) as e:
                logger.debug("Xueqiu page %d failed for %s: %s", page, stock_code, e)
                break

            items = data.get("list", [])
            if not items:
                break

            found = False
            for item in items:
                created_at = item.get("created_at")
                title = item.get("title") or item.get("text", "")
                if not title:
                    continue
                # Truncate long text to headline length
                title = title[:200]
                date_str = (
                    datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d")
                    if created_at else None
                )
                if not date_str:
                    continue
                target = item.get("target", "")
                url = f"https://xueqiu.com{target}" if target else ""
                found = True
                all_items.append({"date": date_str, "title": title, "url": url})

            if not found:
                break

        if not all_items:
            return pd.DataFrame(columns=["date", "title", "url"])

        df = pd.DataFrame(all_items)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["title", "date"])
        df = df.sort_values("date", ascending=False)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)
