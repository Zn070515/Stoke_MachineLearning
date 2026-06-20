"""THS / EastMoney news source with deep-pagination support.

Uses EastMoney's search API directly (bypassing AKShare's thin wrapper)
to fetch up to 500 articles per stock via pagination (100/page × 5 pages).
"""
import json
import logging
import time

import pandas as pd
from curl_cffi import requests

logger = logging.getLogger(__name__)

EM_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Referer": "https://so.eastmoney.com/news/s",
}


class THSNewsSource:
    """Fetch stock news from EastMoney search API with deep pagination."""

    @staticmethod
    def _fetch_em_page(stock_code: str, page: int, page_size: int = 100) -> list[dict]:
        """Fetch one page of EastMoney search results."""
        inner = {
            "uid": "",
            "keyword": stock_code,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": page,
                    "pageSize": page_size,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }
        params = {
            "cb": "jQuery",
            "param": json.dumps(inner, ensure_ascii=False),
            "_": str(int(time.time() * 1000)),
        }
        try:
            resp = requests.get(
                EM_SEARCH_URL, params=params, headers=EM_HEADERS,
                impersonate="chrome120", timeout=15,
            )
            text = resp.text
            if text.startswith("jQuery"):
                data = json.loads(text[text.find("(") + 1 : -1])
                return data.get("result", {}).get("cmsArticleWebOld", [])
        except Exception:
            pass
        return []

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 5,
    ) -> pd.DataFrame:
        """Fetch stock news from EastMoney search with deep pagination.

        Each page returns up to 100 articles. The API typically allows
        ~5 pages before returning empty, yielding up to ~500 articles
        spanning ~6-12 months of history.
        """
        all_rows = []
        for page in range(1, max_pages + 1):
            items = self._fetch_em_page(stock_code, page, page_size=100)
            if not items:
                break
            for it in items:
                title = (it.get("title") or "").replace("<em>", "").replace("</em>", "")
                content = (it.get("content") or "").replace("<em>", "").replace("</em>", "")
                date_str = (it.get("date") or "")[:10]
                code = it.get("code", "")
                url = f"https://finance.eastmoney.com/a/{code}.html" if code else ""
                if title and date_str:
                    all_rows.append({
                        "date": date_str,
                        "title": title,
                        "body": content if len(content) > 20 else "",
                        "url": url,
                    })

        if not all_rows:
            return pd.DataFrame(columns=["date", "title", "body", "url"])

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["title", "date"])
        df = df.sort_values("date", ascending=False)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)
