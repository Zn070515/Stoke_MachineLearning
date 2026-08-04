"""EastMoney company announcements source with full historical coverage.

Unlike the news search API (limited to ~6 months), the EastMoney
announcement API supports date-range queries back to ~2015 with
year-by-year pagination, yielding ~1000 announcements per stock.
"""
import logging
import time

import pandas as pd
from curl_cffi import requests

logger = logging.getLogger(__name__)

ANNOUNCE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://data.eastmoney.com/",
}


class AnnouncementSource:
    """Fetch company announcements from EastMoney with date-range queries.

    Uses year-by-year chunking to bypass the API's ~500-result cap.
    """

    @staticmethod
    def _fetch_page(stock_code: str, begin_date: str, end_date: str,
                    page: int = 1, page_size: int = 100) -> tuple[list[dict], int]:
        """Fetch one page of announcements. Returns (items, total_count)."""
        params = {
            "sr": -1,
            "page_size": page_size,
            "page_index": page,
            "ann_type": "A",
            "client_source": "web",
            "f_node": 0,
            "s_node": 0,
            "stock_list": stock_code,
            "begin_time": begin_date,
            "end_time": end_date,
        }
        try:
            resp = requests.get(
                ANNOUNCE_URL, params=params, headers=HEADERS,
                impersonate="chrome120", timeout=30,
            )
            if resp.status_code != 200:
                return [], 0
            data = resp.json()
            d = data.get("data", {})
            return d.get("list", []), d.get("total", 0)
        except Exception:
            return [], 0

    def fetch_announcements(
        self,
        stock_code: str,
        start_date: str = "2015-01-01",
        end_date: str | None = None,
        meta: dict | None = None,
    ) -> pd.DataFrame:
        """Fetch all announcements for a stock across the full date range.

        Returns DataFrame with columns: date, title, notice_type, url.

        ``meta`` (optional dict) receives ``pages_fetched``, ``expected_rows``
        (the API-reported total across all year windows) and
        ``pagination_exhausted`` (download completion evidence).  A year
        is exhausted only when its declared page count is fully fetched without
        an early empty page; a year whose ``total`` exceeds the 10-page cap is
        truncated and marks the whole fetch NOT exhausted.
        """
        if end_date is None:
            end_date = time.strftime("%Y-%m-%d")

        all_items = []
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        pages_fetched = 0
        total_expected = 0
        pagination_exhausted = True

        for year in range(start_year, end_year + 1):
            y_begin = f"{year}-01-01"
            y_end = f"{year}-12-31" if year < end_year else end_date

            time.sleep(0.05)

            items, total = self._fetch_page(stock_code, y_begin, y_end, page=1)
            pages_fetched += 1
            total_expected += total
            all_items.extend(items)

            if total > 100:
                needed_pages = (total + 99) // 100
                if needed_pages > 10:
                    pagination_exhausted = False
                total_pages = min(needed_pages, 10)
                for page in range(2, total_pages + 1):
                    time.sleep(0.05)
                    more_items, _ = self._fetch_page(stock_code, y_begin, y_end, page=page)
                    pages_fetched += 1
                    if not more_items:
                        pagination_exhausted = False
                        break
                    all_items.extend(more_items)

        if meta is not None:
            meta["pages_fetched"] = pages_fetched
            meta["expected_rows"] = total_expected
            meta["pagination_exhausted"] = pagination_exhausted

        if not all_items:
            return pd.DataFrame(columns=["date", "title", "notice_type", "url"])

        rows = []
        for item in all_items:
            notice_date = (item.get("notice_date") or "")[:10]
            title = item.get("title_short") or item.get("title_ch") or item.get("title") or ""
            art_code = item.get("art_code", "")
            url = f"https://data.eastmoney.com/notices/detail/{stock_code}/{art_code}.html" if art_code else ""
            cols = item.get("columns") or []
            notice_type = (cols[0] if isinstance(cols[0], str) else
                           cols[0].get("column_name", "") if isinstance(cols[0], dict) else "") if cols else ""
            if notice_date and title:
                rows.append({
                    "date": notice_date,
                    "title": title.strip(),
                    "notice_type": notice_type,
                    "url": url,
                })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["title", "date"])
        return df.sort_values("date").reset_index(drop=True)
