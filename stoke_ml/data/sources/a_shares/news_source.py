"""Sina Finance news crawler for A-share stocks."""
import logging
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

SINA_NEWS_URL = (
    "https://vip.stock.finance.sina.com.cn/corp/go.php/"
    "vCB_AllNewsStock/symbol/{prefix}{code}.phtml"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}


class SinaNewsSource:
    """Fetch stock-related news from Sina Finance."""

    @staticmethod
    def _to_sina_prefix(stock_code: str) -> str:
        return "sh" if stock_code.startswith("6") else "sz"

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Fetch news headlines for a stock. Returns DataFrame with date/title/url."""
        prefix = self._to_sina_prefix(stock_code)
        all_items = []

        for page in range(1, max_pages + 1):
            if page == 1:
                url = SINA_NEWS_URL.format(prefix=prefix, code=stock_code)
            else:
                url = SINA_NEWS_URL.format(
                    prefix=prefix, code=stock_code
                ) + f"/{page}.phtml"

            try:
                resp = requests.get(
                    url, headers=HEADERS, impersonate="chrome120", timeout=15,
                )
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                found = False
                for div in soup.find_all("div", class_="datelist"):
                    # date is typically in a span or text node before the links
                    current_date = None
                    for content in div.contents:
                        text = str(content).strip() if hasattr(content, 'strip') else content.get_text(strip=True) if hasattr(content, 'get_text') else str(content)
                        date_match = re.search(
                            r"(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})?", text
                        )
                        if date_match:
                            current_date = date_match.group(1)
                        if hasattr(content, "find_all"):
                            for a in content.find_all("a"):
                                title = a.get_text(strip=True)
                                href = a.get("href", "")
                                if title and href and current_date:
                                    found = True
                                    all_items.append({
                                        "date": current_date,
                                        "title": title,
                                        "body": "",
                                        "url": href,
                                    })
                if not found:
                    break  # no more pages
            except Exception as e:
                logger.warning("Sina news page %d failed for %s: %s", page, stock_code, e)
                break

        if not all_items:
            return pd.DataFrame(columns=["date", "title", "body", "url"])

        df = pd.DataFrame(all_items)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["title"])
        df = df.sort_values("date", ascending=False)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def fetch_article_body(self, url: str) -> str | None:
        """Fetch full text of a Sina news article."""
        try:
            resp = requests.get(
                url, headers=HEADERS, impersonate="chrome120", timeout=15,
            )
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            # Sina articles use <div class="article" id="artibody">
            arti = soup.find("div", class_="article") or soup.find(
                "div", id="artibody"
            )
            if arti:
                paras = arti.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paras)
                return text if len(text) > 20 else None
            return None
        except Exception as e:
            logger.debug("Failed to fetch article body: %s", e)
            return None
