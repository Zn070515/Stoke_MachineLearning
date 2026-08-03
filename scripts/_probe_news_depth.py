"""Probe news sources for historical depth: can we reach back to 2023?

Tests:
1. EastMoney search API (THSNewsSource) with beginTime/endTime window params.
2. EastMoney search pagination depth without date params.
3. Sina AllNewsStock pagination depth.
"""
import json
import re
import time

import pandas as pd
from curl_cffi import requests

EM_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Referer": "https://so.eastmoney.com/news/s",
}
SINA_URL = (
    "http://vip.stock.finance.sina.com.cn/corp/view/"
    "vCB_AllNewsStock.php?symbol=sz000001&Page={page}"
)
SINA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}


def em_page(stock_code: str, page: int, extra: dict | None = None, page_size: int = 100):
    cms = {
        "searchScope": "default",
        "sort": "default",
        "pageIndex": page,
        "pageSize": page_size,
        "preTag": "<em>",
        "postTag": "</em>",
    }
    if extra:
        cms.update(extra)
    inner = {
        "uid": "",
        "keyword": stock_code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {"cmsArticleWebOld": cms},
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
            data = json.loads(text[text.find("(") + 1: -1])
            return data.get("result", {}).get("cmsArticleWebOld", [])
    except Exception as e:
        print(f"    EM ERR: {e}")
    return []


def rows_stats(items):
    if not items:
        return "empty"
    dates = sorted(it.get("date", "")[:10] for it in items)
    return f"{len(items)} items {dates[0]}~{dates[-1]}"


def probe_em_window():
    print("\n=== EastMoney search with beginTime/endTime window (000001, 2023-05) ===")
    for extra in [
        {"beginTime": "2023-05-01", "endTime": "2023-05-31"},
        {"dateRange": "2023-05-01~2023-05-31"},
        {"beginTime": "2023-01-01 00:00:00", "endTime": "2023-12-31 23:59:59"},
    ]:
        items = em_page("000001", 1, extra=extra)
        print(f"  extra={json.dumps(extra)[:60]} -> {rows_stats(items)}")


def probe_em_pagination():
    print("\n=== EastMoney search pagination depth (no date params) ===")
    for page in [1, 3, 5, 10, 20]:
        items = em_page("000001", page)
        print(f"  page {page}: {rows_stats(items)}")


def probe_sina():
    print("\n=== Sina AllNewsStock pagination depth ===")
    import requests as reqs
    from bs4 import BeautifulSoup
    for page in [1, 5, 10, 30, 60]:
        try:
            resp = reqs.get(SINA_URL.format(page=page), headers=SINA_HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            dates = []
            for div in soup.find_all("div", class_="datelist"):
                for content in div.contents:
                    t = str(content)
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
                    if m:
                        dates.append(m.group(1))
            print(f"  page {page}: {len(dates)} dated items {dates[0] if dates else '-'}~{dates[-1] if dates else '-'}")
        except Exception as e:
            print(f"  page {page}: ERR {e}")


if __name__ == "__main__":
    probe_em_window()
    probe_em_pagination()
    probe_sina()
