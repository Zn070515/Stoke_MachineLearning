"""CNINFO (巨潮资讯网) announcement source — CSRC-designated official disclosure platform.

Covers ALL A-shares (SSE + SZSE) through a single API, free, no auth.
Full text extracted from PDFs hosted at static.cninfo.com.cn.

Rate limit: >= 0.1s between requests. Organise ID per stock cached in memory.
PDF download is parallelised via thread pool (PyMuPDF releases GIL).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logger = logging.getLogger(__name__)

CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_SEARCH = "http://www.cninfo.com.cn/new/fulltextSearch/full"
CNINFO_PDF_BASE = "http://static.cninfo.com.cn"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "Accept": "application/json, text/plain, */*",
}

_PDF_CACHE: OrderedDict[str, str | None] = OrderedDict()
_PDF_CACHE_MAX = 2000
_PDF_CACHE_LOCK = threading.Lock()
_ORG_ID_CACHE: dict[str, str] = {}
_thread_local = threading.local()


def _org_id(session: requests.Session, code: str) -> str | None:
    """Discover the organ ID for a stock, cached in memory."""
    if code in _ORG_ID_CACHE:
        return _ORG_ID_CACHE[code]
    try:
        resp = session.get(
            CNINFO_SEARCH, params={"searchkey": code},
            headers=HEADERS, timeout=15,
        )
        data = resp.json()
        for ann in data.get("announcements") or []:
            if str(ann.get("secCode", "")).strip() == code:
                org = str(ann.get("orgId", ""))
                if org:
                    _ORG_ID_CACHE[code] = org
                    return org
        for item in (data.get("classifiedAnnouncements") or {}).values():
            if isinstance(item, list):
                for entry in item:
                    if str(entry.get("secCode", "")).strip() == code:
                        org = str(entry.get("orgId", ""))
                        if org:
                            _ORG_ID_CACHE[code] = org
                            return org
        for item in data.get("securities") or []:
            if str(item.get("code", "")).strip() == code:
                org = str(item.get("orgId", ""))
                if org:
                    _ORG_ID_CACHE[code] = org
                    return org
    except Exception:
        pass
    return None


def _parse_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF byte buffer via PyMuPDF."""
    # online extra — import lazily so core/dev jobs can import this module
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
        doc.close()
        body = "\n".join(pages)
        body = re.sub(r"\s{3,}", "\n", body)
        return body.strip()
    except Exception:
        return ""


def _download_pdfs_parallel(urls: list[str], max_workers: int = 8) -> dict[str, str | None]:
    """Download + parse PDFs in parallel. Cache-aware: skips already-cached URLs."""
    if not urls:
        return {}

    # Bulk cache check under one lock
    with _PDF_CACHE_LOCK:
        cached = {u: _PDF_CACHE.get(u) for u in urls if u in _PDF_CACHE}
        # Enforce cache bound before adding new entries
        needed = len(urls) - len(cached)
        while len(_PDF_CACHE) + needed > _PDF_CACHE_MAX:
            _PDF_CACHE.popitem(last=False)

    uncached = [u for u in urls if u and u not in cached]
    if not uncached:
        return {u: cached.get(u) for u in urls if u}

    results = dict(cached)

    def _fetch_one(url: str) -> tuple[str, str | None]:
        # online extra — import lazily so core/dev jobs can import this module
        from curl_cffi import requests

        try:
            if not hasattr(_thread_local, "session"):
                _thread_local.session = requests.Session()
            full_url = f"{CNINFO_PDF_BASE}/{url}"
            resp = _thread_local.session.get(full_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200 or len(resp.content) < 100:
                return url, None
            text = _parse_pdf_text(resp.content)
            return url, text if len(text) > 20 else None
        except Exception:
            return url, None

    # Suppress MuPDF C-level stderr (corrupt PDF warnings flood logs, 50K+ lines)
    import fitz  # PyMuPDF — online extra, lazy so core/dev jobs can import
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        workers = max(1, min(max_workers, len(uncached)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, u): u for u in uncached}
            for future in as_completed(futures):
                url, text = future.result()
                results[url] = text
    finally:
        fitz.TOOLS.mupdf_display_errors(True)

    # Bulk write cache
    with _PDF_CACHE_LOCK:
        for url, text in results.items():
            if url not in _PDF_CACHE:
                _PDF_CACHE[url] = text

    return results


def _download_pdf(adjunct_url: str) -> str | None:
    """Download + parse a single PDF (sequential fallback)."""
    if not adjunct_url:
        return None

    with _PDF_CACHE_LOCK:
        if adjunct_url in _PDF_CACHE:
            return _PDF_CACHE[adjunct_url]
        while len(_PDF_CACHE) >= _PDF_CACHE_MAX:
            _PDF_CACHE.popitem(last=False)

    url = f"{CNINFO_PDF_BASE}/{adjunct_url}"
    try:
        # online extra — import lazily so core/dev jobs can import this module
        from curl_cffi import requests

        if not hasattr(_thread_local, "session"):
            _thread_local.session = requests.Session()
        resp = _thread_local.session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200 or len(resp.content) < 100:
            with _PDF_CACHE_LOCK:
                _PDF_CACHE[adjunct_url] = None
            return None
        text = _parse_pdf_text(resp.content)
        result = text if len(text) > 20 else None
        with _PDF_CACHE_LOCK:
            _PDF_CACHE[adjunct_url] = result
        return result
    except Exception:
        with _PDF_CACHE_LOCK:
            _PDF_CACHE[adjunct_url] = None
        return None


class CninfoSource:
    """Fetch announcements from CNINFO with optional PDF body extraction."""

    def __init__(self, pdf_bodies: bool = True, pdf_cache_dir: str | None = None,
                 pdf_workers: int = 8):
        self._pdf_bodies = pdf_bodies
        self._pdf_workers = pdf_workers
        self._cache_dir = pdf_cache_dir
        if pdf_cache_dir:
            os.makedirs(pdf_cache_dir, exist_ok=True)
        self._session = None  # curl_cffi session created lazily on first use

    def _get_session(self):
        """Return the per-instance curl_cffi session, created on first use.

        online extra — lazy import so core/dev jobs can construct
        CninfoSource without the online stack installed.
        """
        if self._session is None:
            from curl_cffi import requests

            self._session = requests.Session()
        return self._session

    def fetch_announcements(
        self, stock_code: str, start_date: str = "2015-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all announcements for one stock.

        Columns: date, title, notice_type, url, body (if pdf_bodies=True).
        """
        if end_date is None:
            end_date = time.strftime("%Y-%m-%d")

        org_id = _org_id(self._get_session(), stock_code)
        if not org_id:
            logger.debug("CNINFO: no orgId for %s", stock_code)
            return pd.DataFrame(columns=["date", "title", "notice_type", "url"])

        all_items = []
        stock_param = f"{stock_code},{org_id}"

        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        for year in range(start_year, end_year + 1):
            y_start = start_date if year == start_year else f"{year}-01-01"
            y_end = end_date if year == end_year else f"{year}-12-31"

            page = 1
            while True:
                time.sleep(0.1)
                items, has_more = self._query_page(stock_param, y_start, y_end, page)
                all_items.extend(items)
                if not has_more or not items:
                    break
                page += 1

        if not all_items:
            return pd.DataFrame(columns=["date", "title", "notice_type", "url"])

        df = pd.DataFrame(all_items)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["title", "date"])
        df = df.sort_values("date").reset_index(drop=True)

        if self._pdf_bodies:
            urls = [u for u in df["url"].tolist() if u]
            results = _download_pdfs_parallel(urls, max_workers=self._pdf_workers)
            df["body"] = df["url"].map(lambda u: results.get(u) if u else None)

        return df

    def _query_page(self, stock_param: str, start: str, end: str, page: int):
        """POST one page of results. Returns (items, has_more)."""
        body = {
            "pageNum": page,
            "pageSize": 30,
            "column": "",
            "tabName": "fulltext",
            "plate": "",
            "stock": stock_param,
            "searchkey": "",
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start}~{end}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        last_resp = None
        for attempt in range(3):
            try:
                resp = self._get_session().post(
                    CNINFO_QUERY, data=body, headers=HEADERS,
                    timeout=30, impersonate="chrome120",
                )
                if resp.status_code != 200:
                    last_resp = resp
                    time.sleep(1.0 * (attempt + 1))
                    continue  # retry
                data = resp.json()
                items = []
                for ann in data.get("announcements", []):
                    title = (ann.get("announcementTitle") or "").strip()
                    title = re.sub(r"<[^>]+>", "", title)
                    ts = ann.get("announcementTime", 0)
                    date_str = (
                        time.strftime("%Y-%m-%d", time.localtime(ts / 1000)) if ts
                        else ""
                    )
                    adjunct = ann.get("adjunctUrl") or ""
                    items.append({
                        "date": date_str,
                        "title": title,
                        "notice_type": ann.get("announcementTypeName") or "",
                        "url": adjunct,
                    })
                has_more = bool(data.get("hasMore"))
                return items, has_more
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                # 3 failed attempts — genuinely degraded; surface as failure
                raise
        # 3 attempts all non-200 — treat as hard failure, not silent end-of-data
        raise RuntimeError(
            f"cninfo query failed after 3 attempts: "
            f"HTTP {last_resp.status_code if last_resp is not None else 'unknown'}"
        )


def quick_test(code: str = "000001"):
    """Smoke-test: fetch 2024 announcements for one stock."""
    src = CninfoSource(pdf_bodies=False)
    df = src.fetch_announcements(code, "2024-01-01", "2024-12-31")
    print(f"{code}: {len(df)} announcements")
    if not df.empty:
        print(df.head(3).to_string())
    return df
