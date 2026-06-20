"""Xueqiu (雪球) news source for A-share stocks.

Uses Playwright headless Chromium to bypass Cloudflare WAF, then calls
Xueqiu's internal status API through the authenticated browser session.

Architecture:
  1. Launch Playwright once (singleton browser per process)
  2. Load any stock page to get WAF-cleared cookies
  3. Call /query/v1/symbol/search/status.json directly via fetch()
  4. Parse JSON responses → DataFrame[date, title, url]

The API returns up to 1000 items per stock (20 items/page × 50 pages).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from playwright.sync_api import Page

logger = logging.getLogger(__name__)

_BASE_URL = "https://xueqiu.com"
_API_URL = (
    "https://xueqiu.com/query/v1/symbol/search/status.json"
    "?count=20&comment=0&symbol={symbol}&hl=0&source=all"
    "&sort=time&page={page}&q=&type=11"
)

# Thread-local browser state — Playwright sync API is greenlet-bound,
# so each thread needs its own browser/context instance.
_tls = threading.local()
# Guard browser creation within a single thread (CDP is not thread-safe
# even within the same greenlet).
_lock = threading.Lock()


def _build_symbol(stock_code: str) -> str:
    """Build Xueqiu symbol from 6-digit A-share code."""
    code = str(stock_code).zfill(6)
    if code.startswith(("6", "9")):
        return f"SH{code}"
    elif code.startswith(("0", "3", "2")):
        return f"SZ{code}"
    return f"SH{code}"


def _get_page() -> "Page":
    """Return a page from the thread-local browser context.

    Each OS thread gets its own Playwright instance because the sync
    greenlet is thread-affine.
    """
    from playwright.sync_api import sync_playwright

    if getattr(_tls, "browser", None) is None or not _tls.browser.is_connected():
        with _lock:
            if getattr(_tls, "browser", None) is None or not _tls.browser.is_connected():
                if getattr(_tls, "pw_instance", None) is not None:
                    try:
                        _tls.pw_instance.stop()
                    except Exception:
                        pass
                pw = sync_playwright().start()
                _tls.pw_instance = pw
                _tls.browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                _tls.context = _tls.browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1920, "height": 1080},
                    locale="zh-CN",
                )
                _tls.context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
                    window.chrome = {runtime: {}};
                """)
    return _tls.context.new_page()


_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = _HTML_RE.sub(" ", text)
    return " ".join(text.split())


class XueqiuNewsSource:
    """Fetch stock-related discussions and news from Xueqiu.

    Cloudflare WAF is bypassed by loading the stock page in headless
    Chromium (which auto-solves the JS challenge), then calling the
    internal status API through the authenticated session.
    """

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Fetch posts for a stock from Xueqiu.

        Args:
            stock_code: 6-digit A-share code.
            start_date: YYYY-MM-DD filter (inclusive).
            end_date: YYYY-MM-DD filter (inclusive).
            max_pages: Pages to fetch (20 items/page, max 50).

        Returns:
            DataFrame with columns: date, title, url.
        """
        symbol = _build_symbol(stock_code)
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start_dt = pd.Timestamp(start_date) if start_date else end_dt - pd.Timedelta(days=30)

        page = None
        try:
            page = _get_page()
        except Exception as e:
            logger.warning("Playwright not available for Xueqiu: %s", e)
            return pd.DataFrame(columns=["date", "title", "url"])

        try:
            # Navigate to stock page to get WAF-cleared cookies for this page
            page.goto(
                f"{_BASE_URL}/S/{symbol}",
                timeout=60000,
                wait_until="domcontentloaded",
            )
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            time.sleep(1)
            # Dismiss any modal overlay
            page.evaluate(
                '() => { const m = document.querySelector(".modals.dimmer");'
                " if(m) m.remove(); }"
            )
            time.sleep(0.3)

            # Fetch pages via the internal API
            collected: list[dict] = []
            max_pages = min(max_pages, 50)

            for page_num in range(1, max_pages + 1):
                api_url = _API_URL.format(symbol=symbol, page=page_num)
                js = (
                    "async () => {"
                    f"  const resp = await fetch('{api_url}',"
                    "   {credentials: 'include'});"
                    "  return JSON.stringify(await resp.json());"
                    "}"
                )
                try:
                    raw = page.evaluate(js)
                    data = json.loads(raw)
                except Exception as e:
                    logger.warning(
                        "Xueqiu API page %d failed for %s: %s",
                        page_num, stock_code, e,
                    )
                    break

                items = data.get("list", [])
                if not items:
                    break

                page_collected = 0
                for item in items:
                    created_at = item.get("created_at")
                    if not created_at:
                        continue
                    dt = datetime.fromtimestamp(created_at / 1000.0)

                    # Items are newest-first; stop processing this page
                    # once we reach items older than start_date
                    if dt < start_dt:
                        break

                    text = item.get("text", "") or item.get("title", "") or ""
                    full_text = _strip_html(text)
                    title = full_text[:200]
                    body = full_text[:2000] if len(full_text) > 200 else full_text[200:]
                    if not title:
                        continue

                    post_id = item.get("id")
                    user_id = item.get("user_id") or ""
                    if post_id and user_id:
                        url = f"{_BASE_URL}/{user_id}/{post_id}"
                    elif post_id:
                        url = f"{_BASE_URL}/u/{post_id}"
                    else:
                        url = ""

                    collected.append({"date": dt, "title": title, "body": body, "url": url})
                    page_collected += 1

                # Stop if this page had zero items in range (all older
                # than start_date, so subsequent pages will be older too)
                if page_collected == 0:
                    break

            if not collected:
                return pd.DataFrame(columns=["date", "title", "body", "url"])

            df = pd.DataFrame(collected)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= start_dt]
            # Include all timestamps on end_date (not just midnight)
            df = df[df["date"] < end_dt + pd.Timedelta(days=1)]
            df = df.drop_duplicates(subset=["title", "date"])
            df = df.sort_values("date", ascending=False)

            return df[["date", "title", "body", "url"]].reset_index(drop=True)

        except Exception as e:
            logger.debug("Xueqiu fetch for %s failed: %s", stock_code, e)
            return pd.DataFrame(columns=["date", "title", "body", "url"])
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
