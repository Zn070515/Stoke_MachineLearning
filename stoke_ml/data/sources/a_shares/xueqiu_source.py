"""Xueqiu (雪球) news source for A-share stocks.

Uses Playwright headless Chromium to bypass aliyun WAF.

Architecture (single-threaded, no Timers — avoids cross-thread greenlet bugs):
  1. Browser context reused across stocks (fresh page per stock)
  2. First stock: goto stock page to solve WAF
  3. Subsequent stocks: goto xueqiu.com/ (fast, WAF cookies in context)
  4. Resource blocking for faster loads
  5. storage_state persisted for cross-process WAF cookie reuse
  6. JS-level AbortController (10s) prevents per-page hangs
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from stoke_ml.config import get_project_root

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

logger = logging.getLogger(__name__)

_BASE_URL = "https://xueqiu.com"
_API_URL = (
    "https://xueqiu.com/query/v1/symbol/search/status.json"
    "?count=20&comment=0&symbol={symbol}&hl=0&source=all"
    "&sort=time&page={page}&q=&type=11"
)
_STATE_FILE = str(
    get_project_root() / "data" / "a_shares" / ".xueqiu_browser_state.json"
)

_GOTO_TIMEOUT_MS = 25000
_API_CALL_TIMEOUT_MS = 10000  # JS AbortController timeout

# Module-level browser state (single-threaded — no threading.Timer)
_pw_instance = None
_browser = None
_context: "BrowserContext | None" = None
_waf_solved = False
_browser_lock = None  # created on demand

_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    return " ".join(text.split())


def _build_symbol(stock_code: str) -> str:
    code = str(stock_code).zfill(6)
    if code.startswith(("6", "9")):
        return f"SH{code}"
    elif code.startswith(("0", "3", "2")):
        return f"SZ{code}"
    return f"SH{code}"


def _ensure_context() -> "BrowserContext":
    """Return the module-level browser context, creating it on first call."""
    global _pw_instance, _browser, _context, _browser_lock

    from playwright.sync_api import sync_playwright

    if _browser is not None:
        try:
            if _browser.is_connected():
                return _context
        except Exception:
            pass
        # Browser is dead — clean up, including stale storage_state
        _browser = None
        _context = None
        try:
            os.remove(_STATE_FILE)
        except OSError:
            pass

    if _browser_lock is None:
        import threading
        _browser_lock = threading.Lock()

    with _browser_lock:
        if _browser is not None:
            try:
                if _browser.is_connected():
                    return _context
            except Exception:
                pass

        if _pw_instance is not None:
            try:
                _pw_instance.stop()
            except Exception:
                pass

        _pw_instance = sync_playwright().start()

        storage_state = None
        if os.path.exists(_STATE_FILE):
            try:
                with open(_STATE_FILE) as f:
                    storage_state = json.load(f)
            except Exception:
                pass

        _browser = _pw_instance.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        _context = _browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            storage_state=storage_state,
        )
        _context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
            window.chrome = {runtime: {}};
        """)

        def _block_route(route):
            rt = route.request.resource_type
            if rt in ("image", "font", "media", "stylesheet"):
                route.abort()
            elif "analytics" in route.request.url or "collect" in route.request.url:
                route.abort()
            else:
                route.continue_()

        _context.route("**/*", _block_route)
        return _context


def _save_browser_state(context: "BrowserContext") -> None:
    try:
        state = context.storage_state()
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


class XueqiuNewsSource:
    """Fetch stock discussions from Xueqiu via Playwright."""

    _stock_count = 0
    _BROWSER_RESTART_EVERY = 50

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        global _waf_solved, _browser, _context

        symbol = _build_symbol(stock_code)
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start_dt = pd.Timestamp(start_date) if start_date else end_dt - pd.Timedelta(days=30)
        max_pages = min(max_pages, 50)

        XueqiuNewsSource._stock_count += 1
        if (
            XueqiuNewsSource._stock_count > 1
            and XueqiuNewsSource._stock_count % XueqiuNewsSource._BROWSER_RESTART_EVERY == 0
        ):
            logger.info("Periodic browser restart (stock #%d)", XueqiuNewsSource._stock_count)
            _browser = None
            _context = None
            _waf_solved = False
            try:
                os.remove(_STATE_FILE)
            except OSError:
                pass

        try:
            context = _ensure_context()
        except Exception as e:
            logger.warning("Playwright not available: %s", e)
            return pd.DataFrame(columns=["date", "title", "body", "url"])

        page = None
        try:
            page = context.new_page()
            page.set_default_timeout(15000)

            target_url = f"{_BASE_URL}/S/{symbol}"

            try:
                page.goto(
                    target_url,
                    timeout=_GOTO_TIMEOUT_MS,
                    wait_until="domcontentloaded",
                )
                if not _waf_solved:
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                    time.sleep(0.8)
                    try:
                        page.evaluate(
                            '() => { const m = document.querySelector(".modals.dimmer");'
                            " if(m) m.remove(); }"
                        )
                    except Exception:
                        pass
                    _waf_solved = True
                    logger.debug("WAF solved via %s", symbol)
                _save_browser_state(context)
            except Exception as e:
                logger.debug("goto %s failed: %s", stock_code, e)
                _waf_solved = False

            collected: list[dict] = []
            for waf_round in range(3):
                waf_ok = True
                page_items: list[dict] = []

                for page_num in range(1, max_pages + 1):
                    api_url = _API_URL.format(symbol=symbol, page=page_num)
                    js = (
                        "async () => {"
                        f"  const ctrl = new AbortController();"
                        f"  const t = setTimeout(() => ctrl.abort(), {_API_CALL_TIMEOUT_MS});"
                        "  try {"
                        f"    const r = await fetch('{api_url}',"
                        "      {credentials: 'include', signal: ctrl.signal});"
                        "    const text = await r.text();"
                        "    return JSON.stringify({ok: r.ok, text: text});"
                        "  } catch(e) {"
                        "    return JSON.stringify({ok: false, text: '', err: e.message || String(e)});"
                        "  } finally {"
                        "    clearTimeout(t);"
                        "  }"
                        "}"
                    )
                    try:
                        raw = page.evaluate(js)
                        wrapper = json.loads(raw)
                    except Exception:
                        logger.debug("evaluate failed p%d for %s", page_num, stock_code)
                        break

                    if not wrapper.get("ok"):
                        if page_num == 1:
                            waf_ok = False
                        break

                    text = wrapper.get("text", "")
                    if not text or text.strip().startswith("<"):
                        if page_num == 1:
                            waf_ok = False
                        break

                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        if page_num == 1:
                            waf_ok = False
                        break

                    items = data.get("list", [])
                    if not items:
                        break

                    page_count = 0
                    for item in items:
                        created_at = item.get("created_at")
                        if not created_at:
                            continue
                        dt = datetime.fromtimestamp(created_at / 1000.0)
                        if dt < start_dt:
                            break

                        content = item.get("text", "") or item.get("title", "") or ""
                        clean = _strip_html(content)
                        title = clean[:200]
                        body = clean[:2000] if len(clean) > 200 else ""
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

                        page_items.append({
                            "date": dt, "title": title, "body": body, "url": url,
                        })
                        page_count += 1

                    if page_count == 0:
                        break

                if not waf_ok:
                    _waf_solved = False
                    try:
                        page.goto(
                            f"{_BASE_URL}/S/{symbol}",
                            timeout=_GOTO_TIMEOUT_MS,
                            wait_until="domcontentloaded",
                        )
                        _waf_solved = True
                        _save_browser_state(context)
                    except Exception:
                        pass
                    page_items.clear()
                    continue

                collected = page_items
                break

            if not collected:
                return pd.DataFrame(columns=["date", "title", "body", "url"])

            df = pd.DataFrame(collected)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= start_dt]
            df = df[df["date"] < end_dt + pd.Timedelta(days=1)]
            df = df.drop_duplicates(subset=["title", "date"])
            df = df.sort_values("date", ascending=False)

            return df[["date", "title", "body", "url"]].reset_index(drop=True)

        except Exception as e:
            msg = str(e)
            if "closed" in msg.lower() or "connection" in msg.lower():
                logger.warning("Browser closed for %s, will restart", stock_code)
                _browser = None
                _context = None
                _waf_solved = False
            else:
                logger.debug("Fetch failed for %s: %s", stock_code, e)
            return pd.DataFrame(columns=["date", "title", "body", "url"])
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
