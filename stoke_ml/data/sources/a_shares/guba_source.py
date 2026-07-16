"""EastMoney Guba (股吧) forum post scraper for A-share stocks."""
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

# Use the "f" (发帖时间) sort URL so posts are ordered by publish time
# rather than last-comment time.  This lets us reach deep into history
# (2007+) instead of being limited to the most active recent threads.
#
# NOTE: the "list,f" endpoint has aggressive per-IP rate limiting
# (~200-300 pages before a multi-hour block).  The "topic" endpoint
# (sorted by last-comment time) is more lenient but only surfaces
# recently-active threads, so it can't reach deep history.
GUBA_LIST_URL_F = "https://guba.eastmoney.com/list,{code},f_{page}.html"
GUBA_LIST_URL_TOPIC = "https://guba.eastmoney.com/topic,{code}_{page}.html"
GUBA_DETAIL_URL = "https://guba.eastmoney.com/news,{code},{post_id}.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
}

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0    # seconds base delay
PAGE_DELAY = 2.0     # seconds between pages (keep under ~300 req/session)
BLOCK_COOLDOWN = 90  # seconds to wait when rate-limited
BLOCK_PAGE_MIN_LEN = 5000  # pages shorter than this are likely redirects/blocked
EMPTY_COLUMNS = ["date", "time", "title", "body", "post_id", "url"]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard Guba columns."""
    return pd.DataFrame(columns=EMPTY_COLUMNS)


def _fetch_with_retry(url: str, timeout: int = 20) -> requests.Response | None:
    """GET a URL with retry logic.

    Returns the Response on success, or None after exhausting retries.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                impersonate="chrome146",
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp
            logger.debug(
                "Guba request returned status %d (attempt %d/%d): %s",
                resp.status_code,
                attempt + 1,
                MAX_RETRIES,
                url,
            )
        except Exception as e:
            last_error = e
            logger.debug(
                "Guba request failed (attempt %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                e,
            )

        if attempt < MAX_RETRIES - 1:
            delay = RETRY_DELAY * (2 ** attempt)
            time.sleep(delay)

    if last_error:
        logger.warning("Guba request failed after %d retries: %s", MAX_RETRIES, last_error)
    return None


def _extract_article_list(html: str) -> dict | None:
    """Extract and parse the article_list JSON from a Guba list page.

    Returns the parsed dict or None if extraction fails.
    """
    match = re.search(r"var article_list=\{(.+?)\};\s*var ", html, re.DOTALL)
    if not match:
        # Try alternative delimiter (end of script block)
        match = re.search(
            r"var article_list=(\{.+?\});\s*</script>", html, re.DOTALL
        )
    if not match:
        return None

    raw = match.group(1)

    # Pattern 1 captures content between braces → add braces back
    try:
        return json.loads("{" + raw + "}")
    except (json.JSONDecodeError, KeyError):
        pass

    # Pattern 2 captures content WITH braces → parse directly
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        return None


def _parse_datetime(dt_str: str, year_hint: int | None = None) -> tuple[str, str]:
    """Parse a Guba datetime string into (date, time) tuple.

    Handles formats:
      - "YYYY-MM-DD HH:MM:SS"
      - "MM-DD HH:MM" — year inferred from year_hint or current year

    When paginating deep into history, the f-sort list pages use short
    dates.  The caller tracks the current year across pages and passes it
    as *year_hint*.  This function detects year-boundary crossings by
    checking whether the parsed date would be in the future.

    Returns ("", "") on parse failure.
    """
    dt_str = dt_str.strip()
    if not dt_str:
        return "", ""

    # Full datetime: "2026-06-26 20:05:39"
    full_match = re.match(
        r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}:\d{2}:\d{2})", dt_str
    )
    if full_match:
        return (
            f"{full_match.group(1)}-{full_match.group(2)}-{full_match.group(3)}",
            full_match.group(4),
        )

    # Short datetime: "06-26 20:05" — infer year
    short_match = re.match(r"(\d{2})-(\d{2})\s+(\d{2}:\d{2})", dt_str)
    if short_match:
        month, day = int(short_match.group(1)), int(short_match.group(2))
        yr = year_hint if year_hint is not None else datetime.now().year
        # If the result would be in the future, step back one year.
        # This handles year-boundary crossings when paginating backwards.
        today = datetime.now().date()
        if datetime(yr, month, day).date() > today:
            yr -= 1
        return (
            f"{yr}-{short_match.group(1)}-{short_match.group(2)}",
            short_match.group(3) + ":00",
        )

    logger.debug("Could not parse datetime: %r", dt_str)
    return "", ""


class GubaSource:
    """Fetch stock discussion posts from EastMoney Guba (股吧).

    Two sort modes are available via *sort* parameter:
      - "publish" (default): uses list,f URL, posts sorted by publish time.
        Can reach 2007+ history but has aggressive per-IP rate limits
        (~200-300 pages before multi-hour block).
      - "comment": uses topic URL, posts sorted by last-comment time.
        More lenient rate limits but only surfaces recently-active threads,
        so historical reach is limited (~1 year for popular stocks).

    For deep historical backfill, use sort="publish" with page_delay >= 3.0
    and expect to run in multiple sessions with cooldown periods.
    """

    def __init__(self, sort: str = "publish", page_delay: float | None = None):
        """Args:
            sort: "publish" (post time, deep history) or "comment" (last-comment, lenient).
            page_delay: seconds between pages.  Defaults to 2.0 for publish,
                1.0 for comment.
        """
        self.sort = sort
        self.page_delay = (
            page_delay if page_delay is not None
            else (2.0 if sort == "publish" else 1.0)
        )

    def _build_list_url(self, stock_code: str, page: int = 1) -> str:
        """Build the list page URL based on sort mode.

        Page 1 has no _{N} suffix for either format.
        """
        if self.sort == "publish":
            if page == 1:
                return f"https://guba.eastmoney.com/list,{stock_code},f.html"
            return GUBA_LIST_URL_F.format(code=stock_code, page=page)
        else:
            if page == 1:
                return f"https://guba.eastmoney.com/topic,{stock_code}.html"
            return GUBA_LIST_URL_TOPIC.format(code=stock_code, page=page)

    def _fetch_list_page(
        self, stock_code: str, page: int = 1, year_hint: int | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """Fetch one page of the post list.

        Returns (DataFrame, inferred_year) where inferred_year is the year
        determined from the dates on this page (or the hint if unchanged).
        DataFrame columns: date, time, title, body, post_id, url.
        Returns empty DataFrame on failure.
        """
        url = self._build_list_url(stock_code, page)
        resp = _fetch_with_retry(url)

        if resp is None:
            logger.warning(
                "Guba list page %d failed for %s after retries", page, stock_code
            )
            return _empty_df(), year_hint or datetime.now().year

        # Detect rate-limit / mobile-redirect pages (short HTML, no article_list).
        # EastMoney returns a ~2800-char mobile page when the IP is throttled.
        if len(resp.text) < BLOCK_PAGE_MIN_LEN and "article_list" not in resp.text:
            logger.warning(
                "Guba page %d for %s appears rate-limited (len=%d), cooling down %ds",
                page, stock_code, len(resp.text), BLOCK_COOLDOWN,
            )
            time.sleep(BLOCK_COOLDOWN)
            resp = _fetch_with_retry(url)
            if resp is None or (
                len(resp.text) < BLOCK_PAGE_MIN_LEN and "article_list" not in resp.text
            ):
                logger.error(
                    "Guba page %d for %s still blocked after cooldown", page, stock_code
                )
                return _empty_df(), year_hint or datetime.now().year

        data = _extract_article_list(resp.text)
        if data is None:
            logger.debug(
                "No article_list data on page %d for %s (len=%d)", page, stock_code, len(resp.text),
            )
            return _empty_df(), year_hint or datetime.now().year

        posts = data.get("re", [])
        if not posts:
            return _empty_df(), year_hint or datetime.now().year

        yr = year_hint if year_hint is not None else datetime.now().year
        rows = []
        for post in posts:
            post_id = str(post.get("post_id", ""))
            if not post_id:
                continue

            title = post.get("post_title", "").strip()
            dt_str = post.get("post_publish_time", "").strip()
            date_str, time_str = _parse_datetime(dt_str, year_hint=yr)

            body = post.get("post_content", "").strip() or ""
            detail_url = GUBA_DETAIL_URL.format(
                code=stock_code, post_id=post_id
            )

            rows.append({
                "date": date_str,
                "time": time_str,
                "title": title,
                "body": body,
                "post_id": post_id,
                "url": detail_url,
            })

        # Detect year from the first successfully-parsed full date on this page.
        # Full-format dates (YYYY-MM-DD) give us the ground-truth year.
        for post in posts:
            dt_str = post.get("post_publish_time", "").strip()
            if re.match(r"\d{4}-\d{2}-\d{2}", dt_str):
                yr = int(dt_str[:4])
                break
        else:
            # No full date found — infer from the last (oldest) short date.
            # If the oldest date's month-day is later than today's month-day,
            # we've crossed into the previous year.
            if rows:
                oldest = rows[-1]["date"]
                if oldest and len(oldest) == 10:
                    oldest_mm = int(oldest[5:7])
                    today_mm = datetime.now().month
                    if oldest_mm > today_mm:
                        yr -= 1

        return pd.DataFrame(rows), yr

    def _fetch_post_body(self, stock_code: str, post_id: str) -> str:
        """Fetch the full text of a single post from its detail page.

        Returns the post body text, or empty string on failure.
        """
        url = GUBA_DETAIL_URL.format(code=stock_code, post_id=post_id)
        resp = _fetch_with_retry(url)

        if resp is None:
            logger.debug(
                "Guba detail page failed for %s/%s after retries",
                stock_code,
                post_id,
            )
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Primary: extract from div.newstext
        newstext = soup.find("div", class_="newstext")
        if newstext:
            text = newstext.get_text(strip=True)
            if len(text) > 5:
                return text

        # Fallback: extract from embedded script JSON
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and "post_content" in script.string:
                match = re.search(
                    r'"post_content"\s*:\s*"(.+?)"(?:\s*,\s*"post_abstract"|})',
                    script.string,
                    re.DOTALL,
                )
                if match:
                    import html as html_mod

                    raw = html_mod.unescape(match.group(1))
                    cleaned = BeautifulSoup(
                        raw, "html.parser"
                    ).get_text(strip=True)
                    if len(cleaned) > 5:
                        return cleaned
                break

        return ""

    def fetch_posts(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 10,
        fetch_bodies: bool = True,
    ) -> pd.DataFrame:
        """Fetch Guba forum posts for a stock.

        Args:
            stock_code: 6-digit A-share code.
            start_date: YYYY-MM-DD filter (inclusive). Posts older than this
                are excluded and trigger early termination.
            end_date: YYYY-MM-DD filter (inclusive).
            max_pages: Maximum pages to fetch (~80 posts/page).
            fetch_bodies: If True, fetch full post body from detail pages.

        Returns:
            DataFrame with columns: date, time, title, body, post_id, url.
        """
        all_pages = []
        year_hint = datetime.now().year

        for page in range(1, max_pages + 1):
            df_page, year_hint = self._fetch_list_page(stock_code, page, year_hint)
            if df_page.empty:
                if page == 1:
                    # First page empty means no data at all for this stock
                    break
                # Later pages empty means end of pagination
                break

            # Capture raw page size before any date filtering so the
            # pagination-termination check is not fooled by filtered-out rows.
            raw_page_size = len(df_page)

            # Filter by date range before adding to keep memory low
            if not df_page["date"].str.strip().eq("").all():
                df_page["date_parsed"] = pd.to_datetime(
                    df_page["date"], errors="coerce"
                )

                if start_date:
                    start_ts = pd.Timestamp(start_date)
                    # Stop if the newest post on this page is before start_date
                    valid_dates = df_page["date_parsed"].dropna()
                    if not valid_dates.empty:
                        newest = valid_dates.max()
                        if newest < start_ts:
                            # This page and beyond are too old
                            break

                    df_page = df_page[df_page["date_parsed"] >= start_ts]

                if end_date:
                    end_ts = pd.Timestamp(end_date)
                    df_page = df_page[df_page["date_parsed"] <= end_ts]

                df_page = df_page.drop(columns=["date_parsed"])
            else:
                # All dates empty — keep the page but can't filter
                pass

            if not df_page.empty:
                all_pages.append(df_page)

            # Stop pagination when the raw (unfiltered) page is sparse —
            # this means we've exhausted the available posts regardless of
            # how many rows the date filter kept.
            if raw_page_size < 40 and page > 1:
                break

            # Polite delay between page requests
            if page < max_pages:
                time.sleep(PAGE_DELAY)

        if not all_pages:
            return _empty_df()

        df = pd.concat(all_pages, ignore_index=True)

        # Drop duplicates by post_id
        df = df.drop_duplicates(subset=["post_id"])

        # Sort by date descending (newest first)
        if not df["date"].str.strip().eq("").all():
            df["_date_sort"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("_date_sort", ascending=False)
            df = df.drop(columns=["_date_sort"])

        # Fetch bodies concurrently (5 threads, ~5x speedup)
        if fetch_bodies and not df.empty:
            needs_body = df["body"].str.strip().eq("")
            need_count = needs_body.sum()
            if need_count > 0:
                logger.info(
                    "Fetching bodies for %d/%d posts for %s",
                    need_count, len(df), stock_code,
                )
                indices = df[needs_body].index.tolist()
                post_ids = [str(df.at[i, "post_id"]) for i in indices]
                bodies_result = [""] * len(indices)
                with ThreadPoolExecutor(max_workers=10) as pool:
                    futures = {
                        pool.submit(self._fetch_post_body, stock_code, pid): j
                        for j, pid in enumerate(post_ids)
                    }
                    for fut in as_completed(futures):
                        j = futures[fut]
                        try:
                            bodies_result[j] = fut.result() or ""
                        except Exception:
                            bodies_result[j] = ""
                for j, idx in enumerate(indices):
                    if bodies_result[j]:
                        df.at[idx, "body"] = bodies_result[j]

        return df.reset_index(drop=True)
