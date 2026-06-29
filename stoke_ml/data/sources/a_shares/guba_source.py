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

GUBA_PAGE_URL = "https://guba.eastmoney.com/topic,{code}_{page}.html"
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
RETRY_DELAY = 2.0  # seconds base delay
PAGE_DELAY = 1.0   # seconds between pages
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


def _parse_datetime(dt_str: str) -> tuple[str, str]:
    """Parse a Guba datetime string into (date, time) tuple.

    Handles formats:
      - "YYYY-MM-DD HH:MM:SS"
      - "MM-DD HH:MM"

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
        year = datetime.now().year
        return (
            f"{year}-{short_match.group(1)}-{short_match.group(2)}",
            short_match.group(3) + ":00",
        )

    logger.debug("Could not parse datetime: %r", dt_str)
    return "", ""


class GubaSource:
    """Fetch stock discussion posts from EastMoney Guba (股吧)."""

    @staticmethod
    def _build_list_url(stock_code: str, page: int = 1) -> str:
        """Build the list page URL.

        Page 1: topic,{code}.html (no _1 suffix).
        Pages >= 2: topic,{code}_{N}.html.
        """
        if page == 1:
            return f"https://guba.eastmoney.com/topic,{stock_code}.html"
        return GUBA_PAGE_URL.format(code=stock_code, page=page)

    def _fetch_list_page(self, stock_code: str, page: int = 1) -> pd.DataFrame:
        """Fetch one page of the post list.

        Returns DataFrame with columns: date, time, title, body, post_id, url.
        Returns empty DataFrame on failure.
        """
        url = self._build_list_url(stock_code, page)
        resp = _fetch_with_retry(url)

        if resp is None:
            logger.warning(
                "Guba list page %d failed for %s after retries", page, stock_code
            )
            return _empty_df()

        data = _extract_article_list(resp.text)
        if data is None:
            logger.debug(
                "No article_list data on page %d for %s", page, stock_code
            )
            return _empty_df()

        posts = data.get("re", [])
        if not posts:
            return _empty_df()

        rows = []
        for post in posts:
            post_id = str(post.get("post_id", ""))
            if not post_id:
                continue

            title = post.get("post_title", "").strip()
            dt_str = post.get("post_publish_time", "").strip()
            date_str, time_str = _parse_datetime(dt_str)

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

        return pd.DataFrame(rows)

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

        for page in range(1, max_pages + 1):
            df_page = self._fetch_list_page(stock_code, page)
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
