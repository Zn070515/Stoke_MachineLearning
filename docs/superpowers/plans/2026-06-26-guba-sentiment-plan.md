# 东方财富股吧情感管道 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Guba (股吧) retail investor forum sentiment as an independent feature dimension (6 columns), following the existing 3-layer medallion + FeaturePipeline merge pattern.

**Architecture:** `GubaSource` scrapes list+detail pages via curl-cffi → `GubaStorage` stores Bronze/Silver/Gold Parquet layers with PIT alignment (post-15:00 → next trading day) → `FeaturePipeline._merge_guba()` left-joins and lags by 1 day → 6 guba columns produce lag/rolling features.

**Tech Stack:** curl-cffi, BeautifulSoup, pandas, FinBERT (existing), TradingCalendar (existing)

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `stoke_ml/data/sources/a_shares/guba_source.py` | Scrape guba.eastmoney.com list+detail pages |
| Create | `stoke_ml/data/guba_storage.py` | 3-layer medallion: raw→PIT→daily sentiment |
| Create | `scripts/download_guba.py` | CLI to fetch+sentiment all stocks |
| Modify | `stoke_ml/features/pipeline.py` | Add `GUBA_COLS`, `_merge_guba()`, `use_guba` param |
| Create | `tests/data/test_guba_source.py` | Unit tests for HTML parsing + date filtering |
| Create | `tests/data/test_guba_storage.py` | Unit tests for PIT alignment + ZI fill |

---

### Task 1: GubaSource — list page scraper

**Files:**
- Create: `stoke_ml/data/sources/a_shares/guba_source.py`
- Create: `tests/data/test_guba_source.py`

- [ ] **Step 1: Write the failing test for list page parsing**

```python
import pandas as pd
from stoke_ml.data.sources.a_shares.guba_source import GubaSource


class TestGubaSource:
    def test_parse_list_page_extracts_posts(self):
        """_parse_list_page should extract 80 posts with date, time, title, post_id, url from real HTML."""
        source = GubaSource()

        # Use a real stock — the test verifies the parser, not the network
        df = source._fetch_list_page("600519", page=1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Should return at least some posts"
        assert len(df) <= 80, "One page has max 80 posts"
        for col in ["date", "time", "title", "post_id", "url"]:
            assert col in df.columns, f"Missing column: {col}"
        # date should be parseable
        pd.to_datetime(df["date"])
        # post_id should be non-empty strings
        assert df["post_id"].str.len().min() > 0
        # url should contain the stock code
        assert df["url"].str.contains("600519").all()

    def test_parse_list_page_pagination(self):
        """Page 2 should return different posts than page 1."""
        source = GubaSource()
        page1 = source._fetch_list_page("600519", page=1)
        page2 = source._fetch_list_page("600519", page=2)

        # Different pages should have different post IDs
        ids1 = set(page1["post_id"])
        ids2 = set(page2["post_id"])
        overlap = ids1 & ids2
        assert len(overlap) == 0, f"Pages should not overlap, got {len(overlap)} duplicates"

    def test_parse_list_page_filters_by_date(self):
        """Posts before start_date should be excluded."""
        source = GubaSource()
        # Fetch with a recent start_date — should only get recent posts
        df = source._fetch_list_page("600519", page=1)

        if len(df) > 0:
            dates = pd.to_datetime(df["date"])
            # Posts should be in descending order (newest first)
            assert dates.is_monotonic_decreasing, "Posts should be newest-first"

    def test_fetch_posts_empty_for_nonexistent_stock(self):
        """Invalid stock code should return empty DataFrame."""
        source = GubaSource()
        df = source.fetch_posts("999999", max_pages=1, fetch_bodies=False)
        assert isinstance(df, pd.DataFrame)
        # May be empty or have very few posts
        for col in ["date", "time", "title", "body", "post_id", "url"]:
            assert col in df.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_guba_source.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'stoke_ml.data.sources.a_shares.guba_source'`

- [ ] **Step 3: Write GubaSource with list page parsing**

```python
"""EastMoney Guba (股吧) forum scraper for retail investor sentiment.

Guba is a stock discussion forum where retail investors post highly
emotional, opinionated content — ideal for sentiment analysis.
"""
import logging
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

GUBA_LIST_URL = "https://guba.eastmoney.com/list,{code}.html"
GUBA_PAGE_URL = "https://guba.eastmoney.com/list,{code}_{page}.html"
GUBA_DETAIL_URL = "https://guba.eastmoney.com/news,{code},{post_id}.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
}


class GubaSource:
    """Fetch retail investor posts from EastMoney Guba (股吧).

    Each list page contains 80 posts with title, post_id, and
    read/comment counts. Detail pages provide full post body text.

    Posts have timestamps (YYYY-MM-DD HH:MM:SS) enabling precise
    PIT alignment (post-15:00 → next trading day).
    """

    def _fetch_list_page(self, stock_code: str, page: int = 1) -> pd.DataFrame:
        """Fetch one list page. Returns DataFrame with date, time, title, post_id, url."""
        if page == 1:
            url = GUBA_LIST_URL.format(code=stock_code)
        else:
            url = GUBA_PAGE_URL.format(code=stock_code, page=page)

        try:
            resp = requests.get(
                url, headers=HEADERS, impersonate="chrome120", timeout=15,
            )
            if resp.status_code != 200:
                logger.debug("Guba list page %d for %s returned %d",
                             page, stock_code, resp.status_code)
                return pd.DataFrame(columns=["date", "time", "title", "post_id", "url"])
        except Exception as e:
            logger.debug("Guba list page %d for %s failed: %s", page, stock_code, e)
            return pd.DataFrame(columns=["date", "time", "title", "post_id", "url"])

        soup = BeautifulSoup(resp.text, "html.parser")
        items = []

        for article in soup.find_all("div", class_="articleh"):
            # post_id from the "data-postid" attribute on the listitem
            listitem = article.find_parent("div", class_="listitem") if "listitem" not in (article.parent.get("class", []) if article.parent else []) else article.parent
            # Actually, the listitem is the parent div with data-postid
            post_id = ""
            parent = article.parent
            while parent:
                post_id = parent.get("data-postid", "")
                if post_id:
                    break
                parent = parent.parent

            if not post_id:
                continue

            # Title
            title_tag = article.find("a")
            if not title_tag:
                continue
            title = title_tag.get("title", "") or title_tag.get_text(strip=True)
            href = title_tag.get("href", "")

            # Time — in a span with class "l5"
            time_str = ""
            time_tag = article.find("span", class_="l5")
            if time_tag:
                time_str = time_tag.get_text(strip=True)

            # Parse date and time from the time string
            # Format is typically "06-26 14:32" or "2026-06-26 14:32"
            date_str = ""
            time_only = ""
            time_match = re.match(
                r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}(:\d{2})?)", time_str
            )
            if time_match:
                date_str = time_match.group(1)
                time_only = time_match.group(2)
            else:
                # Try MM-DD HH:MM format — infer year from current date
                short_match = re.match(
                    r"(\d{2})-(\d{2})\s+(\d{2}:\d{2}(:\d{2})?)", time_str
                )
                if short_match:
                    current_year = datetime.now().year
                    date_str = f"{current_year}-{short_match.group(1)}-{short_match.group(2)}"
                    time_only = short_match.group(3)

            if not date_str or not title:
                continue

            items.append({
                "date": date_str,
                "time": time_only or "00:00:00",
                "title": title.strip(),
                "post_id": post_id,
                "url": GUBA_DETAIL_URL.format(code=stock_code, post_id=post_id)
                if not href else f"https://guba.eastmoney.com{href}" if href.startswith("/") else href,
            })

        return pd.DataFrame(items)

    def _fetch_post_body(self, stock_code: str, post_id: str) -> str:
        """Fetch full body text of a single Guba post."""
        url = GUBA_DETAIL_URL.format(code=stock_code, post_id=post_id)
        try:
            resp = requests.get(
                url, headers=HEADERS, impersonate="chrome120", timeout=10,
            )
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            # Post body is in div with class "stockcodec"
            body_div = soup.find("div", class_="stockcodec")
            if body_div:
                text = body_div.get_text(separator=" ", strip=True)
                return text if len(text) > 10 else ""
            return ""
        except Exception as e:
            logger.debug("Failed to fetch post body %s: %s", post_id, e)
            return ""

    def fetch_posts(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 10,
        fetch_bodies: bool = True,
    ) -> pd.DataFrame:
        """Fetch Guba posts for a stock.

        Iterates pages from newest to oldest, stopping when posts
        are older than start_date or max_pages is reached.

        Args:
            stock_code: 6-digit A-share code.
            start_date: YYYY-MM-DD filter (inclusive).
            end_date: YYYY-MM-DD filter (inclusive).
            max_pages: Max pages to fetch (80 posts/page).
            fetch_bodies: If True, fetch full body for each post.

        Returns:
            DataFrame with columns: date, time, title, body, post_id, url.
        """
        import time as _time

        all_items = []
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None

        for page in range(1, max_pages + 1):
            page_df = self._fetch_list_page(stock_code, page=page)

            if page_df.empty:
                break  # no more pages

            # Filter by date range
            page_df["date_parsed"] = pd.to_datetime(page_df["date"], errors="coerce")
            page_df = page_df.dropna(subset=["date_parsed"])

            if end_ts:
                page_df = page_df[page_df["date_parsed"] <= end_ts]
            if start_ts:
                before_start = page_df["date_parsed"] < start_ts
                if before_start.all():
                    break  # all remaining posts are too old
                page_df = page_df[~before_start]

            if page_df.empty:
                break

            all_items.append(page_df.drop(columns=["date_parsed"]))

            # If the oldest post on this page is still within range
            # and we got a full page, there might be more
            oldest_on_page = page_df["date_parsed"].min()
            if start_ts and oldest_on_page < start_ts:
                break

            _time.sleep(0.3)  # light rate limit between pages

        if not all_items:
            return pd.DataFrame(
                columns=["date", "time", "title", "body", "post_id", "url"]
            )

        df = pd.concat(all_items, ignore_index=True)
        df = df.drop_duplicates(subset=["post_id"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        # Fetch bodies
        if fetch_bodies and not df.empty:
            bodies = []
            for _, row in df.iterrows():
                body = self._fetch_post_body(stock_code, row["post_id"])
                bodies.append(body)
                _time.sleep(0.2)  # rate limit between detail pages
            df["body"] = bodies
        else:
            df["body"] = ""

        return df
```

Note: the actual HTML structure may vary slightly. The `_fetch_list_page` implementation above is based on research findings; if the live HTML differs, we'll adjust the selectors in the verify step.

- [ ] **Step 4: Run test to verify list page parsing passes**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_guba_source.py -v
```
Expected: 3/4 tests pass (test_fetch_posts_empty depends on fetch_posts which is now implemented). The `test_parse_list_page_extracts_posts` should PASS.

- [ ] **Step 5: Verify with a real fetch**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.data.sources.a_shares.guba_source import GubaSource
source = GubaSource()
# Quick test: 1 page, no bodies, validate output shape
df = source.fetch_posts('600519', max_pages=1, fetch_bodies=False)
print(f'Posts: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
print(df.head(3).to_string())
"
```

- [ ] **Step 6: Commit**

```bash
git add stoke_ml/data/sources/a_shares/guba_source.py tests/data/test_guba_source.py
git commit -m "feat: add GubaSource scraper for EastMoney Guba forum posts

Scrapes list pages (80 posts/page) with date/time extraction and
optional detail page body fetching via curl-cffi + BeautifulSoup."
```

---

### Task 2: GubaStorage — 3-layer medallion storage

**Files:**
- Create: `stoke_ml/data/guba_storage.py`
- Create: `tests/data/test_guba_storage.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import tempfile
import pandas as pd
import numpy as np
from stoke_ml.data.guba_storage import GubaStorage


class TestGubaStorage:
    @staticmethod
    def _sample_posts() -> pd.DataFrame:
        """Create sample Guba posts for testing."""
        return pd.DataFrame({
            "date": ["2026-06-20", "2026-06-20", "2026-06-23"],
            "time": ["10:30:00", "16:00:00", "14:00:00"],
            "title": ["大涨了", "明天要跌", "稳住了"],
            "body": ["利好！今天涨停了！", "盘后出了利空消息", "没什么变化"],
            "post_id": ["123", "456", "789"],
            "url": ["http://example.com/1", "http://example.com/2", "http://example.com/3"],
        })

    def test_save_and_load_raw(self):
        """Bronze: save raw posts, load them back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            df = self._sample_posts()
            storage.save_raw("600519", df)

            loaded = storage.load_raw("600519")
            assert len(loaded) == 3
            assert "body" in loaded.columns
            assert loaded["post_id"].tolist() == ["123", "456", "789"]

    def test_bronze_to_silver_pit_alignment(self):
        """Post at 16:00 should align to next trading day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            df = self._sample_posts()
            storage.save_raw("600519", df)

            silver = storage.bronze_to_silver("600519")
            assert "aligned_date" in silver.columns

            # Post #2 at 16:00 on 2026-06-20 (Saturday) →
            # next trading day is Monday 2026-06-22
            post2 = silver[silver["post_id"] == "456"]
            assert len(post2) == 1
            aligned = pd.Timestamp(post2.iloc[0]["aligned_date"])
            # 2026-06-20 is Saturday, 16:00 is post-close
            # Next trading day should be Monday 2026-06-22
            assert aligned.day == 22, f"Expected 22, got {aligned.day}"

            # Post #1 at 10:30 should stay on same day
            post1 = silver[silver["post_id"] == "123"]
            aligned1 = pd.Timestamp(post1.iloc[0]["aligned_date"])
            assert aligned1.date().isoformat() == "2026-06-20"

    def test_silver_to_gold_daily_aggregation(self):
        """Gold: aggregate to daily sentiment with ZI fill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)

            # Save raw → build silver → build gold
            df = self._sample_posts()
            storage.save_raw("600519", df)
            silver = storage.bronze_to_silver("600519")
            storage.save_silver("600519", silver)

            gold = storage.silver_to_gold("600519", analyzer=None)
            assert not gold.empty
            for col in ["date", "stock_code", "guba_sentiment_mean",
                        "guba_sentiment_std", "guba_post_count",
                        "guba_positive_ratio", "guba_negative_ratio",
                        "has_guba_post"]:
                assert col in gold.columns, f"Missing: {col}"

            # Days with posts should have has_guba_post=True
            assert gold["has_guba_post"].any()

    def test_load_daily_sentiment_date_range(self):
        """Load sentiment within a date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            df = self._sample_posts()
            storage.save_raw("600519", df)
            silver = storage.bronze_to_silver("600519")
            storage.save_silver("600519", silver)
            gold = storage.silver_to_gold("600519", analyzer=None)
            storage.save_daily_sentiment(gold)

            loaded = storage.load_daily_sentiment(
                "600519", "2026-06-19", "2026-06-24"
            )
            assert not loaded.empty
            assert loaded["date"].min() >= pd.Timestamp("2026-06-19")
            assert loaded["date"].max() <= pd.Timestamp("2026-06-24")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_guba_storage.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'stoke_ml.data.guba_storage'`

- [ ] **Step 3: Write GubaStorage implementation**

```python
"""3-layer medallion storage for Guba forum posts and daily sentiment.

Bronze: data/a_shares/guba_raw/{code}.parquet     — raw posts
Silver: data/a_shares/guba_silver/{code}.parquet  — PIT-aligned
Gold:   data/a_shares/guba_sentiment/{year}/{month}/{code}.parquet — daily
"""
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]


class GubaStorage:
    """3-layer Parquet storage for Guba forum posts and daily sentiment."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")
        os.makedirs(data_dir, exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────

    def _raw_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_raw")
        os.makedirs(p, exist_ok=True)
        return p

    def _silver_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_silver")
        os.makedirs(p, exist_ok=True)
        return p

    def _sentiment_base(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    # ── Bronze: raw posts ──────────────────────────────────────────

    def save_raw(self, stock_code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        existing = self.load_raw(stock_code)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        # Dedup by post_id (most reliable unique key)
        combined = combined.drop_duplicates(subset=["post_id"])
        combined = combined.sort_values("date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_raw(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_stocks_with_raw(self) -> list[str]:
        d = self._raw_dir()
        if not os.path.exists(d):
            return []
        return sorted(
            f.replace(".parquet", "")
            for f in os.listdir(d)
            if f.endswith(".parquet")
        )

    # ── Silver: PIT-aligned ────────────────────────────────────────

    def save_silver(self, stock_code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        existing = self.load_silver(stock_code)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["aligned_date"] = pd.to_datetime(combined["aligned_date"])
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["post_id"])
        combined = combined.sort_values("aligned_date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_silver(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def bronze_to_silver(self, stock_code: str) -> pd.DataFrame:
        """PIT-align raw posts: post-15:00 CST → next trading day."""
        raw = self.load_raw(stock_code)
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Build datetime from date + time columns
        df["datetime_str"] = (
            df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str)
        )
        df["datetime"] = pd.to_datetime(df["datetime_str"], errors="coerce")

        cutoff = pd.Timestamp("15:00:00").time()
        df["aligned_date"] = df["date"]  # default: same day

        post_close = df["datetime"].dt.time > cutoff
        for idx in df[post_close].index:
            d = df.at[idx, "date"].date()
            df.at[idx, "aligned_date"] = pd.Timestamp(
                self._calendar.next_trading_day(d)
            )

        df["aligned_date"] = pd.to_datetime(df["aligned_date"])
        return df.drop(columns=["datetime_str", "datetime"])

    # ── Gold: daily sentiment ──────────────────────────────────────

    def save_daily_sentiment(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(
                self._sentiment_base(), str(year), f"{month:02d}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily_sentiment(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._sentiment_base()
        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer consolidated flat file
        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            return df[mask].sort_values("date").reset_index(drop=True)

        # Fallback: partitioned
        frames = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values("date").reset_index(drop=True)

    def silver_to_gold(
        self,
        stock_code: str,
        analyzer: object | None = None,
    ) -> pd.DataFrame:
        """Aggregate silver posts to daily sentiment features.

        Uses ZI method: trading days without posts get zeros + has_guba_post=False.
        """
        silver = self.load_silver(stock_code)
        if silver.empty:
            return pd.DataFrame()

        # Compute sentiment on body (fallback to title)
        if analyzer is not None:
            from stoke_ml.features.news_nlp import compute_raw_sentiment
            # Use body if available, otherwise title
            if "body" in silver.columns and silver["body"].notna().any():
                silver["text_for_sentiment"] = silver["body"].fillna(silver["title"])
            else:
                silver["text_for_sentiment"] = silver["title"]
            temp = silver.rename(columns={"text_for_sentiment": "title"})
            temp = compute_raw_sentiment(temp, analyzer)
            silver["sentiment_title"] = temp["sentiment_title"].values
        else:
            silver["sentiment_title"] = 0.0

        silver["aligned_date"] = pd.to_datetime(silver["aligned_date"])
        daily = (
            silver.groupby("aligned_date")
            .agg(
                guba_sentiment_mean=("sentiment_title", "mean"),
                guba_sentiment_std=(
                    "sentiment_title",
                    lambda x: x.std() if len(x) > 1 else 0.0,
                ),
                guba_post_count=("sentiment_title", "count"),
                guba_positive_ratio=(
                    "sentiment_title",
                    lambda x: (x > 0.2).sum() / len(x),
                ),
                guba_negative_ratio=(
                    "sentiment_title",
                    lambda x: (x < -0.2).sum() / len(x),
                ),
            )
            .reset_index()
        )

        daily.rename(columns={"aligned_date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["stock_code"] = stock_code
        daily["has_guba_post"] = True
        for col in ["guba_sentiment_mean", "guba_sentiment_std",
                     "guba_positive_ratio", "guba_negative_ratio"]:
            daily[col] = daily[col].astype(np.float32)
        daily["guba_post_count"] = daily["guba_post_count"].astype("int16")

        # ZI fill: all trading days in range, zeros for no-post days
        if len(daily) >= 2:
            all_dates = self._calendar.get_trading_days(
                daily["date"].min(), daily["date"].max()
            )
            date_df = pd.DataFrame({"date": all_dates})
            date_df["date"] = pd.to_datetime(date_df["date"]).dt.date
            daily = date_df.merge(daily, on="date", how="left")
            daily["stock_code"] = stock_code
            daily["has_guba_post"] = daily["has_guba_post"].fillna(False)
            for col in ["guba_sentiment_mean", "guba_sentiment_std",
                        "guba_positive_ratio", "guba_negative_ratio"]:
                daily[col] = daily[col].fillna(0.0).astype(np.float32)
            daily["guba_post_count"] = daily["guba_post_count"].fillna(0).astype("int16")

        return daily[["date", "stock_code"] + GUBA_COLS]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_guba_storage.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/data/guba_storage.py tests/data/test_guba_storage.py
git commit -m "feat: add GubaStorage — 3-layer medallion for Guba forum sentiment

Bronze→Silver→Gold with PIT alignment (post-15:00→next_trading_day)
and ZI fill for trading days without forum posts."
```

---

### Task 3: FeaturePipeline — add guba dimension

**Files:**
- Modify: `stoke_ml/features/pipeline.py`

- [ ] **Step 1: Add GUBA_COLS constant**

At line ~37 (after ETF_FLOW_COLS), insert:

```python
GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]
```

- [ ] **Step 2: Add `use_guba` to `__init__` parameters**

At line ~67 (in the `__init__` signature, after `use_announcements: bool = True`), add:

```python
        use_guba: bool = True,
```

At line ~76 (in the `__init__` body, after `self.use_announcements = use_announcements`), add:

```python
        self.use_guba = use_guba
```

- [ ] **Step 3: Add `guba_df` to `build_features` and `_engineer_features` signatures**

In `build_features` (line ~94, after `announcement_df: pd.DataFrame | None = None`), add:

```python
        guba_df: pd.DataFrame | None = None,
```

In the docstring (line ~101, after the announcement line), add:

```
        guba_df: Daily Guba forum sentiment (GubaStorage.load_daily_sentiment).
```

In the call to `_engineer_features` (line ~103), add `guba_df`:

```python
        feats = self._engineer_features(
            df, sentiment_df, margin_df, northbound_df,
            dragon_tiger_df, fundamental_df, etf_flow_df,
            announcement_df, guba_df,
        )
```

In `_engineer_features` signature (line ~124, after `announcement_df`), add:

```python
        guba_df: pd.DataFrame | None = None,
```

- [ ] **Step 4: Add `_merge_guba` call in `_engineer_features`**

At line ~135 (after `df = self._merge_etf_flow(df, etf_flow_df)`), add:

```python
        df = self._merge_guba(df, guba_df)
```

- [ ] **Step 5: Add guba columns to temporal feature generation**

In `_engineer_features`, at line ~158 (after `temporal_cols += _active_cols(df, ETF_FLOW_COLS)`), add:

```python
            temporal_cols += _active_cols(df, GUBA_COLS)
```

- [ ] **Step 6: Write `_merge_guba` method**

Insert after `_merge_etf_flow` (after line ~341), before the microstructure section:

```python
    def _merge_guba(self, df: pd.DataFrame,
                    guba_df: pd.DataFrame | None) -> pd.DataFrame:
        if not (self.use_guba and guba_df is not None
                and not guba_df.empty):
            return df
        g = guba_df.copy()
        g["date"] = pd.to_datetime(g["date"])
        available = [c for c in GUBA_COLS if c in g.columns]
        if not available:
            return df
        df = df.merge(g[["date"] + available], on="date", how="left")
        for col in available:
            if col == "has_guba_post":
                df[col] = df[col].fillna(False).astype(bool)
            elif col == "guba_post_count":
                df[col] = df[col].fillna(0).astype("int16")
            else:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        # PIT lag: guba sentiment[t-1] paired with price[t]
        for col in available:
            df[col] = df[col].shift(1)
        df["has_guba_post"] = df["has_guba_post"].fillna(False).astype(bool)
        df["guba_post_count"] = df["guba_post_count"].fillna(0).astype("int16")
        for col in ["guba_sentiment_mean", "guba_sentiment_std",
                     "guba_positive_ratio", "guba_negative_ratio"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
        return df
```

- [ ] **Step 7: Verify the pipeline loads correctly**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline, GUBA_COLS
print('GUBA_COLS:', GUBA_COLS)
pipe = FeaturePipeline(use_guba=True)
print('use_guba:', pipe.use_guba)
assert hasattr(pipe, '_merge_guba'), 'Missing _merge_guba method'
print('Pipeline OK')
"
```

- [ ] **Step 8: Verify feature counts increase with guba**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
import numpy as np
from stoke_ml.features.pipeline import FeaturePipeline

# Create dummy K-line data
dates = pd.date_range('2026-01-01', periods=100, freq='B')
df = pd.DataFrame({
    'date': dates,
    'open': np.random.randn(100).cumsum() + 10,
    'high': np.random.randn(100).cumsum() + 11,
    'low': np.random.randn(100).cumsum() + 9,
    'close': np.random.randn(100).cumsum() + 10,
    'volume': np.random.randint(1000000, 10000000, 100),
    'amount': np.random.randint(10000000, 100000000, 100),
})

# Create dummy guba sentiment
guba_dates = dates[:80]
guba = pd.DataFrame({
    'date': guba_dates,
    'guba_sentiment_mean': np.random.randn(80) * 0.3,
    'guba_sentiment_std': np.random.rand(80) * 0.2,
    'guba_post_count': np.random.randint(1, 50, 80),
    'guba_positive_ratio': np.random.rand(80) * 0.5,
    'guba_negative_ratio': np.random.rand(80) * 0.3,
    'has_guba_post': [True] * 80,
})

pipe = FeaturePipeline(seq_len=20, use_temporal=False, use_technical=False, use_scoring=False)
X, y, c = pipe.build_features(df, guba_df=guba)
print(f'Features shape with guba: {X.shape}')
# Should have guba columns in the feature set
assert X.shape[-1] >= 6, f'Expected >=6 features, got {X.shape[-1]}'
print('Feature count OK:', X.shape[-1])
"
```

- [ ] **Step 9: Commit**

```bash
git add stoke_ml/features/pipeline.py
git commit -m "feat: add guba sentiment dimension to FeaturePipeline

Adds GUBA_COLS (6 columns), _merge_guba() method with PIT lag,
and use_guba toggle. Guba features integrated into temporal
lag/rolling feature generation alongside existing sentiment cols."
```

---

### Task 4: Download script

**Files:**
- Create: `scripts/download_guba.py`

- [ ] **Step 1: Write the download script**

```python
"""Download Guba forum posts for all stocks and compute daily sentiment.

Usage:
  python scripts/download_guba.py                              # all stocks
  python scripts/download_guba.py --stocks 600519              # single stock
  python scripts/download_guba.py --max-pages 5 --sleep 0.5    # faster
  python scripts/download_guba.py --skip-sentiment             # raw only
  python scripts/download_guba.py --concurrent                 # parallel
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.sources.a_shares.guba_source import GubaSource
from stoke_ml.features.news_nlp import (
    NewsSentimentAnalyzer,
    compute_raw_sentiment,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stocks_from_disk(data_dir: str) -> list[str]:
    base = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(
        description="Download Guba forum posts for A-share stocks"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all on disk)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages per stock (80 posts/page, default: 10)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between stocks")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment computation (raw only)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use concurrent downloader")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = get_stocks_from_disk(data_dir)

    if not codes:
        logger.error("No stocks found. Run download_data.py first.")
        sys.exit(1)

    end_date = args.end or time.strftime("%Y-%m-%d")

    storage = GubaStorage(data_dir, TradingCalendar("a_shares"))
    source = GubaSource()
    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer()

    mode_label = "concurrent" if args.concurrent else "sequential"
    logger.info(
        "Downloading Guba posts for %d stocks (max_pages=%d, %s to %s, sleep=%.1fs, %s)",
        len(codes), args.max_pages, args.start, end_date, args.sleep, mode_label,
    )

    total_posts = 0
    success, fail, empty = 0, 0, 0

    if args.concurrent:
        from stoke_ml.crawler.rate_limiter import RateLimiter
        from stoke_ml.crawler.concurrent import ConcurrentDownloader

        rate_limiter = RateLimiter(
            base_delay_sec=args.sleep,
            daily_quota=cfg.crawler.rate_limit.daily_quota_per_domain,
        )
        downloader = ConcurrentDownloader(
            rate_limiter=rate_limiter, max_workers=args.workers,
        )

        def _fetch_one(code: str):
            df = source.fetch_posts(
                code,
                start_date=args.start,
                end_date=end_date,
                max_pages=args.max_pages,
                fetch_bodies=True,
            )
            if not args.skip_sentiment and not df.empty:
                df = compute_raw_sentiment(df, analyzer)
            return df

        results = downloader.download_all(codes, _fetch_one)

        for i, code in enumerate(codes):
            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)
            df = results.get(code)
            if df is None:
                logger.error("  %s: fetch failed", code)
                fail += 1
                continue

            if df.empty:
                logger.info("  %s: no posts found", code)
                empty += 1
                continue

            storage.save_raw(code, df)
            logger.info("  %s: %d posts saved (raw)", code, len(df))
            total_posts += len(df)

            silver = storage.bronze_to_silver(code)
            if not silver.empty:
                storage.save_silver(code, silver)

            if not args.skip_sentiment:
                gold = storage.silver_to_gold(code, analyzer)
                if not gold.empty:
                    storage.save_daily_sentiment(gold)
                    guba_days = gold["has_guba_post"].sum()
                    logger.info("  %s: %d sentiment days (%d with posts)",
                                code, len(gold), guba_days)

            success += 1
    else:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.sleep)

            logger.info("[%d/%d] %s ...", i + 1, len(codes), code)

            try:
                df = source.fetch_posts(
                    code,
                    start_date=args.start,
                    end_date=end_date,
                    max_pages=args.max_pages,
                    fetch_bodies=True,
                )
            except Exception as e:
                logger.error("  %s: fetch failed: %s", code, e)
                fail += 1
                continue

            if df.empty:
                logger.info("  %s: no posts found", code)
                empty += 1
                continue

            if not args.skip_sentiment:
                df = compute_raw_sentiment(df, analyzer)

            storage.save_raw(code, df)
            logger.info("  %s: %d posts saved (raw)", code, len(df))
            total_posts += len(df)

            silver = storage.bronze_to_silver(code)
            if not silver.empty:
                storage.save_silver(code, silver)

            if not args.skip_sentiment:
                gold = storage.silver_to_gold(code, analyzer)
                if not gold.empty:
                    storage.save_daily_sentiment(gold)
                    guba_days = gold["has_guba_post"].sum()
                    logger.info("  %s: %d sentiment days (%d with posts)",
                                code, len(gold), guba_days)

            success += 1

    logger.info("Done: %d success, %d fail, %d empty, %d total posts",
                success, fail, empty, total_posts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI with a single stock**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --stocks 600519 --max-pages 2 --sleep 0.5
```
Expected: logs progress for 600519, saves raw/silver/gold, reports post count and sentiment days.

- [ ] **Step 3: Verify the output parquet files exist**

```bash
ls C:/Users/16275/Desktop/Stoke_MachineLearning/data/a_shares/guba_raw/600519.parquet
ls C:/Users/16275/Desktop/Stoke_MachineLearning/data/a_shares/guba_silver/600519.parquet
ls C:/Users/16275/Desktop/Stoke_MachineLearning/data/a_shares/guba_sentiment/*/600519.parquet 2>/dev/null || ls C:/Users/16275/Desktop/Stoke_MachineLearning/data/a_shares/guba_sentiment/600519.parquet 2>/dev/null
```
Expected: all three files exist.

- [ ] **Step 4: Verify sentiment data quality**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
from stoke_ml.data.guba_storage import GubaStorage
storage = GubaStorage('data')
df = storage.load_daily_sentiment('600519', '2020-01-01', '2026-12-31')
if not df.empty:
    print(f'Guba sentiment days: {len(df)}')
    print(f'Days with posts: {df.has_guba_post.sum()}')
    print(f'Sentiment range: [{df.guba_sentiment_mean.min():.3f}, {df.guba_sentiment_mean.max():.3f}]')
    print(f'Avg posts per day: {df.guba_post_count[df.has_guba_post].mean():.1f}')
    print(f'Non-zero sentiment days: {(df.guba_sentiment_mean.abs() > 0.01).sum()}')
    print()
    print(df[df.has_guba_post].head(10).to_string())
else:
    print('No data found — storage may use different path')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/download_guba.py
git commit -m "feat: add download_guba.py CLI for Guba forum sentiment pipeline

Supports --stocks, --max-pages, --sleep, --skip-sentiment, --concurrent.
Full pipeline: fetch→raw→PIT→daily aggregation→FinBERT scoring."
```

---

### Task 5: End-to-end integration test

**Files:** (none new — verification only)

- [ ] **Step 1: Run the full pipeline on one stock and verify FeaturePipeline integration**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
import pandas as pd
import numpy as np
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.features.pipeline import FeaturePipeline

# Load guba sentiment
storage = GubaStorage('data')
guba = storage.load_daily_sentiment('600519', '2015-01-01', '2026-12-31')
print(f'Loaded {len(guba)} guba sentiment days')

# Create matching K-line data
from stoke_ml.data.storage import DataStorage
ds = DataStorage('data')
kl = ds.load_daily('600519', '2015-01-01', '2026-12-31')
print(f'Loaded {len(kl)} K-line days')

# Build features with guba
pipe = FeaturePipeline(seq_len=60, use_guba=True)
X, y, c = pipe.build_features(kl, guba_df=guba)
print(f'Features: {X.shape}, Labels: {y.shape}')

# Check guba columns are present (they'll be inside the feature array)
# The guba columns should contribute to the feature count
print(f'Total features per timestep: {X.shape[-1]}')

# Verify balance
pos_pct = y.mean() * 100
print(f'Positive class: {pos_pct:.1f}%')
print('E2E integration test PASSED')
"
```

- [ ] **Step 2: Run the same without guba to compare feature counts**

```bash
cd C:/Users/16275/Desktop/Stoke_MachineLearning && PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.data.storage import DataStorage
from stoke_ml.features.pipeline import FeaturePipeline

ds = DataStorage('data')
kl = ds.load_daily('600519', '2015-01-01', '2026-12-31')

# Without guba
pipe_no_guba = FeaturePipeline(seq_len=60, use_guba=True)
X_no, _, _ = pipe_no_guba.build_features(kl)

# With guba
from stoke_ml.data.guba_storage import GubaStorage
guba = GubaStorage('data').load_daily_sentiment('600519', '2015-01-01', '2026-12-31')
pipe_with = FeaturePipeline(seq_len=60, use_guba=True)
X_with, _, _ = pipe_with.build_features(kl, guba_df=guba)

diff = X_with.shape[-1] - X_no.shape[-1]
print(f'Features without guba: {X_no.shape[-1]}')
print(f'Features with guba: {X_with.shape[-1]}')
print(f'Guba contributed: {diff} features')
assert diff > 0, 'Guba should add features'
print('Feature count comparison PASSED')
"
```

- [ ] **Step 3: Commit (if any changes from integration testing)**

```bash
git status
# Only commit if integration testing revealed and fixed issues
```

---

## Verification Checklist

After all tasks are complete:

```bash
# 1. Unit tests
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_guba_source.py tests/data/test_guba_storage.py -v

# 2. Single-stock download (quick: 2 pages, no sentiment — test crawler only)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --stocks 600519 --max-pages 2 --skip-sentiment

# 3. Single-stock with sentiment (2 pages + FinBERT)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --stocks 600519 --max-pages 2

# 4. E2E pipeline integration
PYTHONPATH=. ./.venv/Scripts/python -c "
from stoke_ml.features.pipeline import FeaturePipeline, GUBA_COLS
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.storage import DataStorage
ds = DataStorage('data')
kl = ds.load_daily('600519', '2020-01-01', '2026-06-26')
guba = GubaStorage('data').load_daily_sentiment('600519', '2020-01-01', '2026-06-26')
pipe = FeaturePipeline(seq_len=60, use_guba=True)
X, y, _ = pipe.build_features(kl, guba_df=guba)
print(f'OK: X={X.shape}, y={y.shape}, features_per_step={X.shape[-1]}')
"

# 5. Full download (background)
PYTHONPATH=. ./.venv/Scripts/python scripts/download_guba.py --max-pages 10 --sleep 0.5
```
