"""Re-fetch Guba post bodies for stocks with low body coverage.

The detail page URL was changed from topic,{code},{post_id}.html to
news,{code},{post_id}.html, which includes post_content in embedded JSON
for ALL stocks (previously only worked for ~17% of stocks).

Concurrency: stocks are processed in parallel (default 4 workers), and
within each stock, body fetching uses 10 workers per stock.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/refetch_guba_bodies.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/refetch_guba_bodies.py --min-coverage 0.5
  PYTHONPATH=. ./.venv/Scripts/python scripts/refetch_guba_bodies.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/refetch_guba_bodies.py --workers 8
"""
import argparse
import logging
import os
import re
import html as html_mod
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

from stoke_ml.config import load_config
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.features.news_nlp import NewsSentimentAnalyzer, compute_raw_sentiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

GUBA_DETAIL_URL = "https://guba.eastmoney.com/news,{code},{post_id}.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://guba.eastmoney.com/",
}

_print_lock = threading.Lock()
_fetch_lock = threading.Lock()


def _ts_print(*args, **kwargs):
    """Thread-safe print with timestamp."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"{ts} [{args[0]}]", *args[1:], **kwargs)


def fetch_body(stock_code: str, post_id: str) -> str:
    """Fetch post body from Guba detail page."""
    url = GUBA_DETAIL_URL.format(code=stock_code, post_id=post_id)
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome146", timeout=15)
        if resp.status_code != 200:
            return ""

        # Primary: post_content in embedded JSON
        match = re.search(
            r'"post_content"\s*:\s*"(.+?)"(?:\s*,\s*"post_abstract"|})',
            resp.text,
            re.DOTALL,
        )
        if match:
            raw = html_mod.unescape(match.group(1))
            cleaned = BeautifulSoup(raw, "html.parser").get_text(strip=True)
            if len(cleaned) > 5:
                return cleaned

        # Fallback: div.newstext
        soup = BeautifulSoup(resp.text, "html.parser")
        newstext = soup.find("div", class_="newstext")
        if newstext:
            text = newstext.get_text(strip=True)
            if len(text) > 5:
                return text
    except Exception:
        pass
    return ""

import time as _time_mod
_rate_lock = threading.Lock()
_last_request = 0.0
_MIN_INTERVAL = 0.5  # minimum seconds between requests to avoid WAF



def process_stock(code, raw_dir, guba_storage, analyzer, min_coverage):
    """Process a single stock: fetch missing bodies, recompute sentiment, regenerate silver/gold.

    Returns (code, fetched_count).
    """
    raw_path = os.path.join(raw_dir, f"{code}.parquet")
    with _fetch_lock:
        df = pd.read_parquet(raw_path)

    if "body" not in df.columns:
        df["body"] = ""

    needs_body = df["body"].fillna("").str.strip().str.len() == 0
    need_count = needs_body.sum()
    if need_count == 0:
        return code, 0

    coverage = 1 - need_count / len(df)
    if need_count == 0:
        return code, 0
    if min_coverage > 0 and coverage >= min_coverage:
        return code, 0

    logger.info("%s: %d/%d posts need body (%.1f%% coverage)",
                code, need_count, len(df), coverage * 100)

    indices = df[needs_body].index.tolist()
    post_ids = [str(df.at[i, "post_id"]) for i in indices]
    bodies_result = [""] * len(indices)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fetch_body, code, pid): j
            for j, pid in enumerate(post_ids)
        }
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                bodies_result[j] = fut.result() or ""
            except Exception:
                bodies_result[j] = ""

    fetched = 0
    for j, idx in enumerate(indices):
        if bodies_result[j]:
            df.at[idx, "body"] = bodies_result[j]
            fetched += 1

    if fetched > 0:
        df = compute_raw_sentiment(df, analyzer)
        with _fetch_lock:
            df.to_parquet(raw_path, index=False)
        logger.info("  %s: fetched %d bodies, saved", code, fetched)

        # Regenerate silver and gold (these have their own thread safety)
        silver = guba_storage.bronze_to_silver(code)
        if not silver.empty:
            guba_storage.save_silver(code, silver)

        gold = guba_storage.silver_to_gold(code, analyzer)
        if not gold.empty:
            guba_storage.save_daily_sentiment(gold)
            post_days = gold["has_guba_post"].sum()
            logger.info("  %s: %d sentiment days (%d with posts)", code, len(gold), post_days)

    return code, fetched


def main():
    parser = argparse.ArgumentParser(description="Re-fetch Guba post bodies")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="Only refetch stocks below this body coverage (default: 0.8)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of stocks to process in parallel (default: 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    raw_dir = os.path.join(data_dir, "a_shares", "guba_raw")

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted(
            f.replace(".parquet", "")
            for f in os.listdir(raw_dir)
            if f.endswith(".parquet")
        )

    guba_storage = GubaStorage(data_dir)
    analyzer = NewsSentimentAnalyzer()

    total_fetched = 0

    if args.workers > 1 and len(codes) > 1:
        logger.info("Processing %d stocks with %d parallel workers", len(codes), args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_stock, code, raw_dir, guba_storage, analyzer,
                            args.min_coverage): code
                for code in codes
            }
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    _, fetched = fut.result()
                    total_fetched += fetched
                except Exception as e:
                    logger.error("%s: failed — %s", code, e)
    else:
        for code in codes:
            _, fetched = process_stock(code, raw_dir, guba_storage, analyzer, args.min_coverage)
            total_fetched += fetched

    logger.info("Done: %d total bodies fetched", total_fetched)


if __name__ == "__main__":
    main()
