"""Concurrent downloader with shared rate limiting for batch stock operations."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from stoke_ml.crawler.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ConcurrentDownloader:
    """Download per-stock data in parallel with a shared rate limiter."""

    def __init__(self, rate_limiter: RateLimiter | None = None, max_workers: int = 4):
        self._rate_limiter = rate_limiter or RateLimiter(
            base_delay_sec=1.0, daily_quota=10000,
        )
        self._lock = threading.Lock()
        self._max_workers = max_workers

    def download_all(
        self,
        stock_codes: list[str],
        fetch_fn,
        sleep_between: float = 0.5,
    ) -> dict[str, pd.DataFrame | None]:
        """Download data for all stocks concurrently.

        Args:
            stock_codes: List of stock codes to process.
            fetch_fn: Callable(code) -> pd.DataFrame that fetches data for one stock.
            sleep_between: Min seconds between HTTP calls (enforced via rate limiter).

        Returns:
            Dict mapping stock_code -> DataFrame, or None on failure.
        """
        results: dict[str, pd.DataFrame | None] = {}
        failures: dict[str, str] = {}

        def _worker(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
            with self._lock:
                self._rate_limiter.wait()
            try:
                df = fetch_fn(code)
                return code, df, None
            except Exception as e:
                return code, None, str(e)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_worker, code): code for code in stock_codes}
            for future in as_completed(futures):
                code, df, err = future.result()
                if err:
                    failures[code] = err
                    results[code] = None
                else:
                    results[code] = df if df is not None else pd.DataFrame()

        if failures:
            logger.warning(
                "Concurrent download: %d/%d stocks failed",
                len(failures), len(stock_codes),
            )

        return results
