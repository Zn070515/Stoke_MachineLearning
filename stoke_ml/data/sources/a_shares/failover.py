"""Failover orchestrator for A-share data sources.

Tries sources in priority order:
0. Efinance (preferred - fast, reliable)
1. AKShare (fallback - comprehensive)
2. Tushare (optional - requires token)
3. Baostock (last resort - free, limited)
"""
import time
import logging
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource

logger = logging.getLogger(__name__)


class AShareDownloader:
    """Multi-source A-share data downloader with automatic failover."""

    def __init__(self):
        self._sources: list[AShareSourceBase] = [
            EfinanceSource(),
            AKShareSource(),
            TushareSource(),
            BaostockSource(),
        ]
        self._failure_counts: dict[str, int] = {}
        self._circuit_open: dict[str, float] = {}
        self._cooldown_sec = 300
        self._failure_threshold = 15

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        for source in self._sources:
            name = source.SOURCE_NAME
            if not source.is_available():
                logger.debug(f"Source {name} unavailable, skipping")
                continue
            if self._is_circuit_open(name):
                logger.debug(f"Circuit open for {name}, skipping")
                continue

            df = source.fetch_daily(stock_code, start_date, end_date)
            if len(df) > 0:
                self._record_success(name)
                return df
            else:
                self._record_failure(name)
                logger.warning(f"Source {name} returned empty for {stock_code}")

        logger.error(f"All sources failed for {stock_code}")
        return pd.DataFrame()

    def _record_failure(self, name: str):
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if self._failure_counts[name] >= self._failure_threshold:
            self._circuit_open[name] = time.time()

    def _record_success(self, name: str):
        self._failure_counts[name] = 0
        self._circuit_open.pop(name, None)

    def _is_circuit_open(self, name: str) -> bool:
        if name not in self._circuit_open:
            return False
        if time.time() - self._circuit_open[name] >= self._cooldown_sec:
            del self._circuit_open[name]
            self._failure_counts[name] = 0
            return False
        return True
