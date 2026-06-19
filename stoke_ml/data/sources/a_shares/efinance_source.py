"""EastMoney data source using curl-cffi TLS spoofing."""
import logging
import time

import pandas as pd
from curl_cffi import requests

from stoke_ml.data.sources.a_shares.base import AShareSourceBase

logger = logging.getLogger(__name__)

EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# Map EastMoney kline field codes to internal column names.
# Field order in API response may vary; we map by code, not position.
EM_FIELD_MAP = {
    "f51": "date",
    "f52": "open",
    "f53": "close",
    "f54": "high",
    "f55": "low",
    "f56": "volume",
    "f57": "amount",
    "f58": "amplitude",
    "f59": "pct_change",
    "f60": "change",
    "f61": "turnover",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, exponential


class EfinanceSource(AShareSourceBase):
    """A-share data via EastMoney API with TLS fingerprint spoofing."""

    SOURCE_NAME = "efinance"

    @staticmethod
    def _to_secid(stock_code: str) -> str:
        prefix = "1" if stock_code.startswith("6") else "0"
        return f"{prefix}.{stock_code}"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": ",".join(EM_FIELD_MAP.keys()),
            "beg": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "rtntype": "6",
            "secid": self._to_secid(stock_code),
            "klt": "101",
            "fqt": "1",
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    EM_URL, params=params, headers=EM_HEADERS,
                    impersonate="chrome120", timeout=30,
                )
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF ** attempt
                    logger.debug(
                        "EastMoney request failed for %s (attempt %d/%d): %s; "
                        "retrying in %.1fs",
                        stock_code, attempt, MAX_RETRIES, e, wait,
                    )
                    time.sleep(wait)
                continue

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF ** attempt
                    logger.debug(
                        "EastMoney HTTP %d for %s (attempt %d/%d); retrying in %.1fs",
                        resp.status_code, stock_code, attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                continue

            try:
                data = resp.json()
            except ValueError:
                last_error = "invalid JSON"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** attempt)
                continue

            # Check API-level error codes
            rc = data.get("rc")
            if rc is not None and rc != 0:
                msg = data.get("msg", "unknown error")
                logger.warning(
                    "EastMoney API error for %s: rc=%s msg=%s", stock_code, rc, msg,
                )
                return pd.DataFrame()

            klines = data.get("data", {})
            if not isinstance(klines, dict):
                klines = {}
            klines = klines.get("klines")
            if not klines:
                return pd.DataFrame()

            # Map fields by code, not position
            field_codes = params["fields2"].split(",")
            rows = [k.split(",") for k in klines]
            df = pd.DataFrame(rows, columns=field_codes)
            df.rename(columns=EM_FIELD_MAP, inplace=True)
            return self._normalize(df, stock_code)

        logger.warning(
            "Efinance fetch failed for %s after %d attempts: %s",
            stock_code, MAX_RETRIES, last_error,
        )
        return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].copy()
        for col in ["open", "high", "low", "close", "volume", "amount", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def is_available(self) -> bool:
        try:
            from curl_cffi import requests  # noqa: F811
            return True
        except ImportError:
            return False
