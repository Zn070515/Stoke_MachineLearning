"""EastMoney HTTP client with serial throttling and TLS impersonation.

Ported from a-stock-data V3.4.0 em_get() pattern. EastMoney's WAF enforces:
    >5 QPS / >=10 concurrent connections / >=200 req/min → temporary IP ban.

Uses curl-cffi for browser TLS fingerprint impersonation (Chrome 120) because
EastMoney rejects non-browser JA3/JA4 fingerprints at the TCP level.

All eastmoney.com calls MUST go through this module to avoid getting blocked.
Module-level state guarantees serial execution across all callers — even
concurrent threads — because EastMoney cannot handle parallel connections.

Usage:
    from stoke_ml.crawler.eastmoney import EastMoneyClient
    client = EastMoneyClient()
    resp = client.get("https://push2.eastmoney.com/api/qt/slist/get", params={...})
    data = client.datacenter("RPT_DAILYBILLBOARD_DETAILSNEW", filter_str="...")
"""

import logging
import random
import threading
import time
from typing import Optional

from curl_cffi import requests as curl_requests

from stoke_ml.crawler.fingerprint import FingerprintGenerator

logger = logging.getLogger(__name__)

# ── Module-level state (serial guarantee across all EastMoneyClient instances) ──
_lock = threading.Lock()
_session: Optional[curl_requests.Session] = None
_last_call: float = 0.0
_ref_count: int = 0

DEFAULT_MIN_INTERVAL = 1.0  # seconds
DEFAULT_TIMEOUT = 15  # seconds
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_BACKOFF = 60.0  # base seconds — EastMoney WAF cooldown is 60-300s

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


class EastMoneyClient:
    """HTTP client with EastMoney-specific serial throttling + TLS spoofing.

    All instances share the same underlying curl-cffi session and throttle
    state, guaranteeing at most one in-flight EastMoney request at any time.
    """

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
        impersonate: str = "chrome120",
    ):
        global _ref_count
        self._min_interval = min_interval
        self._timeout = timeout
        self._impersonate = impersonate
        self._fingerprint = FingerprintGenerator(
            browser="chrome", device="desktop", os="windows",
        )
        with _lock:
            _ref_count += 1
        self._init_session(impersonate=impersonate)

    @staticmethod
    def _init_session(impersonate: str = "chrome120", _lock_held: bool = False):
        """Create or re-create the shared curl-cffi session (idempotent)."""
        global _session
        if _lock_held:
            if _session is not None:
                return
            _session = curl_requests.Session(impersonate=impersonate)
            return
        with _lock:
            if _session is not None:
                return
            _session = curl_requests.Session(impersonate=impersonate)

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        """GET request through the EastMoney throttle gate."""
        return self._request("GET", url, params=params, headers=headers,
                             timeout=timeout, **kwargs)

    def post(
        self,
        url: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        """POST request through the EastMoney throttle gate."""
        return self._request("POST", url, params=params, json=json,
                             headers=headers, timeout=timeout, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # session is shared — leave it open for other instances

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[int] = None,
        _attempt: int = 0,
        **kwargs,
    ):
        global _last_call

        with _lock:
            if _session is None:
                self._init_session(impersonate=self._impersonate, _lock_held=True)

            elapsed = time.time() - _last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                jitter = random.uniform(0.1, 0.5)
                time.sleep(wait + jitter)

            merged_headers = self._fingerprint.generate()
            if headers:
                merged_headers.update(headers)

            timeout_val = timeout or self._timeout

            resp = None
            try:
                if method == "GET":
                    resp = _session.get(
                        url, params=params, headers=merged_headers,
                        timeout=timeout_val, **kwargs,
                    )
                else:
                    resp = _session.post(
                        url, params=params, headers=merged_headers,
                        timeout=timeout_val, **kwargs,
                    )
            except Exception:
                logger.debug("EastMoney request exception (attempt %d/%d)",
                             _attempt + 1, MAX_RETRIES + 1, exc_info=True)

        # Only bump _last_call on success so failed requests don't
        # needlessly delay the retry / next caller.
        if resp is not None:
            _last_call = time.time()

        # Connection errors (resp is None): EastMoney intermittently drops TCP
        # connections as a WAF measure. Retry with backoff, same as HTTP errors.
        if resp is None and _attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF * (2 ** _attempt)
            time.sleep(backoff + random.uniform(0, 0.3))
            return self._request(
                method, url, params=params, headers=headers,
                timeout=timeout, _attempt=_attempt + 1, **kwargs,
            )

        if resp is None:
            raise ConnectionError(
                f"EastMoney request failed after {MAX_RETRIES + 1} attempts: {url}"
            )

        # Manual retry for HTTP errors.
        # 403 is deliberately excluded — it signals WAF blocking.
        if resp.status_code in RETRY_STATUSES and _attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF * (2 ** _attempt)
            time.sleep(backoff + random.uniform(0, 0.3))
            return self._request(
                method, url, params=params, headers=headers,
                timeout=timeout, _attempt=_attempt + 1, **kwargs,
            )

        return resp

    # ── Convenience wrappers ──────────────────────────────────────────────

    def datacenter(
        self,
        report_name: str,
        columns: str = "ALL",
        filter_str: str = "",
        page_size: int = 50,
        page_number: int = 1,
        sort_columns: str = "",
        sort_types: str = "-1",
    ) -> list[dict]:
        """Query EastMoney datacenter unified API.

        Covers: 龙虎榜, 解禁, 融资融券, 大宗交易, 股东户数, 分红.

        Returns list of record dicts, or empty list on failure.
        """
        params = {
            "reportName": report_name,
            "columns": columns,
            "filter": filter_str,
            "pageNumber": str(page_number),
            "pageSize": str(page_size),
            "sortColumns": sort_columns,
            "sortTypes": sort_types,
            "source": "WEB",
            "client": "WEB",
        }
        try:
            r = self.get(DATACENTER_URL, params=params)
            r.raise_for_status()
            d = r.json()
            if d.get("result") and d["result"].get("data"):
                return d["result"]["data"]
            return []
        except (OSError, ValueError, KeyError):
            logger.debug("datacenter query failed: %s", report_name, exc_info=True)
            return []

    def close(self):
        """Decrement refcount; close shared session only when last client exits."""
        global _session, _ref_count
        with _lock:
            _ref_count = max(0, _ref_count - 1)
            if _ref_count == 0 and _session is not None:
                _session.close()
                _session = None
