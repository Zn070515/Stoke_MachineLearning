"""Adaptive rate limiter with per-host throttling + circuit breaker.

Per-host throttling (HostThrottle) ensures different domains never block each
other — an EastMoney call spaced at 1.0 s doesn't delay a Sina call at 0.3 s.
The lock is held only during bookkeeping, not across sleep, so distinct buckets
fire concurrently without contention.

Ported from Vibe-Trading's HostThrottle pattern (backtest/loaders/_http.py).
"""
import logging
import random
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Per-host minimum interval defaults (seconds) ─────────────────────────
# These are conservative defaults tuned from real-world WAF behavior.
# Override per host by setting env var STOKE_ML_<HOST>_MIN_INTERVAL.

HOST_CONFIGS: dict[str, float] = {
    "eastmoney.com": 1.0,
    "push2.eastmoney.com": 1.0,
    "push2his.eastmoney.com": 1.0,
    "push2ex.eastmoney.com": 1.0,
    "datacenter-web.eastmoney.com": 1.0,
    "sina.com.cn": 0.3,
    "sina.com": 0.3,
    "10jqka.com.cn": 0.5,
    "tushare.pro": 0.3,
    "akshare": 0.5,
    "baostock": 0.3,
}
DEFAULT_HOST_INTERVAL = 2.0

# Upper bound on random jitter added on top of min_interval so parallel callers
# de-synchronize instead of all firing the instant the interval elapses.
_JITTER_MAX_S = 0.4


def _extract_domain(url: str) -> str:
    """Extract bare domain (no port) from URL for host-key lookup.

    >>> _extract_domain("https://push2.eastmoney.com/api/qt/slist/get")
    'push2.eastmoney.com'
    >>> _extract_domain("http://localhost:8080/path")
    'localhost'
    """
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc or "default"
    return netloc.partition(":")[0]


def _resolve_interval(domain: str) -> float:
    """Resolve minimum interval for a domain.

    Checks HOST_CONFIGS for exact and suffix matches (e.g. 'push2.eastmoney.com'
    matches 'eastmoney.com' via suffix), then env var override, then default.
    """
    if domain in HOST_CONFIGS:
        base = HOST_CONFIGS[domain]
    else:
        # Suffix match: push2.eastmoney.com matches eastmoney.com
        base = DEFAULT_HOST_INTERVAL
        for suffix, interval in HOST_CONFIGS.items():
            if domain.endswith(suffix):
                base = interval
                break

    # Env var override: STOKE_ML_<DOMAIN_WITH_DOTS_AS_UNDERSCORES>_MIN_INTERVAL
    env_key = f"STOKE_ML_{domain.replace('.', '_').replace('-', '_')}_MIN_INTERVAL"
    import os
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            val = float(env_val)
            if val > 0:
                return val
            logger.warning(
                "Env %s=%s is not positive, using default %.1fs for %s",
                env_key, env_val, base, domain,
            )
        except ValueError:
            logger.warning(
                "Env %s=%s is not a valid float, using default %.1fs for %s",
                env_key, env_val, base, domain,
            )
    return base


# ── HostThrottle ─────────────────────────────────────────────────────────

class HostThrottle:
    """Process-wide per-bucket minimum-spacing gate.

    One instance guards all callers. ``wait(bucket, min_interval)`` blocks until
    at least ``min_interval`` seconds (plus jitter) have elapsed since the last
    request tagged with the same ``bucket``. The lock is held only for the
    bookkeeping arithmetic, not across the sleep, so distinct buckets never
    block one another.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, bucket: str, min_interval: float) -> None:
        """Block until ``bucket`` is allowed to fire, then record the slot.

        The *reserved fire time* (jitter included) is what gets stored, so the
        next caller spaces off this caller's actual fire instant. Jitter only
        pushes a slot later, never earlier — consecutive requests stay at least
        ``min_interval`` apart even during concurrent bursts.
        """
        if min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last = self._last.get(bucket)
            if last is None or now >= last + min_interval:
                fire_at = now
            else:
                fire_at = last + min_interval + random.uniform(0.0, _JITTER_MAX_S)
            self._last[bucket] = fire_at
        sleep_for = fire_at - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


# ── CircuitBreaker (unchanged) ───────────────────────────────────────────

class CircuitBreaker:
    """Stops requests to a domain after consecutive failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 300,
    ):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}

    def record_failure(self, domain: str):
        self._failures[domain] += 1
        if self._failures[domain] >= self._threshold:
            self._opened_at[domain] = time.time()

    def record_success(self, domain: str):
        self._failures[domain] = 0
        self._opened_at.pop(domain, None)

    def is_open(self, domain: str) -> bool:
        if domain not in self._opened_at:
            return False
        elapsed = time.time() - self._opened_at[domain]
        if elapsed >= self._cooldown:
            self._failures[domain] = 0
            del self._opened_at[domain]
            return False
        return True


# ── RateLimiter (per-host upgrade) ───────────────────────────────────────

class RateLimiter:
    """Adaptive request rate limiter with per-host throttling.

    Uses HostThrottle internally so different domains get independent
    spacing. The legacy ``wait()`` (no domain) still works for backward
    compatibility with ConcurrentDownloader — it uses a shared "default"
    bucket with the configured base delay.
    """

    def __init__(
        self,
        base_delay_sec: float = 2.0,
        jitter_factor: float = 0.5,
        max_backoff_sec: float = 300,
        daily_quota: int = 10000,
        failure_threshold: int = 5,
        cooldown_seconds: float = 300,
    ):
        self._base_delay = base_delay_sec
        self._jitter = jitter_factor
        self._max_backoff = max_backoff_sec
        self._daily_quota = daily_quota
        self._current_delay = base_delay_sec
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._day_start = time.time()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )
        self._throttle = HostThrottle()

    @property
    def current_delay(self) -> float:
        return self._current_delay

    def wait(self, domain: Optional[str] = None):
        """Block until the next request is allowed.

        Args:
            domain: Optional host domain. When provided, uses per-host
                throttling (spacing from HOST_CONFIGS). When None, uses
                the legacy global delay (backward-compatible).
        """
        if domain:
            interval = _resolve_interval(domain)
            self._throttle.wait(domain, interval)
        else:
            # Legacy mode: global delay with jitter
            jitter = self._current_delay * self._jitter * (0.5 + random.random())
            time.sleep(self._current_delay + jitter)

    def report_429(self):
        self._current_delay = min(self._current_delay * 2, self._max_backoff)

    def report_success(self):
        self._current_delay = self._base_delay

    def can_request(self, domain: str) -> bool:
        self._reset_daily_if_needed()
        if self._daily_counts[domain] >= self._daily_quota:
            return False
        if self._circuit_breaker.is_open(domain):
            return False
        return True

    def record_request(self, domain: str):
        self._daily_counts[domain] += 1

    def record_failure(self, domain: str):
        self._circuit_breaker.record_failure(domain)

    def record_success(self, domain: str):
        self._circuit_breaker.record_success(domain)

    def _reset_daily_if_needed(self):
        if time.time() - self._day_start > 86400:
            self._daily_counts.clear()
            self._day_start = time.time()
