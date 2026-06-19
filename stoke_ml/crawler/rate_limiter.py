"""Adaptive rate limiter with circuit breaker pattern.

Features:
- Random jitter delays between requests
- Exponential backoff on 429/503 responses
- Circuit breaker: stop requesting a domain after N consecutive failures
- Daily quota tracking per domain
"""
import time
import random
from collections import defaultdict


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


class RateLimiter:
    """Adaptive request rate limiter."""

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

    @property
    def current_delay(self) -> float:
        return self._current_delay

    def wait(self):
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
