"""Tiered proxy pool with per-domain error tracking.

Proxies are organized in tiers (free, paid). On error, the pool
rotates to the next available proxy. Each proxy tracks errors
per domain — a proxy blocked on sina.com may still work on eastmoney.com.
"""
import random
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Proxy:
    """A single proxy with error tracking per domain."""

    url: str
    tier: str = "free"
    max_errors: int = 5
    error_count: int = 0
    domain_errors: Dict[str, int] = field(default_factory=dict)

    def is_usable(self) -> bool:
        return self.error_count < self.max_errors

    def mark_error(self, domain: str | None = None):
        self.error_count += 1
        if domain:
            self.domain_errors[domain] = self.domain_errors.get(domain, 0) + 1

    def mark_success(self):
        self.error_count = max(0, self.error_count - 1)

    def domain_error_count(self, domain: str) -> int:
        return self.domain_errors.get(domain, 0)


class ProxyPool:
    """Pool of proxies with tiered rotation and error tracking."""

    def __init__(
        self,
        proxies: List[Proxy] | None = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._proxies: List[Proxy] = proxies or []
        self._current: Proxy | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_proxy(self) -> Proxy | None:
        if not self._enabled:
            return None
        usable = [p for p in self._proxies if p.is_usable()]
        if not usable:
            raise RuntimeError("No usable proxies available")
        self._current = random.choice(usable)
        return self._current

    def mark_current_bad(self, domain: str | None = None):
        if self._current:
            self._current.mark_error(domain)
            self._current = None

    def mark_current_good(self):
        if self._current:
            self._current.mark_success()

    def add_proxy(self, proxy: Proxy):
        self._proxies.append(proxy)

    def usable_count(self) -> int:
        return len([p for p in self._proxies if p.is_usable()])
