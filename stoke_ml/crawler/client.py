"""Unified crawler client combining all anti-block layers.

Integrates: TLS spoofing + fingerprint headers + session pool
+ proxy rotation + rate limiting. Provides a simple get/post
interface that routes through all defense layers transparently.
"""
from stoke_ml.crawler.tls import TLSSession
from stoke_ml.crawler.fingerprint import FingerprintGenerator
from stoke_ml.crawler.session_pool import SessionPool
from stoke_ml.crawler.proxy_pool import ProxyPool, Proxy
from stoke_ml.crawler.rate_limiter import RateLimiter, _extract_domain


class CrawlerClient:
    """HTTP client with defense-in-depth anti-blocking."""

    def __init__(
        self,
        impersonate: str = "chrome120",
        browser: str = "chrome",
        device: str = "desktop",
        os: str = "windows",
        session_pool_size: int = 50,
        proxy_enabled: bool = False,
        proxies: list | None = None,
        base_delay_sec: float = 2.0,
        daily_quota: int = 10000,
    ):
        self._fingerprint = FingerprintGenerator(
            browser=browser, device=device, os=os
        )
        self._session_pool = SessionPool(
            max_sessions=session_pool_size,
            impersonate=impersonate,
        )
        proxy_list = [Proxy(url=p) for p in (proxies or [])]
        self._proxy_pool = ProxyPool(
            proxies=proxy_list, enabled=proxy_enabled
        )
        self._rate_limiter = RateLimiter(
            base_delay_sec=base_delay_sec,
            daily_quota=daily_quota,
        )

    def get(self, url: str, **kwargs):
        """Perform a GET request through all defense layers."""
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        """Perform a POST request through all defense layers."""
        return self._request("POST", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs):
        domain = _extract_domain(url)

        if not self._rate_limiter.can_request(domain):
            raise RuntimeError(f"Rate limit or circuit breaker: {domain}")

        headers = self._fingerprint.generate()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        try:
            proxy = self._proxy_pool.get_proxy()
        except RuntimeError:
            proxy = None
        if proxy:
            kwargs["proxies"] = {"http": proxy.url, "https": proxy.url}

        self._rate_limiter.wait(domain)
        session = self._session_pool.get_session()

        try:
            resp = session.http.request(
                method, url, headers=headers, **kwargs
            )
            session.mark_used()
            self._rate_limiter.record_request(domain)

            if resp.status_code >= 400:
                session.mark_bad()
                self._rate_limiter.record_failure(domain)
                if proxy:
                    self._proxy_pool.mark_current_bad(domain)
                if resp.status_code == 429:
                    self._rate_limiter.report_429()
                return resp

            session.mark_good()
            self._rate_limiter.report_success()
            self._rate_limiter.record_success(domain)
            if proxy:
                self._proxy_pool.mark_current_good()
            return resp
        except Exception as e:
            session.mark_bad()
            self._rate_limiter.record_failure(domain)
            if proxy:
                self._proxy_pool.mark_current_bad(domain)
            raise e

    def close(self):
        self._session_pool.close_all()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
