"""TLS fingerprint spoofing via curl-cffi.

Uses curl-impersonate patched libcurl to mimic browser TLS handshakes
(JA3/JA4 fingerprints), making Python requests indistinguishable
from real Chrome/Firefox/Safari browsers at the TLS level.
"""
from curl_cffi import requests

SUPPORTED_IMPERSONATE = {
    "chrome110", "chrome116", "chrome120", "chrome123", "chrome124",
    "safari15_5", "safari17_0",
    "firefox",
    "edge99", "edge101", "edge110",
}


class TLSSession:
    """HTTP session with browser TLS fingerprint impersonation."""

    def __init__(self, impersonate: str = "chrome120"):
        if impersonate not in SUPPORTED_IMPERSONATE:
            raise ValueError(
                f"Unsupported impersonate target: {impersonate}. "
                f"Choose from: {sorted(SUPPORTED_IMPERSONATE)}"
            )
        self.impersonate = impersonate
        self._session = requests.Session(impersonate=impersonate)

    def get(self, url: str, **kwargs):
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs):
        return self._session.post(url, **kwargs)

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
