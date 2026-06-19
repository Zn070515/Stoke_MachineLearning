"""Browser fingerprint-consistent header generation.

Uses browserforge to generate HTTP header sets that are internally
consistent — UA version matches sec-ch-ua, Accept-Language matches
OS locale, etc. Anti-bot systems flag inconsistent header sets.
"""
from typing import Dict
from browserforge.headers import HeaderGenerator

SUPPORTED_BROWSERS = {"chrome", "firefox", "safari", "edge"}
SUPPORTED_DEVICES = {"desktop", "mobile"}
SUPPORTED_OS = {"windows", "macos", "linux", "android", "ios"}


class FingerprintGenerator:
    """Generates browser-consistent HTTP headers."""

    def __init__(
        self,
        browser: str = "chrome",
        device: str = "desktop",
        os: str = "windows",
    ):
        if browser not in SUPPORTED_BROWSERS:
            raise ValueError(
                f"Unsupported browser: {browser}. "
                f"Choose from: {sorted(SUPPORTED_BROWSERS)}"
            )
        self.browser = browser
        self.device = device
        self.os = os
        self._generator = HeaderGenerator(
            browser=browser, device=device, os=os
        )
        self._cached_headers: Dict[str, str] | None = None

    def generate(self) -> Dict[str, str]:
        """Generate headers, cached per instance like a real browser."""
        if self._cached_headers is None:
            self._cached_headers = self._generator.generate()
        return dict(self._cached_headers)

    def refresh(self):
        """Force regeneration (simulates a new browser profile)."""
        self._cached_headers = None
        return self.generate()
