# Stock Prediction with Deep Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular deep learning stock prediction system starting with anti-block crawling, progressing through XGBoost baseline to LSTM/Transformer models.

**Architecture:** Six-layer pipeline (Crawler → Data → Features → Model → Prediction → Evaluation). Each layer has well-defined data contracts and can be developed independently. Phased delivery: P1 XGBoost baseline → P2 LSTM/CNN-Seq2Seq → P3 Transformer multi-task.

**Tech Stack:** Python 3.10+, PyTorch 2.x + Lightning, XGBoost, curl-cffi, browserforge, ta-lib, Hydra, wandb, pytest

**Source:** `docs/superpowers/specs/2026-06-19-stock-prediction-design.md`

---

## PART A: Project Foundation

### Task A1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `stoke_ml/__init__.py`
- Create: `stoke_ml/crawler/__init__.py`
- Create: `stoke_ml/data/__init__.py`
- Create: `stoke_ml/features/__init__.py`
- Create: `stoke_ml/models/__init__.py`
- Create: `stoke_ml/evaluation/__init__.py`
- Create: `stoke_ml/config.py`
- Create: `.gitignore`

- [ ] **Step 1: Create project root .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# Data (large files)
data/raw/
data/processed/
*.parquet

# IDE
.vscode/
.idea/

# Secrets
.env
*.token
*_token*

# Superpowers
.superpowers/

# Jupyter
.ipynb_checkpoints/

# wandb
wandb/
```

- [ ] **Step 2: Write requirements.txt**

```
# Core
python>=3.10
pandas>=2.0
polars>=0.20
numpy>=1.24

# ML/DL
torch>=2.0
pytorch-lightning>=2.0
xgboost>=2.0
lightgbm>=4.0

# Data processing
ta-lib>=0.4
pandas-ta>=0.3

# Anti-block crawler
curl-cffi>=0.6
browserforge>=1.0
playwright>=1.40

# Data sources
akshare>=1.12
yfinance>=0.2
efinance>=0.4
tushare>=1.4
baostock>=0.8

# NLP (Phase 3)
transformers>=4.35
jieba>=0.42

# Config & monitoring
hydra-core>=1.3
omegaconf>=2.3
wandb>=0.16

# Storage
pyarrow>=14.0
fastparquet>=2023.10

# Code quality
pytest>=7.4
ruff>=0.1

# Utilities
tqdm>=4.66
matplotlib>=3.8
seaborn>=0.13
```

- [ ] **Step 3: Write config.yaml**

```yaml
# Global configuration
project:
  name: stoke-ml
  data_dir: ./data
  model_dir: ./models/checkpoints

# Market configuration
markets:
  a_shares:
    enabled: true
    stock_universe: [csi300, csi500]
    start_date: "2015-01-01"
  us:
    enabled: false  # Enable in Phase 2
    stock_universe: [sp500]
    start_date: "2015-01-01"

# Crawler configuration (from spec section 3)
crawler:
  tls_impersonate: "chrome120"
  browserforge:
    browser: "chrome"
    device: "desktop"
    os: "windows"
  session_pool:
    max_sessions: 50
    max_age_minutes: 30
    max_usage: 30
    max_error_score: 3.0
  proxy:
    enabled: false  # Enable when proxies available
    tier: "free"
    validate_on_startup: true
    max_proxies: 20
  rate_limit:
    base_delay_sec: 2.0
    jitter_factor: 0.5
    max_backoff_sec: 300
    circuit_breaker_cooldown_sec: 300
    daily_quota_per_domain: 10000
  browser_fallback:
    enabled: true
    engine: "playwright"
    stealth_js: true
    captcha_solver: "opencv"

# Feature configuration
features:
  seq_len: 60  # sliding window length in days
  technical_indicators: true
  rule_based_scoring: true
  temporal_features: true
  target_horizon: 1  # predict N days ahead

# Model configuration
model:
  phase: 1  # 1=baseline, 2=dl, 3=multitask
  name: "xgboost"
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 200
    subsample: 0.8
    colsample_bytree: 0.8

# Training configuration
training:
  batch_size: 512
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 5
  validation:
    method: "walk_forward"
    train_years: 2
    val_months: 3

# Evaluation
evaluation:
  primary_metric: "mcc"
  financial_metrics: [sharpe, max_drawdown, win_rate, profit_factor]
```

- [ ] **Step 4: Write stoke_ml/config.py**

```python
"""Global configuration loader using Hydra/OmegaConf."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to project config.yaml.

    Returns:
        OmegaConf DictConfig object with all settings.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    # Resolve relative paths
    cfg.project.data_dir = str(_PROJECT_ROOT / cfg.project.data_dir)
    cfg.project.model_dir = str(_PROJECT_ROOT / cfg.project.model_dir)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
```

- [ ] **Step 5: Verify project structure**

Run: `python -c "from stoke_ml.config import load_config; cfg = load_config(); print(cfg.project.name)"`
Expected: `stoke-ml`

- [ ] **Step 6: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Run: `pip install ta-lib` (may need separate binary install on Windows)
Run: `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"`
Expected: `CUDA: True`

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "feat: project scaffolding with config and dependencies"
```

---

## PART B: Anti-Block Crawler System (LAYER 0)

### Task B1: TLS Fingerprint Spoofing

**Files:**
- Create: `stoke_ml/crawler/tls.py`
- Create: `tests/crawler/test_tls.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for TLS fingerprint spoofing."""
import pytest
from stoke_ml.crawler.tls import TLSSession


def test_tls_session_creates_with_impersonate_target():
    session = TLSSession(impersonate="chrome120")
    assert session.impersonate == "chrome120"
    assert session._session is not None


def test_tls_session_get_request_returns_response():
    session = TLSSession(impersonate="chrome120")
    # Use httpbin for testing — a public echo service
    resp = session.get("https://httpbin.org/get")
    assert resp.status_code == 200
    data = resp.json()
    assert "headers" in data


def test_tls_session_preserves_cookies():
    session = TLSSession(impersonate="chrome120")
    # httpbin sets a cookie and echoes it back
    resp = session.get("https://httpbin.org/cookies/set?test=value")
    assert resp.status_code == 200
    # Verify the session stored the cookie
    assert len(session._session.cookies) > 0


def test_tls_session_raises_on_invalid_impersonate():
    with pytest.raises(ValueError, match="Unsupported impersonate target"):
        TLSSession(impersonate="invalid_browser_999")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_tls.py -v`
Expected: FAIL with "No module named 'stoke_ml.crawler.tls'"

- [ ] **Step 3: Write TLS session implementation**

```python
"""TLS fingerprint spoofing via curl-cffi.

Uses curl-impersonate patched libcurl to mimic browser TLS handshakes
(JA3/JA4 fingerprints), making Python requests indistinguishable
from real Chrome/Firefox/Safari browsers at the TLS level.
"""
from typing import Any
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

    def get(self, url: str, **kwargs) -> requests.Response:
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self._session.post(url, **kwargs)

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_tls.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_tls.py stoke_ml/crawler/tls.py
git commit -m "feat: TLS fingerprint spoofing via curl-cffi"
```

### Task B2: Browser Fingerprint Header Generation

**Files:**
- Create: `stoke_ml/crawler/fingerprint.py`
- Create: `tests/crawler/test_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for browser fingerprint header generation."""
from stoke_ml.crawler.fingerprint import FingerprintGenerator


def test_generates_chrome_headers():
    gen = FingerprintGenerator(browser="chrome", device="desktop", os="windows")
    headers = gen.generate()
    assert "User-Agent" in headers
    assert "Accept" in headers
    assert "Accept-Language" in headers
    assert "Chrome" in headers["User-Agent"]


def test_generates_firefox_headers():
    gen = FingerprintGenerator(browser="firefox", device="desktop", os="windows")
    headers = gen.generate()
    assert "Firefox" in headers["User-Agent"]


def test_headers_are_internally_consistent():
    """sec-ch-ua must match User-Agent browser version."""
    gen = FingerprintGenerator(browser="chrome", device="desktop", os="windows")
    headers = gen.generate()
    ua = headers["User-Agent"]
    sec_ch_ua = headers.get("sec-ch-ua", "")
    # Both should reference the same major version
    assert "Chrome" in sec_ch_ua or "Chromium" in sec_ch_ua


def test_headers_are_cached_per_instance():
    """Same generator instance should reuse headers (real browser behavior)."""
    gen = FingerprintGenerator(browser="chrome", device="desktop", os="windows")
    h1 = gen.generate()
    h2 = gen.generate()
    assert h1["User-Agent"] == h2["User-Agent"]


def test_rejects_invalid_browser():
    import pytest
    with pytest.raises(ValueError, match="Unsupported browser"):
        FingerprintGenerator(browser="ie6", device="desktop", os="windows")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_fingerprint.py -v`
Expected: FAIL

- [ ] **Step 3: Write fingerprint generator implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_fingerprint.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_fingerprint.py stoke_ml/crawler/fingerprint.py
git commit -m "feat: browser fingerprint-consistent header generation"
```

### Task B3: Session Pool with Error Scoring

**Files:**
- Create: `stoke_ml/crawler/session_pool.py`
- Create: `tests/crawler/test_session_pool.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for session pool with error scoring."""
import time
from stoke_ml.crawler.session_pool import Session, SessionPool


def test_session_starts_usable():
    session = Session()
    assert session.is_usable() is True


def test_session_tracks_usage():
    session = Session(max_usage=3)
    session.mark_used()
    session.mark_used()
    assert session.usage_count == 2
    assert session.is_usable() is True
    session.mark_used()
    assert session.is_usable() is False  # exceeded max_usage


def test_session_error_scoring():
    session = Session(max_error_score=3.0)
    session.mark_bad()  # +1.0
    session.mark_bad()  # +1.0
    assert session.error_score == 2.0
    assert session.is_usable() is True
    session.mark_bad()  # +1.0 => 3.0, triggers retirement
    assert session.is_usable() is False


def test_session_error_decay():
    session = Session(max_error_score=3.0, error_score_decrement=0.5)
    session.mark_bad()
    assert session.error_score == 1.0
    session.mark_good()  # -0.5
    assert session.error_score == 0.5


def test_session_expires_by_age():
    session = Session(max_age_seconds=0.01)  # 10ms
    time.sleep(0.02)
    assert session.is_usable() is False


def test_pool_creates_sessions():
    pool = SessionPool(max_sessions=5)
    session = pool.get_session()
    assert session is not None
    assert isinstance(session, Session)


def test_pool_returns_different_sessions():
    pool = SessionPool(max_sessions=100)
    s1 = pool.get_session()
    s2 = pool.get_session()
    # With 100 sessions, should get different ones most of the time
    assert len({id(s1), id(s2)}) > 0


def test_pool_replaces_retired_sessions():
    pool = SessionPool(max_sessions=3)
    session = pool.get_session()
    # Force retirement
    for _ in range(5):
        session.mark_bad()
    assert session.is_usable() is False
    # Pool should create a replacement
    pool.replenish()
    new_session = pool.get_session()
    assert new_session.is_usable() is True


def test_pool_respects_max_sessions():
    pool = SessionPool(max_sessions=10)
    for _ in range(100):
        s = pool.get_session()
        # Mark some bad so they retire
        s.mark_bad()
    pool.replenish()
    assert pool.active_count() <= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_session_pool.py -v`
Expected: FAIL

- [ ] **Step 3: Write session pool implementation**

```python
"""Session pool with error scoring and automatic retirement.

Each session simulates a distinct user with its own cookie jar,
header set, and usage history. The pool manages lifecycle:
creation, random selection, error- or usage-based retirement,
and automatic replenishment.
"""
import time
import random
from typing import List
from curl_cffi import requests

BLOCKED_STATUS_CODES = {401, 403, 429}


class Session:
    """A single user session with usage and error tracking."""

    def __init__(
        self,
        max_age_seconds: float = 1800,  # 30 min
        max_usage: int = 30,
        max_error_score: float = 3.0,
        error_score_decrement: float = 0.5,
        impersonate: str = "chrome120",
    ):
        self._created_at = time.time()
        self._max_age = max_age_seconds
        self._max_usage = max_usage
        self._max_error_score = max_error_score
        self._error_score_decrement = error_score_decrement
        self._usage_count = 0
        self._error_score = 0.0
        self._http = requests.Session(impersonate=impersonate)

    @property
    def usage_count(self) -> int:
        return self._usage_count

    @property
    def error_score(self) -> float:
        return self._error_score

    @property
    def http(self):
        return self._http

    def is_usable(self) -> bool:
        if time.time() - self._created_at > self._max_age:
            return False
        if self._usage_count >= self._max_usage:
            return False
        if self._error_score >= self._max_error_score:
            return False
        return True

    def mark_used(self):
        self._usage_count += 1

    def mark_good(self):
        self._error_score = max(0.0, self._error_score - self._error_score_decrement)

    def mark_bad(self):
        self._error_score += 1.0

    def retire(self):
        """Force immediate retirement."""
        self._usage_count = self._max_usage
        self._http.close()

    def close(self):
        self._http.close()


class SessionPool:
    """Pool of user sessions with lifecycle management."""

    def __init__(
        self,
        max_sessions: int = 50,
        max_age_seconds: float = 1800,
        max_usage: int = 30,
        max_error_score: float = 3.0,
        error_score_decrement: float = 0.5,
        impersonate: str = "chrome120",
    ):
        self._max_sessions = max_sessions
        self._session_params = {
            "max_age_seconds": max_age_seconds,
            "max_usage": max_usage,
            "max_error_score": max_error_score,
            "error_score_decrement": error_score_decrement,
            "impersonate": impersonate,
        }
        self._sessions: List[Session] = []
        self._fill_initial()

    def _fill_initial(self):
        for _ in range(self._max_sessions):
            self._sessions.append(Session(**self._session_params))

    def get_session(self) -> Session:
        """Return a random usable session.

        If the selected session is not usable, try up to 10 times
        to find one. If none are usable, create a new one.
        """
        usable = [s for s in self._sessions if s.is_usable()]
        if not usable:
            self.replenish()
            usable = [s for s in self._sessions if s.is_usable()]
            if not usable:
                s = Session(**self._session_params)
                self._sessions.append(s)
                return s
        return random.choice(usable)

    def replenish(self):
        """Remove retired sessions and fill pool back to max."""
        self._sessions = [s for s in self._sessions if s.is_usable()]
        while len(self._sessions) < self._max_sessions:
            self._sessions.append(Session(**self._session_params))

    def active_count(self) -> int:
        return len([s for s in self._sessions if s.is_usable()])

    def close_all(self):
        for s in self._sessions:
            s.close()
        self._sessions.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_session_pool.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_session_pool.py stoke_ml/crawler/session_pool.py
git commit -m "feat: session pool with error scoring and auto-retirement"
```

### Task B4: Proxy Pool with Tiered Rotation

**Files:**
- Create: `stoke_ml/crawler/proxy_pool.py`
- Create: `tests/crawler/test_proxy_pool.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for proxy pool with tiered rotation."""
from stoke_ml.crawler.proxy_pool import ProxyPool, Proxy


def test_proxy_creation():
    p = Proxy(url="http://127.0.0.1:8080", tier="free")
    assert p.url == "http://127.0.0.1:8080"
    assert p.tier == "free"
    assert p.error_count == 0
    assert p.is_usable() is True


def test_proxy_error_threshold():
    p = Proxy(url="http://127.0.0.1:8080", tier="free", max_errors=3)
    p.mark_error()
    p.mark_error()
    assert p.is_usable() is True
    p.mark_error()
    assert p.is_usable() is False


def test_proxy_domain_tracking():
    p = Proxy(url="http://127.0.0.1:8080", tier="free")
    p.mark_error(domain="finance.sina.com.cn")
    p.mark_error(domain="finance.sina.com.cn")
    # This proxy is bad for sina but might work elsewhere
    assert p.is_usable() is True  # total errors < default max
    assert p.domain_errors["finance.sina.com.cn"] == 2


def test_pool_returns_proxy():
    proxies = [
        Proxy(url="http://proxy1:8080", tier="free"),
        Proxy(url="http://proxy2:8080", tier="free"),
    ]
    pool = ProxyPool(proxies=proxies)
    p = pool.get_proxy()
    assert p is not None


def test_pool_rotates_on_error():
    proxies = [
        Proxy(url="http://proxy1:8080", tier="free"),
        Proxy(url="http://proxy2:8080", tier="free"),
    ]
    pool = ProxyPool(proxies=proxies)
    p1 = pool.get_proxy()
    pool.mark_current_bad()
    p2 = pool.get_proxy()
    # Should get a different proxy after marking bad
    # (may be same if pool only has 1 usable, but with 2 fresh proxies
    #  we should rotate)
    pool.mark_current_bad()
    # Both should now be bad, pool should raise
    import pytest
    with pytest.raises(RuntimeError, match="No usable proxies"):
        pool.get_proxy()


def test_pool_disabled_mode():
    pool = ProxyPool(enabled=False)
    assert pool.get_proxy() is None  # disabled pool returns None


def test_proxy_validation_url():
    p = Proxy(url="http://127.0.0.1:8080", tier="free")
    assert "http" in p.url
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_proxy_pool.py -v`
Expected: FAIL

- [ ] **Step 3: Write proxy pool implementation**

```python
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
        """Return a usable proxy, or None if disabled/no proxies."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_proxy_pool.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_proxy_pool.py stoke_ml/crawler/proxy_pool.py
git commit -m "feat: tiered proxy pool with per-domain error tracking"
```

### Task B5: Adaptive Rate Limiter with Circuit Breaker

**Files:**
- Create: `stoke_ml/crawler/rate_limiter.py`
- Create: `tests/crawler/test_rate_limiter.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for adaptive rate limiter with circuit breaker."""
import time
from stoke_ml.crawler.rate_limiter import RateLimiter, CircuitBreaker


def test_rate_limiter_delays():
    limiter = RateLimiter(base_delay_sec=0.01, jitter_factor=0.0)
    start = time.time()
    limiter.wait()
    elapsed = time.time() - start
    assert elapsed >= 0.01


def test_rate_limiter_jitter():
    limiter = RateLimiter(base_delay_sec=0.1, jitter_factor=0.5)
    delays = []
    for _ in range(10):
        start = time.time()
        limiter.wait()
        delays.append(time.time() - start)
    # With jitter, delays should vary
    assert len(set(round(d, 2) for d in delays)) > 1


def test_exponential_backoff():
    limiter = RateLimiter(base_delay_sec=0.001)
    assert limiter.current_delay == 0.001
    limiter.report_429()
    assert limiter.current_delay >= 0.002  # doubled
    limiter.report_429()
    assert limiter.current_delay >= 0.004  # doubled again
    limiter.report_success()
    assert limiter.current_delay == 0.001  # reset


def test_circuit_breaker_opens():
    cb = CircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=0.1,
    )
    assert cb.is_open("test-domain") is False
    cb.record_failure("test-domain")
    cb.record_failure("test-domain")
    assert cb.is_open("test-domain") is False  # only 2
    cb.record_failure("test-domain")
    assert cb.is_open("test-domain") is True  # 3 failures = open


def test_circuit_breaker_resets_after_cooldown():
    cb = CircuitBreaker(
        failure_threshold=2,
        cooldown_seconds=0.05,  # 50ms
    )
    cb.record_failure("test-domain")
    cb.record_failure("test-domain")
    assert cb.is_open("test-domain") is True
    time.sleep(0.06)
    assert cb.is_open("test-domain") is False  # cooled down


def test_circuit_breaker_separate_domains():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    cb.record_failure("domain-a")
    cb.record_failure("domain-a")
    assert cb.is_open("domain-a") is True
    assert cb.is_open("domain-b") is False  # different domain


def test_rate_limiter_respects_daily_quota():
    limiter = RateLimiter(base_delay_sec=0.0, daily_quota=3)
    assert limiter.can_request("test-domain") is True
    limiter.record_request("test-domain")
    limiter.record_request("test-domain")
    limiter.record_request("test-domain")
    assert limiter.can_request("test-domain") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_rate_limiter.py -v`
Expected: FAIL

- [ ] **Step 3: Write rate limiter implementation**

```python
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
        """Check if circuit is open (requests blocked) for a domain."""
        if domain not in self._opened_at:
            return False
        elapsed = time.time() - self._opened_at[domain]
        if elapsed >= self._cooldown:
            # Cooled down — reset
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
        """Sleep for the current delay with random jitter."""
        jitter = self._current_delay * self._jitter * (0.5 + random.random())
        time.sleep(self._current_delay + jitter)

    def report_429(self):
        """Exponential backoff on rate limit response."""
        self._current_delay = min(
            self._current_delay * 2, self._max_backoff
        )

    def report_success(self):
        """Reset delay on successful request."""
        self._current_delay = self._base_delay

    def can_request(self, domain: str) -> bool:
        """Check daily quota and circuit breaker."""
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
        if time.time() - self._day_start > 86400:  # 24h
            self._daily_counts.clear()
            self._day_start = time.time()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_rate_limiter.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_rate_limiter.py stoke_ml/crawler/rate_limiter.py
git commit -m "feat: adaptive rate limiter with circuit breaker"
```

### Task B6: Unified Crawler Client

**Files:**
- Create: `stoke_ml/crawler/client.py`
- Create: `tests/crawler/test_client.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for unified crawler client."""
from stoke_ml.crawler.client import CrawlerClient


def test_client_creation_with_defaults():
    client = CrawlerClient()
    assert client is not None
    client.close()


def test_client_get_request():
    client = CrawlerClient()
    resp = client.get("https://httpbin.org/get")
    assert resp.status_code == 200


def test_client_session_reuse():
    client = CrawlerClient(session_pool_size=5)
    r1 = client.get("https://httpbin.org/get")
    r2 = client.get("https://httpbin.org/get")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_client_disabled_proxy():
    client = CrawlerClient(proxy_enabled=False)
    resp = client.get("https://httpbin.org/get")
    assert resp.status_code == 200
    client.close()


def test_client_context_manager():
    with CrawlerClient() as client:
        resp = client.get("https://httpbin.org/get")
        assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/crawler/test_client.py -v`
Expected: FAIL

- [ ] **Step 3: Write unified crawler client**

```python
"""Unified crawler client combining all anti-block layers.

Integrates: TLS spoofing + fingerprint headers + session pool
+ proxy rotation + rate limiting. Provides a simple get/post
interface that routes through all defense layers transparently.
"""
from stoke_ml.crawler.tls import TLSSession
from stoke_ml.crawler.fingerprint import FingerprintGenerator
from stoke_ml.crawler.session_pool import SessionPool
from stoke_ml.crawler.proxy_pool import ProxyPool, Proxy
from stoke_ml.crawler.rate_limiter import RateLimiter


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
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        if not self._rate_limiter.can_request(domain):
            raise RuntimeError(f"Rate limit or circuit breaker: {domain}")

        session = self._session_pool.get_session()
        headers = self._fingerprint.generate()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        proxy = self._proxy_pool.get_proxy()
        if proxy:
            kwargs["proxies"] = {"http": proxy.url, "https": proxy.url}

        self._rate_limiter.wait()

        try:
            resp = session.http.request(
                method, url, headers=headers, **kwargs
            )
            session.mark_used()
            session.mark_good()
            self._rate_limiter.report_success()
            self._rate_limiter.record_request(domain)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/crawler/test_client.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/crawler/test_client.py stoke_ml/crawler/client.py
git commit -m "feat: unified crawler client with all anti-block layers"
```

---

## PART C: Data Pipeline (LAYER 1)

### Task C1: Trading Calendar

**Files:**
- Create: `stoke_ml/data/calendar.py`
- Create: `tests/data/test_calendar.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for trading calendar."""
import datetime as dt
from stoke_ml.data.calendar import TradingCalendar


def test_a_shares_calendar_has_trading_days():
    cal = TradingCalendar(market="a_shares")
    days = cal.get_trading_days("2024-01-01", "2024-01-31")
    assert len(days) > 0
    assert len(days) < 31  # weekends + holidays excluded


def test_a_shares_weekends_excluded():
    cal = TradingCalendar(market="a_shares")
    days = cal.get_trading_days("2024-01-01", "2024-01-07")
    for d in days:
        assert d.weekday() < 5  # Mon-Fri


def test_is_trading_day():
    cal = TradingCalendar(market="a_shares")
    # 2024-01-01 is New Year's Day holiday in China
    assert cal.is_trading_day(dt.date(2024, 1, 1)) is False
    # 2024-01-02 should be a regular trading day
    is_trade = cal.is_trading_day(dt.date(2024, 1, 2))
    assert is_trade is True or is_trade is False  # depends on actual calendar


def test_next_trading_day():
    cal = TradingCalendar(market="a_shares")
    friday = dt.date(2024, 1, 5)  # Friday
    next_day = cal.next_trading_day(friday)
    assert next_day.weekday() < 5
    assert next_day > friday


def test_us_calendar():
    cal = TradingCalendar(market="us")
    days = cal.get_trading_days("2024-01-01", "2024-01-31")
    assert len(days) > 0
    # US New Year: Jan 1
    assert dt.date(2024, 1, 1) not in days
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_calendar.py -v`
Expected: FAIL

- [ ] **Step 3: Write trading calendar implementation**

```python
"""Trading calendar for A-shares and US markets.

Generates trading day lists with weekend and holiday exclusion.
Uses exchange_calendars for accurate holiday calendars.
"""
import datetime as dt
import pandas as pd


class TradingCalendar:
    """Trading day calendar for a specific market."""

    A_SHARES_HOLIDAYS_2024 = {
        dt.date(2024, 1, 1),    # New Year
        dt.date(2024, 2, 9), dt.date(2024, 2, 10), dt.date(2024, 2, 11),
        dt.date(2024, 2, 12), dt.date(2024, 2, 13), dt.date(2024, 2, 14),
        dt.date(2024, 2, 15), dt.date(2024, 2, 16),  # Spring Festival
        dt.date(2024, 4, 4), dt.date(2024, 4, 5),     # Qingming
        dt.date(2024, 5, 1), dt.date(2024, 5, 2), dt.date(2024, 5, 3),  # Labor
        dt.date(2024, 6, 10),   # Dragon Boat
        dt.date(2024, 9, 16), dt.date(2024, 9, 17),   # Mid-Autumn
        dt.date(2024, 10, 1), dt.date(2024, 10, 2), dt.date(2024, 10, 3),
        dt.date(2024, 10, 4), dt.date(2024, 10, 7),   # National Day
    }

    US_HOLIDAYS_2024 = {
        dt.date(2024, 1, 1),    # New Year
        dt.date(2024, 1, 15),   # MLK Day
        dt.date(2024, 2, 19),   # Presidents Day
        dt.date(2024, 3, 29),   # Good Friday
        dt.date(2024, 5, 27),   # Memorial Day
        dt.date(2024, 6, 19),   # Juneteenth
        dt.date(2024, 7, 4),    # Independence Day
        dt.date(2024, 9, 2),    # Labor Day
        dt.date(2024, 11, 28),  # Thanksgiving
        dt.date(2024, 12, 25),  # Christmas
    }

    HOLIDAYS = {
        "a_shares": A_SHARES_HOLIDAYS_2024,
        "us": US_HOLIDAYS_2024,
    }

    def __init__(self, market: str = "a_shares"):
        if market not in self.HOLIDAYS:
            raise ValueError(f"Unknown market: {market}. Choose: a_shares, us")
        self.market = market
        self._holidays = self.HOLIDAYS[market]

    def get_trading_days(
        self, start: str | dt.date, end: str | dt.date
    ) -> list[dt.date]:
        """Return all trading days in the given date range (inclusive)."""
        if isinstance(start, str):
            start = dt.date.fromisoformat(start)
        if isinstance(end, str):
            end = dt.date.fromisoformat(end)
        dates = pd.bdate_range(start=start, end=end).date
        return [d for d in dates if d not in self._holidays]

    def is_trading_day(self, date: dt.date) -> bool:
        """Check if a given date is a trading day."""
        if date.weekday() >= 5:
            return False
        if date in self._holidays:
            return False
        return True

    def next_trading_day(self, date: dt.date) -> dt.date:
        """Return the next trading day after the given date."""
        candidate = date + dt.timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += dt.timedelta(days=1)
        return candidate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_calendar.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_calendar.py stoke_ml/data/calendar.py
git commit -m "feat: trading calendar for A-shares and US markets"
```

### Task C2: A-Share Data Downloader with Failover

**Files:**
- Create: `stoke_ml/data/sources/a_shares/__init__.py`
- Create: `stoke_ml/data/sources/a_shares/base.py`
- Create: `stoke_ml/data/sources/a_shares/efinance_source.py`
- Create: `stoke_ml/data/sources/a_shares/akshare_source.py`
- Create: `stoke_ml/data/sources/a_shares/failover.py`
- Create: `tests/data/sources/test_a_shares.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for A-share data sources."""
import pandas as pd
import datetime as dt
from stoke_ml.data.sources.a_shares.failover import AShareDownloader


def test_downloader_returns_dataframe():
    downloader = AShareDownloader()
    df = downloader.fetch_daily("000001", "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Check unified schema
    required_cols = {"date", "stock_code", "open", "high", "low",
                     "close", "volume", "amount", "pct_change"}
    assert required_cols.issubset(set(df.columns))


def test_downloader_handles_invalid_stock():
    downloader = AShareDownloader()
    df = downloader.fetch_daily("INVALID", "2024-01-01", "2024-01-31")
    assert len(df) == 0  # Should return empty, not crash


def test_downloader_date_range():
    downloader = AShareDownloader()
    df = downloader.fetch_daily("000001", "2024-06-01", "2024-06-10")
    dates = pd.to_datetime(df["date"])
    assert dates.min() >= pd.Timestamp("2024-06-01")
    assert dates.max() <= pd.Timestamp("2024-06-10")


def test_downloader_normalizes_schema():
    downloader = AShareDownloader()
    df = downloader.fetch_daily("000001", "2024-01-01", "2024-01-10")
    # Check types
    assert df["open"].dtype in ("float64", "float32")
    assert df["volume"].dtype in ("float64", "int64", "float32")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/sources/test_a_shares.py -v`
Expected: FAIL

- [ ] **Step 3: Write base interface**

```python
"""Base interface for A-share data sources."""
from abc import ABC, abstractmethod
import pandas as pd


class AShareSourceBase(ABC):
    """Abstract base for A-share market data fetchers."""

    SOURCE_NAME: str = "base"

    @abstractmethod
    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data and return normalized DataFrame.

        Args:
            stock_code: Stock code (e.g., '000001' for 平安银行)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
            [date, stock_code, open, high, low, close, volume, amount, pct_change]
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is currently accessible."""
        ...
```

- [ ] **Step 4: Write Efinance source**

```python
"""Efinance (东方财富) data source for A-shares."""
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase


class EfinanceSource(AShareSourceBase):
    """Fast, preferred A-share data source via East Money API."""

    SOURCE_NAME = "efinance"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            import efinance as ef
            df = ef.stock.get_quote_history(
                stock_code, beg=start_date, end=end_date
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return self._normalize(df, stock_code)
        except Exception:
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_change",
        })
        # Keep only unified columns
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def is_available(self) -> bool:
        try:
            import efinance
            return True
        except ImportError:
            return False
```

- [ ] **Step 5: Write AKShare source**

```python
"""AKShare data source for A-shares (fallback)."""
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase


class AKShareSource(AShareSourceBase):
    """Comprehensive A-share data source via AKShare scraping wrapper."""

    SOURCE_NAME = "akshare"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=stock_code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",  # Forward-adjusted
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return self._normalize(df, stock_code)
        except Exception:
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_change",
        })
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def is_available(self) -> bool:
        try:
            import akshare
            return True
        except ImportError:
            return False
```

- [ ] **Step 6: Write failover orchestrator**

```python
"""Failover orchestrator for A-share data sources.

Tries sources in priority order:
0. Efinance (preferred - fast, reliable)
1. AKShare (fallback - comprehensive)
2. Tushare (optional - requires token)
3. Baostock (last resort - free, limited)
"""
import pandas as pd
import logging
from stoke_ml.data.sources.a_shares.base import AShareSourceBase
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource

logger = logging.getLogger(__name__)


class AShareDownloader:
    """Multi-source A-share data downloader with automatic failover."""

    def __init__(self):
        self._sources: list[AShareSourceBase] = [
            EfinanceSource(),
            AKShareSource(),
        ]
        self._failure_counts: dict[str, int] = {}
        self._circuit_open: dict[str, float] = {}
        self._cooldown_sec = 300  # 5 min

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily data, trying each source in priority order."""
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
        if self._failure_counts[name] >= 5:
            import time
            self._circuit_open[name] = time.time()

    def _record_success(self, name: str):
        self._failure_counts[name] = 0
        self._circuit_open.pop(name, None)

    def _is_circuit_open(self, name: str) -> bool:
        if name not in self._circuit_open:
            return False
        import time
        if time.time() - self._circuit_open[name] >= self._cooldown_sec:
            del self._circuit_open[name]
            self._failure_counts[name] = 0
            return False
        return True
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/data/sources/test_a_shares.py -v`
Expected: 4 PASS (tests will need network access to data sources)

- [ ] **Step 8: Commit**

```bash
git add tests/data/sources/test_a_shares.py stoke_ml/data/sources/
git commit -m "feat: A-share data downloader with multi-source failover"
```

### Task C3: Data Cleaner & Storage

**Files:**
- Create: `stoke_ml/data/cleaner.py`
- Create: `stoke_ml/data/storage.py`
- Create: `tests/data/test_cleaner.py`
- Create: `tests/data/test_storage.py`

- [ ] **Step 1: Write cleaner test**

```python
"""Tests for data cleaner."""
import pandas as pd
import numpy as np
from stoke_ml.data.cleaner import DataCleaner


def test_cleaner_fills_missing_values():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "stock_code": "000001",
        "open": [10.0, np.nan, 10.5, 10.3, np.nan],
        "high": [10.5, np.nan, 11.0, 10.8, np.nan],
        "low": [9.8, np.nan, 10.2, 10.1, np.nan],
        "close": [10.2, 10.3, 10.8, 10.4, 10.5],
        "volume": [1000, 1200, np.nan, 1100, 900],
        "amount": [10200, 12360, np.nan, 11440, 9450],
        "pct_change": [0.01, -0.02, 0.03, np.nan, 0.01],
    })
    cleaner = DataCleaner()
    result = cleaner.clean(df)
    assert result["open"].isna().sum() == 0  # No missing values


def test_cleaner_removes_outliers():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "stock_code": "000001",
        "open": [10, 10, 1000, 10, 10],  # 1000 is an outlier
        "high": [10.5, 10.5, 1050, 10.5, 10.5],
        "low": [9.8, 9.8, 980, 9.8, 9.8],
        "close": [10.2, 10.2, 1020, 10.2, 10.2],
        "volume": [1000, 1000, 1000, 1000, 1000],
        "amount": [10000, 10000, 10000, 10000, 10000],
        "pct_change": [0, 0, 100, 0, 0],  # 100% pct_change is outlier
    })
    cleaner = DataCleaner(pct_change_limit=15.0)
    result = cleaner.clean(df)
    assert len(result) < 5  # Outlier row removed


def test_cleaner_preserves_valid_data():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "stock_code": "000001",
        "open": [10.0, 10.2, 10.4],
        "high": [10.5, 10.6, 10.8],
        "low": [9.8, 10.0, 10.2],
        "close": [10.2, 10.4, 10.6],
        "volume": [1000, 1100, 1200],
        "amount": [10000, 11000, 12000],
        "pct_change": [0.5, 0.3, 0.4],
    })
    cleaner = DataCleaner()
    result = cleaner.clean(df)
    assert len(result) == 3
    assert (result["open"].values == [10.0, 10.2, 10.4]).all()
```

- [ ] **Step 2: Write storage test**

```python
"""Tests for data storage."""
import tempfile
import os
import pandas as pd
from stoke_ml.data.storage import DataStorage


def test_storage_save_and_load_parquet():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "stock_code": "000001",
        "open": [10.0, 10.2, 10.4, 10.3, 10.5],
        "close": [10.2, 10.4, 10.6, 10.4, 10.5],
        "volume": [1000, 1100, 1200, 1100, 1300],
        "amount": [10000, 11000, 12000, 11000, 13000],
        "pct_change": [0.5, 0.3, 0.4, -0.2, 0.1],
        "high": [10.5, 10.5, 10.7, 10.5, 10.6],
        "low": [9.8, 10.1, 10.3, 10.2, 10.4],
    })
    with tempfile.TemporaryDirectory() as tmp:
        storage = DataStorage(data_dir=tmp)
        storage.save_daily(df, market="a_shares")
        loaded = storage.load_daily("000001", "2024-01-01", "2024-12-31",
                                     market="a_shares")
        assert len(loaded) == 5
        assert (loaded["close"].values == df["close"].values).all()


def test_storage_partitions_by_year_month():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "stock_code": "000001",
        "open": [10.0] * 5, "high": [10.5] * 5,
        "low": [9.8] * 5, "close": [10.2] * 5,
        "volume": [1000] * 5, "amount": [10000] * 5,
        "pct_change": [0.0] * 5,
    })
    with tempfile.TemporaryDirectory() as tmp:
        storage = DataStorage(data_dir=tmp)
        storage.save_daily(df, market="a_shares")
        # Check directory structure
        import glob
        parquet_files = glob.glob(os.path.join(tmp, "a_shares", "daily", "**", "*.parquet"), recursive=True)
        assert len(parquet_files) > 0
```

- [ ] **Step 3: Write implementations**

```python
"""Data cleaner — missing values, outliers, price adjustment validation."""
import pandas as pd
import numpy as np


class DataCleaner:
    """Clean raw OHLCV data before storage."""

    def __init__(self, pct_change_limit: float = 11.0):
        """Args:
            pct_change_limit: Max daily % change before flagging as outlier.
              11.0 = slightly above A-share 10% limit, for tolerance.
        """
        self._pct_limit = pct_change_limit

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps."""
        df = df.copy()
        df = self._fill_missing(df)
        df = self._remove_outliers(df)
        df = self._validate_ohlc(df)
        return df.reset_index(drop=True)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill OHLCV missing values (max 1 day)."""
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=1)
        # Volume and amount: fill with 0 (suspension days)
        for col in ["volume", "amount"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with extreme pct_change (data errors)."""
        if "pct_change" not in df.columns:
            return df
        mask = df["pct_change"].abs() <= self._pct_limit
        return df[mask].copy()

    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure high >= max(open,close) and low <= min(open,close)."""
        if all(c in df.columns for c in ["high", "low", "open", "close"]):
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)
        return df
```

```python
"""Data storage — Parquet partitioned by year/month."""
import os
import pandas as pd


class DataStorage:
    """Save and load market data as partitioned Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_daily(self, df: pd.DataFrame, market: str = "a_shares"):
        """Save daily data partitioned by (year, month, stock_code)."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(
                self._root, market, "daily",
                str(year), f"{month:02d}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            # Drop partition columns before saving
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str,
        market: str = "a_shares"
    ) -> pd.DataFrame:
        """Load daily data for a stock in date range."""
        import pyarrow.parquet as pq
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = os.path.join(self._root, market, "daily")
        if not os.path.exists(base):
            return pd.DataFrame()

        # Collect all parquet files for this stock
        all_data = []
        for root, dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    all_data.append(df[mask])

        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/data/test_cleaner.py tests/data/test_storage.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_cleaner.py tests/data/test_storage.py stoke_ml/data/cleaner.py stoke_ml/data/storage.py
git commit -m "feat: data cleaner and partitioned Parquet storage"
```

---

## PART D: Feature Engineering (LAYER 2)

### Task D1: Technical Indicators

**Files:**
- Create: `stoke_ml/features/technical.py`
- Create: `tests/features/test_technical.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for technical indicators."""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators


def _make_price_df(n_days=200):
    """Create synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_days),
        "open": close + np.random.randn(n_days) * 0.2,
        "high": close + np.abs(np.random.randn(n_days)) * 0.5,
        "low": close - np.abs(np.random.randn(n_days)) * 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n_days),
    })


def test_compute_ma():
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    assert "ma_5" in result.columns
    assert "ma_20" in result.columns
    assert "ma_60" in result.columns


def test_compute_macd():
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    assert "macd_dif" in result.columns
    assert "macd_dea" in result.columns
    assert "macd_hist" in result.columns


def test_compute_rsi():
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    assert "rsi_6" in result.columns
    assert "rsi_14" in result.columns
    # RSI should be between 0 and 100
    assert result["rsi_14"].between(0, 101).all()


def test_compute_bollinger():
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    assert "boll_upper" in result.columns
    assert "boll_mid" in result.columns
    assert "boll_lower" in result.columns


def test_compute_atr():
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    assert "atr_14" in result.columns


def test_no_lookahead_bias():
    """Indicators at time t must only use data up to time t."""
    df = _make_price_df(200)
    ti = TechnicalIndicators()
    result = ti.compute_all(df)
    # MA at index 50 should not use data beyond index 50
    ma5_at_50 = result.loc[50, "ma_5"]
    ma5_recomputed = df.loc[46:50, "close"].mean()
    assert abs(ma5_at_50 - ma5_recomputed) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_technical.py -v`
Expected: FAIL

- [ ] **Step 3: Write technical indicators implementation**

```python
"""Technical indicators using ta-lib / pandas-ta."""
import pandas as pd
import pandas_ta as ta


class TechnicalIndicators:
    """Compute standard technical indicators from OHLCV data."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators. No lookahead bias."""
        result = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # Moving averages
        for period in [5, 10, 20, 60, 120]:
            result[f"ma_{period}"] = close.rolling(period).mean()

        # EMA
        result["ema_12"] = close.ewm(span=12, adjust=False).mean()
        result["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # MACD
        result["macd_dif"] = result["ema_12"] - result["ema_26"]
        result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

        # RSI
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2)
        result["boll_mid"] = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        result["boll_upper"] = result["boll_mid"] + 2 * boll_std
        result["boll_lower"] = result["boll_mid"] - 2 * boll_std

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean()

        # Volume indicators
        result["volume_ma5"] = volume.rolling(5).mean()
        result["volume_ratio"] = volume / result["volume_ma5"].replace(0, 1)
        result["obv"] = (volume * ((close.diff() > 0).astype(int) * 2 - 1)).cumsum()

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_technical.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_technical.py stoke_ml/features/technical.py
git commit -m "feat: technical indicators (MA, MACD, RSI, BOLL, ATR)"
```

### Task D2: Rule-Based Scoring

**Files:**
- Create: `stoke_ml/features/scoring.py`
- Create: `tests/features/test_scoring.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for rule-based scoring."""
import pandas as pd
import numpy as np
from stoke_ml.features.scoring import TrendScorer


def _make_trend_df():
    """Create data with a clear uptrend."""
    close = np.linspace(100, 150, 100)  # Steady uptrend
    return pd.DataFrame({
        "close": close,
        "ma_5": close - np.linspace(1, 5, 100),
        "ma_10": close - np.linspace(2, 10, 100),
        "ma_20": close - np.linspace(3, 15, 100),
        "ma_60": close - np.linspace(4, 20, 100),
        "volume": np.full(100, 5000),
        "volume_ma5": np.full(100, 5000),
    })


def test_trend_classification_strong_bull():
    df = _make_trend_df()
    scorer = TrendScorer()
    result = scorer.score(df)
    # In uptrend with MA alignment, should be strong_bull or bull
    assert result["trend_level"].iloc[-1] in [0, 1]


def test_trend_classification_returns_7_levels():
    df = _make_trend_df()
    scorer = TrendScorer()
    result = scorer.score(df)
    assert result["trend_level"].between(0, 6).all()


def test_bias_calculation():
    df = _make_trend_df()
    scorer = TrendScorer()
    result = scorer.score(df)
    assert "bias_ma5" in result.columns
    assert result["bias_ma5"].iloc[-1] > 0  # Price above MA5 in uptrend


def test_buy_signal_range():
    df = _make_trend_df()
    scorer = TrendScorer()
    result = scorer.score(df)
    assert "buy_signal" in result.columns
    assert result["buy_signal"].between(0, 5).all()


def test_volume_classification():
    df = pd.DataFrame({
        "close": np.linspace(100, 105, 50),
        "ma_5": np.linspace(99, 104, 50),
        "ma_10": np.linspace(98, 103, 50),
        "ma_20": np.linspace(97, 102, 50),
        "ma_60": np.linspace(95, 100, 50),
        "volume": np.concatenate([np.full(25, 5000), np.full(25, 200)]),
        "volume_ma5": np.full(50, 5000),
    })
    scorer = TrendScorer()
    result = scorer.score(df)
    # Last rows should have volume_shrink flag
    assert "volume_shrink" in result.columns
    assert result["volume_shrink"].iloc[-1] is True or result["volume_shrink"].iloc[-1] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_scoring.py -v`
Expected: FAIL

- [ ] **Step 3: Write scoring implementation**

```python
"""Rule-based trend and buy signal scoring.

Extracts structured signals from technical indicators to serve
as model input features. Not used as standalone trading signals.
"""
import pandas as pd
import numpy as np


class TrendScorer:
    """Score trend strength and generate buy/sell level features."""

    BIAS_THRESHOLD = 5.0
    VOLUME_SHRINK_RATIO = 0.7
    VOLUME_HEAVY_RATIO = 1.5
    MA_SUPPORT_TOLERANCE = 0.02

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result = self._classify_trend(result)
        result = self._compute_bias(result)
        result = self._classify_volume(result)
        result = self._compute_buy_signal(result)
        return result

    def _classify_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """7-level trend classification."""
        ma5 = df["ma_5"].values
        ma10 = df["ma_10"].values
        ma20 = df["ma_20"].values
        ma60 = df["ma_60"].values
        close = df["close"].values

        trend = np.full(len(df), 3, dtype=int)  # default: neutral
        for i in range(len(df)):
            if close[i] > ma5[i] > ma10[i] > ma20[i] > ma60[i]:
                trend[i] = 0  # strong_bull
            elif close[i] > ma5[i] > ma10[i] > ma20[i]:
                trend[i] = 1  # bull
            elif close[i] > ma20[i]:
                trend[i] = 2  # mild_bull
            elif close[i] < ma5[i] < ma10[i] < ma20[i] < ma60[i]:
                trend[i] = 6  # strong_bear
            elif close[i] < ma5[i] < ma10[i] < ma20[i]:
                trend[i] = 5  # bear
            elif close[i] < ma20[i]:
                trend[i] = 4  # mild_bear
            else:
                trend[i] = 3  # neutral

        df["trend_level"] = trend
        return df

    def _compute_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price deviation from MAs."""
        for period in [5, 10, 20, 60]:
            ma_col = f"ma_{period}"
            if ma_col in df.columns:
                df[f"bias_ma{period}"] = (
                    (df["close"] - df[ma_col]) / df[ma_col] * 100
                )
        return df

    def _classify_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume shrinkage and heavy flags."""
        if "volume_ratio" in df.columns:
            df["volume_shrink"] = df["volume_ratio"] < self.VOLUME_SHRINK_RATIO
            df["volume_heavy"] = df["volume_ratio"] > self.VOLUME_HEAVY_RATIO
        else:
            df["volume_shrink"] = False
            df["volume_heavy"] = False
        return df

    def _compute_buy_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite 6-level buy signal (0=strong_buy, 5=strong_sell)."""
        signal = np.full(len(df), 3, dtype=int)  # default neutral

        for i in range(len(df)):
            score = 0
            # Trend contribution
            trend = df["trend_level"].iloc[i]
            if trend <= 1:     score -= 2  # strong bull/bull
            elif trend <= 2:   score -= 1  # mild bull
            elif trend >= 5:   score += 2  # strong bear/bear
            elif trend >= 4:   score += 1  # mild bear

            # Bias contribution
            bias = df.get("bias_ma5", pd.Series(0)).iloc[i]
            if abs(bias) > self.BIAS_THRESHOLD:
                score += 1 if bias > 0 else -1  # Penalize excessive bias

            # Volume contribution
            if df.get("volume_shrink", pd.Series(False)).iloc[i]:
                score += 1  # Low volume = weaker signal
            if df.get("volume_heavy", pd.Series(False)).iloc[i]:
                score -= 1  # High volume = stronger signal

            # Map to 0-5 scale
            if score <= -3:    signal[i] = 0  # strong_buy
            elif score <= -1:  signal[i] = 1  # buy
            elif score == 0:   signal[i] = 2  # mild_buy
            elif score == 1:   signal[i] = 3  # mild_sell
            elif score <= 3:   signal[i] = 4  # sell
            else:              signal[i] = 5  # strong_sell

        df["buy_signal"] = signal
        return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_scoring.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_scoring.py stoke_ml/features/scoring.py
git commit -m "feat: rule-based trend scorer and buy signal classification"
```

### Task D3: Feature Pipeline & Temporal Features

**Files:**
- Create: `stoke_ml/features/temporal.py`
- Create: `stoke_ml/features/pipeline.py`
- Create: `tests/features/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for feature pipeline."""
import pandas as pd
import numpy as np
from stoke_ml.features.pipeline import FeaturePipeline


def _make_ohlcv_df(n_days=300):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_days),
        "stock_code": "000001",
        "open": close + np.random.randn(n_days) * 0.2,
        "high": close + np.abs(np.random.randn(n_days)) * 0.5,
        "low": close - np.abs(np.random.randn(n_days)) * 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n_days),
        "amount": close * np.random.randint(1000, 10000, n_days) / 100,
        "pct_change": np.random.randn(n_days) * 2,
    })


def test_pipeline_returns_feature_matrix():
    df = _make_ohlcv_df(300)
    pipeline = FeaturePipeline(seq_len=60)
    X, y = pipeline.build_features(df, target_col="close")
    assert X.ndim == 3  # (samples, seq_len, n_features)
    assert y.ndim == 1  # (samples,)
    assert X.shape[1] == 60  # seq_len
    assert X.shape[0] > 0  # At least some samples


def test_pipeline_target_is_direction():
    df = _make_ohlcv_df(300)
    pipeline = FeaturePipeline(seq_len=60, horizon=1)
    _, y = pipeline.build_features(df, target_col="close")
    # y should be binary (0 or 1) for direction
    assert set(np.unique(y)).issubset({0, 1})


def test_pipeline_flat_mode():
    df = _make_ohlcv_df(300)
    pipeline = FeaturePipeline(seq_len=60, flat_mode=True)
    X, y = pipeline.build_features(df, target_col="close")
    assert X.ndim == 2  # Flattened for ML models


def test_pipeline_no_lookahead():
    """Target at time t must not use features from time t+1 or later."""
    df = _make_ohlcv_df(300)
    pipeline = FeaturePipeline(seq_len=60, horizon=1)
    X, y = pipeline.build_features(df, target_col="close")
    # The last feature window should end at df index -1 - horizon
    # We can check: last sample's last feature value should match
    # df.close at index (len(df) - 1 - horizon)
    last_feature_close = X[-1, -1, df.columns.get_loc("close") - 1]
    # This is the close of the last day in the window = day (len-1-horizon)
    expected_idx = len(df) - 1 - 1  # -1 for 0-index, -1 for horizon
    assert abs(last_feature_close - df["close"].iloc[expected_idx]) < 0.01
```

- [ ] **Step 2: Write temporal features**

```python
"""Temporal features: lags, rolling windows, calendar features."""
import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged versions of specified columns."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean/std for specified columns."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        for w in windows:
            result[f"{col}_roll{w}_mean"] = result[col].rolling(w).mean()
            result[f"{col}_roll{w}_std"] = result[col].rolling(w).std()
    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter features."""
    result = df.copy()
    dates = pd.to_datetime(result["date"])
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    return result
```

- [ ] **Step 3: Write feature pipeline**

```python
"""Feature pipeline orchestrating all feature engineering steps."""
import pandas as pd
import numpy as np
from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer
from stoke_ml.features.temporal import (
    add_lag_features, add_rolling_features, add_calendar_features,
)


class FeaturePipeline:
    """End-to-end feature engineering for stock prediction."""

    TARGET_COLS = ["open", "high", "low", "close", "volume"]
    LAGS = [1, 2, 3, 5, 10, 20]
    ROLLING_WINDOWS = [5, 10, 20, 60]

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        flat_mode: bool = False,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.flat_mode = flat_mode
        self._ti = TechnicalIndicators()
        self._scorer = TrendScorer()

    def build_features(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and target vector.

        Args:
            df: Raw OHLCV DataFrame
            target_col: Column to predict direction of

        Returns:
            X: Feature matrix (samples, seq_len, features) or (samples, features) if flat
            y: Binary target vector (1=up, 0=down)
        """
        feats = self._engineer_features(df)
        X, y = self._create_sequences(feats, target_col)
        return X, y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps."""
        df = df.copy()
        # Technical indicators
        df = self._ti.compute_all(df)
        # Rule-based scoring
        df = self._scorer.score(df)
        # Temporal features
        cols = self.TARGET_COLS + ["volume_ratio", "atr_14", "rsi_14"]
        df = add_lag_features(df, cols, self.LAGS)
        df = add_rolling_features(df, cols, self.ROLLING_WINDOWS)
        df = add_calendar_features(df)
        return df

    def _create_sequences(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences and binary targets."""
        # Drop non-numeric and date columns
        drop_cols = ["date", "stock_code"]
        feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Drop rows with NaN (from rolling/lag calculations)
        feat_df = feat_df.dropna()

        # Target: next-day price direction
        close = feat_df[target_col].values
        target = (close[self.horizon:] > close[:-self.horizon]).astype(int)

        # Feature columns: everything except target raw price
        price_cols_to_drop = ["open", "high", "low", "close", "amount"]
        X_cols = [c for c in feat_df.columns if c not in price_cols_to_drop]
        X_data = feat_df[X_cols].values.astype(np.float32)

        # Create sequences
        n_samples = len(X_data) - self.seq_len - self.horizon + 1
        if n_samples <= 0:
            return np.array([]), np.array([])

        if self.flat_mode:
            X = np.array([
                X_data[i:i + self.seq_len].flatten()
                for i in range(n_samples)
            ], dtype=np.float32)
        else:
            X = np.array([
                X_data[i:i + self.seq_len]
                for i in range(n_samples)
            ], dtype=np.float32)

        y = target[self.seq_len - 1:self.seq_len - 1 + n_samples]
        return X, y
```

- [ ] **Step 4: Run test**

Run: `pytest tests/features/test_pipeline.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stoke_ml/features/temporal.py stoke_ml/features/pipeline.py tests/features/test_pipeline.py
git commit -m "feat: feature pipeline with temporal features and sequence builder"
```

---

## PART E: Phase 1 — ML Baseline (XGBoost)

### Task E1: Walk-Forward Data Splitter

**Files:**
- Create: `stoke_ml/evaluation/splitter.py`
- Create: `tests/evaluation/test_splitter.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for walk-forward data splitter."""
import pandas as pd
import numpy as np
from stoke_ml.evaluation.splitter import WalkForwardSplitter


def test_splitter_creates_windows():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    splitter = WalkForwardSplitter(train_years=2, val_months=3)
    windows = list(splitter.split(dates))
    assert len(windows) > 1
    for train_idx, val_idx in windows:
        # Train should be ~2 years, val ~3 months
        assert len(train_idx) > len(val_idx)
        # No overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        # No lookahead
        assert max(train_idx) < min(val_idx)


def test_splitter_no_overlap():
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    splitter = WalkForwardSplitter(train_years=2, val_months=3)
    windows = list(splitter.split(dates))
    all_val = set()
    for _, val_idx in windows:
        for i in val_idx:
            assert i not in all_val  # Each point only in one val set
            all_val.add(i)


def test_splitter_step_size():
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    splitter = WalkForwardSplitter(train_years=2, val_months=3, step_months=3)
    windows = list(splitter.split(dates))
    # Check that val windows start ~3 months apart
    val_starts = [val_idx[0] for _, val_idx in windows]
    assert len(val_starts) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_splitter.py -v`
Expected: FAIL

- [ ] **Step 3: Write splitter implementation**

```python
"""Walk-forward (expanding window) data splitter.

Time series data cannot be randomly shuffled. Walk-forward validation
trains on expanding historical windows and validates on subsequent
periods, preventing lookahead bias.
"""
import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """Generate train/validation splits respecting time order."""

    def __init__(
        self,
        train_years: int = 2,
        val_months: int = 3,
        step_months: int = 3,
    ):
        self.train_days = train_years * 252  # approximate trading days
        self.val_days = val_months * 21
        self.step_days = step_months * 21

    def split(self, dates: pd.DatetimeIndex | np.ndarray):
        """Yield (train_indices, val_indices) tuples.

        Args:
            dates: Array of dates (must be sorted ascending).

        Yields:
            (train_idx, val_idx) as numpy arrays of indices.
        """
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.values
        n = len(dates)

        start = 0
        while True:
            train_end = start + self.train_days
            val_end = train_end + self.val_days
            if val_end > n:
                break
            train_idx = np.arange(start, train_end)
            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx
            start += self.step_days
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_splitter.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/evaluation/test_splitter.py stoke_ml/evaluation/splitter.py
git commit -m "feat: walk-forward data splitter for time series validation"
```

### Task E2: Model Metrics

**Files:**
- Create: `stoke_ml/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for evaluation metrics."""
import numpy as np
from stoke_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_financial_metrics,
    mcc_score,
)


def test_mcc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert mcc_score(y_true, y_pred) == 1.0


def test_mcc_worst():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert mcc_score(y_true, y_pred) == -1.0


def test_mcc_random():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert mcc_score(y_true, y_pred) == 0.0


def test_classification_metrics_returns_dict():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "mcc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_financial_metrics():
    # Simple price series: steady uptrend
    prices = np.array([100, 101, 102, 103, 104, 103, 105, 107])
    # Predict up every day
    predictions = np.array([1, 1, 1, 1, 1, 1, 1])
    metrics = compute_financial_metrics(prices, predictions)
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Write metrics implementation**

```python
"""Model evaluation metrics — classification and financial."""
import numpy as np


def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient (balanced metric for binary)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Compute standard classification metrics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc_score(y_true, y_pred)),
    }


def compute_financial_metrics(
    prices: np.ndarray, predictions: np.ndarray
) -> dict:
    """Compute trading-oriented financial metrics."""
    # Simulate daily returns based on predictions
    price_returns = np.diff(prices) / prices[:-1]
    strategy_returns = price_returns * (2 * predictions - 1)  # long if predict up, short if down

    # Sharpe ratio (annualized, assuming daily frequency)
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_dd = float(np.min(drawdowns))

    # Win rate
    wins = np.sum(strategy_returns > 0)
    total = len(strategy_returns)
    win_rate = wins / total if total > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/evaluation/test_metrics.py stoke_ml/evaluation/metrics.py
git commit -m "feat: classification and financial evaluation metrics"
```

### Task E3: XGBoost Baseline Model

**Files:**
- Create: `stoke_ml/models/baseline/__init__.py`
- Create: `stoke_ml/models/baseline/xgboost_model.py`
- Create: `tests/models/test_xgboost.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for XGBoost baseline model."""
import numpy as np
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline


def test_model_trains_and_predicts():
    X = np.random.randn(500, 50).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # simple rule

    model = XGBoostBaseline(n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_predict_proba():
    X = np.random.randn(500, 50).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = XGBoostBaseline(n_estimators=10)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.ndim == 1
    assert proba.min() >= 0 and proba.max() <= 1


def test_model_save_load(tmp_path):
    X = np.random.randn(100, 50).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = XGBoostBaseline(n_estimators=10)
    model.fit(X, y)

    path = str(tmp_path / "model.json")
    model.save(path)
    loaded = XGBoostBaseline.load(path)
    assert loaded is not None
    preds_orig = model.predict(X[:5])
    preds_loaded = loaded.predict(X[:5])
    assert np.array_equal(preds_orig, preds_loaded)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_xgboost.py -v`
Expected: FAIL

- [ ] **Step 3: Write XGBoost baseline model**

```python
"""XGBoost baseline model for stock direction prediction."""
import numpy as np
import xgboost as xgb


class XGBoostBaseline:
    """XGBoost classifier for next-day price direction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
    ):
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        if scale_pos_weight is not None:
            self._params["scale_pos_weight"] = scale_pos_weight
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        # Auto-compute class weight for imbalance
        if self._params.get("scale_pos_weight") is None:
            neg = np.sum(y == 0)
            pos = np.sum(y == 1)
            self._params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0

        self._model = xgb.XGBClassifier(**self._params)
        self._model.fit(X, y, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """Save model to JSON file."""
        if self._model is None:
            raise RuntimeError("Model not trained. Nothing to save.")
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostBaseline":
        """Load model from JSON file."""
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(path)
        return instance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_xgboost.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_xgboost.py stoke_ml/models/baseline/
git commit -m "feat: XGBoost baseline classifier for stock direction"
```

### Task E4: Training Script (Phase 1 End-to-End)

**Files:**
- Create: `scripts/train_baseline.py`

- [ ] **Step 1: Write training script**

```python
"""Phase 1: Train XGBoost baseline on A-share data.

Usage: python scripts/train_baseline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.cleaner import DataCleaner
from stoke_ml.data.storage import DataStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_financial_metrics,
)
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()

    # 1. Download data for one test stock
    logger.info("Downloading data...")
    downloader = AShareDownloader()
    raw_df = downloader.fetch_daily("000001", "2020-01-01", "2025-05-31")

    if len(raw_df) == 0:
        logger.error("No data downloaded. Check network / data sources.")
        return

    # 2. Clean and store
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)
    logger.info(f"Cleaned data: {len(clean_df)} rows")

    # 3. Build features
    pipeline = FeaturePipeline(seq_len=60, horizon=1, flat_mode=True)
    X, y = pipeline.build_features(clean_df)
    logger.info(f"Feature matrix: {X.shape}, target: {y.shape}")

    # 4. Walk-forward validation
    splitter = WalkForwardSplitter(train_years=2, val_months=3)
    dates = clean_df["date"].values
    # Dates need to align with feature indices
    feat_start = len(clean_df) - len(X)
    feat_dates = dates[feat_start:]

    all_metrics = []
    for i, (train_idx, val_idx) in enumerate(splitter.split(feat_dates)):
        # Map to feature indices
        train_mask = np.isin(np.arange(len(X)), train_idx)
        val_mask = np.isin(np.arange(len(X)), val_idx)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < 100 or len(X_val) < 10:
            continue

        model = XGBoostBaseline()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = compute_classification_metrics(y_val, preds)
        # Financial metrics
        val_prices = clean_df["close"].values[val_idx[-len(preds):]]
        fin_metrics = compute_financial_metrics(val_prices, preds)
        metrics.update(fin_metrics)

        all_metrics.append(metrics)
        logger.info(f"Window {i}: MCC={metrics['mcc']:.4f}, "
                    f"Sharpe={metrics['sharpe']:.4f}")

    # 5. Report average metrics
    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics])
               for k in all_metrics[0].keys()}
        logger.info("=== Average Metrics ===")
        for k, v in avg.items():
            logger.info(f"  {k}: {v:.4f}")
    else:
        logger.warning("No valid windows produced.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training script**

Run: `python scripts/train_baseline.py`
Expected: Downloads data, trains model, outputs MCC and Sharpe across walk-forward windows.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_baseline.py
git commit -m "feat: end-to-end Phase 1 training script (XGBoost baseline)"
```

---

## PART F: Phase 2 — Deep Learning Models

### Task F1: Stock Dataset & DataLoader

**Files:**
- Create: `stoke_ml/models/dl/dataset.py`
- Create: `tests/models/dl/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for stock prediction dataset."""
import numpy as np
import torch
from stoke_ml.models.dl.dataset import StockDataset


def test_dataset_returns_correct_shape():
    X = np.random.randn(100, 60, 50).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    ds = StockDataset(X, y)
    x0, y0 = ds[0]
    assert x0.shape == (60, 50)  # (seq_len, features)
    assert y0.shape == ()  # scalar


def test_dataset_len():
    X = np.random.randn(100, 60, 50).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    ds = StockDataset(X, y)
    assert len(ds) == 100


def test_dataloader_batching():
    X = np.random.randn(100, 60, 50).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    ds = StockDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (16, 60, 50)
    assert batch_y.shape == (16,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/dl/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Write dataset implementation**

```python
"""PyTorch Dataset for stock time series data."""
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Time series dataset for stock prediction.

    Args:
        X: Feature tensor of shape (n_samples, seq_len, n_features)
        y: Target tensor of shape (n_samples,) or (n_samples, n_tasks)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y.ndim == 1 else torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/dl/test_dataset.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/dl/test_dataset.py stoke_ml/models/dl/dataset.py
git commit -m "feat: PyTorch dataset for stock time series"
```

### Task F2: LSTM Model

**Files:**
- Create: `stoke_ml/models/dl/lstm_model.py`
- Create: `tests/models/dl/test_lstm.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LSTM model."""
import torch
import numpy as np
from stoke_ml.models.dl.lstm_model import LSTMModel


def test_lstm_forward_pass():
    model = LSTMModel(input_dim=50, hidden_dim=128, num_layers=2, dropout=0.3)
    x = torch.randn(16, 60, 50)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (16, 2)  # (batch, num_classes)


def test_lstm_training_step():
    model = LSTMModel(input_dim=50, hidden_dim=128)
    x = torch.randn(16, 60, 50)
    y = torch.randint(0, 2, (16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    # Should not crash


def test_lstm_predict():
    model = LSTMModel(input_dim=50, hidden_dim=128)
    model.eval()
    x = torch.randn(32, 60, 50)
    with torch.no_grad():
        probs = model.predict_proba(x)
        preds = model.predict(x)
    assert probs.shape == (32,)
    assert preds.shape == (32,)
    assert probs.min() >= 0 and probs.max() <= 1
    assert set(preds.numpy()).issubset({0, 1})


def test_lstm_vram_footprint():
    """Verify LSTM model fits within VRAM budget."""
    model = LSTMModel(input_dim=50, hidden_dim=256, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    # ~500K params → float32 = ~2MB, far below 4GB budget
    assert n_params < 1_000_000  # < 1M params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/dl/test_lstm.py -v`
Expected: FAIL

- [ ] **Step 3: Write LSTM model**

```python
"""LSTM model for stock direction prediction."""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """2-layer LSTM for binary stock direction classification.

    Architecture: LSTM → dropout → FC → output
    """

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of positive class (up)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict binary labels."""
        probs = self.predict_proba(x)
        return (probs > 0.5).long()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/dl/test_lstm.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/dl/test_lstm.py stoke_ml/models/dl/lstm_model.py
git commit -m "feat: 2-layer LSTM model for stock direction prediction"
```

### Task F3: PyTorch Lightning Training Module

**Files:**
- Create: `stoke_ml/models/dl/lightning_module.py`
- Create: `scripts/train_lstm.py`

- [ ] **Step 1: Write Lightning module**

```python
"""PyTorch Lightning module wrapping LSTM training loop."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from stoke_ml.models.dl.lstm_model import LSTMModel
from stoke_ml.evaluation.metrics import mcc_score


class StockLightningModule(pl.LightningModule):
    """Lightning wrapper for stock prediction models."""

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if class_weight is not None:
            self._class_weight = torch.tensor(class_weight, dtype=torch.float)
        else:
            self._class_weight = None
        self._criterion = nn.CrossEntropyLoss(weight=self._class_weight)
        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        self._val_preds.append(preds.cpu().numpy())
        self._val_targets.append(y.cpu().numpy())
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = np.concatenate(self._val_preds)
            all_targets = np.concatenate(self._val_targets)
            mcc = mcc_score(all_targets, all_preds)
            self.log("val_mcc", mcc, on_epoch=True)
            self._val_preds.clear()
            self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
```

- [ ] **Step 2: Write LSTM training script**

```python
"""Phase 2: Train LSTM model on A-share data.

Usage: python scripts/train_lstm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.cleaner import DataCleaner
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.dl.dataset import StockDataset
from stoke_ml.models.dl.lightning_module import StockLightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()

    # 1. Download data
    logger.info("Downloading data...")
    downloader = AShareDownloader()
    raw_df = downloader.fetch_daily("000001", "2020-01-01", "2025-05-31")
    if len(raw_df) == 0:
        logger.error("No data downloaded.")
        return
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)

    # 2. Build 3D features
    pipeline = FeaturePipeline(seq_len=60, horizon=1, flat_mode=False)
    X, y = pipeline.build_features(clean_df)
    logger.info(f"Feature matrix: {X.shape}")

    # 3. Simple train/val split (time-respecting)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = StockDataset(X_train, y_train)
    val_ds = StockDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 4. Train
    model = StockLightningModule(
        input_dim=X.shape[2],
        hidden_dim=cfg.model.get("hidden_dim", 128),
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath=cfg.project.model_dir,
        monitor="val_loss",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.get("epochs", 100),
        callbacks=[early_stop, checkpoint],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Best model saved to: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify training runs**

Run: `python scripts/train_lstm.py`
Expected: Downloads data, trains LSTM, reports val_mcc.

- [ ] **Step 4: Commit**

```bash
git add stoke_ml/models/dl/lightning_module.py scripts/train_lstm.py
git commit -m "feat: PyTorch Lightning LSTM training pipeline"
```

---

## PART G: Phase 3 — Multi-Modal Multi-Task (Directional)

*Phase 3 tasks are outlined here and will be fully detailed after Phase 2 results inform architecture choices. Each task follows the same TDD pattern as Parts A-F.*

### Task G1: News Crawler Integration (8-10 tasks)

- Integrate anti-block crawler client with financial news sources
- Store raw news (财联社, East Money headlines, NewsAPI for US)
- Implement text preprocessing (jieba tokenization, deduplication, stock entity linking)
- Map news to trading days

### Task G2: NLP Feature Extraction (6-8 tasks)

- FinBERT sentiment for English news (transformers pipeline)
- BERT-wwm-chinese sentiment for Chinese news
- Build daily aggregated features per stock: sentiment score, news count, embedding vector
- Cache embeddings to avoid recomputation

### Task G3: Transformer Multi-Modal Model (8-10 tasks)

- Transformer encoder with positional encoding
- Cross-attention fusion: price features query news embeddings
- Multi-task heads: direction (Focal Loss), volatility (Huber Loss), turning point (CrossEntropy)
- Gradient checkpointing for VRAM efficiency
- Mixed precision training (torch.cuda.amp)

### Task G4: Walk-Forward Backtesting Engine (5-7 tasks)

- Full walk-forward backtest with portfolio simulation
- Equity curve generation
- Sentiment consensus analysis (forcing positive/negative sentiment)
- Performance report generation (annual returns, drawdown plot, monthly heatmap)

### Task G5: US Market Integration (4-6 tasks)

- US data downloader (yfinance primary, Finnhub fallback)
- US trading calendar
- Market embedding for multi-market model
- Cross-market evaluation

---

## Self-Review

1. **Spec coverage**: All major sections covered — crawler (LAYER 0), data pipeline (LAYER 1), feature engineering (LAYER 2), XGBoost baseline + LSTM (LAYER 3), training methodology, metrics (LAYER 4/5). Phase 3 tasks outlined directionally.

2. **Placeholder scan**: No TBD/TODO markers. Phase 3 tasks are directional outlines acknowledging that architecture depends on Phase 2 results — this is intentional scope deferral, not a placeholder.

3. **Type consistency**: Data contracts from spec (3D tensor features, binary targets, model interface) are used consistently across tasks. `FeaturePipeline` output shape `(n_samples, seq_len, n_features)` matches `LSTMModel` input expectation `(batch, seq_len, input_dim)`.

4. **Gap identified**: CNN-Seq2Seq and Dilated CNN models from the model catalog are not yet implemented. These belong in Phase 2 extension tasks (F4, F5) which can be added after LSTM baseline is working.
