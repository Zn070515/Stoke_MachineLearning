"""§P2-13: lazy optional imports in the data chain.

``requests`` and ``akshare`` are heavy optional dependencies.  The online
sources must import gracefully WITHOUT them at module-import time — they may
only be required at actual network-call time (function-local imports).

This test runs in a FRESH subprocess that blocks ``requests`` and ``akshare``
BEFORE importing the data chain.  It must be a subprocess test — nulling a
module in-process would poison the shared sys.modules cache for every other
test in the session.

The four lazified sites (see plan §P2-13):

- ``stoke_ml/data/sources/a_shares/sector_source.py``
- ``stoke_ml/data/sources/a_shares/minute_source_tencent.py``
- ``stoke_ml/data/sources/a_shares/minute_source_sina_direct.py``
- ``scripts/production/download_data.py``

A top-level ``import requests`` / ``import akshare`` (regression) would raise
ModuleNotFoundError here; the function-local imports must not.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK_REQUESTS_AKSHARE_IMPORT = (
    "import sys;"
    "sys.modules['requests'] = None;"
    "sys.modules['akshare'] = None;"
    "import stoke_ml.data.sources.a_shares.sector_source;"
    "import stoke_ml.data.sources.a_shares.minute_source_tencent;"
    "import stoke_ml.data.sources.a_shares.minute_source_sina_direct;"
    "import scripts.production.download_data;"
    "print('OK')"
)


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"  # Windows: keep stdout UTF-8
    return env


def test_import_chain_succeeds_without_requests_akshare():
    """Importing the data chain must succeed even when requests/akshare are blocked.

    A fresh subprocess nulls requests and akshare in sys.modules BEFORE
    importing anything, then imports the chain.  A top-level ``import requests``
    or ``import akshare`` (regression) would raise ModuleNotFoundError here; the
    function-local imports must not.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_REQUESTS_AKSHARE_IMPORT],
        capture_output=True, text=True,
        # The child sets PYTHONIOENCODING=utf-8, so its stderr (which may carry
        # non-ASCII bytes from a transitive import warning) must be decoded as
        # UTF-8.  Decoding with the Windows locale (GBK) crashes the reader
        # thread on those bytes.  errors="replace" is a belt-and-suspenders.
        encoding="utf-8", errors="replace",
        env=_subprocess_env(),
    )
    assert proc.returncode == 0, (
        f"import chain failed without requests/akshare:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout
