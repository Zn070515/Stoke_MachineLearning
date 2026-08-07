"""§十五: lazy pyarrow import in train_panel_panel.

PyArrow is a heavy optional dependency.  ``scripts.production.train_panel_panel``
must import gracefully WITHOUT pyarrow (no top-level ``import pyarrow.parquet``),
and ``_estimate_panel_memory`` must degrade to None (skip the early guard) when
pyarrow is unavailable.

Two tests here:

- ``test_import_chain_succeeds_without_pyarrow`` runs in a FRESH subprocess that
  blocks pyarrow BEFORE importing the module.  It must be a subprocess test —
  blocking pyarrow in-process would poison the shared module cache for every
  other test in the session.
- ``test_estimate_panel_memory_none_when_no_schema`` exercises the None-on-
  failure path directly in-process (pyarrow present): no prebuilt / no parquets
  → None, never raising.

The happy-path schema-read estimate is covered by
``tests/scripts/test_train_panel_universe.py`` (test_estimate_panel_memory_*).
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import scripts.production.train_panel_panel as tpp

REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK_PYARROW_IMPORT = (
    "import sys;"
    "sys.modules['pyarrow'] = None;"
    "sys.modules['pyarrow.parquet'] = None;"
    "import scripts.production.train_panel_panel;"
    "print('OK')"
)


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"  # Windows: keep stdout UTF-8
    return env


def test_import_chain_succeeds_without_pyarrow():
    """Importing train_panel_panel must succeed even when pyarrow is blocked.

    A fresh subprocess nulls pyarrow in sys.modules BEFORE importing anything,
    then imports the module chain.  A top-level ``import pyarrow.parquet``
    (regression) would raise ModuleNotFoundError here; the lazy import must not.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_PYARROW_IMPORT],
        capture_output=True, text=True,
        env=_subprocess_env(),
    )
    assert proc.returncode == 0, (
        f"import chain failed without pyarrow:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout


def _panel_args(prebuilt):
    """Minimal args stub — only the fields _estimate_panel_memory touches on the
    early-return paths are needed."""
    return types.SimpleNamespace(prebuilt=prebuilt)


def test_estimate_panel_memory_none_no_prebuilt():
    """Live build (--prebuilt unset) → None without raising."""
    args = _panel_args(prebuilt=None)
    assert tpp._estimate_panel_memory(args, ["000001"], "irrelevant") is None


def test_estimate_panel_memory_none_no_parquets(tmp_path):
    """A prebuilt dir with no *.parquet → None without raising (never crash the
    estimate path)."""
    args = _panel_args(prebuilt=str(tmp_path))
    assert tpp._estimate_panel_memory(args, ["000001"], str(tmp_path)) is None
