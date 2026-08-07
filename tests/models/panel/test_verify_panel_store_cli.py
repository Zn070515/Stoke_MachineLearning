"""§十九-12: ``verify_panel_store.py`` CLI exit-code contract.

The DEEP chunk verify itself (``verify_panel_store_chunks``) is exercised in
``test_panel_store.py``; these tests pin the CLI WRAPPER contract that a
lockbox / research run actually gates on:

- exit 0 + a JSON summary on a clean store,
- exit 1 + ``ERROR:`` on stderr on a tampered store and on a store with no
  chunk manifest.

The script is run as a REAL subprocess (``sys.executable``, ``PYTHONPATH`` set
to the repo root) so the exit code is asserted honestly — an in-process
``main()`` call could not prove the ``sys.exit`` wiring.  The store builders
are reused from ``test_panel_store`` (same directory → the same module object
under pytest's rootdir sys.path insertion).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_panel_store import _meta, _storeable_panel
from stoke_ml.models.panel import panel_store
from stoke_ml.models.panel.panel_store import (
    _PANEL_ARRAY_KEYS,
    save_panel_memmap,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "production" / "verify_panel_store.py"


def _run_cli(store_dir):
    """Run the actual script in a fresh subprocess; return the CompletedProcess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"  # Windows: keep child stdout UTF-8
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(store_dir)],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=env,
    )


def _tamper_byte(path, offset):
    """Flip a single byte at ``offset`` in an npy file, in place."""
    with open(path, "r+b") as fh:
        fh.seek(offset)
        b = fh.read(1)
        fh.seek(offset)
        fh.write(bytes([b[0] ^ 0xFF]))


class TestVerifyPanelStoreCli:
    def test_clean_store_exit_0_and_json_summary(self, tmp_path):
        """A clean store → exit 0, with a single JSON line carrying the
        verified / arrays_verified / chunks_verified contract keys."""
        save_panel_memmap(_storeable_panel(), tmp_path, meta=_meta())
        proc = _run_cli(tmp_path)
        assert proc.returncode == 0, (
            f"expected exit 0 on clean store:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}")
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
        assert len(lines) >= 1
        summary = json.loads(lines[-1])  # the JSON line is the last stdout line
        assert set(summary) == {"verified", "arrays_verified", "chunks_verified"}
        assert summary["verified"] is True
        assert summary["arrays_verified"] >= len(_PANEL_ARRAY_KEYS)
        assert summary["chunks_verified"] >= len(_PANEL_ARRAY_KEYS)

    def test_tampered_store_exit_1(self, tmp_path):
        """A value-level byte flip (shape/dtype untouched, so the cheap load
        check passes) → exit 1 + ERROR on stderr naming the array."""
        save_panel_memmap(_storeable_panel(seed=3), tmp_path, meta=_meta())
        pk = tmp_path / "past_known.npy"
        _tamper_byte(pk, panel_store._npy_data_offset(pk) + 1)
        # the tamper is VALUE-level: the cheap load still succeeds (header
        # bindings + manifest root are untouched) — only the deep verify catches it
        loaded = panel_store.load_panel_memmap(tmp_path)
        assert isinstance(loaded["past_known"], np.memmap)
        proc = _run_cli(tmp_path)
        assert proc.returncode == 1, (
            f"expected exit 1 on tampered store:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}")
        assert "ERROR" in proc.stderr
        assert "past_known" in proc.stderr

    def test_no_manifest_exit_1(self, tmp_path):
        """A store with no chunk_manifest.json cannot be deep-verified →
        exit 1 + ERROR naming the manifest."""
        save_panel_memmap(_storeable_panel(), tmp_path)  # no meta → no manifest
        proc = _run_cli(tmp_path)
        assert proc.returncode == 1, (
            f"expected exit 1 on manifest-less store:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}")
        assert "ERROR" in proc.stderr
        assert "chunk_manifest" in proc.stderr
