"""Minimal local CI — the 4 stages a private research repo needs (§十八-4).

Runs, in order:
  1. compileall      — every module/script compiles (catches syntax/dead-import)
  2. docs consistency — check_docs_consistency.py (exit 1 on drift)
  3. fast pytest     — default pytest (excludes slow/network markers)
  4. production smoke— real production chain on tiny synthetic OHLCV

A stage failure stops the run and the script exits non-zero, so it can gate a
manual pre-commit pass or a GitHub Actions wrapper.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from stoke_ml.config import get_project_root

ROOT = get_project_root()
PY = sys.executable
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)
ENV["PYTHONIOENCODING"] = "utf-8"

STAGES = [
    ("compileall", [PY, "-m", "compileall", "-q", "stoke_ml", "scripts", "tests"]),
    ("docs consistency", [PY, "scripts/production/check_docs_consistency.py"]),
    ("fast pytest", [PY, "-m", "pytest", "tests/", "-q"]),
    ("production smoke", [PY, "-m", "pytest", "tests/models/panel/test_production_smoke.py", "-q"]),
]


def main() -> int:
    failed = False
    for name, cmd in STAGES:
        t0 = time.time()
        print(f"\n=== {name} ===")
        rc = subprocess.run(cmd, cwd=ROOT, env=ENV).returncode
        dt = time.time() - t0
        ok = rc == 0
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
        failed = failed or not ok
        if not ok:
            print(f"aborting after {name} failed (rc={rc})")
            break
    print("\nCI " + ("PASSED" if not failed else "FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
