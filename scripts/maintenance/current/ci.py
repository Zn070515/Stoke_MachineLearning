"""Local CI — mirrors the GitHub Actions split (§十三-4) for a pre-commit pass.

GitHub runs the same slices as parallel jobs (core-fast / storage-parquet / ml
on PR+push; slow-nightly / optional-online on schedule + workflow_dispatch).
The remote jobs install via `uv sync --frozen` against the committed uv.lock
(§十三-3); this script runs the requested slice(s) sequentially in the local
venv so the local pass and the remote run cannot diverge:

  core-fast      lockfile freshness + compileall + docs consistency +
                 tests/utils + tests/evaluation
  storage-parquet  tests/data (storage / parquet / manifests / sources)
  ml             tests/models + tests/features + tests/preprocessing + tests/scripts
  slow-nightly   pytest -m slow tests/ (full deps)
  optional-online  pytest -m network tests/ (online deps)

The old standalone production-smoke stage is gone: fast pytest already includes
the non-slow smoke tests, so running the file again was pure duplication.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py --groups ml
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py --slow
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/ci.py --online
"""
import argparse
import os
import subprocess
import sys
import time

from stoke_ml.config import get_project_root

ROOT = get_project_root()
PY = sys.executable
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)
ENV["PYTHONIOENCODING"] = "utf-8"

GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "core-fast": [
        ("lockfile freshness", ["uv", "lock", "--check"]),
        ("compileall", [PY, "-m", "compileall", "-q", "stoke_ml", "scripts", "tests"]),
        ("docs consistency", [PY, "scripts/production/check_docs_consistency.py"]),
        ("pytest utils+evaluation", [PY, "-m", "pytest", "tests/utils", "tests/evaluation", "-q"]),
    ],
    "storage-parquet": [
        ("pytest tests/data", [PY, "-m", "pytest", "tests/data", "-q"]),
    ],
    "ml": [
        ("pytest models+features+preprocessing+scripts", [
            PY, "-m", "pytest",
            "tests/models", "tests/features", "tests/preprocessing", "tests/scripts", "-q",
        ]),
    ],
    "slow-nightly": [
        ("pytest slow", [PY, "-m", "pytest", "-m", "slow", "tests/", "-q"]),
    ],
    "optional-online": [
        ("pytest network", [PY, "-m", "pytest", "-m", "network", "tests/", "-q"]),
    ],
}

# Default pre-commit pass: the three fast groups (the PR critical path).
DEFAULT_GROUPS = ["core-fast", "storage-parquet", "ml"]


def _run_stage(name: str, cmd: list[str]) -> bool:
    t0 = time.time()
    print(f"\n=== {name} ===")
    rc = subprocess.run(cmd, cwd=ROOT, env=ENV).returncode
    dt = time.time() - t0
    ok = rc == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CI slices locally (§十三-4).")
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(GROUPS),
        default=DEFAULT_GROUPS,
        help="CI slice(s) to run; default: %(default)s",
    )
    parser.add_argument(
        "--slow", action="store_true", help="shortcut for --groups slow-nightly",
    )
    parser.add_argument(
        "--online", action="store_true", help="shortcut for --groups optional-online",
    )
    args = parser.parse_args()

    groups = args.groups
    if args.slow:
        groups = ["slow-nightly"]
    if args.online:
        groups = ["optional-online"]
    if args.slow and args.online:
        groups = ["slow-nightly", "optional-online"]

    failed = False
    for group in groups:
        print(f"\n##### GROUP: {group} #####")
        for name, cmd in GROUPS[group]:
            ok = _run_stage(name, cmd)
            failed = failed or not ok
            if not ok:
                print(f"aborting after {name} failed (group {group})")
                break
        if failed:
            break

    print("\nCI " + ("PASSED" if not failed else "FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
