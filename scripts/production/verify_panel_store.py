#!/usr/bin/env python
"""Formal CLI for the §十九-12 deep panel-store chunk verification.

Exposes ``verify_panel_store_chunks`` (``stoke_ml.models.panel.panel_store``)
as an exit-code-bearing command so a lockbox / research run can gate on store
integrity: exit 0 on a fully verified store, exit 1 on corruption / tamper /
missing manifest.  The check itself (a DEEP re-hash of every chunk of every
.npy content array against ``chunk_manifest.json``) is expensive — run it ON
DEMAND before a lockbox run, never on load.

Usage:
    PYTHONPATH=. ./.venv/Scripts/python scripts/production/verify_panel_store.py <dir>

Success (stdout, exit 0) — a human line plus a single machine-readable JSON
line with true lowercase JSON booleans:
    Verified 3 arrays / 42 chunks in <dir>
    {"arrays_verified": 3, "chunks_verified": 42, "verified": true}

Failure (stderr, exit 1):
    ERROR: panel store at <dir>: chunk 32 of past_known.npy does not match ...

Only ``RuntimeError`` (corruption / tamper / missing or malformed manifest) and
``ValueError`` (a corrupt / unparseable manifest or meta — ``JSONDecodeError``)
are translated to a non-zero exit; unexpected exceptions (coding bugs, OSError)
propagate with their traceback so a real failure is never masked as a clean
"store not verified" exit.
"""
import argparse
import json
import sys
from pathlib import Path

from stoke_ml.models.panel.panel_store import verify_panel_store_chunks


def main(argv=None) -> int:
    """Run the deep chunk verification; return the process exit code (0/1)."""
    ap = argparse.ArgumentParser(
        description=(
            "Deep chunk verification of a panel store: re-hash every chunk of "
            "every .npy content array against chunk_manifest.json.  Exit 0 on "
            "a fully verified store, 1 on corruption/tamper/missing manifest."
        ),
    )
    ap.add_argument(
        "dir", metavar="DIR",
        help="path to the panel store (dir with meta.json, "
             "chunk_manifest.json and *.npy arrays)",
    )
    args = ap.parse_args(argv)

    try:
        result = verify_panel_store_chunks(args.dir)
    except (RuntimeError, ValueError) as exc:
        # The verification failure IS the contract: report it and exit non-zero.
        # RuntimeError covers the check's corruption/tamper/missing-manifest
        # guards; ValueError covers a corrupt or unparseable manifest/meta
        # (JSONDecodeError) — both are operator-facing errors, not tracebacks.
        # Other exception types (OSError, coding bugs) still propagate so a real
        # failure is never masked as a clean "store not verified" exit.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.dir)
    print(
        f"Verified {result['arrays_verified']} arrays / "
        f"{result['chunks_verified']} chunks in {out}"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
