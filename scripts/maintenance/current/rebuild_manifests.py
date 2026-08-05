"""One-time migration: stamp contract manifests for canonical flat parquets.

Parquets written before the manifest feature or before the
strong schema hash / provenance fields carry no sidecar
manifest, so a formal read with ``require_valid_manifest=True`` would refuse to
load them.  This reads each ``daily/{code}.parquet`` as-is and writes a
self-consistent manifest.  Non-destructive: parquet bytes are never touched;
legacy rows are honestly recorded with ``unknown`` source/adjust (their real
provenance is not recoverable).

Run::

    PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/current/rebuild_manifests.py [data_dir]
"""
import sys

from stoke_ml.data.storage import DataStorage


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    store = DataStorage(data_dir)
    codes = store.list_stocks()
    for code in codes:
        store.rebuild_manifest(code)
    print(f"rebuilt {len(codes)} manifests under {data_dir}/a_shares/daily")
    return 0


if __name__ == "__main__":
    sys.exit(main())
