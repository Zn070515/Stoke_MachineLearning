"""Download-run manifest (review v8 §二-2).

Records what a download run REQUESTED vs what actually landed on disk, so a
partially-successful run ("4990 of 5000 succeeded, 10 failed, program said
success") is never mistaken for complete.  The manifest is the artifact
training / QA reads to know the true coverage of the universe.
"""
import datetime as dt
import json
import os

SCHEMA_VERSION = "1.0"


def default_path(data_dir: str) -> str:
    return os.path.join(data_dir, "a_shares", "download_manifest.json")


def write_manifest(
    path: str,
    *,
    market: str,
    start_date: str,
    end_date: str,
    requested: list[str],
    failed: list[str],
    on_disk: set[str],
    success_count: int,
    skipped_existing_count: int = 0,
) -> dict:
    """Persist a download-run manifest and return it.

    `requested` is the full universe this run set out to fetch; `failed` is the
    subset whose fetch returned empty; `on_disk` is the set of codes actually
    present in the flat store at the end of the run.  `missing` (requested but
    not on disk) is the number that matters — it cannot be silently zero.
    """
    requested_sorted = sorted(set(requested))
    failed_sorted = sorted(set(failed))
    missing = sorted(set(requested_sorted) - set(on_disk))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "requested_count": len(requested_sorted),
        "success_count": int(success_count),
        "failed_count": len(failed_sorted),
        "skipped_existing_count": int(skipped_existing_count),
        "on_disk_count": len(on_disk),
        "missing_count": len(missing),
        "failed": failed_sorted,
        "missing": missing,
        "all_complete": len(missing) == 0,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return manifest


def load_manifest(path: str) -> dict | None:
    """Read a download-run manifest, or None if it has not been written yet."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
