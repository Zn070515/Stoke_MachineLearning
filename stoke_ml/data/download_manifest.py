"""Download-run manifest.

Records what a download run REQUESTED vs what actually landed on disk, so a
partially-successful run ("4990 of 5000 succeeded, 10 failed, program said
success") is never mistaken for complete.  The manifest is the artifact
training / QA reads to know the true coverage of the universe.
"""
import datetime as dt
import json
import os

SCHEMA_VERSION = "1.3"


def default_path(data_dir: str) -> str:
    return os.path.join(data_dir, "a_shares", "download_manifest.json")


def run_manifest_path(data_dir: str, dataset: str) -> str:
    """Where a dataset-scoped download run manifest lives.

    Every download script records its run here so a partial run ("2998 of 3000
    fetched, 2 failed, script said done") is never mistaken for complete — the
    repo-wide download-state invariant (§五-5).
    """
    return os.path.join(data_dir, dataset, "download_manifest.json")


def write_run_manifest(
    data_dir: str,
    dataset: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    requested: list[str],
    failed: list[str],
    complete: set[str],
    success_count: int | None = None,
    skipped_existing_count: int = 0,
) -> dict:
    """Dataset-scoped wrapper over :func:`write_manifest`.

    ``dataset`` is the subdirectory under ``data_dir`` that owns the artifact
    (e.g. ``a_shares/pledge`` → ``pledge``).  ``complete`` must be the set of
    requested units that actually landed validly this run; ``failed`` the units
    whose fetch returned empty or raised.  ``missing`` (requested but not
    complete) can never be silently zero, so ``all_complete`` is honest.
    """
    return write_manifest(
        run_manifest_path(data_dir, dataset),
        market=dataset,
        start_date=start_date,
        end_date=end_date,
        requested=requested,
        failed=failed,
        complete=complete,
        success_count=success_count if success_count is not None else len(complete),
        skipped_existing_count=skipped_existing_count,
    )


def write_manifest(
    path: str,
    *,
    market: str,
    start_date: str | None = None,
    end_date: str | None = None,
    requested_end: str | None = None,
    effective_end: str | None = None,
    latest_available_end: str | None = None,
    status: str | None = None,
    requested: list[str],
    failed: list[str],
    complete: set[str],
    success_count: int,
    skipped_existing_count: int = 0,
) -> dict:
    """Persist a download-run manifest and return it.

    `requested` is the full universe this run set out to fetch; `failed` is the
    subset whose fetch returned empty.  `complete` MUST be the set of requested
    codes that both validate against their contract manifest AND cover the
    requested date range — the caller computes it via
    ``DataStorage.validate_manifest`` plus a range check, never from mere file
    presence.  `missing` (requested but not complete) is the number that
    matters: a parquet on disk does not mean the history is complete, so
    `all_complete` only holds when every requested stock is validated AND
    range-covered (§五-4).  The ``requested`` and ``complete`` lists are
    persisted so "is the ENTIRE requested universe complete" is auditable after
    the fact — a run that skipped already-complete stocks still reports the
    full request (§P0-4).

    §七-2 (bounded end): an explicit future ``requested_end`` (past the latest
    available trading day) is recorded verbatim, while ``effective_end`` and
    ``latest_available_end`` say what was actually achievable.  A request whose
    end was bounded is reported as ``status="bounded_complete"`` and
    ``all_complete=False`` — the run must NOT claim coverage of a date range no
    source can serve (§七-2).
    """
    requested_sorted = sorted(set(requested))
    failed_sorted = sorted(set(failed))
    missing = sorted(set(requested_sorted) - set(complete))
    bounded = bool(
        requested_end and effective_end and requested_end > effective_end
    )
    status_field = status or ("bounded_complete" if bounded else "complete")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "requested_end": requested_end,
        "effective_end": effective_end,
        "latest_available_end": latest_available_end,
        "status": status_field,
        "requested_count": len(requested_sorted),
        "success_count": int(success_count),
        "failed_count": len(failed_sorted),
        "skipped_existing_count": int(skipped_existing_count),
        "complete_count": len(complete),
        "missing_count": len(missing),
        "failed": failed_sorted,
        "missing": missing,
        "requested": requested_sorted,
        "complete": sorted(complete),
        "all_complete": (
            len(missing) == 0 and status_field == "complete"
        ),
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
