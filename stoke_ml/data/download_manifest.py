"""Download-run manifest.

Records what a download run REQUESTED vs what actually landed on disk, so a
partially-successful run ("4990 of 5000 succeeded, 10 failed, program said
success") is never mistaken for complete.  The manifest is the artifact
training / QA reads to know the true coverage of the universe.
"""
import datetime as dt
import json
import logging
import os
import sys

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
    universe_status: str | None = None,
    bounded_reason: str | None = None,
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
    if status is not None:
        # Explicit caller status wins — kept for migration / in-progress use.
        status_field = status
    elif len(missing) == 0 and len(failed) == 0:
        # Every requested unit is complete (and none failed) → the only honest
        # status is a full-completion one.  §九-1: a bounded end never claims a
        # date range no source can serve (§七-2).
        status_field = "bounded_complete" if bounded else "complete"
    else:
        # Un-reconciled missing/failed units: nothing landed → "failed", else
        # the run only partially covered its request → "partial".  §九-1: this
        # replaces the old derivation that read ONLY `bounded` and so reported
        # "complete" for a run that silently dropped failed stocks.
        status_field = "failed" if len(complete) == 0 else "partial"
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
    # §九-3: honesty about universe coverage.  When the PIT listing/delisting
    # artifact is absent the run cannot claim to cover a stock's pre-IPO
    # window, so it records the fact rather than pretending full coverage.
    if universe_status is not None:
        manifest["universe_status"] = universe_status
    if bounded_reason is not None:
        manifest["bounded_reason"] = bounded_reason
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return manifest


def write_run_manifest_or_exit(
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
    """Like :func:`write_run_manifest`, but a write failure is FATAL.

    A run that cannot record its own coverage must fail loudly — the parquet
    landing on disk is worthless if the manifest that proves the run's true
    coverage is silently missing (§五-5).  On success returns the manifest dict
    normally; on failure logs at error and exits non-zero.  Used by the
    production downloaders so a manifest write is never swallowed (§十一).
    """
    try:
        return write_run_manifest(
            data_dir, dataset,
            start_date=start_date, end_date=end_date,
            requested=requested, failed=failed, complete=complete,
            success_count=success_count,
            skipped_existing_count=skipped_existing_count,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("run manifest write failed: %s", exc)
        sys.exit(1)


def load_manifest(path: str) -> dict | None:
    """Read a download-run manifest, or None if it has not been written yet."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
