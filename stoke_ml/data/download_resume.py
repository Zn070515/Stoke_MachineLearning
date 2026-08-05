"""Download resume helpers — manifest-based completion.

A download is *complete* only when a written manifest says so.  Resume must
never decide "this stock/year is complete" by guessing from file presence or a
small sample: an existing parquet can start at the right date yet be missing the
middle, carry a changed schema, or come from a different source/adjustment, and
a year directory with one partial file does not make the whole year complete.

Every successful download writes ``<base>/.manifests/<name>.json`` via
``mark_stock_result`` / ``write_year_manifest``.  ``skip_completed_stocks`` and
``skip_completed_years`` trust ONLY a COMPLETE manifest that matches the current
request (range / schema / explicit coverage).  Anything else — missing manifest,
PARTIAL / FAILED / DEGRADED status, schema mismatch, or a request the manifest
does not cover — is re-downloaded.

A non-empty frame is no longer COMPLETE by default.  For event-type
data (news / guba / comments / announcements) min/max dates cannot prove
completeness — a capped page loop looks identical to a fully-downloaded
history.  ``mark_stock_result`` therefore defaults an evidence-less non-empty
result to PARTIAL, and only upgrades to COMPLETE when the caller supplies
proof: ``provider_exhausted=True`` (provider has no more data),
``provider_range_guaranteed=True``, or ``expected_rows == actual_rows``.

Provider exhaustion and request coverage are separate facts (§五-1).  A
bounded provider that only keeps the recent N years reports
``provider_exhausted=True`` yet its coverage is NOT complete, and the manifest
is recorded BOUNDED_COMPLETE (never a COMPLETE status).

v14 §十一 unifies the coverage model around one per-stock window:

    requested_start/end   the user's original global request, verbatim
    effective_start/end   the stock's ACTUAL ask — the requested window
                          clipped by the caller to its own lifecycle
                          (``max(global_start, listing_date)`` to
                          ``min(global_end, delist_date, latest_available)``);
                          degenerates to the requested window when the caller
                          knows no lifecycle bounds
    actual_start/end      the dates actually fetched
    effective_range_covered   THE single completion conclusion, derived
                          ``actual covers effective`` from the dates.

No boolean override can bypass the date facts: the ``covers_request`` /
``request_covered`` keys that legacy manifests carry are now DERIVED on write
(``covers_request`` mirrors the effective-range conclusion; ``request_covered``
is informational actual-vs-requested) and ``_manifest_matches`` re-verifies
every match against the stored effective range rather than trusting any stored
boolean (§五-2).
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)

STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_FAILED = "FAILED"
STATUS_DEGRADED = "DEGRADED"
# Provider has nothing more to give but the fetched data does not reach the
# requested window (bounded history / later listing).  Never trusted for skip.
STATUS_BOUNDED_COMPLETE = "BOUNDED_COMPLETE"


def evidence_says_complete(
    df: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    pagination_exhausted: bool | None = None,
    provider_exhausted: bool | None = None,
    provider_range_guaranteed: bool | None = None,
) -> bool:
    """Whether the supplied evidence proves a fetched frame is complete.

    This answers only "did we reach the end of what this provider has" —
    NOT whether the request was covered (that is ``request_covered``).

    Only three proofs are accepted:
      * ``provider_exhausted=True`` (legacy alias ``pagination_exhausted``) —
        a paged fetch reached the end of the provider's history (last page
        short / empty / older than the window).
      * ``provider_range_guaranteed=True`` — the source returns the complete
        record set for a date-range query in a single call (dense data).
      * ``expected_rows == actual rows`` — an independent count matched.

    An empty frame is never complete.
    """
    if df.empty:
        return False
    if provider_exhausted is True or pagination_exhausted is True:
        return True
    if provider_range_guaranteed is True:
        return True
    if expected_rows is not None and len(df) == expected_rows:
        return True
    return False


def request_covered(
    *,
    requested_start: str | None = None,
    requested_end: str | None = None,
    actual_start=None,
    actual_end=None,
    missing_intervals: list | None = None,
) -> bool | None:
    """Whether fetched dates cover the requested window (§五-1 formula).

    ``actual_start <= requested_start and actual_end >= requested_end and
    missing_intervals == []``.  Returns None when no requested range was
    supplied (coverage not assessable).  Pre-listing no-data — a stock listed
    after ``requested_start`` — is NOT counted as covered here; callers that
    know the listing date pin ``covers_request`` explicitly.
    """
    if requested_start is None and requested_end is None:
        return None
    if missing_intervals:
        return False
    if requested_start is not None:
        if actual_start is None or pd.Timestamp(actual_start) > pd.Timestamp(requested_start):
            return False
    if requested_end is not None:
        if actual_end is None or pd.Timestamp(actual_end) < pd.Timestamp(requested_end):
            return False
    return True


def schema_hash(df: pd.DataFrame) -> str:
    """Stable short hash of a DataFrame's column set (schema-change detector)."""
    cols = "|".join(sorted(str(c) for c in df.columns))
    return hashlib.sha1(cols.encode("utf-8")).hexdigest()[:16]


def _iso(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _manifest_path(base_dir: str, name: str) -> str:
    return os.path.join(base_dir, ".manifests", f"{name}.json")


def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning(
            "manifest write failed (category=%s): %s",
            classify_error(exc).value, exc,
        )
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# per-stock manifests
# ---------------------------------------------------------------------------

def write_stock_manifest(
    raw_dir: str,
    stock_code: str,
    *,
    dataset: str | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    effective_start=None,
    effective_end=None,
    actual_start=None,
    actual_end=None,
    expected_rows: int | None = None,
    actual_rows: int | None = None,
    missing_intervals: list | None = None,
    pages_requested: int | None = None,
    pages_fetched: int | None = None,
    pagination_exhausted: bool | None = None,
    provider_exhausted: bool | None = None,
    provider_range_guaranteed: bool | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    schema_hash: str | None = None,
    status: str | None = None,
) -> str:
    """Atomically write a per-stock completion manifest. Returns its path.

    v14 §十一: the manifest records ONE date-derived completion conclusion.
    ``effective_start/effective_end`` is the window the stock actually needed
    (the requested window clipped by the caller to listing/delist/latest);
    it degenerates to the requested window when not supplied.  The payload's
    ``effective_range_covered`` is derived ``actual covers effective`` and is
    the ONLY fact ``_manifest_matches`` trusts.  The legacy ``covers_request``
    / ``request_covered`` keys are still written (backward compatibility with
    old consumers / on-disk manifests) but are DERIVED — no override parameter
    is accepted, so a boolean can never bypass the date facts.

    ``status`` derives from the dates when not supplied: no actual rows →
    DEGRADED; effective range covered → COMPLETE; data present but the
    effective window not reached → BOUNDED_COMPLETE (never skipped by resume).
    An explicitly-passed ``COMPLETE`` that contradicts a date-derived
    not-covered conclusion is downgraded to BOUNDED_COMPLETE rather than
    recording a self-contradictory manifest.
    """
    eff_start = _iso(effective_start if effective_start is not None else requested_start)
    eff_end = _iso(effective_end if effective_end is not None else requested_end)
    eff_cov = request_covered(
        requested_start=eff_start,
        requested_end=eff_end,
        actual_start=actual_start,
        actual_end=actual_end,
        missing_intervals=missing_intervals,
    )
    req_cov = request_covered(
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
        missing_intervals=missing_intervals,
    )
    no_data = actual_rows == 0 or (actual_start is None and actual_end is None)
    if status is None:
        if no_data:
            status = STATUS_DEGRADED
        elif eff_cov is not False:
            status = STATUS_COMPLETE
        else:
            status = STATUS_BOUNDED_COMPLETE
    elif status == STATUS_COMPLETE and eff_cov is False:
        status = STATUS_BOUNDED_COMPLETE
    effective_range_covered = (
        eff_cov if eff_cov is not None else status == STATUS_COMPLETE
    )
    payload = {
        "dataset": dataset,
        "stock_code": stock_code,
        "requested_start": _iso(requested_start),
        "requested_end": _iso(requested_end),
        "effective_start": eff_start,
        "effective_end": eff_end,
        "actual_start": _iso(actual_start),
        "actual_end": _iso(actual_end),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "missing_intervals": missing_intervals or [],
        "pages_requested": pages_requested,
        "pages_fetched": pages_fetched,
        # provider_exhausted is the §五-1 canonical field; pagination_exhausted
        # is kept as a legacy alias so old consumers still read it.
        "provider_exhausted": provider_exhausted if provider_exhausted is not None else pagination_exhausted,
        "pagination_exhausted": pagination_exhausted if pagination_exhausted is not None else provider_exhausted,
        "provider_range_guaranteed": provider_range_guaranteed,
        "source": source,
        "adjustment": adjustment,
        "schema_hash": schema_hash,
        "status": status,
        # v14 §十一: the single completion conclusion, derived from dates.
        "effective_range_covered": effective_range_covered,
        # Legacy aliases — DERIVED, never overridable.
        "covers_request": effective_range_covered,
        "request_covered": req_cov,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _manifest_path(raw_dir, stock_code)
    _write_json(path, payload)
    return path


def read_stock_manifest(raw_dir: str, stock_code: str) -> dict | None:
    return _read_json(_manifest_path(raw_dir, stock_code))


def mark_stock_result(
    raw_dir: str,
    stock_code: str,
    df: pd.DataFrame,
    *,
    dataset: str | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    effective_start=None,
    effective_end=None,
    expected_rows: int | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    status: str | None = None,
    date_col: str = "date",
    pages_requested: int | None = None,
    pages_fetched: int | None = None,
    pagination_exhausted: bool | None = None,
    provider_exhausted: bool | None = None,
    provider_range_guaranteed: bool | None = None,
    missing_intervals: list | None = None,
) -> str:
    """Record one stock's download outcome from the fetched frame.

    Derives actual coverage, row count and schema hash from ``df``.

    Status rules:
      * empty ``df`` → DEGRADED (attempt recorded, never trusted for resume);
      * non-empty with proof (``provider_exhausted`` / legacy
        ``pagination_exhausted=True`` / ``provider_range_guaranteed=True`` /
        ``expected_rows == actual_rows``) AND the *effective* window covered →
        COMPLETE;
      * fetch proved complete but the effective window not reached (bounded
        provider / later listing) → BOUNDED_COMPLETE — never skipped by resume;
      * non-empty without proof → PARTIAL — the fetch may have been truncated
        by a page cap or rate limit, so resume must NOT skip it.

    v14 §十一: completion is judged against the stock's EFFECTIVE window —
    ``effective_start/effective_end`` (the requested range clipped by the caller
    to listing/delist/latest-available), degenerating to the requested range
    when the caller knows no lifecycle bounds.  This is what replaces the old
    ``covers_request`` override: a stock that listed after ``requested_start``
    is expressed by passing ``effective_start=listing_date``, and the manifest
    derives ``effective_range_covered = actual covers effective``.  No boolean
    parameter can promote a frame whose effective window was not reached.
    """
    if date_col in df.columns and not df.empty:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        actual_start = dates.min()
        actual_end = dates.max()
    else:
        actual_start = actual_end = None
    eff_start = effective_start if effective_start is not None else requested_start
    eff_end = effective_end if effective_end is not None else requested_end
    eff_cov = request_covered(
        requested_start=eff_start,
        requested_end=eff_end,
        actual_start=actual_start,
        actual_end=actual_end,
        missing_intervals=missing_intervals,
    )
    if status is None:
        if df.empty:
            status = STATUS_DEGRADED
        elif not evidence_says_complete(
            df,
            expected_rows=expected_rows,
            pagination_exhausted=pagination_exhausted,
            provider_exhausted=provider_exhausted,
            provider_range_guaranteed=provider_range_guaranteed,
        ):
            status = STATUS_PARTIAL
        elif eff_cov is not False:
            # v14 §十一: coverage is the date-derived conclusion ONLY — no
            # boolean override can promote a frame whose effective window was
            # not reached.
            status = STATUS_COMPLETE
        else:
            status = STATUS_BOUNDED_COMPLETE
    return write_stock_manifest(
        raw_dir, stock_code,
        dataset=dataset,
        requested_start=requested_start,
        requested_end=requested_end,
        effective_start=eff_start,
        effective_end=eff_end,
        actual_start=actual_start,
        actual_end=actual_end,
        expected_rows=expected_rows,
        actual_rows=len(df),
        missing_intervals=missing_intervals,
        pages_requested=pages_requested,
        pages_fetched=pages_fetched,
        pagination_exhausted=pagination_exhausted,
        provider_exhausted=provider_exhausted,
        provider_range_guaranteed=provider_range_guaranteed,
        source=source,
        adjustment=adjustment,
        schema_hash=schema_hash(df),
        status=status,
    )


# ---------------------------------------------------------------------------
# year-level manifests
# ---------------------------------------------------------------------------

def write_year_manifest(
    storage_base: str,
    data_type: str,
    year: int,
    *,
    requested_start: str | None = None,
    requested_end: str | None = None,
    effective_start=None,
    effective_end=None,
    actual_start=None,
    actual_end=None,
    expected_rows: int | None = None,
    actual_rows: int | None = None,
    n_stocks: int | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    schema_hash: str | None = None,
    status: str | None = None,
) -> str:
    """Atomically write a year-level completion manifest. Returns its path.

    Mirrors :func:`write_stock_manifest`: ``effective_range_covered`` is the
    derived completion conclusion (actual covers the effective window, which
    degenerates to the requested window), and ``status`` derives from it when
    not supplied — a year whose fetched dates do not reach the requested range
    is BOUNDED_COMPLETE, never a COMPLETE that a stale boolean would skip.
    """
    base = os.path.join(storage_base, data_type)
    eff_start = _iso(effective_start if effective_start is not None else requested_start)
    eff_end = _iso(effective_end if effective_end is not None else requested_end)
    eff_cov = request_covered(
        requested_start=eff_start,
        requested_end=eff_end,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    no_data = actual_rows == 0 or (actual_start is None and actual_end is None)
    if status is None:
        if no_data:
            status = STATUS_DEGRADED
        elif eff_cov is not False:
            status = STATUS_COMPLETE
        else:
            status = STATUS_BOUNDED_COMPLETE
    elif status == STATUS_COMPLETE and eff_cov is False:
        status = STATUS_BOUNDED_COMPLETE
    effective_range_covered = (
        eff_cov if eff_cov is not None else status == STATUS_COMPLETE
    )
    payload = {
        "dataset": data_type,
        "year": year,
        "requested_start": _iso(requested_start),
        "requested_end": _iso(requested_end),
        "effective_start": eff_start,
        "effective_end": eff_end,
        "actual_start": _iso(actual_start),
        "actual_end": _iso(actual_end),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "n_stocks": n_stocks,
        "source": source,
        "adjustment": adjustment,
        "schema_hash": schema_hash,
        "status": status,
        "effective_range_covered": effective_range_covered,
        "covers_request": effective_range_covered,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _manifest_path(base, str(year))
    _write_json(path, payload)
    return path


def read_year_manifest(storage_base: str, data_type: str, year: int) -> dict | None:
    return _read_json(_manifest_path(os.path.join(storage_base, data_type), str(year)))


# ---------------------------------------------------------------------------
# matching + skip logic
# ---------------------------------------------------------------------------

def _manifest_matches(
    manifest: dict | None,
    *,
    requested_start: str | None = None,
    requested_end: str | None = None,
    schema_hash: str | None = None,
) -> bool:
    """A manifest satisfies the request only when it is COMPLETE and covers it.

    Coverage is RE-VERIFIED against the stored *effective* range every time
    (§五-2 / v14 §十一): no stored boolean is trusted on its own, because a
    stale, partial or hand-edited manifest could claim coverage it does not
    have.  For new manifests the effective window is stored on the manifest; a
    legacy manifest (no effective fields) falls back to the current request's
    bounds.  The stored booleans only serve as a denial veto (explicit False
    still forces a re-download), and a request that extends past the stored
    effective end is never skipped.
    """
    if not manifest or manifest.get("status") != STATUS_COMPLETE:
        return False
    if schema_hash and manifest.get("schema_hash") != schema_hash:
        return False
    # v14 §十一: effective_range_covered is the single completion conclusion.
    # A stored False always forces a re-download, regardless of dates.
    if manifest.get("effective_range_covered") is False:
        return False
    # Legacy denial veto (pre-v14 writers): a stored covers_request False, or a
    # request_covered False on a manifest that predates effective_range_covered,
    # still forces a re-download.  New manifests derive covers_request from
    # effective_range_covered (already handled) and only record request_covered
    # informatively — a new manifest whose *requested* window was never reached
    # (e.g. pre-listing) must not be denied when its effective coverage is
    # complete, so the request_covered veto is legacy-only.
    if manifest.get("covers_request") is False:
        return False
    if "effective_range_covered" not in manifest and manifest.get("request_covered") is False:
        return False
    # Re-derive coverage against the effective window the stock actually needed.
    eff_start = manifest.get("effective_start") or requested_start
    eff_end = manifest.get("effective_end") or requested_end
    if eff_start:
        a = manifest.get("actual_start")
        if not a or pd.Timestamp(a) > pd.Timestamp(eff_start):
            return False
    if eff_end:
        b = manifest.get("actual_end")
        if not b or pd.Timestamp(b) < pd.Timestamp(eff_end):
            return False
    # A request that extends past the stored effective end means the stock may
    # need data this manifest never covered → re-download (v14 §十一).
    if requested_end and eff_end and pd.Timestamp(requested_end) > pd.Timestamp(eff_end):
        return False
    return True


def skip_completed_stocks(
    raw_dir: str,
    codes: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    schema_hash: str | None = None,
) -> tuple[list[str], int]:
    """Filter out stocks whose download is recorded COMPLETE for this request.

    Returns ``(pending_codes, n_skipped)``.  A stock is skipped ONLY when
    ``<raw_dir>/.manifests/{code}.json`` says COMPLETE and matches the requested
    range / schema.  File presence alone is never trusted.
    """
    pending: list[str] = []
    skipped = 0
    for code in codes:
        manifest = read_stock_manifest(raw_dir, code)
        if _manifest_matches(
            manifest,
            requested_start=start_date,
            requested_end=end_date,
            schema_hash=schema_hash,
        ):
            skipped += 1
        else:
            pending.append(code)
    if skipped:
        logger.info(
            "Skipping %d already-complete stocks, %d remaining", skipped, len(pending),
        )
    return pending, skipped


def skip_completed_years(
    storage_base: str,
    years: list[int],
    data_type: str,
) -> tuple[list[int], int]:
    """Filter out years recorded COMPLETE for the given data type.

    Returns ``(pending_years, n_skipped)``.  A year is skipped ONLY when its
    ``{data_type}/.manifests/{year}.json`` says COMPLETE.  Detecting a parquet
    file in a year directory — or sampling a few flat files — is never enough
    to call a year complete.
    """
    pending: list[int] = []
    skipped = 0
    for year in years:
        manifest = read_year_manifest(storage_base, data_type, year)
        if _manifest_matches(manifest):
            skipped += 1
        else:
            pending.append(year)
    if skipped:
        logger.info(
            "Skipping %d already-complete years for %s, %d remaining",
            skipped, data_type, len(pending),
        )
    return pending, skipped
