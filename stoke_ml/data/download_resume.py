"""Download resume helpers — manifest-based completion (v6 §八).

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
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"
STATUS_DEGRADED = "DEGRADED"


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
    except Exception:
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
    actual_start=None,
    actual_end=None,
    expected_rows: int | None = None,
    actual_rows: int | None = None,
    missing_intervals: list | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    schema_hash: str | None = None,
    covers_request: bool | None = None,
    status: str = STATUS_COMPLETE,
) -> str:
    """Atomically write a per-stock completion manifest. Returns its path."""
    payload = {
        "dataset": dataset,
        "stock_code": stock_code,
        "requested_start": _iso(requested_start),
        "requested_end": _iso(requested_end),
        "actual_start": _iso(actual_start),
        "actual_end": _iso(actual_end),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "missing_intervals": missing_intervals or [],
        "source": source,
        "adjustment": adjustment,
        "schema_hash": schema_hash,
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # covers_request is only recorded when the writer decided it; absence means
    # coverage falls back to actual_start/actual_end comparison (legacy).
    if covers_request is not None:
        payload["covers_request"] = covers_request
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
    expected_rows: int | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    covers_request: bool | None = None,
    status: str | None = None,
    date_col: str = "date",
) -> str:
    """Record one stock's download outcome from the fetched frame.

    Derives actual coverage, row count and schema hash from ``df``.  A
    non-empty ``df`` defaults to COMPLETE and ``covers_request=True`` (the
    downloader just fetched and saved it successfully — trust the fetch); an
    empty one defaults to DEGRADED — recorded so the attempt is visible, but
    never trusted by ``skip_completed_stocks``.  Pass ``covers_request=None``
    to fall back to date-range coverage comparison instead.
    """
    if status is None:
        status = STATUS_COMPLETE if not df.empty else STATUS_DEGRADED
    if covers_request is None:
        covers_request = not df.empty
    if date_col in df.columns and not df.empty:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        actual_start = dates.min()
        actual_end = dates.max()
    else:
        actual_start = actual_end = None
    return write_stock_manifest(
        raw_dir, stock_code,
        dataset=dataset,
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
        expected_rows=expected_rows,
        actual_rows=len(df),
        source=source,
        adjustment=adjustment,
        schema_hash=schema_hash(df),
        covers_request=covers_request,
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
    actual_start=None,
    actual_end=None,
    expected_rows: int | None = None,
    actual_rows: int | None = None,
    n_stocks: int | None = None,
    source: str | None = None,
    adjustment: str | None = None,
    schema_hash: str | None = None,
    covers_request: bool | None = None,
    status: str = STATUS_COMPLETE,
) -> str:
    """Atomically write a year-level completion manifest. Returns its path."""
    base = os.path.join(storage_base, data_type)
    payload = {
        "dataset": data_type,
        "year": year,
        "requested_start": _iso(requested_start),
        "requested_end": _iso(requested_end),
        "actual_start": _iso(actual_start),
        "actual_end": _iso(actual_end),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "n_stocks": n_stocks,
        "source": source,
        "adjustment": adjustment,
        "schema_hash": schema_hash,
        "covers_request": covers_request,
        "status": status,
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
    """A manifest satisfies the request only when it is COMPLETE and covers it."""
    if not manifest or manifest.get("status") != STATUS_COMPLETE:
        return False
    if schema_hash and manifest.get("schema_hash") != schema_hash:
        return False
    # New manifests declare coverage explicitly (handles empty-but-successful
    # windows and bounded sources).  Legacy manifests fall back to dates.
    if "covers_request" in manifest:
        return manifest["covers_request"] is True
    if requested_start:
        a = manifest.get("actual_start")
        if not a or pd.Timestamp(a) > pd.Timestamp(requested_start):
            return False
    if requested_end:
        b = manifest.get("actual_end")
        if not b or pd.Timestamp(b) < pd.Timestamp(requested_end):
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
    range / schema.  File presence alone is never trusted (v6 §八).
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
    to call a year complete (v6 §八).
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
