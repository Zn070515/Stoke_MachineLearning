"""Download resume helpers.

Common pattern for download scripts: skip already-downloaded stocks (or date
ranges) by checking existing raw data on disk.  This avoids re-fetching data
that was already downloaded before an interruption (OOM, timeout, rate-limit).
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def skip_completed_stocks(
    raw_dir: str,
    codes: list[str],
    start_date: str | None = None,
    date_col: str = "date",
    suffix: str = ".parquet",
) -> tuple[list[str], int]:
    """Filter out stocks whose raw data already covers the requested date range.

    Returns ``(pending_codes, n_skipped)``.

    A stock is *complete* (skipped) when:
    1. ``{raw_dir}/{code}{suffix}`` exists, is readable, and non-empty
    2. If *start_date* is given, the oldest date in *date_col* reaches back
       to *start_date* or earlier

    Corrupted / unreadable files are silently deleted so the stock is
    re-downloaded.  Files that have data but donʼt reach *start_date* are
    treated as complete and kept (bounded-pagination sources only have a
    limited history) — never deleted on resume.
    """
    pending: list[str] = []
    skipped = 0
    start_ts = pd.Timestamp(start_date) if start_date else None

    for code in codes:
        path = os.path.join(raw_dir, f"{code}{suffix}")
        if not os.path.exists(path):
            pending.append(code)
            continue

        try:
            existing = pd.read_parquet(path)
        except Exception:
            logger.debug("  %s: unreadable file, re-downloading", code)
            _safe_unlink(path)
            pending.append(code)
            continue

        if existing.empty:
            pending.append(code)
            continue

        if start_ts is not None and date_col in existing.columns:
            dates = pd.to_datetime(existing[date_col], errors="coerce")
            oldest = dates.min()
            if pd.isna(oldest):
                # dates are garbled → re-download to be safe
                _safe_unlink(path)
                pending.append(code)
                continue
            if oldest <= start_ts:
                skipped += 1
                continue
            # Data exists but doesn't reach start_date (e.g. bounded
            # pagination on news).  Do NOT delete — just skip this run so
            # we don't destroy existing data and re-fetch it next time.
            skipped += 1
            logger.debug(
                "  %s: oldest %s is after %s — treating as complete (bounded source)",
                code, str(oldest.date()), str(start_ts.date()),
            )
            continue
        else:
            # No date-column or no date filter — existing file is enough
            skipped += 1
            continue

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
    """For year-by-year downloads, check which years already have data saved.

    Returns ``(pending_years, n_skipped)``.

    Supports two storage layouts:

    *Partitioned* — ``{storage_base}/{data_type}/{year}/`` contains parquet files.
    *Flat* — ``{storage_base}/{data_type}/*.parquet`` are per-stock flat files
      (all years merged).  In this mode the function reads a few files to
      determine the latest date on disk, then skips all years before it.
    """
    pending: list[int] = []
    skipped = 0
    type_dir = os.path.join(storage_base, data_type)

    # --- partitioned layout (year subdirs) ---
    if _has_year_dirs(type_dir):
        for year in years:
            year_dir = os.path.join(type_dir, str(year))
            if _dir_has_parquet(year_dir):
                skipped += 1
            else:
                pending.append(year)
        if skipped:
            logger.info(
                "Skipping %d already-complete years for %s, %d remaining",
                skipped, data_type, len(pending),
            )
        return pending, skipped

    # --- flat layout (per-stock files, all years merged) ---
    parquet_files = [f for f in os.listdir(type_dir) if f.endswith(".parquet")]
    if not parquet_files:
        return years, 0

    max_date = _max_date_in_dir(type_dir, parquet_files)
    if max_date is None:
        return years, 0

    for year in years:
        if year < max_date.year:
            skipped += 1
        else:
            pending.append(year)

    if skipped:
        logger.info(
            "Skipping %d already-complete years for %s (latest data: %s), %d remaining",
            skipped, data_type, max_date.date(), len(pending),
        )
    return pending, skipped


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _safe_unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _has_year_dirs(type_dir: str) -> bool:
    """Return True if the directory has year-named (4-digit) subdirectories."""
    if not os.path.isdir(type_dir):
        return False
    for name in os.listdir(type_dir):
        if name.isdigit() and len(name) == 4 and os.path.isdir(
            os.path.join(type_dir, name)
        ):
            return True
    return False


def _dir_has_parquet(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    for _root, _dirs, files in os.walk(dir_path):
        if any(f.endswith(".parquet") for f in files):
            return True
    return False


def _max_date_in_dir(
    type_dir: str, parquet_files: list[str], date_col: str = "date", sample: int = 5
) -> pd.Timestamp | None:
    """Read a sample of flat parquet files and return the latest date found."""
    import random

    random.shuffle(parquet_files)
    max_ts = pd.Timestamp.min
    for fname in parquet_files[:sample]:
        try:
            df = pd.read_parquet(os.path.join(type_dir, fname), columns=[date_col])
        except Exception:
            continue
        if df.empty or date_col not in df.columns:
            continue
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if not dates.empty and dates.max() > max_ts:
            max_ts = dates.max()
    return None if max_ts is pd.Timestamp.min else max_ts
