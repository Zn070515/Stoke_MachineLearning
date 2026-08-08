"""File-level asset contract for auxiliary data stores (§十三).

The daily K-line store (``stoke_ml/data/storage.py``) already carries the full
governance stack: a per-file contract manifest, a schema hash, atomic
read-modify-write, formal reads that raise on a missing/mismatched manifest.
The auxiliary data stores — fundamentals, market-wide, ETF flow, announcements
— historically wrote raw Parquet directly with no manifest, no atomic
replacement and no source/version record, so corruption or a layout drift went
silently trusted.

``asset_contract.py`` abstracts the FILE-LEVEL piece of that governance into a
small reusable layer, leaving the column-schema contract (``contract.py``), the
content-validation layer (``generation_store.py``, §十三-2) and the daily store
itself untouched:

* :class:`AtomicCommit` — write a file via temp + ``os.replace``, removing the
  temp on failure, so a reader never observes a torn file and a failed write
  leaves the prior file untouched.
* :func:`acquire_lock` / :func:`release_lock` / :func:`file_lock` — the shared
  single-writer lock (atomic ``os.mkdir`` of ``<target>.lock``, blocking with
  timeout + stale-lock steal) that serializes read-modify-write on the same
  target across processes.  Used by ``generation_store`` and
  ``market_wide_storage``.
* :class:`DataAssetContract` — frozen description of one file-level asset: its
  ``data_type``, partition scheme, optional ``column_contract`` name, and the
  column whose values bound the manifest's ``start``/``end`` extent.
* :func:`write_asset_manifest` — atomically write the sidecar
  ``{parquet}.manifest.json`` pinning rows / extent / schema hash / source /
  written_at alongside the parquet.  A crash between the two ``os.replace``
  calls leaves a stale pair that the next validated read catches.
* :func:`validate_asset_manifest` / :func:`check_asset_read` — the cross-check
  that flags a tampered file: a parquet whose content hash / row count / data
  type no longer matches its manifest is ``ok=False``.

The Formal Asset Interface (§十七)
----------------------------------
A manifest pins SEVEN aspects of the file it guards.  Six are recorded at write
and cross-checked at read; ``source`` is provenance (reported, not re-derivable
from the file, so not cross-checked):

1. **Content identity** — ``schema_hash``, a value-level checksum of every
   column (stable across parquet round-trip rewrites).
2. **Source identity** — ``source``, drawn from ``df.attrs["source"]``
   (default ``"unknown"`` when the writing code did not declare it).
3. **Coverage** — ``start`` / ``end``, the ``extent_column``'s min/max
   (``"date"``, ``"report_date"``, ...).  A single-file asset whose dates live
   in the row **index** (broadcast industry / market-env files) is covered the
   same way: ``_extent`` falls back to a DatetimeIndex when the named column is
   absent.
4. **Effective-date policy** — ``effective_date_policy``, HOW the effective
   date of a stored value is determined.  Vocabulary: ``"record_date"`` (value
   recorded for a trading day), ``"event_date"`` (schedule/event-list rows
   keyed by event date), ``"post_close_next_trading_day"`` (post-close text
   events PIT-mapped to the next trading day at storage),
   ``"index_date"`` (dates live in the file's DatetimeIndex).
5. **Vintage status** — ``vintage_source`` / ``vintage_transform`` /
   ``vintage_pit``, the channel's 3-dim classification drawn from
   ``channel_vintage.declaration_of`` (:func:`contract_for_channel` fills them
   automatically).  The manifest of a channel carries the SAME labels its
   training admission is judged against — a vintage re-declaration is a
   manifest-visible event, not a silent re-label.
6. **Schema** — ``column_contract``, the name of a column-schema contract in
   ``contract.py`` (recorded for provenance; enforcement is a separate gate).
7. **Atomic commit** — write via temp + ``os.replace``; the manifest itself is
   written atomically with ``atomic_write_json``.

The original three assets (fundamentals / announcement / ETF flow, v15 T9)
declare only the first three aspects; the new fields default to ``None`` and
stay out of their manifests, so nothing written before this extension is
re-validated differently.

Backward compatibility: existing on-disk aux data has NO manifests, so default
reads are lenient — a manifest-less file is read (logged at debug) and only a
PRESENT-but-mismatched manifest is flagged (warning).  ``require_valid_manifest=True``
(opt-in, mirroring the daily store's formal-load rule) raises for a missing OR
mismatched manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from stoke_ml.data import channel_vintage as _cv
from stoke_ml.data.download_resume import read_stock_manifest

logger = logging.getLogger(__name__)

_ASSET_MANIFEST_SUFFIX = ".manifest.json"


class AtomicCommit:
    """Same-directory temp + ``os.replace`` atomic file writer.

    Usage::

        with AtomicCommit(dest) as ac:
            df.to_parquet(ac.tmp_path, index=False)

    The temp lives in the destination directory so the final ``os.replace`` is
    a same-filesystem rename.  On success the temp is swapped over ``dest``;
    on any exception the temp is removed and ``dest`` is left untouched.
    """

    def __init__(self, dest: str):
        self.dest = dest
        self.tmp_path = f"{dest}.tmp.{os.getpid()}"

    def __enter__(self) -> "AtomicCommit":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._cleanup()
            return False
        try:
            os.replace(self.tmp_path, self.dest)
        except BaseException:
            self._cleanup()
            raise
        return False

    def _cleanup(self) -> None:
        try:
            if os.path.isfile(self.tmp_path):
                os.unlink(self.tmp_path)
        except OSError:
            pass


def atomic_write_json(path: str, payload: dict) -> None:
    """Write ``payload`` as pretty JSON via temp + ``os.replace``."""
    with AtomicCommit(path) as ac:
        with open(ac.tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def manifest_body_digest(path: str) -> str:
    """Content digest of ONE ``*.manifest.json`` sidecar, ``written_at`` excluded.

    ``written_at`` is a per-write TIMESTAMP, so a content-identical rewrite of
    the same data (a re-download that changed nothing) must NOT change any
    hash built from this sidecar — the binding is WHAT the data is, not when it
    was written.  The sidecar is parsed as JSON and the timestamp key dropped
    before the canonical digest; a non-JSON / unreadable sidecar degrades to
    hashing its RAW BYTES (a partial rewrite still visible, and a JSON-level
    nicety like written_at exclusion is moot when the file is not JSON anyway).
    Returns the hex digest — never None: an unreadable sidecar maps to
    ``"<unreadable>"`` so the entry stays present (fail-closed).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return "<unreadable>"
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k != "written_at"}
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── Single-writer lock ─────────────────────────────────────────────────────
# A read-modify-write on the same target from two processes (parallel macro
# downloads, parallel market-wide backfills) must be exclusive.  Atomic rename
# alone only protects readers from torn files, not concurrent writers from
# interleaving their updates.  The lock is an atomic ``os.mkdir`` of
# ``<target>.lock`` — exactly one process wins the mkdir, the rest block up to
# ``timeout`` and then raise :class:`TimeoutError`.  A lock dir whose mtime is
# older than ``_LOCK_STALE`` is assumed to be a crashed process's leftover and
# is stolen (``os.rmdir``) so writers can never deadlock on a stale lock.
_LOCK_TIMEOUT = 600.0
_LOCK_STALE = 900.0


def acquire_lock(target: str, timeout: float = _LOCK_TIMEOUT) -> str:
    """Exclusive per-target lock via atomic mkdir. Returns the lock dir path.

    Stale-lock steal: a lock dir older than ``_LOCK_STALE`` (900s) is assumed
    to be a crashed writer's leftover and is stolen, so a LIVE writer holding
    the lock that long would be displaced — the trade is crash recovery over
    long-write safety.  generation_store holds the lock only for the sub-second
    write body, so this is not a practical hazard there.
    """
    lock_dir = target + ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.mkdir(lock_dir)
            return lock_dir
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_dir) > _LOCK_STALE:
                    os.rmdir(lock_dir)  # steal stale lock from a crashed process
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire lock: {lock_dir}")
            time.sleep(0.05)


def release_lock(lock_dir: str) -> None:
    try:
        os.rmdir(lock_dir)
    except OSError:
        pass


@contextmanager
def file_lock(target: str, timeout: float = _LOCK_TIMEOUT):
    """Blocking exclusive lock on ``target`` as a context manager.

    Acquires ``<target>.lock`` on entry and releases it on exit (even on
    exception).  Raises :class:`TimeoutError` if the lock cannot be acquired
    within ``timeout`` seconds.
    """
    lock_dir = acquire_lock(target, timeout=timeout)
    try:
        yield lock_dir
    finally:
        release_lock(lock_dir)


@dataclass(frozen=True)
class DataAssetContract:
    """File-level description of one auxiliary data asset.

    Complements the column-schema :class:`stoke_ml.data.contract.DataContract`
    (what columns a dataset must carry) with the FILE-level facts a manifest
    pins: rows, primary-key extent, schema hash, source, written_at — plus,
    for headline_v1-adopted channels (§十七), the effective-date policy and the
    channel's 3-dim vintage status (see the module docstring for the full
    seven-aspect interface).
    """

    data_type: str
    partition: str
    #: Column whose values bound the manifest's start/end (e.g. "date",
    #: "report_date").  None when the asset has no meaningful extent.  For a
    #: single-file asset whose dates live in the row INDEX (broadcast
    #: industry / market-env), name the index — ``_extent`` falls back to a
    #: DatetimeIndex when the column is absent.
    extent_column: str | None = None
    #: Name of a column-schema contract in ``contract.py`` governing this
    #: asset.  Recorded in the manifest for provenance; schema enforcement is
    #: a separate gate and is deliberately NOT applied here.
    column_contract: str | None = None
    #: HOW the effective date of a stored value is determined — one of
    #: ``"record_date"`` / ``"event_date"`` / ``"post_close_next_trading_day"``
    #: / ``"index_date"`` (see module docstring).  Recorded in the manifest for
    #: provenance; ``None`` (the v15 T9 assets) stays out of the manifest.
    effective_date_policy: str | None = None
    #: The channel's declared ``source_vintage`` (``immutable_snapshot`` /
    #: ``latest_revised``) per ``channel_vintage``.  Filled by
    #: :func:`contract_for_channel`; ``None`` for assets that predate the
    #: vintage extension.
    vintage_source: str | None = None
    #: The channel's declared ``transform`` (``raw`` / ``model_versioned`` /
    #: ``formula_versioned``).  A RECORDING dimension, not a deny axis.
    vintage_transform: str | None = None
    #: The channel's declared ``pit_alignment`` (``verified`` / ``proxy``).
    #: A RECORDING dimension; does not gate admission.
    vintage_pit: str | None = None


def contract_for_channel(
    channel: str,
    *,
    data_type: str,
    partition: str,
    extent_column: str | None = None,
    column_contract: str | None = None,
    effective_date_policy: str | None = None,
) -> DataAssetContract:
    """A ``DataAssetContract`` for ``channel`` with its 3-dim vintage filled.

    The vintage fields (``vintage_source`` / ``vintage_transform`` /
    ``vintage_pit``) are drawn from ``channel_vintage.declaration_of(channel)``
    — the T7 curated declaration — so the manifest of a channel carries the
    SAME labels its training admission is judged against.  An undeclared
    channel falls back to the reserved ``"unknown"`` label on each axis
    (recorded in the manifest, never silently omitted), matching the
    deny-by-default policy.
    """
    return DataAssetContract(
        data_type=data_type,
        partition=partition,
        extent_column=extent_column,
        column_contract=column_contract,
        effective_date_policy=effective_date_policy,
        vintage_source=_cv.source_vintage_of(channel),
        vintage_transform=_cv.transform_of(channel),
        vintage_pit=_cv.pit_alignment_of(channel),
    )


def _canonical_dtype(dtype) -> str:
    """A dtype string stable across parquet's harmless round-trip rewrites.

    Parquet does not guarantee to preserve a column's pandas dtype exactly:
    an ``object`` string column may read back as ``str`` (StringDtype), a
    ``str`` column as ``object``, and a ``datetime64[s]`` column may come back
    as ``datetime64[ms]``.  These are representation changes, not value
    changes, so the hash treats string-likes as ``"string"`` and every
    datetime64 resolution as ``"datetime64"`` — a genuine column-type drift
    (string → number) still changes the tag.
    """
    if dtype.kind == "M":
        return "datetime64"
    if dtype.kind == "O" or str(dtype).startswith("string"):
        return "string"
    return str(dtype)


def _content_checksum(df: pd.DataFrame, columns: list[str]) -> str:
    """Deterministic hash of the actual values, stable across parquet round-trip.

    Numeric/bool/datetime columns hash raw bytes where cheap; datetime64 is
    normalized to ``datetime64[ns]`` so a unit rewrite (s→ms) hashes
    identically.  String/object columns hash the joined UTF-8 text
    (``.values.tobytes()`` on object dtype would hash Python pointers, which
    differ between processes), and ``str()`` absorbs the object↔StringDtype
    representation change because both sides produce the same text.
    """
    h = hashlib.sha256()
    for c in columns:
        s = df[c]
        if s.dtype.kind in "biufc":
            h.update(s.to_numpy().tobytes())
        elif s.dtype.kind == "M":
            h.update(s.astype("datetime64[ns]").to_numpy().tobytes())
        else:
            h.update(b"\x00".join(str(v).encode("utf-8") for v in s.tolist()))
    return h.hexdigest()


def schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of columns + dtypes + content, recomputable from disk.

    A parquet whose columns were renamed/dropped, whose dtypes drifted, or
    whose values were edited no longer hashes to the manifest's recorded value
    and :func:`validate_asset_manifest` flags it.  Provenance is deliberately
    NOT part of the hash (unlike the daily store's ``_schema_hash``) because
    auxiliary parquets carry no ``df.attrs``, so the hash must be recomputable
    purely from the file on disk.
    """
    columns = sorted(map(str, df.columns))
    dtypes = [f"{c}:{_canonical_dtype(df[c].dtype)}" for c in columns]
    sig = "|".join([
        "cols=" + ",".join(columns),
        "dtypes=" + ",".join(dtypes),
        "content=" + _content_checksum(df, columns),
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def _extent(df: pd.DataFrame, column: str | None) -> dict:
    """{start, end} iso-date extent of ``column``; empty when not derivable.

    When the named column is absent but the frame carries a DatetimeIndex
    (single-file broadcast assets like industry / market-env whose dates live
    in the index), the index bounds the extent instead — a documented, strictly
    additive fallback that does not change column-based assets' behavior.
    """
    if not column or not len(df):
        return {}
    if column in df.columns:
        values = df[column]
    elif isinstance(df.index, pd.DatetimeIndex):
        values = df.index
    else:
        return {}
    extent = pd.to_datetime(values, errors="coerce").dropna()
    if not len(extent):
        return {}
    return {
        "start": extent.min().strftime("%Y-%m-%d"),
        "end": extent.max().strftime("%Y-%m-%d"),
    }


def asset_manifest_path(parquet_path: str) -> str:
    """Sidecar manifest path for a parquet file."""
    return parquet_path + _ASSET_MANIFEST_SUFFIX


def write_asset_manifest(
    parquet_path: str,
    asset: DataAssetContract,
    df: pd.DataFrame,
    *,
    entity: str | None = None,
    **extra: object,
) -> dict:
    """Atomically write the sidecar manifest for a parquet asset."""
    manifest = {
        "data_type": asset.data_type,
        "partition": asset.partition,
        "entity": entity,
        "rows": int(len(df)),
        "schema_hash": schema_hash(df),
        "source": df.attrs.get("source", "unknown"),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest.update(_extent(df, asset.extent_column))
    if asset.column_contract:
        manifest["column_contract"] = asset.column_contract
    for key in (
        "effective_date_policy",
        "vintage_source",
        "vintage_transform",
        "vintage_pit",
    ):
        value = getattr(asset, key)
        if value is not None:
            manifest[key] = value
    manifest.update(extra)
    atomic_write_json(asset_manifest_path(parquet_path), manifest)
    return manifest


def validate_asset_manifest(
    parquet_path: str,
    asset: DataAssetContract,
    *,
    df: pd.DataFrame | None = None,
) -> dict:
    """Cross-check a parquet file against its sidecar asset manifest.

    Returns a report dict and never raises for a mismatch (callers choose how
    to react)::

        {
          "exists": bool,          # parquet file present?
          "ok": bool,
          "reason": str | None,    # missing manifest / unreadable file
          "mismatches": [str, ...],
          "manifest": dict | None,
          "actual": dict | None,
        }

    ``df`` is an already-loaded frame to avoid a second read; it must be the
    RAW parquet bytes (no dtype/date coercion) so the recomputed schema hash
    matches what the manifest recorded.
    """
    if not os.path.isfile(parquet_path):
        return {"exists": False, "ok": False, "reason": "parquet missing"}
    path = asset_manifest_path(parquet_path)
    if not os.path.isfile(path):
        return {"exists": True, "ok": False,
                "reason": "manifest missing — file predates manifests"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        return {"exists": True, "ok": False, "reason": f"manifest unreadable: {exc}"}
    try:
        actual_df = pd.read_parquet(parquet_path) if df is None else df
    except (KeyError, OSError, ValueError) as exc:
        return {"exists": True, "ok": False, "reason": f"unreadable: {exc}"}
    actual = {"rows": int(len(actual_df)), "schema_hash": schema_hash(actual_df)}
    actual.update(_extent(actual_df, asset.extent_column))
    # manifest-only keys (e.g. start/end when _extent is un-derivable and returns
    # {}) intentionally go unchecked — the rows check covers the practical tamper.
    mismatches = [
        f"{key}: manifest={manifest.get(key)!r} actual={value!r}"
        for key, value in actual.items()
        if manifest.get(key) != value
    ]
    for key in ("data_type", "partition"):
        expected = getattr(asset, key)
        if manifest.get(key) != expected:
            mismatches.append(
                f"{key}: manifest={manifest.get(key)!r} expected={expected!r}"
            )
    for key in (
        "effective_date_policy",
        "vintage_source",
        "vintage_transform",
        "vintage_pit",
    ):
        expected = getattr(asset, key)
        if expected is None:
            continue  # asset predates / does not declare this aspect
        if manifest.get(key) != expected:
            mismatches.append(
                f"{key}: manifest={manifest.get(key)!r} expected={expected!r}"
            )
    return {
        "exists": True,
        "ok": not mismatches,
        "mismatches": mismatches,
        "manifest": manifest,
        "actual": actual,
    }


def check_asset_read(
    parquet_path: str,
    asset: DataAssetContract,
    df: pd.DataFrame,
    *,
    require_valid_manifest: bool = False,
) -> None:
    """Validate a just-loaded parquet against its manifest and react.

    Lenient (default): a manifest-less file is a legacy read (debug log); a
    PRESENT-but-mismatched manifest is a tamper signal (warning log) but the
    data is still returned.  ``require_valid_manifest=True`` raises for a
    missing OR mismatched manifest (formal read).
    """
    if not os.path.isfile(asset_manifest_path(parquet_path)):
        if require_valid_manifest:
            raise ValueError(
                f"refusing to read {parquet_path}: asset manifest missing — "
                f"file predates manifests"
            )
        logger.debug("asset_contract[%s]: %s has no manifest (legacy file)",
                     asset.data_type, parquet_path)
        return
    report = validate_asset_manifest(parquet_path, asset, df=df)
    if report["ok"]:
        return
    reason = report.get("reason") or "; ".join(report["mismatches"])
    if require_valid_manifest:
        raise ValueError(
            f"refusing to read {parquet_path} with require_valid_manifest=True: "
            f"{reason}"
        )
    logger.warning("asset_contract[%s]: manifest mismatch for %s: %s",
                   asset.data_type, parquet_path, reason)


# ── §T8 provider-era fields (no_event vs not_observed) ─────────────────────
#
# A sparse channel's coverage gate must distinguish a stock that GENUINELY had
# no events on a day (``no_event`` — legitimate) from an era we NEVER observed
# (``not_observed`` — a data gap).  The write-end record lives in the asset
# manifest of the channel's GOLD (daily) files:
#
#   provider_available_start/end — the provider era this stock needed, i.e. the
#       downloader manifest's ``effective_start/end`` (the requested window
#       clipped by the caller to listing/delist/latest), degenerating to
#       ``requested_start/end``.
#   retrieved_ranges              — the ACTUAL date ranges fetched inside that
#       era (``[actual_start, actual_end]`` ∩ era, minus ``known_gaps``).
#   known_gaps                    — the downloader manifest's (normalized)
#       ``missing_intervals``.
#
# A stock whose gold manifest carries a provider era is ``era-observable``: a
# calendar day inside ``retrieved_ranges`` counts as OBSERVED whether or not an
# event happened that day — ``no_event`` is not a gap.  A stock with no gold
# manifest / no provider era is ``not_observed`` and is excluded from the
# era-coverage numerator (never counted as zero), reported separately.
#
# ``era_coverage`` is the calendar-day fraction of the provider era covered by
# ``retrieved_ranges`` — a simple, deterministic proxy for "how much of the era
# did we actually look at".  Calendar days (not trading days) keep the helper
# calendar-agnostic; the aggregation to the channel-level gate metric lives in
# the training probe (``train_panel_panel._probe_era_coverage``).


def _iso(value) -> str | None:
    """A value's ISO date, or None when missing / unparseable."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _normalize_ranges(value) -> list[list[str]]:
    """Coerce a range-ish value to a list of ``[start, end]`` ISO-date pairs.

    Accepts ``None`` (→ []), a single ``[start, end]`` pair, or a list of
    pairs; ranges with a missing / unparseable bound are dropped.
    """
    if not value:
        return []
    pairs = (
        value
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple))
        else [value]
    )
    out: list[list[str]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        lo, hi = _iso(pair[0]), _iso(pair[1])
        if lo and hi:
            out.append([lo, hi])
    return out


def _subtract_ranges(
    ranges: list[list[str]], gaps: list[list[str]],
) -> list[list[str]]:
    """Subtract ``gaps`` from ``ranges`` (both ``[start, end]`` ISO lists)."""
    result: list[list[str]] = []
    for lo, hi in ranges:
        segs = [(pd.Timestamp(lo), pd.Timestamp(hi))]
        for glo, ghi in gaps:
            g_lo, g_hi = pd.Timestamp(glo), pd.Timestamp(ghi)
            next_segs: list = []
            for s_lo, s_hi in segs:
                if g_hi < s_lo or g_lo > s_hi:
                    next_segs.append((s_lo, s_hi))
                else:
                    if g_lo > s_lo:
                        next_segs.append((s_lo, g_lo - pd.Timedelta(days=1)))
                    if g_hi < s_hi:
                        next_segs.append((g_hi + pd.Timedelta(days=1), s_hi))
            segs = next_segs
        for s_lo, s_hi in segs:
            result.append([s_lo.strftime("%Y-%m-%d"), s_hi.strftime("%Y-%m-%d")])
    return result


def provider_era_fields(downloader_manifest: dict | None) -> dict:
    """Map a downloader per-stock manifest (§五, ``download_resume.py``) to the
    three asset-manifest provider-era fields (§T8).

    ``provider_available_start/end`` is the provider era this stock needed — the
    manifest's ``effective_start/end`` (the requested window clipped by the
    caller to listing/delist/latest), degenerating to ``requested_start/end``.
    ``retrieved_ranges`` is what was actually fetched inside that era
    (``[actual_start, actual_end]`` ∩ era, minus ``known_gaps``).
    ``known_gaps`` is the normalized ``missing_intervals``.

    Returns ``{}`` when the manifest is missing or carries no provider era — the
    stock is ``not_observed`` (we cannot distinguish "no events that day" from
    "we never observed that era").
    """
    if not downloader_manifest:
        return {}
    win_start = _iso(downloader_manifest.get("effective_start")
                     or downloader_manifest.get("requested_start"))
    win_end = _iso(downloader_manifest.get("effective_end")
                   or downloader_manifest.get("requested_end"))
    if not win_start or not win_end:
        return {}
    gaps = _normalize_ranges(downloader_manifest.get("missing_intervals"))
    fetched: list[list[str]] = []
    actual_start = _iso(downloader_manifest.get("actual_start"))
    actual_end = _iso(downloader_manifest.get("actual_end"))
    if actual_start and actual_end:
        lo = max(actual_start, win_start)  # ISO strings sort chronologically
        hi = min(actual_end, win_end)
        if lo <= hi:
            fetched = [[lo, hi]]
    return {
        "provider_available_start": win_start,
        "provider_available_end": win_end,
        "retrieved_ranges": _subtract_ranges(fetched, gaps),
        "known_gaps": gaps,
    }


def downloader_era_fields(
    raw_dir: str, stock_code: str, cache: dict | None = None,
) -> dict:
    """Provider-era fields for a stock's gold asset manifest.

    Reads the stock's downloader manifest at ``<raw_dir>/.manifests/{code}.json``
    (§五) and maps it via :func:`provider_era_fields`; ``{}`` when there is no
    downloader manifest (the stock was never downloaded → ``not_observed``).
    ``cache`` (a dict, typically per storage save call) avoids re-reading the
    same manifest when one stock is written in many partitions.
    """
    if cache is None:
        cache = {}
    if stock_code not in cache:
        cache[stock_code] = provider_era_fields(
            read_stock_manifest(raw_dir, stock_code))
    return cache[stock_code]


def parse_era_coverage(asset_manifest: dict) -> dict:
    """Parse a gold asset manifest's provider-era fields into an era-coverage
    report (§T8)::

        {
          "provider_available_start": str | None,
          "provider_available_end":   str | None,
          "retrieved_ranges":         [[start, end], ...],
          "known_gaps":               [[start, end], ...],
          "era_covered":              float | None,  # fraction of era days retrieved
          "not_observed":             bool,
        }

    ``era_covered`` is the calendar-day fraction of ``[provider_available_start,
    provider_available_end]`` that ``retrieved_ranges`` covers (clamped to the
    window).  A day inside a retrieved range is OBSERVED even when the stock had
    no event that day — ``no_event`` is legitimate, not a gap.  ``None`` when
    the manifest records no provider era (``not_observed=True``): such a stock is
    excluded from the era-coverage numerator, never counted as zero.
    """
    win_start = asset_manifest.get("provider_available_start")
    win_end = asset_manifest.get("provider_available_end")
    retrieved = _normalize_ranges(asset_manifest.get("retrieved_ranges"))
    report = {
        "provider_available_start": win_start,
        "provider_available_end": win_end,
        "retrieved_ranges": retrieved,
        "known_gaps": _normalize_ranges(asset_manifest.get("known_gaps")),
        "era_covered": None,
        "not_observed": True,
    }
    if not win_start or not win_end:
        return report
    total = (pd.Timestamp(win_end) - pd.Timestamp(win_start)).days + 1
    if total <= 0:
        return report
    covered = 0
    for lo, hi in retrieved:
        lo_d = max(pd.Timestamp(lo), pd.Timestamp(win_start))
        hi_d = min(pd.Timestamp(hi), pd.Timestamp(win_end))
        if lo_d <= hi_d:
            covered += (hi_d - lo_d).days + 1
    report["era_covered"] = min(covered / total, 1.0)
    report["not_observed"] = False
    return report
