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
per-file lock + content-validation layer (``generation_store.py``, §十三-2) and
the daily store itself untouched:

* :class:`AtomicCommit` — write a file via temp + ``os.replace``, removing the
  temp on failure, so a reader never observes a torn file and a failed write
  leaves the prior file untouched.
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
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

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


@dataclass(frozen=True)
class DataAssetContract:
    """File-level description of one auxiliary data asset.

    Complements the column-schema :class:`stoke_ml.data.contract.DataContract`
    (what columns a dataset must carry) with the FILE-level facts a manifest
    pins: rows, primary-key extent, schema hash, source, written_at.
    """

    data_type: str
    partition: str
    #: Column whose values bound the manifest's start/end (e.g. "date",
    #: "report_date").  None when the asset has no meaningful extent.
    extent_column: str | None = None
    #: Name of a column-schema contract in ``contract.py`` governing this
    #: asset.  Recorded in the manifest for provenance; schema enforcement is
    #: a separate gate and is deliberately NOT applied here.
    column_contract: str | None = None


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
    """{start, end} iso-date extent of ``column``; empty when not derivable."""
    if not column or column not in df.columns or not len(df):
        return {}
    extent = pd.to_datetime(df[column], errors="coerce").dropna()
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
