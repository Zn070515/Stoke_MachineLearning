"""Canonical ``datetime64[us]`` date normalization for storage read layers.

pandas 3.0 is strict about datetime64 units on merge keys: ``merge_asof`` (and
in some paths plain ``.merge``) raises ``MergeError`` when a left key is
``datetime64[ms]`` and a right key is ``datetime64[us]``.  On-disk parquets are
a mixture — e.g. daily K-line is largely ``timestamp[ms]`` while aux channels
(industry_ranking, sector_membership, …) are ``timestamp[us]`` — so a
read-then-merge pipeline crashes on the first stock whose daily parquet happens
to be ms.

Every storage read method that feeds a ``.merge(on="date")`` /
``pd.merge_asof(on="date")`` call MUST normalize ``date`` to the canonical
``datetime64[us]`` unit here.  ``astype("datetime64[us]")`` is a lossless
promotion from any coarser unit and preserves NaT.  Files on disk are never
rewritten — the coercion is in-memory only, at the read layer.
"""
import pandas as pd

_CANONICAL_DATE_DTYPE = "datetime64[us]"


def as_date_us(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df["date"]`` to canonical ``datetime64[us]`` in place.

    A frame without a ``date`` column is returned unchanged (no-op), so callers
    may normalize unconditionally without a guard.
    """
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).astype(_CANONICAL_DATE_DTYPE)
    return df
