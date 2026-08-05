"""Canonical A-share stock-code sanitizer (§八-1).

A code read back from Parquet as float (``600001.0``), exchange-prefixed
(``SH600001`` / ``600001.SH``), numeric (600001), or a zero-padded string must
all collapse to the same ``"600001"``.  A bare ``str(code).zfill(6)`` turns the
float ``600001.0`` into the illegal ``"600001.0"`` and ``1.0`` into ``"0001.0"``
— never a legal six-digit code.  Every source, Universe, Storage and Panel code
path normalizes through this single function so the canonical key is identical
everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EXCHANGE_PREFIXES = ("SH", "SZ", "BJ", "sh", "sz", "bj")
_EXCHANGE_SUFFIXES = (".SH", ".SZ", ".BJ", ".sh", ".sz", ".bj")


def normalize_stock_code(value) -> str | None:
    """Return the canonical zero-padded 6-digit code, or ``None`` if unusable.

    Handles int, float (``600001.0``), ``"600001.0"``, ``"SH600001"``,
    ``"600001.SH"``, empty values and illegal characters.  ``None``/NaN/blank
    and non-numeric garbage are not a legal code — the caller drops or treats
    them as missing rather than propagating a corrupted key.
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return None
    if isinstance(value, float):
        if not np.isfinite(value) or value != int(value):
            return None
        return f"{int(value):06d}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):06d}"
    s = str(value).strip()
    if not s:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if s[:2] in _EXCHANGE_PREFIXES:
        s = s[2:]
    elif s[-3:] in _EXCHANGE_SUFFIXES:
        s = s[:-3]
    if not s.isdigit():
        return None
    return s.zfill(6)


def normalize_stock_code_series(s) -> pd.Series:
    """Vectorized column normalization.

    Dtype-agnostic: a float/object ``stock_code`` column of raw Parquet/API
    values maps to canonical 6-digit codes (``None`` where unusable).
    """
    return s.map(normalize_stock_code)


def is_valid_stock_code(value) -> bool:
    """True iff ``value`` normalizes to a legal code."""
    return normalize_stock_code(value) is not None
