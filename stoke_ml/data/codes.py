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

import re

import numpy as np
import pandas as pd

_EXCHANGE_PREFIXES = ("SH", "SZ", "BJ", "sh", "sz", "bj")
_EXCHANGE_SUFFIXES = (".SH", ".SZ", ".BJ", ".sh", ".sz", ".bj")


def _code_market(code6: str) -> str | None:
    """Infer the exchange market from the leading digits of a 6-digit code."""
    if code6[0] == "6":
        return "SH"          # 600/601/603/605/688/689 Shanghai
    if code6[0] in ("0", "3"):
        return "SZ"          # 000/001/002/003/300/301 Shenzhen
    if code6[0] in ("4", "8") or code6.startswith("92"):
        return "BJ"          # 4xxxxx / 8xxxxx / 920xxx Beijing BSE
    return None


def normalize_stock_code(value) -> str | None:
    """Return the canonical zero-padded 6-digit code, or ``None`` if unusable.

    Handles int, float (``600001.0``), ``"600001.0"``, ``"SH600001"``,
    ``"600001.SH"``, empty values and illegal characters.  ``None``/NaN/blank
    and non-numeric garbage are not a legal code — the caller drops or treats
    them as missing rather than propagating a corrupted key.

    §六 (strict): a legal code must satisfy ``re.fullmatch(r"\\d{6}", code)``
    and be in the plausible A-share range — so negative, ``> 999999``,
    ``000000``, non-integer fractions, illegal characters, and an exchange
    prefix/suffix that contradicts the market implied by the code's leading
    digits are all rejected.
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return None
    if isinstance(value, float):
        # Reject NaN/±inf and non-integer-valued floats (600001.5 is not a code).
        if not np.isfinite(value) or value != int(value):
            return None
        value = int(value)
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if v <= 0 or v > 999999:
            return None
        return f"{v:06d}"
    # String path — strip whitespace, then exchange prefix/suffix BEFORE the
    # integer-.0 cleanup, so "sh600001.0" resolves to "600001" (§六).
    s = str(value).strip()
    if not s:
        return None
    prefix: str | None = None
    if s[:2] in _EXCHANGE_PREFIXES:
        prefix = s[:2].upper()
        s = s[2:]
    elif s[-3:] in _EXCHANGE_SUFFIXES:
        prefix = s[-2:].upper()
        s = s[:-3]
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if not s.isdigit():
        return None
    code = s.zfill(6)
    if len(code) != 6 or code == "000000":
        return None
    market = _code_market(code)
    if prefix is not None and market is not None and prefix != market:
        return None
    return code


def normalize_stock_code_series(s) -> pd.Series:
    """Vectorized column normalization.

    Dtype-agnostic: a float/object ``stock_code`` column of raw Parquet/API
    values maps to canonical 6-digit codes (``None`` where unusable).
    """
    return s.map(normalize_stock_code)


def is_valid_stock_code(value) -> bool:
    """True iff ``value`` normalizes to a legal code."""
    return normalize_stock_code(value) is not None
