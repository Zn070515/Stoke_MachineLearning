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

# BSE (北交所) equity prefixes — shared by market_of_code and
# a_share_equity_segment so the provider router and the equity filter cannot
# drift apart (§六).  Real BSE ranges: 43xxxx (新三板 精选层), 83xxxx/87xxxx/
# 88xxxx (旧挂牌), 920xxx (2024+ 新号段).
_BJ_PREFIXES = ("43", "83", "87", "88", "920")


class UnsupportedMarketError(ValueError):
    """Raised when a provider is asked to route a code it cannot serve.

    The code is either not an A-share common equity code (``market_of_code``
    returned ``None``) or is in a market the provider deliberately does not
    support.  Providers MUST NOT translate such a code into a guessed
    SH/SZ/BJ request — that is exactly how ``920001`` used to leak onto the
    Shenzhen exchange as a bogus ``.SZ`` / ``sz920001`` symbol.
    """


def market_of_code(code6: str) -> str | None:
    """Return the exchange market of a normalized 6-digit code.

    THE single authoritative market router for A-share providers (§六).
    Every provider derives its exchange prefix from this — never from its own
    leading-digit heuristic.  Caliber:
      * SH 上海: ``6xxxxx`` (600/601/603/605/688/689 主板 / 科创板)
      * SZ 深圳: ``0xxxxx`` / ``3xxxxx`` (000/001/002/003/300/301/302 主板 / 创业板)
      * BJ 北交所: ``43 / 83 / 87 / 88 / 920``
    Returns ``None`` for anything else (indices, B-shares, funds, ...).

    Consistent with :func:`a_share_equity_segment` on every A-share common
    equity prefix; only wider on the ``3xxxxx`` range (e.g. ``399001``
    深证成指 routes SZ here but is not equity there).

    ``code6`` must already be a normalized 6-digit code — callers normalize
    via :func:`normalize_stock_code` first.
    """
    if code6.startswith("6"):
        return "SH"
    if code6.startswith(("0", "3")):
        return "SZ"
    if code6.startswith(_BJ_PREFIXES):
        return "BJ"
    return None


def _code_market(code6: str) -> str | None:
    """Infer the exchange market from the leading digits of a 6-digit code.

    Thin shim over :func:`market_of_code` used by :func:`normalize_stock_code`
    to reject an exchange prefix/suffix that contradicts the code.  Keeps the
    legacy broad ``4xxxxx / 8xxxxx / 92xxxx → BJ`` catch (e.g. 老三板 400xxx)
    that the strict equity router deliberately does not claim, so the format
    check keeps accepting historical 老三板 spellings.
    """
    market = market_of_code(code6)
    if market is not None:
        return market
    if code6[0] in ("4", "8") or code6.startswith("92"):
        return "BJ"
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
    """True iff ``value`` normalizes to a legal code.

    This is the FORMAT layer only (§十): any plausible six-digit code in the
    A-share range passes, including non-equity instruments (100xxx indices,
    200xxx/900xxx B-shares, 500xxx funds, ...).  Use
    :func:`is_a_share_equity_code` when the canonical daily K-line store needs
    the A-share common-equity layer on top — a format-legal code is not
    automatically a stock.
    """
    return normalize_stock_code(value) is not None


def a_share_equity_segment(code6: str) -> str | None:
    """Return the A-share common-equity segment of a normalized 6-digit code,
    or ``None`` if it is not an A-share common stock.

    Prefix-based classification (§十), the practical filter for a daily K-line
    store:
      * SH 主板 / 科创板: ``600 / 601 / 603 / 605 / 688 / 689``
      * SZ 主板 / 创业板: ``000 / 001 / 002 / 003 / 300 / 301 / 302``
      * BJ 北交所:        ``43 / 83 / 87 / 88 / 920``
    Everything else — 100xxx indices, 200xxx (SZ B股), 500xxx funds,
    900xxx (SH B股), the 000300/399xxx index ranges that fall outside the
    equity prefixes, etc. — returns ``None``.

    ``code6`` must already be a normalized 6-digit code (see
    :func:`normalize_stock_code`); callers normalize first.
    """
    if code6.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code6.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "SZ"
    if code6.startswith(_BJ_PREFIXES):
        return "BJ"
    return None


def is_a_share_equity_code(value) -> bool:
    """True iff ``value`` is an A-share common-stock code (two-layer test).

    Layer 1 — format: :func:`normalize_stock_code` accepts any legal six-digit
    code in the plausible A-share range.  Layer 2 — equity filter:
    :func:`a_share_equity_segment` must classify the code as SH/SZ/BJ common
    equity.  Use this gate on the canonical daily K-line store, which only ever
    holds common-equity series (100xxx indices, 200xxx/900xxx B-shares,
    500xxx funds are format-legal but not stocks and are rejected here).
    """
    code = normalize_stock_code(value)
    if code is None:
        return False
    return a_share_equity_segment(code) is not None
