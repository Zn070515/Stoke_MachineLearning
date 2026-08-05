"""CLI stock-code argument normalization (§九-2).

Every download script accepts a comma-separated ``--stocks``/``--stock_list``
argument whose items may arrive exchange-prefixed (``SH600001``), suffix-suffixed
(``600001.SH``), numeric, or bare.  Each script used to pass those items straight
into fetch/storage, so a non-canonical key drifted from the canonical key that
``stoke_ml.data.codes`` normalizes everywhere else (Run Manifest / requested
universe / filters / storage).  This single helper routes every CLI entry
through :func:`normalize_stock_code` so the canonical six-digit code is identical
regardless of how the user typed it.
"""
from __future__ import annotations

from stoke_ml.data.codes import normalize_stock_code


def parse_stock_codes_arg(raw: str | None) -> list[str]:
    """Normalize a comma-separated ``--stocks`` value to canonical six-digit codes.

    Splits on commas, normalizes each item via ``normalize_stock_code`` (dropping
    unusable/illegal entries such as ``"abc"``), then dedupes and sorts.  ``None``
    and empty input return an empty list.  The returned list is the canonical
    requested-universe the script should fetch, filter and record.
    """
    if not raw:
        return []
    codes: set[str] = set()
    for item in raw.split(","):
        norm = normalize_stock_code(item)
        if norm is not None:
            codes.add(norm)
    return sorted(codes)
