"""Text quality filter: HTML stripping, dedup, short/noise text removal.

Operates on per-post DataFrames (Silver layer). All operations are per-row
or within narrow temporal windows — no future-data dependency (PIT-safe).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

_EMOJI_ONLY_RE = re.compile(r"^[\W_]*$", re.UNICODE)
_HTML_RE = re.compile(r"<[^>]*>")


def _clean_text(val):
    """Strip HTML and return cleaned string, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return _HTML_RE.sub("", str(val)).strip()


def _is_noise(text: str) -> bool:
    """True if text is empty or consists solely of non-word characters."""
    if not text:
        return True
    return bool(_EMOJI_ONLY_RE.match(text))


class QualityFilter(PreprocessingStep):
    """Remove low-quality posts: HTML tags, duplicates, noise, short texts.

    Operates on per-post DataFrames with ``title`` and optionally ``body``
    columns.  The dedup scan uses a sliding 3-day window to avoid O(n²)
    while catching near-duplicates that cluster in time.
    """

    def __init__(
        self,
        min_text_length: int = 5,
        max_duplicate_similarity: float = 0.9,
        remove_html: bool = True,
    ):
        self.min_text_length = min_text_length
        self.max_duplicate_similarity = max_duplicate_similarity
        self.remove_html = remove_html

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if df.empty:
            return df.copy()

        df = df.copy()

        # --- 1. Strip HTML ---------------------------------------------------
        if self.remove_html:
            for col in ("title", "body"):
                if col in df.columns:
                    df[col] = df[col].apply(_clean_text)

        # --- 2. Drop pure emoji / symbol rows --------------------------------
        has_title = "title" in df.columns
        has_body = "body" in df.columns
        if has_title or has_body:
            keep = np.ones(len(df), dtype=bool)
            for i in range(len(df)):
                title_ok = not _is_noise(
                    str(df.iloc[i]["title"]) if has_title else ""
                )
                body_ok = not _is_noise(
                    str(df.iloc[i]["body"]) if has_body else ""
                )
                if not title_ok and not body_ok:
                    keep[i] = False
            df = df.loc[keep].copy()

        # --- 3. Drop short texts ---------------------------------------------
        if has_title:
            title_len = df["title"].str.len()
            body_filled = (
                df["body"].fillna("").str.len() > 0
                if has_body
                else pd.Series(False, index=df.index)
            )
            keep = (title_len >= self.min_text_length) | body_filled
            df = df.loc[keep].copy()

        # --- 4. Content-based dedup (sliding 3-day window) -------------------
        date_col = (
            "aligned_date" if "aligned_date" in df.columns
            else "date" if "date" in df.columns
            else None
        )
        if date_col is not None and len(df) > 1:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)

            texts = []
            for i in range(len(df)):
                parts = []
                if has_title:
                    parts.append(str(df.iloc[i]["title"]))
                if has_body:
                    body_val = df.iloc[i]["body"]
                    if not isinstance(body_val, float) or not np.isnan(body_val):
                        parts.append(str(body_val))
                texts.append(" ".join(parts))

            df["_text_key"] = texts
            dates = df[date_col].values

            keep = np.ones(len(df), dtype=bool)
            window_days = np.timedelta64(3, "D")

            for i in range(len(df)):
                if not keep[i]:
                    continue
                # Scan forward within window
                j = i + 1
                while j < len(df) and (dates[j] - dates[i]) <= window_days:
                    if keep[j]:
                        sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
                        if sim > self.max_duplicate_similarity:
                            keep[j] = False
                    j += 1

            df = df.loc[keep].drop(columns=["_text_key"])

        return df.reset_index(drop=True)
