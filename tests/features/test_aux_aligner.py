"""Integration tests for aux_aligner._load_macro_features (§十三-2).

The macro channel reads through the generation layout first and only falls
back to the legacy flat file when no generation layout exists at all.  A torn
generation layout is refused (GenerationStoreError) in every mode — never
silently masked by a legacy flat file that happens to sit alongside it.
"""
import os

import pandas as pd
import pytest

from stoke_ml.data.generation_store import GenerationStoreError, write_generation
from stoke_ml.features.aux_aligner import _load_macro_features

REL = "a_shares/macro/macro_daily"


def _macro_df(rows=10):
    # Explicit dates (freq=None): pyarrow round-trips the datetime index
    # exactly only when freq is None, so assert_frame_equal stays strict.
    dates = pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(rows)])
    return pd.DataFrame(
        {"rate": [float(i) for i in range(rows)]}, index=pd.Index(dates, name="date"),
    )


def _legacy_path(data_dir):
    return os.path.join(data_dir, "a_shares", "macro", "macro_daily.parquet")


def _write_legacy(data_dir, rows=10):
    os.makedirs(os.path.dirname(_legacy_path(data_dir)), exist_ok=True)
    _macro_df(rows).to_parquet(_legacy_path(data_dir))
    return _legacy_path(data_dir)


def test_valid_generation_returns_df(tmp_path):
    data_dir = str(tmp_path)
    df = _macro_df()
    write_generation(
        data_dir, REL, df,
        {"dataset": "macro_daily", "rows": len(df), "columns": list(df.columns)},
    )
    pd.testing.assert_frame_equal(_load_macro_features(data_dir), df)


def test_legacy_fallback_when_no_generation(tmp_path):
    data_dir = str(tmp_path)
    df = _macro_df()
    _write_legacy(data_dir, rows=len(df))
    pd.testing.assert_frame_equal(_load_macro_features(data_dir), df)


def test_torn_generation_refused_even_with_legacy(tmp_path):
    """The generation layout is authoritative once present: a torn gen (CURRENT
    pointing at a nonexistent dir) must raise, not silently fall back to a
    legacy flat file sitting alongside it."""
    data_dir = str(tmp_path)
    _write_legacy(data_dir)
    gen_root = os.path.join(data_dir, "a_shares", "macro", "macro_daily_gen")
    os.makedirs(gen_root, exist_ok=True)
    with open(os.path.join(gen_root, "CURRENT"), "w", encoding="utf-8") as f:
        f.write("gen_00000001")
    with pytest.raises(GenerationStoreError):
        _load_macro_features(data_dir)


def test_neither_returns_none(tmp_path):
    assert _load_macro_features(str(tmp_path)) is None
