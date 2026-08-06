"""T10 tests: generation-store single-writer lock + content validation (§十三).

write_generation now runs the whole read-modify-write (generation numbering +
parquet + manifest + CURRENT flip) under a single-writer lock keyed on the
generation root, and stamps ``schema_hash`` computed over the FULL content
including the index.  read_generation validates the active parquet against that
hash and refuses a tampered generation with :class:`GenerationStoreError`;
a legacy manifest (no schema_hash) still reads (with a warning).

The macro frame is INDEX-indexed content (a DatetimeIndex named "date" is the
primary key), so the index tamper test proves the reset_index() canonicalization
really covers the index values, not just the columns.
"""
import json
import logging
import os

import pandas as pd
import pytest

from stoke_ml.data.asset_contract import acquire_lock, release_lock
from stoke_ml.data.generation_store import (
    CURRENT_NAME,
    GenerationStoreError,
    read_generation,
    write_generation,
)

REL = "a_shares/macro/macro_daily"


def _gen_root(data_dir):
    return os.path.join(data_dir, "a_shares", "macro", "macro_daily_gen")


def _macro_df(rows=10):
    # Explicit dates (freq=None): pyarrow round-trips the datetime index exactly
    # only when freq is None, so assert_frame_equal stays strict.
    dates = pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(rows)])
    return pd.DataFrame(
        {"rate": [float(i) for i in range(rows)]}, index=pd.Index(dates, name="date"),
    )


def _manifest(**over):
    m = {"dataset": "macro_daily", "rows": 10, "columns": ["rate"], "source": "akshare"}
    m.update(over)
    return m


def _active_gen_dir(data_dir):
    with open(os.path.join(_gen_root(data_dir), CURRENT_NAME), encoding="utf-8") as f:
        gen_name = f.read().strip()
    return os.path.join(_gen_root(data_dir), gen_name)


def _rewrite_manifest(data_dir, mutator):
    gen_dir = _active_gen_dir(data_dir)
    manifest_path = os.path.join(gen_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    mutator(manifest)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def test_roundtrip_exact(tmp_path):
    df = _macro_df()
    write_generation(str(tmp_path), REL, df, _manifest())
    out = read_generation(str(tmp_path), REL)
    pd.testing.assert_frame_equal(out, df)


def test_concurrent_write_refused(tmp_path):
    """Another writer holding the generation-root lock must be refused — no new
    generation dir, no CURRENT flip, no partial write."""
    data_dir = str(tmp_path)
    gen_root = _gen_root(data_dir)
    os.makedirs(gen_root, exist_ok=True)
    lock_dir = acquire_lock(gen_root)
    try:
        with pytest.raises(GenerationStoreError, match="lock"):
            write_generation(data_dir, REL, _macro_df(), _manifest(), lock_timeout=0.2)
        names = os.listdir(gen_root)
        assert not any(n.startswith("gen_") for n in names)
        assert not os.path.exists(os.path.join(gen_root, CURRENT_NAME))
    finally:
        release_lock(lock_dir)


def test_tampered_manifest_schema_hash_raises(tmp_path):
    data_dir = str(tmp_path)
    write_generation(data_dir, REL, _macro_df(), _manifest())
    _rewrite_manifest(data_dir, lambda m: m.update(schema_hash="deadbeef"))
    with pytest.raises(GenerationStoreError, match="schema_hash"):
        read_generation(data_dir, REL)


def test_tampered_data_value_raises(tmp_path):
    """Rewrite the active parquet with an edited value byte (equivalent to an
    in-place corruption): the recomputed schema_hash no longer matches."""
    data_dir = str(tmp_path)
    df = _macro_df()
    write_generation(data_dir, REL, df, _manifest())
    tampered = df.copy()
    tampered.loc[tampered.index[0], "rate"] += 1.0
    tampered.to_parquet(os.path.join(_active_gen_dir(data_dir), "data.parquet"))
    with pytest.raises(GenerationStoreError, match="schema_hash"):
        read_generation(data_dir, REL)


def test_tampered_index_detected(tmp_path):
    """The DatetimeIndex is primary-key content; shifting a date must be caught
    — proving the reset_index() canonicalization covers index values."""
    data_dir = str(tmp_path)
    write_generation(data_dir, REL, _macro_df(), _manifest())
    shifted = _macro_df()
    shifted.index = pd.to_datetime(
        [f"2024-01-{i + 2:02d}" for i in range(len(shifted))]  # +1 day
    )
    shifted.to_parquet(os.path.join(_active_gen_dir(data_dir), "data.parquet"))
    with pytest.raises(GenerationStoreError, match="schema_hash"):
        read_generation(data_dir, REL)


def test_legacy_unhashed_manifest_proceeds(tmp_path, caplog):
    """A generation written before T10 has no schema_hash; read_generation
    cannot verify it and must proceed (with a warning), not raise."""
    data_dir = str(tmp_path)
    df = _macro_df()
    write_generation(data_dir, REL, df, _manifest())
    _rewrite_manifest(data_dir, lambda m: m.pop("schema_hash", None))
    with caplog.at_level(logging.WARNING, logger="stoke_ml.data.generation_store"):
        out = read_generation(data_dir, REL)
    pd.testing.assert_frame_equal(out, df)
    assert any("schema_hash" in r.getMessage() for r in caplog.records)


def test_tampered_index_detected_when_name_collides(tmp_path):
    """Regression: when the index name collides with a column name, the index
    values must STILL be hashed.  The pre-review fallback (hash without the
    index on reset_index ValueError) let a tampered index pass validation —
    exactly the guarantee T10 exists to give."""
    data_dir = str(tmp_path)
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03", "2024-01-04"])},
        index=pd.Index(dates, name="date"),  # index name collides with column
    )
    write_generation(data_dir, REL, df, {"dataset": "macro_daily", "rows": 2})
    # Tamper ONLY the index values (shift the dates), leave the column intact.
    tampered = df.copy()
    tampered.index = pd.to_datetime(["2024-02-01", "2024-02-02"])
    tampered.to_parquet(os.path.join(_active_gen_dir(data_dir), "data.parquet"))
    with pytest.raises(GenerationStoreError, match="schema_hash"):
        read_generation(data_dir, REL)


def test_refused_first_write_leaves_no_gen_dir(tmp_path):
    """A refused first write must not leave a ``..._gen/`` dir behind — that
    would flip read_generation from None (legacy fallback) to a CURRENT-missing
    GenerationStoreError for aux_helpers._load_macro_features."""
    data_dir = str(tmp_path)
    gen_root = _gen_root(data_dir)
    os.makedirs(os.path.dirname(gen_root), exist_ok=True)  # only the PARENT
    lock_dir = acquire_lock(gen_root)
    try:
        with pytest.raises(GenerationStoreError, match="lock"):
            write_generation(data_dir, REL, _macro_df(), _manifest(), lock_timeout=0.2)
        # gen_root itself must not exist → read_generation still returns None.
        assert not os.path.isdir(gen_root)
        assert read_generation(data_dir, REL) is None
    finally:
        release_lock(lock_dir)
