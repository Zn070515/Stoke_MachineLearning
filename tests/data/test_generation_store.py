"""Tests for the generation-directory + CURRENT-pointer atomic aux-data write (§十三-2).

data.parquet + manifest.json are NOT a single atomic object — a crash between
the two ``os.replace`` calls would leave a torn active pair.  A generation
directory plus a CURRENT pointer makes the pair switch as one unit: a partial
next generation is never followed because CURRENT is flipped last, only once
both files are in place.
"""
import json
import os

import pandas as pd
import pytest

from stoke_ml.data.generation_store import (
    GenerationStoreError,
    read_generation,
    write_generation,
)

REL = "a_shares/macro/macro_daily"


def _gen_root(data_dir):
    return os.path.join(data_dir, "a_shares", "macro", "macro_daily_gen")


def _macro_df(rows=10):
    # Explicit dates (freq=None): pyarrow round-trips the datetime index
    # exactly only when freq is None, so assert_frame_equal stays strict.
    dates = pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(rows)])
    return pd.DataFrame(
        {"rate": [float(i) for i in range(rows)]}, index=pd.Index(dates, name="date"),
    )


def _manifest(**over):
    m = {"dataset": "macro_daily", "rows": 10, "columns": ["rate"], "source": "akshare"}
    m.update(over)
    return m


def test_roundtrip(tmp_path):
    df = _macro_df()
    write_generation(str(tmp_path), REL, df, _manifest())
    out = read_generation(str(tmp_path), REL)
    pd.testing.assert_frame_equal(out, df)


def test_no_generation_layout_returns_none(tmp_path):
    assert read_generation(str(tmp_path), REL) is None


def test_write_twice_current_points_to_second(tmp_path):
    data_dir = str(tmp_path)
    df1 = _macro_df(rows=5)
    df2 = _macro_df(rows=9)
    g1 = write_generation(data_dir, REL, df1, _manifest(rows=5))
    g2 = write_generation(data_dir, REL, df2, _manifest(rows=9))
    assert g1 == "gen_00000001"
    assert g2 == "gen_00000002"
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), encoding="utf-8") as f:
        assert f.read().strip() == g2
    # Both generations remain on disk, each with data.parquet + manifest.json.
    for name in (g1, g2):
        for fname in ("data.parquet", "manifest.json"):
            assert os.path.isfile(
                os.path.join(_gen_root(data_dir), name, fname)
            )
    pd.testing.assert_frame_equal(
        read_generation(data_dir, REL), df2,
    )


def test_manifest_generation_key_stamped(tmp_path):
    data_dir = str(tmp_path)
    gen = write_generation(data_dir, REL, _macro_df(), _manifest())
    with open(
        os.path.join(_gen_root(data_dir), gen, "manifest.json"),
        encoding="utf-8",
    ) as f:
        m = json.load(f)
    assert m["generation"] == gen
    assert m["dataset"] == "macro_daily"
    assert m["columns"] == ["rate"]


def test_two_phase_crash_isolated(tmp_path):
    """Crash mid-write of gen_2 (data.parquet landed, manifest did not, CURRENT
    never flipped) must not affect reads: CURRENT still points at the complete
    gen_1, so the partial gen_2 is invisible."""
    data_dir = str(tmp_path)
    df1 = _macro_df(rows=5)
    write_generation(data_dir, REL, df1, _manifest(rows=5))
    gen2 = os.path.join(_gen_root(data_dir), "gen_00000002")
    os.makedirs(gen2, exist_ok=True)
    _macro_df(rows=3).to_parquet(os.path.join(gen2, "data.parquet"))
    pd.testing.assert_frame_equal(
        read_generation(data_dir, REL), df1,
    )


def test_torn_missing_current_raises(tmp_path):
    """gen_root exists but CURRENT was never flipped -> half-initialized."""
    data_dir = str(tmp_path)
    os.makedirs(os.path.join(_gen_root(data_dir), "gen_00000001"), exist_ok=True)
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)


def test_torn_invalid_current_raises(tmp_path):
    data_dir = str(tmp_path)
    os.makedirs(_gen_root(data_dir), exist_ok=True)
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), "w", encoding="utf-8") as f:
        f.write("not-a-gen")
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)


def test_torn_current_non_digit_suffix_raises(tmp_path):
    """Regression: 'gen_' is 4 chars, so the 8-digit suffix must start at index
    4 — a non-digit at index 4 (e.g. 'gen_a00000001') must be rejected as an
    invalid name, not slip through to the missing-dir check."""
    data_dir = str(tmp_path)
    os.makedirs(_gen_root(data_dir), exist_ok=True)
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), "w", encoding="utf-8") as f:
        f.write("gen_a00000001")
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)


def test_torn_current_missing_gen_dir_raises(tmp_path):
    data_dir = str(tmp_path)
    os.makedirs(_gen_root(data_dir), exist_ok=True)
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), "w", encoding="utf-8") as f:
        f.write("gen_00000001")
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)


def test_torn_gen_missing_data_raises(tmp_path):
    """The old flat+manifest torn pair mapped into a generation: manifest.json
    present but data.parquet absent."""
    data_dir = str(tmp_path)
    gen = os.path.join(_gen_root(data_dir), "gen_00000001")
    os.makedirs(gen, exist_ok=True)
    with open(os.path.join(gen, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generation": "gen_00000001"}, f)
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), "w", encoding="utf-8") as f:
        f.write("gen_00000001")
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)


def test_torn_gen_missing_manifest_raises(tmp_path):
    data_dir = str(tmp_path)
    gen = os.path.join(_gen_root(data_dir), "gen_00000001")
    os.makedirs(gen, exist_ok=True)
    _macro_df().to_parquet(os.path.join(gen, "data.parquet"))
    with open(os.path.join(_gen_root(data_dir), "CURRENT"), "w", encoding="utf-8") as f:
        f.write("gen_00000001")
    with pytest.raises(GenerationStoreError):
        read_generation(data_dir, REL)
