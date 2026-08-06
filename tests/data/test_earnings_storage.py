"""EarningsStorage snapshot-load error aggregation tests (§T18).

A corrupt snapshot parquet must be SKIPPED (continue), but the failure must be
counted and reported through the ErrorSummary — not silently swallowed.
"""
import os

import pandas as pd

from stoke_ml.data.earnings_storage import EarningsStorage


def _storage(tmp_path):
    a_shares = tmp_path / "a_shares"
    (a_shares / "earnings").mkdir(parents=True, exist_ok=True)
    return EarningsStorage(str(tmp_path))


def test_load_snapshot_corrupt_file_skipped_and_reported(tmp_path, caplog):
    """A corrupt snapshot is skipped (continue) but aggregated into the
    ErrorSummary, so the failure is visible rather than silently dropped."""
    es = _storage(tmp_path)
    (tmp_path / "a_shares" / "earnings" / "forecasts.parquet").write_bytes(
        b"not a parquet"
    )
    # A valid-but-empty second snapshot keeps the loop path exercised.
    pd.DataFrame(columns=["stock_code", "announce_date"]).to_parquet(
        tmp_path / "a_shares" / "earnings" / "express.parquet", index=False
    )

    with caplog.at_level("WARNING", logger="stoke_ml.data.earnings_storage"):
        snap = es._load_snapshot()

    assert snap.empty  # continue semantics preserved — corrupt file skipped
    assert any("Error summary" in r.getMessage() for r in caplog.records)
    assert any("earnings_snapshot" in r.getMessage() for r in caplog.records)


def test_load_snapshot_all_missing_returns_empty(tmp_path):
    """No snapshot files at all → empty frame, no error summary emitted."""
    es = _storage(tmp_path)
    snap = es._load_snapshot()
    assert snap.empty


def test_load_snapshot_valid_rows_survive(tmp_path):
    """A valid snapshot is normalized into the common schema."""
    es = _storage(tmp_path)
    pd.DataFrame({
        "stock_code": ["000001"],
        "announce_date": ["2024-01-05"],
        "net_profit_yoy": [12.5],
        "net_profit": [1000.0],
    }).to_parquet(tmp_path / "a_shares" / "earnings" / "forecasts.parquet",
                  index=False)
    snap = es._load_snapshot()
    assert not snap.empty
    assert snap["stock_code"].iloc[0] == "000001"
