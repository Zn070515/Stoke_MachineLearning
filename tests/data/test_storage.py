"""DataStorage partition merge / flat-shadowing tests (v6 §七).

save_daily must be NON-destructive: read the existing month partition, merge by
date, dedup, sort, then atomically replace.  load_daily must never let a stale
flat ``daily/{code}.parquet`` shadow a fresher partitioned increment — the
partition wins per-date while the flat still supplies dates partitions lack.
"""
import os
import time

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.storage import DataStorage, _acquire_lock, _release_lock


def _frame(dates, closes=None, code="000001"):
    dates = list(pd.to_datetime(dates))
    n = len(dates)
    if closes is None:
        closes = [10.0 + 0.1 * i for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": [float(c) for c in closes],
        "high": [float(c) + 0.5 for c in closes],
        "low": [float(c) - 0.5 for c in closes],
        "close": [float(c) for c in closes],
        "volume": [1e6] * n,
        "amount": [1e8] * n,
        "stock_code": code,
    })


def _write_flat(tmp_path, code, dates, closes):
    """Simulate the legacy flat file, with an mtime older than any partition."""
    base = os.path.join(str(tmp_path), "a_shares", "daily")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{code}.parquet")
    _frame(dates, closes=closes, code=code).to_parquet(path, index=False)
    os.utime(path, (time.time() - 3600.0,) * 2)


class TestSaveDaily:
    def test_incremental_merge_no_overwrite(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06", "2024-01-07"]))
        store.save_daily(_frame(["2024-01-07", "2024-01-08"]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["date"].tolist() == pd.to_datetime(
            ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]
        ).tolist()

    def test_same_date_keeps_last_write(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05"], closes=[10.0]))
        store.save_daily(_frame(["2024-01-05"], closes=[99.0]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert len(out) == 1
        assert out["close"].iloc[0] == pytest.approx(99.0)

    def test_no_tmp_residue(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        store.save_daily(_frame(["2024-01-06", "2024-01-07"]))
        residue = []
        for root, _, files in os.walk(str(tmp_path)):
            residue += [f for f in files if ".tmp" in f]
        assert residue == []

    def test_multi_stock_multi_month(self, tmp_path):
        store = DataStorage(str(tmp_path))
        df = pd.concat([
            _frame(["2023-12-29", "2023-12-30"], code="000001"),
            _frame(["2024-01-02"], code="000001"),
            _frame(["2024-01-02"], code="600519"),
        ])
        store.save_daily(df)
        a = store.load_daily("000001", "2023-12-01", "2024-01-31")
        assert len(a) == 3
        b = store.load_daily("600519", "2023-12-01", "2024-01-31")
        assert len(b) == 1


class TestLoadDaily:
    def test_flat_fast_path_when_flat_fresh(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06"], closes=[10.0, 11.0])
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([10.0, 11.0])

    def test_partition_wins_over_stale_flat(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06"], closes=[10.0, 11.0])
        store = DataStorage(str(tmp_path))
        # fresh partition increment lands after the stale flat file
        store.save_daily(_frame(
            ["2024-01-05", "2024-01-06", "2024-01-07"], closes=[20.0, 21.0, 22.0]
        ))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([20.0, 21.0, 22.0])

    def test_flat_supplies_dates_partitions_lack(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06", "2024-01-07"],
                    closes=[10.0, 11.0, 12.0])
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-07"], closes=[99.0]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([10.0, 11.0, 99.0])

    def test_date_range_filter(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06", "2024-01-07"],
                    closes=[10.0, 11.0, 12.0])
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-06", "2024-01-06")
        assert len(out) == 1
        assert out["date"].iloc[0] == pd.Timestamp("2024-01-06")

    def test_empty_when_nothing_present(self, tmp_path):
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-01", "2024-01-31")
        assert out.empty

    def test_partition_only_when_no_flat(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["date"].tolist() == pd.to_datetime(["2024-01-05", "2024-01-06"]).tolist()


class TestPartitionLock:
    def test_acquire_release(self, tmp_path):
        lock = os.path.join(str(tmp_path), "x.lock")
        _acquire_lock(lock)
        assert os.path.exists(lock)
        _release_lock(lock)
        assert not os.path.exists(lock)

    def test_second_acquire_times_out(self, tmp_path, monkeypatch):
        monkeypatch.setattr("stoke_ml.data.storage._LOCK_TIMEOUT", 0.1)
        lock = os.path.join(str(tmp_path), "x.lock")
        _acquire_lock(lock)
        with pytest.raises(TimeoutError):
            _acquire_lock(lock)
        _release_lock(lock)

    def test_stale_lock_recovered(self, tmp_path, monkeypatch):
        monkeypatch.setattr("stoke_ml.data.storage._LOCK_STALE", 0.05)
        lock = os.path.join(str(tmp_path), "x.lock")
        with open(lock, "w", encoding="utf-8") as f:
            f.write("99999")
        os.utime(lock, (1.0, 1.0))  # crashed-writer leftover from 1970
        _acquire_lock(lock)
        assert os.path.exists(lock)
        _release_lock(lock)
