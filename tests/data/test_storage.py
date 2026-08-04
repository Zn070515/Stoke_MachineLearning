"""DataStorage flat-canonical tests (review v7 §五 + v8 §二-1).

The canonical store is a single flat ``daily/{code}.parquet`` per stock.
``save_daily`` is NON-destructive: read existing flat, merge by date, dedup,
sort, atomically replace under a per-file lock.  ``load_daily`` reads only the
flat file; legacy year/month partition directories on disk are ignored (never
read, never written).  ``list_stocks`` discovers codes from flat files only.

Each parquet carries a sidecar ``daily/{code}.manifest.json`` (review v8 §二-1)
pinning stock / start / end / rows / source / adjust / schema hash so "file
exists" never silently implies "data complete".
"""
import os

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
    base = os.path.join(str(tmp_path), "a_shares", "daily")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{code}.parquet")
    _frame(dates, closes=closes, code=code).to_parquet(path, index=False)


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
    def test_reads_flat_file(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06"], closes=[10.0, 11.0])
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([10.0, 11.0])

    def test_merge_appends_to_existing_flat(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06"], closes=[10.0, 11.0])
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-07"], closes=[12.0]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([10.0, 11.0, 12.0])

    def test_date_range_filter(self, tmp_path):
        _write_flat(tmp_path, "000001", ["2024-01-05", "2024-01-06", "2024-01-07"],
                    closes=[10.0, 11.0, 12.0])
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-06", "2024-01-06")
        assert len(out) == 1
        assert out["date"].iloc[0] == pd.Timestamp("2024-01-06")

    def test_empty_when_nothing_present(self, tmp_path):
        out = DataStorage(str(tmp_path)).load_daily("000001", "2024-01-01", "2024-01-31")
        assert out.empty

    def test_save_load_roundtrip(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["date"].tolist() == pd.to_datetime(["2024-01-05", "2024-01-06"]).tolist()

    def test_stale_partition_dirs_ignored(self, tmp_path):
        # Legacy year/month partition dirs must never be read or shadow the flat file
        legacy = os.path.join(str(tmp_path), "a_shares", "daily", "2024", "01")
        os.makedirs(legacy, exist_ok=True)
        _frame(["2024-01-05"], closes=[42.0]).to_parquet(
            os.path.join(legacy, "000001.parquet"), index=False
        )
        store = DataStorage(str(tmp_path))
        assert store.load_daily("000001", "2024-01-01", "2024-01-31").empty
        store.save_daily(_frame(["2024-01-06"], closes=[11.0]))
        out = store.load_daily("000001", "2024-01-01", "2024-01-31")
        assert out["close"].tolist() == pytest.approx([11.0])


class TestListStocks:
    def test_empty_when_no_daily_dir(self, tmp_path):
        assert DataStorage(str(tmp_path)).list_stocks() == []

    def test_lists_flat_codes_only(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05"], code="000001"))
        store.save_daily(_frame(["2024-01-05"], code="600519"))
        legacy = os.path.join(str(tmp_path), "a_shares", "daily", "2024", "01")
        os.makedirs(legacy, exist_ok=True)
        _frame(["2024-01-05"], code="000002").to_parquet(
            os.path.join(legacy, "000002.parquet"), index=False
        )
        assert store.list_stocks() == ["000001", "600519"]

    def test_ignores_non_parquet_files(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05"], code="000001"))
        base = os.path.join(str(tmp_path), "a_shares", "daily")
        with open(os.path.join(base, "README.txt"), "w", encoding="utf-8") as f:
            f.write("x")
        assert store.list_stocks() == ["000001"]


class TestManifest:
    """review v8 §二-1: save_daily writes a per-stock contract manifest and the
    storage can validate that the on-disk parquet still matches it.  The whole
    point: "file exists" must never silently imply "data complete"."""

    def test_save_writes_full_manifest(self, tmp_path):
        store = DataStorage(str(tmp_path))
        df = _frame(["2024-01-05", "2024-01-06"])
        df.attrs["source"] = "efinance"
        df.attrs["adjustment_mode"] = "qfq"
        store.save_daily(df)
        m = store.manifest("000001")
        assert m is not None
        assert m["stock"] == "000001"
        assert m["start"] == "2024-01-05"
        assert m["end"] == "2024-01-06"
        assert m["rows"] == 2
        assert m["source"] == "efinance"
        assert m["adjust"] == "qfq"
        assert len(m["schema_hash"]) == 16
        assert "updated" in m

    def test_incremental_merge_updates_manifest(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        store.save_daily(_frame(["2024-01-07", "2024-01-08"]))
        m = store.manifest("000001")
        assert m["rows"] == 4
        assert m["start"] == "2024-01-05" and m["end"] == "2024-01-08"

    def test_validate_ok_after_save(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        report = store.validate_manifest("000001")
        assert report["ok"], report
        assert report["exists"] and report["mismatches"] == []

    def test_validate_flags_missing_manifest(self, tmp_path):
        # A bare parquet with no manifest is NOT "complete" — validate must say so.
        _write_flat(tmp_path, "000001", ["2024-01-05"], closes=[10.0])
        report = DataStorage(str(tmp_path)).validate_manifest("000001")
        assert not report["ok"]
        assert report["reason"] == "manifest missing — file exists ≠ data complete"

    def test_validate_flags_schema_drift(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05"]))
        # Drop a column directly on disk → schema_hash no longer matches.
        path = os.path.join(str(tmp_path), "a_shares", "daily", "000001.parquet")
        df = pd.read_parquet(path).drop(columns=["amount"])
        df.to_parquet(path, index=False)
        report = store.validate_manifest("000001")
        assert not report["ok"]
        assert any("schema_hash" in m for m in report["mismatches"]), report

    def test_validate_flags_row_drift(self, tmp_path):
        store = DataStorage(str(tmp_path))
        store.save_daily(_frame(["2024-01-05", "2024-01-06"]))
        # Overwrite the parquet with fewer rows behind the manifest's back.
        path = os.path.join(str(tmp_path), "a_shares", "daily", "000001.parquet")
        _frame(["2024-01-05"], code="000001").to_parquet(path, index=False)
        report = store.validate_manifest("000001")
        assert not report["ok"]
        assert any("rows" in m for m in report["mismatches"]), report

    def test_manifest_none_for_missing_stock(self, tmp_path):
        assert DataStorage(str(tmp_path)).manifest("600519") is None
        report = DataStorage(str(tmp_path)).validate_manifest("600519")
        assert not report["ok"] and report["reason"] == "parquet missing"


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
