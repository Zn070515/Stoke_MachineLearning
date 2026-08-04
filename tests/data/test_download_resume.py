"""Manifest-based download resume tests (v6 §八).

Resume must NEVER decide a stock/year is complete by guessing from file
presence or sampling — only an explicit COMPLETE manifest that matches the
request can cause a skip.
"""
import os

import pandas as pd

from stoke_ml.data.download_resume import (
    STATUS_COMPLETE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    mark_stock_result,
    read_stock_manifest,
    schema_hash,
    skip_completed_stocks,
    skip_completed_years,
    write_stock_manifest,
    write_year_manifest,
)


def _frame(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "stock_code": "000001",
        "value": [1.0] * len(dates),
    })


class TestManifestWriteRead:
    def test_roundtrip(self, tmp_path):
        write_stock_manifest(
            str(tmp_path), "000001", dataset="news_raw",
            requested_start="2024-01-01", requested_end="2024-12-31",
            actual_start="2019-03-04", actual_end="2024-12-31",
            actual_rows=10, source="sina", adjustment="qfq",
            schema_hash="abc123", status=STATUS_COMPLETE,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE
        assert m["dataset"] == "news_raw"
        assert m["requested_start"] == "2024-01-01"
        assert m["actual_start"] == "2019-03-04"
        assert m["actual_rows"] == 10
        assert m["source"] == "sina"
        assert m["schema_hash"] == "abc123"
        assert "completed_at" in m

    def test_missing_manifest_returns_none(self, tmp_path):
        assert read_stock_manifest(str(tmp_path), "000001") is None

    def test_no_tmp_residue(self, tmp_path):
        write_stock_manifest(str(tmp_path), "000001", status=STATUS_COMPLETE)
        write_stock_manifest(str(tmp_path), "000001", status=STATUS_PARTIAL)
        manifests = os.path.join(str(tmp_path), ".manifests")
        files = os.listdir(manifests)
        assert files == ["000001.json"]

    def test_schema_hash_stable_and_sensitive(self):
        a = _frame(["2024-01-01"])
        b = _frame(["2024-01-02"])
        assert schema_hash(a) == schema_hash(b)
        c = a.assign(extra_col=1.0)
        assert schema_hash(c) != schema_hash(a)


class TestMarkStockResult:
    def test_nonempty_defaults_to_complete(self, tmp_path):
        df = _frame(["2024-01-01", "2024-01-02", "2024-01-03"])
        mark_stock_result(str(tmp_path), "000001", df, dataset="news_raw")
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE
        assert m["actual_start"] == "2024-01-01"
        assert m["actual_end"] == "2024-01-03"
        assert m["actual_rows"] == 3
        assert m["schema_hash"] == schema_hash(df)

    def test_empty_defaults_to_degraded(self, tmp_path):
        mark_stock_result(str(tmp_path), "000001", _frame([]), dataset="news_raw")
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_DEGRADED


class TestSkipCompletedStocks:
    def test_no_manifest_means_pending(self, tmp_path):
        pending, skipped = skip_completed_stocks(str(tmp_path), ["000001", "600519"])
        assert pending == ["000001", "600519"]
        assert skipped == 0

    def test_file_on_disk_without_manifest_is_not_trusted(self, tmp_path):
        """Core v6 fix: a parquet file alone must NOT skip a stock."""
        raw = str(tmp_path)
        _frame(["2024-01-01", "2024-01-02"]).to_parquet(
            os.path.join(raw, "000001.parquet"), index=False
        )
        pending, skipped = skip_completed_stocks(raw, ["000001"])
        assert pending == ["000001"]
        assert skipped == 0

    def test_complete_manifest_covers_request(self, tmp_path):
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", requested_start="2024-01-01", covers_request=True,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2024-01-01"
        )
        assert pending == []
        assert skipped == 1

    def test_covers_request_false_forces_redownload(self, tmp_path):
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", requested_start="2024-01-01", covers_request=False,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2024-01-01"
        )
        assert pending == ["000001"]
        assert skipped == 0

    @staticmethod
    def _write_status(tmp_path, status):
        write_stock_manifest(
            str(tmp_path), "000001", status=status, covers_request=True,
        )

    def test_non_complete_statuses_force_redownload(self, tmp_path):
        for status in (STATUS_PARTIAL, STATUS_FAILED, STATUS_DEGRADED):
            raw = str(tmp_path) + "_" + status.lower()
            os.makedirs(raw, exist_ok=True)
            self._write_status(raw, status)
            pending, skipped = skip_completed_stocks(raw, ["000001"])
            assert pending == ["000001"], status
            assert skipped == 0, status

    def test_schema_hash_mismatch_forces_redownload(self, tmp_path):
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", covers_request=True,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], schema_hash="different-hash"
        )
        assert pending == ["000001"]
        assert skipped == 0

    def test_legacy_manifest_uses_actual_coverage(self, tmp_path):
        # Old manifests have no covers_request: coverage is decided by dates.
        write_stock_manifest(
            str(tmp_path), "000001", status=STATUS_COMPLETE,
            actual_start="2020-01-01", actual_end="2024-12-31",
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2024-01-01"
        )
        assert skipped == 1 and pending == []
        # A request the manifest does not reach is NOT covered.
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2019-01-01"
        )
        assert pending == ["000001"] and skipped == 0


class TestSkipCompletedYears:
    def _year_manifest(self, tmp_path, data_type, year, status=STATUS_COMPLETE):
        write_year_manifest(
            str(tmp_path), data_type, year, status=status, covers_request=True,
        )

    def test_no_manifest_means_pending(self, tmp_path):
        pending, skipped = skip_completed_years(
            str(tmp_path), [2020, 2021], "margin"
        )
        assert pending == [2020, 2021]
        assert skipped == 0

    def test_parquet_file_without_manifest_not_trusted(self, tmp_path):
        """v6 fix #2: any parquet in a year dir no longer marks it complete."""
        year_dir = os.path.join(str(tmp_path), "margin", "2020")
        os.makedirs(year_dir, exist_ok=True)
        _frame(["2020-01-02"]).to_parquet(
            os.path.join(year_dir, "000001.parquet"), index=False
        )
        pending, skipped = skip_completed_years(str(tmp_path), [2020], "margin")
        assert pending == [2020]
        assert skipped == 0

    def test_complete_manifest_skips_year(self, tmp_path):
        self._year_manifest(tmp_path, "margin", 2020)
        pending, skipped = skip_completed_years(str(tmp_path), [2020, 2021], "margin")
        assert pending == [2021]
        assert skipped == 1

    def test_partial_manifest_does_not_skip(self, tmp_path):
        self._year_manifest(tmp_path, "margin", 2020, status=STATUS_PARTIAL)
        pending, skipped = skip_completed_years(str(tmp_path), [2020], "margin")
        assert pending == [2020]
        assert skipped == 0
