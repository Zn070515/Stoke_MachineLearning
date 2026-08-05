"""Manifest-based download resume tests.

Resume must NEVER decide a stock/year is complete by guessing from file
presence or sampling — only an explicit COMPLETE manifest that matches the
request can cause a skip.
"""
import os

import pandas as pd

from stoke_ml.data.download_resume import (
    STATUS_BOUNDED_COMPLETE,
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
    def test_nonempty_without_evidence_defaults_to_partial(self, tmp_path):
        """A non-empty frame proves nothing for event data — a page-cap
        truncation looks identical to a complete download, so default PARTIAL."""
        df = _frame(["2024-01-01", "2024-01-02", "2024-01-03"])
        mark_stock_result(str(tmp_path), "000001", df, dataset="news_raw")
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_PARTIAL
        assert m["covers_request"] is False
        assert m["actual_start"] == "2024-01-01"
        assert m["actual_end"] == "2024-01-03"
        assert m["actual_rows"] == 3
        assert m["schema_hash"] == schema_hash(df)

    def test_pagination_exhausted_defaults_to_complete(self, tmp_path):
        df = _frame(["2024-01-01", "2024-01-02"])
        mark_stock_result(
            str(tmp_path), "000001", df, dataset="news_raw",
            pages_requested=5, pages_fetched=3, pagination_exhausted=True,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE
        assert m["covers_request"] is True
        assert m["pages_requested"] == 5
        assert m["pages_fetched"] == 3
        assert m["pagination_exhausted"] is True
        assert m["provider_exhausted"] is True  # §五-1 canonical field

    def test_bounded_provider_exhausted_not_covered(self, tmp_path):
        """A provider that only keeps the recent 3 years: provider has no more
        data (exhausted) but the requested window is NOT reached -> status must
        be BOUNDED_COMPLETE and covers_request must be False (§五-1)."""
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01", "2024-01-02"]),
            dataset="news_raw", requested_start="2015-01-01",
            pagination_exhausted=True,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_BOUNDED_COMPLETE
        assert m["covers_request"] is False
        assert m["request_covered"] is False
        assert m["provider_exhausted"] is True

    def test_provider_range_guaranteed_defaults_to_complete(self, tmp_path):
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="margin", provider_range_guaranteed=True,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE
        assert m["provider_range_guaranteed"] is True

    def test_expected_rows_match_defaults_to_complete(self, tmp_path):
        df = _frame(["2024-01-01", "2024-01-02"])
        mark_stock_result(
            str(tmp_path), "000001", df, dataset="announcements", expected_rows=2,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE

    def test_expected_rows_mismatch_defaults_to_partial(self, tmp_path):
        df = _frame(["2024-01-01", "2024-01-02"])
        mark_stock_result(
            str(tmp_path), "000001", df, dataset="announcements", expected_rows=7,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_PARTIAL

    def test_pagination_not_exhausted_defaults_to_partial(self, tmp_path):
        """Hit the page cap without reaching the end -> truncated -> PARTIAL."""
        df = _frame(["2024-01-01", "2024-01-02"])
        mark_stock_result(
            str(tmp_path), "000001", df, dataset="news_raw",
            pages_requested=500, pages_fetched=500, pagination_exhausted=False,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_PARTIAL
        assert m["pages_fetched"] == 500

    def test_empty_defaults_to_degraded(self, tmp_path):
        mark_stock_result(str(tmp_path), "000001", _frame([]), dataset="news_raw")
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_DEGRADED
        assert m["covers_request"] is False

    def test_effective_start_derives_complete(self, tmp_path):
        """v14 §十一: a bounded provider with no pre-listing data cannot reach
        the global requested_start, but the caller clips the ask to the stock's
        own lifecycle (effective_start=listing_date) so the effective window is
        genuinely covered -> status derives COMPLETE and
        effective_range_covered is True.  No boolean override is involved."""
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", requested_start="2015-01-01",
            effective_start="2024-01-01", pagination_exhausted=True,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_COMPLETE
        assert m["effective_range_covered"] is True
        assert m["effective_start"] == "2024-01-01"
        # covers_request mirrors the effective-range conclusion; request_covered
        # stays informational (the raw global request was never reached).
        assert m["covers_request"] is True
        assert m["request_covered"] is False


class TestSkipCompletedStocks:
    def test_no_manifest_means_pending(self, tmp_path):
        pending, skipped = skip_completed_stocks(str(tmp_path), ["000001", "600519"])
        assert pending == ["000001", "600519"]
        assert skipped == 0

    def test_file_on_disk_without_manifest_is_not_trusted(self, tmp_path):
        """Core invariant: a parquet file alone must NOT skip a stock."""
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
            dataset="news_raw", requested_start="2024-01-01",
            pagination_exhausted=True,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2024-01-01"
        )
        assert pending == []
        assert skipped == 1

    def test_effective_range_not_covered_forces_redownload(self, tmp_path):
        """v14 §十一: no effective_start supplied -> the effective window
        degenerates to the request, and actual dates that do NOT reach the
        effective end mean the manifest is BOUNDED_COMPLETE
        (effective_range_covered False) -> resume must re-download."""
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", requested_start="2024-01-01",
            requested_end="2024-12-31", pagination_exhausted=True,
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["status"] == STATUS_BOUNDED_COMPLETE
        assert m["effective_range_covered"] is False
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"],
            start_date="2024-01-01", end_date="2024-12-31",
        )
        assert pending == ["000001"]
        assert skipped == 0

    def test_stored_covers_true_but_dates_lie_not_skipped(self, tmp_path):
        """§五-2/v14 §十一: a COMPLETE manifest whose derived covers_request is
        True but whose actual dates do NOT reach the requested start is
        re-verified against the effective range and re-downloaded — the date
        facts, not any stored boolean, decide."""
        write_stock_manifest(
            str(tmp_path), "000001", status=STATUS_COMPLETE,
            actual_start="2024-01-01", actual_end="2024-12-31",
        )
        m = read_stock_manifest(str(tmp_path), "000001")
        assert m["covers_request"] is True  # derived, yet the dates still lie
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2020-01-01"
        )
        assert pending == ["000001"]
        assert skipped == 0
        # The same manifest DOES cover a request inside its actual range.
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2024-06-01"
        )
        assert pending == []
        assert skipped == 1

    def test_bounded_provider_never_skipped(self, tmp_path):
        """BOUNDED_COMPLETE (provider exhausted but request not reached) is
        never trusted for skip: resume must try another provider / retry."""
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01", "2024-01-02"]),
            dataset="news_raw", requested_start="2015-01-01",
            pagination_exhausted=True,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2015-01-01"
        )
        assert pending == ["000001"]
        assert skipped == 0

    def test_truncated_fetch_not_skipped(self, tmp_path):
        """Hit the page cap without reaching the end -> PARTIAL ->
        resume must re-download, never treat the truncated history as done."""
        mark_stock_result(
            str(tmp_path), "000001", _frame(["2024-01-01"]),
            dataset="news_raw", requested_start="2015-01-01",
            pages_requested=500, pages_fetched=500, pagination_exhausted=False,
        )
        pending, skipped = skip_completed_stocks(
            str(tmp_path), ["000001"], start_date="2015-01-01"
        )
        assert pending == ["000001"]
        assert skipped == 0

    @staticmethod
    def _write_status(tmp_path, status):
        write_stock_manifest(str(tmp_path), "000001", status=status)

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
            dataset="news_raw", pagination_exhausted=True,
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
        # v14 §十一: write_year_manifest derives coverage from dates — no
        # covers_request override.  With no actual dates supplied the effective
        # window degenerates to the request and status derives from it.
        write_year_manifest(str(tmp_path), data_type, year, status=status)

    def test_no_manifest_means_pending(self, tmp_path):
        pending, skipped = skip_completed_years(
            str(tmp_path), [2020, 2021], "margin"
        )
        assert pending == [2020, 2021]
        assert skipped == 0

    def test_parquet_file_without_manifest_not_trusted(self, tmp_path):
        """A parquet in a year dir alone does not mark it complete."""
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
