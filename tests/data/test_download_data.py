"""Download-data script resume tests (§五-3).

``--skip-existing`` must NOT trust file presence: a stock is skipped only
when its parquet has a VALID contract manifest (``DataStorage.validate_manifest``
ok) AND that manifest's actual dates cover the requested range.  A bare file,
a missing/corrupt manifest, or coverage that ends before the requested window
forces a re-download.
"""
import os

import pandas as pd

from scripts.production.download_data import (
    filter_existing,
    get_all_a_share_codes,
    get_stock_codes,
)
from stoke_ml.data.storage import DataStorage


def _write_membership(root, rows):
    """Write a membership.parquet so the historical-index path is exercised."""
    base = root / "a_shares" / "index_constituents_hist"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(base / "membership.parquet", index=False)


def _write_delisted(root, codes):
    """Write delisted.parquet in the SSE convention (公司代码 carries the code)."""
    base = root / "a_shares" / "universe"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "公司代码": codes,
        "stock_code": [None] * len(codes),
        "暂停上市日期": pd.to_datetime(["2015-06-30"] * len(codes)),
    }).to_parquet(base / "delisted.parquet", index=False)


def _frame(dates, code="000001"):
    dates = list(pd.to_datetime(dates))
    n = len(dates)
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


def _save(root, code, dates):
    """Write a parquet + full contract manifest via DataStorage.save_daily."""
    DataStorage(str(root)).save_daily(_frame(dates, code=code))


class TestFilterExisting:
    def test_no_daily_dir_means_all_pending(self, tmp_path):
        pending, complete = filter_existing(["000001", "600519"], str(tmp_path))
        assert pending == ["000001", "600519"]
        assert complete == set()

    def test_file_without_manifest_is_not_trusted(self, tmp_path):
        daily = str(tmp_path / "a_shares" / "daily")
        os.makedirs(daily, exist_ok=True)
        _frame(["2024-01-02", "2024-01-03"]).to_parquet(
            os.path.join(daily, "000001.parquet"), index=False
        )
        pending, complete = filter_existing(["000001"], str(tmp_path))
        assert pending == ["000001"]
        assert complete == set()

    def test_valid_manifest_full_coverage_skips(self, tmp_path):
        _save(tmp_path, "000001", ["2023-12-29", "2024-01-15", "2024-01-31"])
        pending, complete = filter_existing(
            ["000001"], str(tmp_path), "2024-01-01", "2024-01-31"
        )
        assert pending == []
        assert complete == {"000001"}

    def test_manifest_end_before_request_forces_redownload(self, tmp_path):
        _save(tmp_path, "000001", ["2023-12-29", "2024-01-02", "2024-01-03"])
        pending, complete = filter_existing(
            ["000001"], str(tmp_path), "2024-01-01", "2024-02-28"
        )
        assert pending == ["000001"]
        assert complete == set()

    def test_manifest_start_after_request_forces_redownload(self, tmp_path):
        _save(tmp_path, "000001", ["2024-06-03", "2024-12-31"])
        pending, complete = filter_existing(
            ["000001"], str(tmp_path), "2024-01-01", "2024-12-31"
        )
        assert pending == ["000001"]
        assert complete == set()

    def test_corrupt_manifest_forces_redownload(self, tmp_path):
        _save(tmp_path, "000001", ["2023-12-29", "2024-01-15", "2024-12-30"])
        df = pd.read_parquet(str(tmp_path / "a_shares" / "daily" / "000001.parquet"))
        df.drop(columns=["amount"]).to_parquet(
            str(tmp_path / "a_shares" / "daily" / "000001.parquet"), index=False
        )
        pending, complete = filter_existing(
            ["000001"], str(tmp_path), "2024-01-01", "2024-12-31"
        )
        assert pending == ["000001"]
        assert complete == set()

    def test_no_requested_bounds_skips_on_valid_manifest(self, tmp_path):
        _save(tmp_path, "000001", ["2024-01-02", "2024-01-03"])
        pending, complete = filter_existing(["000001"], str(tmp_path))
        assert pending == []
        assert complete == {"000001"}

    def _write_ipo(self, root, rows):
        base = root / "a_shares" / "universe"
        base.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(base / "ipo.parquet", index=False)

    def test_late_listed_stock_complete_to_list_date(self, tmp_path):
        """§P0-3: a 2018 IPO can never start in 2000 — the effective start is
        max(requested_start, list_date), so a global 2000→2024 request does not
        force a late-listed stock to re-download forever."""
        self._write_ipo(tmp_path, [
            {"stock_code": "000001",
             "list_date": pd.Timestamp("2018-03-01")},
        ])
        _save(tmp_path, "000001", ["2018-03-01", "2024-12-31"])
        pending, complete = filter_existing(
            ["000001"], str(tmp_path), "2000-01-01", "2024-12-31"
        )
        assert complete == {"000001"}
        assert pending == []

    def test_delisted_stock_complete_to_delist_date(self, tmp_path):
        """§P0-3: a stock delisted in 2015 has no data past 2015-06-30 — the
        effective end is min(requested_end, delist_date), so the requested
        2024 end does not mark it incomplete."""
        _write_delisted(tmp_path, ["000002"])
        _save(tmp_path, "000002", ["2013-01-02", "2015-06-30"])
        pending, complete = filter_existing(
            ["000002"], str(tmp_path), "2013-01-02", "2024-12-31"
        )
        assert complete == {"000002"}
        assert pending == []

    def test_end_caps_at_last_fully_closed_day(self, tmp_path):
        """§P0-3: the default end is the last fully closed trading day, never
        today (which has not closed) — so a request reaching into the future is
        satisfied by data through that last closed session and no further."""
        from datetime import timedelta

        from scripts.production.download_data import _last_fully_closed_trading_day
        from stoke_ml.data.calendar import TradingCalendar

        cal = TradingCalendar("a_shares", calendar_dir=str(tmp_path))
        last_closed = _last_fully_closed_trading_day(cal)
        prev_day = last_closed - timedelta(days=1)
        _save(tmp_path, "000001", [last_closed.isoformat()])
        _save(tmp_path, "600519", [prev_day.isoformat()])
        pending, complete = filter_existing(
            ["000001", "600519"], str(tmp_path),
            last_closed.isoformat(), "2099-12-31",
        )
        assert complete == {"000001"}
        assert pending == ["600519"]

    def test_get_stock_codes_uses_historical_member_union(self, tmp_path):
        """§七-2: the index download universe is the HISTORICAL member union, not
        today's constituents.  Both default indices covered by membership.parquet
        → no AKShare call, only stocks that ever belonged to CSI300/CSI500."""
        _write_membership(tmp_path, [
            {"stock_code": "600000", "index_code": "000300",
             "in_date": pd.Timestamp("2015-01-31"),
             "out_date": pd.Timestamp("2019-01-31")},
            {"stock_code": "600009", "index_code": "000300",
             "in_date": pd.Timestamp("2016-01-29"),
             "out_date": pd.NaT},
            {"stock_code": "000001", "index_code": "000905",
             "in_date": pd.Timestamp("2015-01-31"),
             "out_date": pd.Timestamp("2017-01-31")},
        ])
        codes = get_stock_codes(str(tmp_path))
        assert codes == ["000001", "600000", "600009"]

    def test_get_stock_codes_index_filter_no_network_for_covered(self, tmp_path):
        """Requesting an index with membership data must not fall back to AKShare
        for that index; an uncovered index keeps the AKShare path (not exercised
        here)."""
        _write_membership(tmp_path, [
            {"stock_code": "600000", "index_code": "000300",
             "in_date": pd.Timestamp("2015-01-31"),
             "out_date": pd.NaT},
        ])
        codes = get_stock_codes(str(tmp_path), ["000300"])
        assert codes == ["600000"]

    def test_get_stock_codes_missing_artifact_falls_back_to_current(
            self, tmp_path, monkeypatch):
        """No membership.parquet → the historical path yields nothing, so
        get_stock_codes falls through to the AKShare current-constituent
        default (documented behavior for indices without historical data).
        Mock the network call so this test is deterministic and offline."""
        import scripts.production.download_data as dd
        monkeypatch.setattr(
            dd.ak, "index_stock_cons_csindex",
            lambda symbol: pd.DataFrame({"成分券代码": ["000001", "600519"]}),
        )
        assert get_stock_codes(str(tmp_path), ["000300"]) == ["000001", "600519"]

    def test_run_manifest_all_complete_requires_range_coverage(self, tmp_path):
        """§五-4: a parquet on disk that does NOT cover the requested range must
        not count as complete — the run manifest's all_complete stays False even
        though the file exists."""
        from stoke_ml.data.download_manifest import default_path, write_manifest

        # Stock present on disk, but only two days of history vs a 12-month ask.
        _save(tmp_path, "000001", ["2024-01-02", "2024-01-03"])
        _, complete = filter_existing(
            ["000001"], str(tmp_path), "2024-01-01", "2024-12-31"
        )
        assert complete == set()  # file on disk, but not range-covered
        manifest = write_manifest(
            default_path(str(tmp_path)), market="a_shares",
            start_date="2024-01-01", end_date="2024-12-31",
            requested=["000001"], failed=[], complete=complete, success_count=1,
        )
        assert manifest["missing"] == ["000001"]
        assert manifest["all_complete"] is False


class TestGetAllAShareCodes:
    """§十六: `--all` must download delisted stocks' history too, or the
    2000-2026 "全 A" panel silently becomes the survivor set of stocks still
    visible today (survivorship bias, §七-1)."""

    def test_unions_delisted_codes_into_download_universe(self, tmp_path, monkeypatch):
        import scripts.production.download_data as dd

        monkeypatch.setattr(
            dd.ak, "stock_info_a_code_name",
            lambda: pd.DataFrame({"code": ["000001", "600519"]}),
        )
        _write_delisted(tmp_path, ["000002", "600002"])
        codes = get_all_a_share_codes(str(tmp_path))
        # Currently-listed ∪ delisted records — the delisted stock is NOT
        # dropped from the historical universe.
        assert codes == ["000001", "000002", "600002", "600519"]

    def test_without_data_dir_only_currently_listed(self, tmp_path, monkeypatch):
        import scripts.production.download_data as dd

        monkeypatch.setattr(
            dd.ak, "stock_info_a_code_name",
            lambda: pd.DataFrame({"code": ["000001", "600519"]}),
        )
        _write_delisted(tmp_path, ["000002"])
        # No data_dir → the delisted union is skipped (legacy behavior).
        assert get_all_a_share_codes() == ["000001", "600519"]
