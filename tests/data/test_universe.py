"""PIT universe-status tests (§七-1).

The universe module normalizes ipo.parquet + delisted.parquet into a
per-stock (stock_code, list_date, delist_date) table and translates a
delisting date onto a trading-calendar grid.  These tests guard the
survivorship-free universe construction — a delisted stock must never be
silently dropped from the download universe nor held past its delisting.
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.universe import (
    delist_global_index,
    delisted_codes,
    index_membership_mask,
    load_universe_status,
    not_delisted_mask,
)


def _write_universe(data_dir: str) -> None:
    """Write realistic ipo/delisted parquets into `data_dir/a_shares/universe/`.

    Two SZ (000001/000002) + two SSE (600001/600002) stocks; 000002 and 600002
    are delisted; 600999 is delisted but NEVER listed in ipo.parquet and its
    delisted row carries only the Chinese `公司代码` (SSE convention — the
    `stock_code` column is NaN there).
    """
    import os

    uni = os.path.join(data_dir, "a_shares", "universe")
    os.makedirs(uni, exist_ok=True)

    pd.DataFrame({
        "stock_code": ["000001", "000002", "600001", "600002"],
        "list_date": pd.to_datetime([
            "1991-04-03", "1991-06-25", "1990-12-19", "1997-01-08"]),
    }).to_parquet(os.path.join(uni, "ipo.parquet"))

    pd.DataFrame({
        "公司代码": ["000002", "600002", "600999"],
        "stock_code": ["000002", "600002", None],
        "暂停上市日期": pd.to_datetime([
            "2024-05-15", "2019-08-23", "2015-06-30"]),
    }).to_parquet(os.path.join(uni, "delisted.parquet"))


class TestLoadUniverseStatus:
    def test_empty_data_dir_returns_empty_table(self, tmp_path):
        status = load_universe_status(str(tmp_path))
        assert status.empty
        assert list(status.columns) == [
            "stock_code", "list_date", "delist_date"]

    def test_listing_and_delisting_merge(self, tmp_path):
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        rows = status.set_index("stock_code")
        # Never-delisted stocks keep NaN delist_date.
        assert pd.isna(rows.loc["000001", "delist_date"])
        assert pd.isna(rows.loc["600001", "delist_date"])
        # Delisted stocks carry both dates.
        assert rows.loc["000002", "list_date"] == pd.Timestamp("1991-06-25")
        assert rows.loc["000002", "delist_date"] == pd.Timestamp("2024-05-15")
        assert rows.loc["600002", "delist_date"] == pd.Timestamp("2019-08-23")
        # SSE stock whose delisted row has NaN stock_code is keyed by 公司代码.
        assert rows.loc["600999", "delist_date"] == pd.Timestamp("2015-06-30")
        assert pd.isna(rows.loc["600999", "list_date"])

    def test_delisted_codes(self, tmp_path):
        _write_universe(str(tmp_path))
        assert delisted_codes(str(tmp_path)) == ["000002", "600002", "600999"]

    def test_no_universe_dir_delisted_codes_empty(self, tmp_path):
        assert delisted_codes(str(tmp_path)) == []


class TestDelistGlobalIndex:
    def test_maps_delist_dates_onto_grid(self, tmp_path):
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        grid = np.array(
            ["2019-08-22", "2019-08-23", "2024-05-14", "2024-05-15"],
            dtype="datetime64[ns]")
        idx = delist_global_index(
            grid, status, ["000001", "000002", "600001", "600002", "600999"])
        assert idx.tolist() == [-1, 3, -1, 1, -1]

    def test_off_grid_delist_lands_on_last_prior_day(self, tmp_path):
        """A delist date that is NOT a trading day (e.g. a holiday) maps to the
        last grid day before it — the effective last trading day."""
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        # 2024-05-15 is missing from the grid; 2024-05-14 is the last prior day.
        grid = np.array(
            ["2024-05-13", "2024-05-14", "2024-05-16"], dtype="datetime64[ns]")
        idx = delist_global_index(grid, status, ["000002"])
        assert idx.tolist() == [1]

    def test_delist_before_grid_start_is_never(self, tmp_path):
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        grid = np.array(
            ["2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        # 600999 delisted 2015-06-30 < grid start → -1.
        idx = delist_global_index(grid, status, ["600999"])
        assert idx.tolist() == [-1]

    def test_empty_status_returns_all_never(self, tmp_path):
        status = load_universe_status(str(tmp_path))
        grid = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        idx = delist_global_index(grid, status, ["000001", "600001"])
        assert idx.tolist() == [-1, -1]


class TestIndexMembershipMask:
    """Per-day index-membership eligibility grid (§七-3)."""

    GRID = np.array(
        ["2020-01-01", "2020-02-03", "2020-03-02", "2020-04-01",
         "2020-05-04", "2020-06-01", "2020-07-01"],
        dtype="datetime64[ns]",
    )

    def test_half_open_interval(self):
        codes = ["000001"]
        mem = pd.DataFrame({
            "stock_code": ["000001"],
            "index_code": ["000300"],
            "in_date": pd.to_datetime(["2020-02-03"]),
            "out_date": pd.to_datetime(["2020-06-01"]),
        })
        mask = index_membership_mask(self.GRID, codes, mem)
        # Member on [in_date, out_date): cols 1..4 inclusive.
        assert mask.tolist() == [[False, True, True, True, True, False, False]]

    def test_no_out_date_members_forever(self):
        codes = ["000001"]
        mem = pd.DataFrame({
            "stock_code": ["000001"],
            "index_code": ["000905"],
            "in_date": pd.to_datetime(["2020-03-02"]),
            "out_date": pd.to_datetime([None]),
        })
        mask = index_membership_mask(self.GRID, codes, mem)
        assert mask.tolist() == [[False, False, True, True, True, True, True]]

    def test_in_date_not_on_grid_rounds_down_to_next(self):
        """in_date not itself a grid day → first grid day at/after it is member."""
        codes = ["000001"]
        mem = pd.DataFrame({
            "stock_code": ["000001"],
            "index_code": ["000300"],
            "in_date": pd.to_datetime(["2020-01-15"]),
            "out_date": pd.to_datetime(["2020-04-15"]),
        })
        mask = index_membership_mask(self.GRID, codes, mem)
        assert mask.tolist() == [[False, True, True, True, False, False, False]]

    def test_union_across_intervals_and_indices(self):
        codes = ["000001", "000002"]
        mem = pd.DataFrame({
            "stock_code": ["000001", "000001", "000002"],
            "index_code": ["000300", "000905", "000300"],
            "in_date": pd.to_datetime(["2020-01-01", "2020-05-04", "2020-01-01"]),
            "out_date": pd.to_datetime(["2020-04-01", None, "2020-06-01"]),
        })
        mask = index_membership_mask(self.GRID, codes, mem)
        # 000001: [Jan, Apr) OR [May, forever) → cols 0,1,2,4,5,6.
        assert mask[0].tolist() == [True, True, True, False, True, True, True]
        # 000002: [Jan, Jun) → cols 0..4.
        assert mask[1].tolist() == [True, True, True, True, True, False, False]

    def test_code_not_in_membership_stays_false(self):
        codes = ["600519"]
        mem = pd.DataFrame({
            "stock_code": ["000001"],
            "index_code": ["000300"],
            "in_date": pd.to_datetime(["2020-01-01"]),
            "out_date": pd.to_datetime([None]),
        })
        mask = index_membership_mask(self.GRID, codes, mem)
        assert mask.tolist() == [[False] * len(self.GRID)]

    def test_empty_membership_all_false(self):
        mask = index_membership_mask(
            self.GRID, ["000001"], pd.DataFrame())
        assert mask.shape == (1, len(self.GRID))
        assert not mask.any()

    def test_empty_grid_zero_width(self):
        mask = index_membership_mask(
            np.array([], dtype="datetime64[ns]"), ["000001"],
            pd.DataFrame({"stock_code": ["000001"], "index_code": ["000300"],
                          "in_date": pd.to_datetime(["2020-01-01"]),
                          "out_date": pd.to_datetime([None])}))
        assert mask.shape == (1, 0)


class TestNotDelistedMask:
    """未退市 gate: a known-delisted stock can never enter after its delisting day."""

    def test_delisted_blocked_from_delist_day_on(self, tmp_path):
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        # 000002 delisted 2024-05-15, 600002 delisted 2019-08-23.
        grid = np.array(
            ["2019-08-22", "2019-08-23", "2024-05-14", "2024-05-15"],
            dtype="datetime64[ns]")
        codes = ["000001", "000002", "600001", "600002"]
        nd = not_delisted_mask(grid, codes, status)
        # 000001 (never delisted) stays eligible everywhere.
        assert nd[0].tolist() == [True, True, True, True]
        # 000002: delist col = index of 2024-05-15 = 3 → blocked from col 3 on.
        assert nd[1].tolist() == [True, True, True, False]
        # 600001 (never delisted) eligible everywhere.
        assert nd[2].tolist() == [True, True, True, True]
        # 600002: delist col = 2019-08-23 = 1 → blocked from col 1 on.
        assert nd[3].tolist() == [True, False, False, False]

    def test_delist_day_off_grid_maps_to_last_prior(self, tmp_path):
        _write_universe(str(tmp_path))
        status = load_universe_status(str(tmp_path))
        # 2024-05-15 not on grid → 2024-05-14 (last prior) is the block start.
        grid = np.array(
            ["2024-05-13", "2024-05-14", "2024-05-16"], dtype="datetime64[ns]")
        nd = not_delisted_mask(grid, ["000002"], status)
        assert nd.tolist() == [[True, False, False]]

    def test_empty_status_all_eligible(self, tmp_path):
        status = load_universe_status(str(tmp_path))
        grid = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        nd = not_delisted_mask(grid, ["000001", "600001"], status)
        assert nd.tolist() == [[True, True], [True, True]]
