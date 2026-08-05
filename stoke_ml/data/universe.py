"""PIT universe status: listing / delisting records used to build a
survivorship-free universe (§七-1).

The download universe, panel universe, candidate eligibility and exit policy
all read the same normalized status table so a delisted stock is never silently
dropped (survivorship bias) nor held past its last trading day (a real
delisting is a forced exit, not an unresolved data gap).

Sources (both already downloaded to ``{data_dir}/a_shares/universe/``):
  - ``ipo.parquet``      — every stock's listing date.
  - ``delisted.parquet`` — delisted stocks; ``暂停上市日期`` (suspension) is the
    last day the stock actually traded, i.e. the effective delisting date.

The delisted parquet stores the SSE code only in the Chinese ``公司代码`` column
(stock_code is NaN for SSE), so the canonical code is taken from ``公司代码``
when present.  All codes are zero-padded to 6 digits.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DELISTED_DATE_COL = "暂停上市日期"  # effective last trading day (suspension)
CODE_COL = "公司代码"              # canonical code (present for every row)

INDEX_MEMBERSHIP_PATH = ("a_shares", "index_constituents_hist", "membership.parquet")
DEFAULT_INDICES = ("000300", "000905")  # CSI300 + CSI500 default download universe


def _read_universe_parquet(data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "a_shares", "universe", name)
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_universe_status(data_dir: str) -> pd.DataFrame:
    """Normalized per-stock ``(stock_code, list_date, delist_date)`` table.

    ``delist_date`` is NaN for stocks still listed.  ``list_date`` may be NaN
    when the IPO record is missing.  Both dates are ``datetime64[ns]``.
    """
    ipo = _read_universe_parquet(data_dir, "ipo.parquet")
    delisted = _read_universe_parquet(data_dir, "delisted.parquet")

    recs: dict[str, dict[str, object]] = {}
    if not ipo.empty:
        codes = ipo["stock_code"].astype(str).str.zfill(6)
        for code, ld in zip(codes, ipo["list_date"]):
            recs.setdefault(code, {})["list_date"] = ld
    if not delisted.empty:
        raw = delisted[CODE_COL].copy()
        raw = raw.where(raw.notna(), delisted.get("stock_code"))
        codes = raw.astype(str).str.zfill(6)
        for code, dd in zip(codes, delisted[DELISTED_DATE_COL]):
            code = code.zfill(6)  # codes dropped by .where stay NaN → "nan"
            recs.setdefault(code, {})["delist_date"] = dd

    rows = [
        {
            "stock_code": code,
            "list_date": pd.to_datetime(v.get("list_date"), errors="coerce"),
            "delist_date": pd.to_datetime(v.get("delist_date"), errors="coerce"),
        }
        for code, v in recs.items()
        if code != "nan"
    ]
    return pd.DataFrame(rows, columns=["stock_code", "list_date", "delist_date"])


def delisted_codes(data_dir: str) -> list[str]:
    """Codes of every stock with a delisting record (download-universe fix)."""
    status = load_universe_status(data_dir)
    if status.empty:
        return []
    return sorted(status.loc[status["delist_date"].notna(), "stock_code"])


def delist_global_index(
    global_dates: np.ndarray,
    status_df: pd.DataFrame,
    codes: list[str],
) -> np.ndarray:
    """Per-stock index (into ``global_dates``) of the last trading day before or
    at its delisting; ``-1`` when never delisted / date not on the grid.

    ``global_dates`` is the union trading calendar the panel is indexed by
    (datetime64).  ``codes`` must be in panel-stock order so the returned array
    aligns 1:1 with the panel's stock axis.  Used to translate a stock's
    suspension date into the sleeve simulator's column space so a real
    delisting force-sells at the delisting close instead of dangling as an
    unresolved hold.
    """
    out = np.full(len(codes), -1, dtype=int)
    if status_df.empty or len(global_dates) == 0:
        return out
    dd = status_df.set_index("stock_code")["delist_date"]
    grid = np.asarray(global_dates)
    for i, code in enumerate(codes):
        d = dd.get(code)
        if d is None or pd.isna(d):
            continue
        ts = np.datetime64(pd.Timestamp(d))
        if ts < grid[0]:
            continue
        out[i] = int(np.searchsorted(grid, ts, side="right")) - 1
    return out


def load_index_membership(
    data_dir: str, indices: list[str] | None = None
) -> pd.DataFrame:
    """Historical index-membership intervals from ``membership.parquet``.

    Long-form ``(stock_code, index_code, in_date, out_date)``; ``out_date`` NaT
    means the stock is still a member.  A stock is a member on trading day d iff
    ``in_date <= d < out_date`` (half-open).  Returns an empty frame when the
    artifact is missing or no requested index is covered.
    """
    path = os.path.join(data_dir, *INDEX_MEMBERSHIP_PATH)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["stock_code", "index_code", "in_date", "out_date"])
    mem = pd.read_parquet(path)
    if mem.empty:
        return mem
    for col in ("in_date", "out_date"):
        mem[col] = pd.to_datetime(mem[col], errors="coerce")
    if indices:
        mem = mem[mem["index_code"].astype(str).isin([str(i) for i in indices])]
    return mem.reset_index(drop=True)


def historical_index_members(
    data_dir: str, indices: list[str] | None = None
) -> list[str]:
    """Union of every stock that EVER belonged to the given indices (§七-2).

    This is the download-universe default: restricting the backtest to today's
    constituents is survivorship-adjacent — a stock that left CSI300 in 2019
    still contributes legitimate index-universe history for 2015-2019.  Empty
    when no historical membership data exists.
    """
    mem = load_index_membership(data_dir, indices)
    if mem.empty:
        return []
    return sorted(mem["stock_code"].astype(str).str.zfill(6).unique())


def index_membership_mask(
    global_dates: np.ndarray,
    codes: list[str],
    membership: pd.DataFrame,
) -> np.ndarray:
    """Per-day index-membership eligibility ``(n_stocks, T)`` (§七-3).

    ``mask[i, t]`` is True iff ``in_date <= global_dates[t] < out_date`` for at
    least one membership interval of stock ``codes[i]`` (half-open).  ``codes``
    must be in panel-stock order so the returned grid aligns 1:1 with the
    panel's stock axis.  A stock with no interval, or a date off the calendar
    grid, stays False.  This is what turns a "historical member union" universe
    into a true PIT "member that day" candidate pool.
    """
    T = len(global_dates)
    mask = np.zeros((len(codes), T), dtype=bool)
    if membership is None or membership.empty or T == 0:
        return mask
    row_of = {str(c).zfill(6): i for i, c in enumerate(codes)}
    grid = np.asarray(global_dates)
    in_dates = pd.to_datetime(membership["in_date"], errors="coerce")
    out_dates = pd.to_datetime(membership["out_date"], errors="coerce")
    for code, in_d, out_d in zip(
        membership["stock_code"].astype(str).str.zfill(6),
        in_dates, out_dates,
    ):
        row = row_of.get(code)
        if row is None or pd.isna(in_d):
            continue
        lo = int(np.searchsorted(grid, np.datetime64(in_d), side="left"))
        hi = T if pd.isna(out_d) else int(
            np.searchsorted(grid, np.datetime64(out_d), side="left"))
        lo = max(lo, 0)
        hi = min(hi, T)
        if lo < hi:
            mask[row, lo:hi] = True
    return mask


def not_delisted_mask(
    global_dates: np.ndarray, codes: list[str], status_df: pd.DataFrame
) -> np.ndarray:
    """``(n_stocks, T)`` grid blocking ENTRY from a stock's delisting day on.

    ``delist_global_index`` maps each stock to its last trading column; columns
    ``>= delist_col`` are disqualified (§七-3 未退市) so a known-delisted stock
    can never be ranked as a fresh candidate after its delisting, even if its
    K-line data extends past the record date.  Stocks never delisted (or with
    no status record) stay eligible everywhere.
    """
    T = len(global_dates)
    nd = np.ones((len(codes), T), dtype=bool)
    if status_df is None or status_df.empty or T == 0:
        return nd
    dcol = delist_global_index(global_dates, status_df, codes)
    for i, c in enumerate(dcol):
        if 0 <= c < T:
            nd[i, c:] = False
    return nd
