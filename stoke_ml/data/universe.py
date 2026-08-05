"""PIT universe status: listing / delisting records used to build a
survivorship-free universe (§七-1).

The download universe, panel universe, candidate eligibility and exit policy
all read the same normalized status table so a delisted stock is never silently
dropped (survivorship bias) nor held past its last trading day (a real
delisting is a forced exit, not an unresolved data gap).

Sources (both already downloaded to ``{data_dir}/a_shares/universe/``):
  - ``ipo.parquet``    — every stock's listing date.
  - ``delisted.parquet`` — delisted stocks with SEPARATE exit fields:
    ``suspension_date`` (暂停上市日期 — SSE's trading-stop date),
    ``delist_effective_date`` (终止上市日期 — SZSE's formal-removal date) and
    ``delist_reason``.  §八-2: the two date fields are NEVER collapsed into one
    Chinese column; the reader prefers the trading-stop date and falls back to
    the formal-removal date only when the former is absent, so a SZSE stock
    (which reports only 终止上市日期) is no longer silently treated as
    never-delisted.

The delisted parquet stores the SSE code only in the Chinese ``公司代码`` column
(stock_code is NaN for SSE), so the canonical code is taken from ``公司代码``
when present.  All codes are zero-padded to 6 digits.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code, normalize_stock_code_series

# New-format delisted.parquet semantic columns (written by IPOStSource).
SUSPENSION_COL = "suspension_date"      # 暂停上市日期 — day trading stopped (SSE)
EFFECTIVE_COL = "delist_effective_date"  # 终止上市日期 — formal removal (SZSE)
REASON_COL = "delist_reason"            # 终止上市原因

# Legacy raw-AKShare column names (older on-disk delisted.parquet formats).
LEGACY_SUSPENSION_COL = "暂停上市日期"
LEGACY_EFFECTIVE_COL = "终止上市日期"
LEGACY_REASON_COL = "终止上市原因"
LEGACY_CODE_COL = "公司代码"

INDEX_MEMBERSHIP_PATH = ("a_shares", "index_constituents_hist", "membership.parquet")
DEFAULT_INDICES = ("000300", "000905")  # CSI300 + CSI500 default download universe


def _read_universe_parquet(data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "a_shares", "universe", name)
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _delisted_codes(delisted: pd.DataFrame) -> pd.Series:
    """Canonical codes from the new-format ``stock_code`` column, else the legacy
    Chinese ``公司代码`` (SSE rows have no stock_code)."""
    if "stock_code" in delisted.columns:
        codes = delisted["stock_code"].copy()
    else:
        codes = pd.Series(np.nan, index=delisted.index)
    if LEGACY_CODE_COL in delisted.columns:
        codes = codes.where(codes.notna(), delisted[LEGACY_CODE_COL])
    return normalize_stock_code_series(codes)


def _coerce_date(src: pd.Series) -> pd.Series:
    """``to_datetime`` on a column of datetime.date + literal ``"None"`` strings
    (AKShare's missing-date representation) — the strings are masked to NaN
    first so parsing neither warns about an un-inferable format nor leaks a
    corrupt timestamp.  Masking is dtype-agnostic: a parquet round-trip can
    store these as pandas 2.x ``str`` dtype, not ``object``."""
    src = src.where(~src.isin(["None", "nan", "NaN", "", "NaT"]), np.nan)
    return pd.to_datetime(src, errors="coerce")


def _delisted_date_col(
    delisted: pd.DataFrame, new_col: str, legacy_col: str
) -> pd.Series:
    """A delist date column from the new format, else the legacy Chinese column."""
    if new_col in delisted.columns:
        src = delisted[new_col]
    elif legacy_col in delisted.columns:
        src = delisted[legacy_col]
    else:
        src = pd.Series(np.nan, index=delisted.index)
    return _coerce_date(src)


def _delisted_col(delisted: pd.DataFrame, col: str, legacy_col: str | None = None) -> pd.Series:
    if col in delisted.columns:
        return delisted[col]
    if legacy_col is not None and legacy_col in delisted.columns:
        return delisted[legacy_col]
    return pd.Series(np.nan, index=delisted.index)


def load_universe_status(data_dir: str) -> pd.DataFrame:
    """Normalized per-stock universe table.

    Columns: ``stock_code, list_date, delist_date, suspension_date,
    delist_effective_date, delist_reason``.  ``delist_date`` is NaN for stocks
    still listed; ``list_date`` may be NaN when the IPO record is missing.
    All dates are ``datetime64[ns]``.

    ``delist_date`` is the resolved EXIT date the force-sell logic consumes:
    the trading-stop (suspension) date when present, else the formal-removal
    (delist-effective) date.  The two source fields stay in SEPARATE columns
    (§八-2) so no single Chinese field silently decides every exit — SSE and
    SZSE report different fields, and collapsing them into one assumption was
    dropping every SZSE delisting as "never delisted".
    """
    ipo = _read_universe_parquet(data_dir, "ipo.parquet")
    delisted = _read_universe_parquet(data_dir, "delisted.parquet")

    recs: dict[str, dict[str, object]] = {}
    if not ipo.empty:
        for code, ld in zip(
            normalize_stock_code_series(ipo["stock_code"]), ipo["list_date"]
        ):
            if pd.isna(code):
                continue
            recs.setdefault(code, {})["list_date"] = ld
    if not delisted.empty:
        codes = _delisted_codes(delisted)
        susp = _delisted_date_col(delisted, SUSPENSION_COL, LEGACY_SUSPENSION_COL)
        eff = _delisted_date_col(delisted, EFFECTIVE_COL, LEGACY_EFFECTIVE_COL)
        reason = _delisted_col(delisted, REASON_COL, LEGACY_REASON_COL)
        for code, sp, ef, rs in zip(codes, susp, eff, reason):
            if pd.isna(code):
                continue
            rec = recs.setdefault(code, {})
            if pd.notna(sp) or pd.notna(ef):
                rec["suspension_date"] = sp
                rec["delist_effective_date"] = ef
                # Conservative exit: a stock stops being tradeable at its
                # suspension (SSE); SZSE reports only the formal removal, so
                # that date stands in as the last-trading-day proxy.
                rec["delist_date"] = sp if pd.notna(sp) else ef
            if pd.notna(rs):
                rec["delist_reason"] = rs

    rows = [
        {
            "stock_code": code,
            "list_date": pd.to_datetime(v.get("list_date"), errors="coerce"),
            "delist_date": pd.to_datetime(v.get("delist_date"), errors="coerce"),
            "suspension_date": pd.to_datetime(
                v.get("suspension_date"), errors="coerce"),
            "delist_effective_date": pd.to_datetime(
                v.get("delist_effective_date"), errors="coerce"),
            "delist_reason": v.get("delist_reason"),
        }
        for code, v in recs.items()
    ]
    return pd.DataFrame(rows, columns=[
        "stock_code", "list_date", "delist_date",
        "suspension_date", "delist_effective_date", "delist_reason",
    ])


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
    codes = normalize_stock_code_series(mem["stock_code"]).dropna()
    return sorted(codes.astype(str).unique())


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
    row_of = {}
    for i, c in enumerate(codes):
        norm = normalize_stock_code(c)
        if norm is not None:
            row_of[norm] = i
    grid = np.asarray(global_dates)
    in_dates = pd.to_datetime(membership["in_date"], errors="coerce")
    out_dates = pd.to_datetime(membership["out_date"], errors="coerce")
    for code, in_d, out_d in zip(
        normalize_stock_code_series(membership["stock_code"]),
        in_dates, out_dates,
    ):
        row = row_of.get(code) if not pd.isna(code) else None
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
