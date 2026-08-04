"""Data contracts — frozen, machine-enforced dataset schemas.

Each dataset (daily K-line, margin, northbound, ...) is described by one
``DataContract`` so downloaders, storage, the quality gate and feature builders
share a single source of truth for schema, primary key, units, price basis,
timezone and calendar — instead of each module relying on local comments and
experience.

The contracts harden the schemas from documentation into real constraints:

* ``validate_finite`` — key columns carry a required finite ratio and the frame
  a minimum valid row count, so an all-NaN OHLC file is corrupt, not "valid".
  Suspension days are represented by ABSENT rows, never NaN OHLC rows.
* ``validate_ohlc`` — ``low <= open/close <= high`` on every bar.
* ``validate_source_metadata`` — a ``source`` column, when present, must be
  non-empty, and an ``adjustment_mode`` column must hold a legal value.
* ``validate_dates(..., trading_days=...)`` — optional membership in the
  official trading calendar.
* Price basis is split across DISTINCT contracts (``raw_unadjusted_daily`` /
  ``adjustment_factor`` / ``research_qfq_daily``) instead of one ``daily_equity``
  silently covering every price system.  On-disk daily K-line is the qfq
  research series, so ``daily_equity`` aliases it.

``CONTRACTS`` holds the frozen contracts; ``get_contract(name)`` looks one up.
The validation helpers return a flat list of violation strings (empty == valid)
so callers can both fail a quality gate and print the offending rows cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataContract:
    """One dataset's frozen contract."""

    dataset_name: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    units: dict[str, str]
    price_basis: str
    timezone: str
    calendar: str
    source_priority: tuple[str, ...] = ()
    allowed_missingness: dict[str, str] = field(default_factory=dict)
    # Enforced numeric constraints.
    price_columns: tuple[str, ...] = ()
    required_finite_ratio: dict[str, float] = field(default_factory=dict)
    minimum_valid_rows: int = 0
    adjustment_mode: str = "n/a"


# ── validators ────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Missing required columns."""
    return [f"missing_column:{c}" for c in contract.required_columns
            if c not in df.columns]


def validate_primary_key(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Null or duplicated primary-key values."""
    pk = list(contract.primary_key)
    missing = [c for c in pk if c not in df.columns]
    if missing:
        return [f"pk_missing_column:{c}" for c in missing]
    sub = df[pk]
    out = []
    n_null = int(sub.isna().any(axis=1).sum())
    if n_null:
        out.append(f"pk_null:{n_null}")
    n_dup = int(sub.duplicated().sum())
    if n_dup:
        out.append(f"pk_dup:{n_dup}")
    return out


def validate_dates(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    trading_days: set | None = None,
) -> list[str]:
    """Dates parse cleanly, and — for date-keyed daily datasets — are sorted
    and never fall on a weekend (A-shares never trade weekends).

    The date column comes from the contract (``date`` for daily datasets,
    ``report_date`` for quarterly fundamentals); contracts with no date column
    skip date validation entirely.

    ``trading_days``: an optional set of ``datetime.date`` objects
    (or ISO date strings) from the official calendar.  When supplied, dates
    outside it are flagged — catching rows on exchange holidays.  Passing a
    prebuilt set avoids per-file calendar construction in batch loops.
    """
    date_col = next(
        (c for c in ("date", "report_date") if c in contract.required_columns), None
    )
    if date_col is None:
        return []
    if date_col not in df.columns:
        return ["missing_date_column"]
    d = pd.to_datetime(df[date_col], errors="coerce")
    if d.isna().any():
        return [f"na_date:{int(d.isna().sum())}"]
    out = []
    if not d.is_monotonic_increasing:
        out.append("dates_not_sorted")
    if "date" in contract.primary_key:
        n_wk = int(d.dt.dayofweek.isin([5, 6]).sum())
        if n_wk:
            out.append(f"weekend_dates:{n_wk}")
        if trading_days is not None:
            if trading_days and isinstance(next(iter(trading_days)), str):
                keyed = d.dt.strftime("%Y-%m-%d")
            else:
                keyed = d.dt.date
            n_off = int((~keyed.isin(trading_days)).sum())
            if n_off:
                out.append(f"non_trading_day:{n_off}")
    return out


def validate_units(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Sign/range sanity derived from the unit mapping.

    Fully verifying a unit's magnitude (e.g. that ``volume`` is really shares,
    not lots) needs an independent reference, but the sign constraints catch the
    classic A-share corruption signatures: non-positive prices, negative
    volume/amount.  An all-NaN column is caught by ``validate_finite`` rather
    than silently skipped here.
    """
    out = []
    for col, unit in contract.units.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite = x[~pd.isna(x)]
        if finite.size == 0:
            continue
        if unit == "price":
            n = int((finite <= 0).sum())
            if n:
                out.append(f"{col}<=0:{n}")
        elif unit in ("shares", "CNY", "yuan", "shares_float", "lots"):
            n = int((finite < 0).sum())
            if n:
                out.append(f"{col}<0:{n}")
    return out


def validate_finite(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Required finite ratios + minimum valid rows.

    Previously an all-NaN OHLC file passed because schema only checked column
    existence and units skipped zero-finite columns.  Suspension is represented
    by absent rows, not NaN OHLC rows, so a frame below the finite threshold is
    corrupt, not "suspended".
    """
    out = []
    if contract.minimum_valid_rows and len(df) < contract.minimum_valid_rows:
        out.append(f"too_few_rows:{len(df)}<{contract.minimum_valid_rows}")
    for col, min_ratio in contract.required_finite_ratio.items():
        if col not in df.columns:
            continue  # validate_schema reports the missing column
        if len(df) == 0:
            out.append(f"{col}_finite_ratio:0.0<{min_ratio}")
            continue
        ratio = float(df[col].notna().mean())
        if ratio < min_ratio:
            out.append(f"{col}_finite_ratio:{ratio:.4f}<{min_ratio}")
    return out


def validate_ohlc(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """low <= open/close <= high on every bar."""
    if "low" not in contract.price_columns or "high" not in contract.price_columns:
        return []
    cols = [c for c in contract.price_columns if c in df.columns]
    if not cols:
        return []
    out = []
    TOL = 1e-9
    low = pd.to_numeric(df.get("low"), errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(df.get("high"), errors="coerce").to_numpy(dtype="float64")
    for c in cols:
        if c in ("low", "high"):
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        nan = np.isnan(low) | np.isnan(high) | np.isnan(v)
        n_low = int(((v < low - TOL) & ~nan).sum())
        n_high = int(((v > high + TOL) & ~nan).sum())
        if n_low:
            out.append(f"{c}<low:{n_low}")
        if n_high:
            out.append(f"{c}>high:{n_high}")
    return out


VALID_ADJUSTMENT_MODES = {"raw", "none", "qfq", "hfq", "n/a"}


def validate_source_metadata(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """source column non-empty; adjustment_mode column legal."""
    out = []
    if "source" in df.columns:
        s = df["source"].astype("string")
        n_empty = int((s.isna() | (s.str.strip() == "")).sum())
        if n_empty:
            out.append(f"source_empty:{n_empty}")
    if "adjustment_mode" in df.columns:
        modes = df["adjustment_mode"].dropna().unique()
        bad = sorted(str(m) for m in modes if m not in VALID_ADJUSTMENT_MODES)
        if bad:
            out.append(f"adjustment_mode_invalid:{','.join(bad)}")
    return out


def validate_contract(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    code: str | None = None,
    trading_days: set | None = None,
) -> list[str]:
    """Run every validator; return a flat list of violation strings."""
    out = []
    out += validate_schema(df, contract)
    out += validate_primary_key(df, contract)
    out += validate_dates(df, contract, trading_days=trading_days)
    out += validate_units(df, contract)
    out += validate_finite(df, contract)
    out += validate_ohlc(df, contract)
    out += validate_source_metadata(df, contract)
    return [f"{code}:{v}" if code else v for v in out]


# ── frozen contracts ──────────────────────────────────────────────────────

# Price basis is split across distinct contracts.  On-disk daily
# K-line is the forward-adjusted (qfq) research series, so
# ``daily_equity`` aliases ``research_qfq_daily`` rather than pretending the
# same schema covers unadjusted data too.
DAILY_PRICE_COLUMNS = ("open", "high", "low", "close")
DAILY_REQUIRED_FINITE = {"open": 0.99, "high": 0.99, "low": 0.99, "close": 0.99}

RESEARCH_QFQ_DAILY = DataContract(
    dataset_name="research_qfq_daily",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "open", "high", "low", "close",
        "volume", "amount", "pct_change",
    ),
    units={
        "open": "price", "high": "price", "low": "price", "close": "price",
        "volume": "shares", "amount": "CNY", "turnover": "percent",
    },
    price_basis="qfq",
    adjustment_mode="qfq",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    source_priority=("efinance", "akshare", "tushare", "baostock"),
    allowed_missingness={
        "turnover": (
            "optional column; absent for ~850 STAR-market stocks, the feature "
            "pipeline derives turnover_proxy from amount/close"
        ),
        "pct_change": "NaN on the first listing day",
    },
    price_columns=DAILY_PRICE_COLUMNS,
    required_finite_ratio=DAILY_REQUIRED_FINITE,
    minimum_valid_rows=1,
)

# daily_equity == the research qfq series (see contract docstring).
DAILY_EQUITY = RESEARCH_QFQ_DAILY

RAW_UNQUOTED_DAILY = DataContract(
    dataset_name="raw_unadjusted_daily",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "open", "high", "low", "close",
        "volume", "amount",
    ),
    units={
        "open": "price", "high": "price", "low": "price", "close": "price",
        "volume": "shares", "amount": "CNY",
    },
    price_basis="unadjusted",
    adjustment_mode="raw",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    source_priority=("efinance", "akshare", "tushare", "baostock"),
    price_columns=DAILY_PRICE_COLUMNS,
    required_finite_ratio=DAILY_REQUIRED_FINITE,
    minimum_valid_rows=1,
)

ADJUSTMENT_FACTOR = DataContract(
    dataset_name="adjustment_factor",
    primary_key=("stock_code", "date"),
    required_columns=("stock_code", "date", "qfq_factor", "hfq_factor"),
    units={"qfq_factor": "ratio", "hfq_factor": "ratio"},
    price_basis="n/a",
    adjustment_mode="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    allowed_missingness={
        "hfq_factor": "optional; absent for stocks with no dividends",
    },
)

MARGIN = DataContract(
    dataset_name="margin",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "margin_balance", "margin_buy", "margin_repay",
        "short_balance", "short_sell_vol", "short_repay_vol", "margin_net",
    ),
    units={
        "margin_balance": "CNY", "margin_buy": "CNY", "margin_repay": "CNY",
        "short_balance": "CNY", "short_sell_vol": "shares",
        "short_repay_vol": "shares", "margin_net": "CNY",
    },
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    source_priority=("eastmoney",),
)

NORTHBOUND = DataContract(
    dataset_name="northbound",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "north_hold_shares", "north_hold_value",
        "north_hold_pct", "north_net_buy",
    ),
    units={
        "north_hold_shares": "shares", "north_hold_value": "CNY",
        "north_hold_pct": "percent", "north_net_buy": "CNY",
    },
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    source_priority=("eastmoney",),
)

DRAGON_TIGER = DataContract(
    dataset_name="dragon_tiger",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "type", "reason", "net_buy",
    ),
    units={"net_buy": "CNY"},
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    source_priority=("eastmoney",),
)

FUNDAMENTALS = DataContract(
    dataset_name="fundamentals",
    primary_key=("stock_code", "report_date"),
    required_columns=(
        "stock_code", "report_date", "disclose_date",
        "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
        "debt_ratio", "current_ratio", "gross_margin", "net_margin",
        "total_revenue", "net_profit",
    ),
    units={},
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="quarterly",
    source_priority=("eastmoney", "akshare"),
    allowed_missingness={
        "roe": "NaN before first disclosure",
        "eps": "NaN for loss-making periods without EPS",
    },
)

CONTRACTS = {
    RESEARCH_QFQ_DAILY.dataset_name: RESEARCH_QFQ_DAILY,
    "daily_equity": RESEARCH_QFQ_DAILY,  # alias
    RAW_UNQUOTED_DAILY.dataset_name: RAW_UNQUOTED_DAILY,
    ADJUSTMENT_FACTOR.dataset_name: ADJUSTMENT_FACTOR,
    MARGIN.dataset_name: MARGIN,
    NORTHBOUND.dataset_name: NORTHBOUND,
    DRAGON_TIGER.dataset_name: DRAGON_TIGER,
    FUNDAMENTALS.dataset_name: FUNDAMENTALS,
}


def get_contract(name: str) -> DataContract:
    try:
        return CONTRACTS[name]
    except KeyError:
        raise KeyError(
            f"unknown data contract: {name!r} (available: {sorted(CONTRACTS)})"
        ) from None
