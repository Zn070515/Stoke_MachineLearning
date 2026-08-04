"""Data contracts — frozen, machine-enforced dataset schemas (v6 §十六).

Each dataset (daily K-line, margin, northbound, ...) is described by one
``DataContract`` so downloaders, storage, the quality gate and feature builders
share a single source of truth for schema, primary key, units, price basis,
timezone and calendar — instead of each module relying on local comments and
experience.

``CONTRACTS`` holds the frozen contracts; ``get_contract(name)`` looks one up.
The validation helpers return a flat list of violation strings (empty == valid)
so callers can both fail a quality gate and print the offending rows cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DataContract:
    """One dataset's frozen contract (field names mirror v6 §十六)."""

    dataset_name: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    units: dict[str, str]
    price_basis: str
    timezone: str
    calendar: str
    source_priority: tuple[str, ...] = ()
    allowed_missingness: dict[str, str] = field(default_factory=dict)


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


def validate_dates(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Dates parse cleanly, and — for date-keyed daily datasets — are sorted
    and never fall on a weekend (A-shares never trade weekends).

    The date column comes from the contract (``date`` for daily datasets,
    ``report_date`` for quarterly fundamentals); contracts with no date column
    skip date validation entirely.
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
    return out


def validate_units(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Sign/range sanity derived from the unit mapping.

    Fully verifying a unit's magnitude (e.g. that ``volume`` is really shares,
    not lots) needs an independent reference, but the sign constraints catch the
    classic A-share corruption signatures: non-positive prices, negative
    volume/amount.
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


def validate_contract(
    df: pd.DataFrame, contract: DataContract, *, code: str | None = None
) -> list[str]:
    """Run every validator; return a flat list of violation strings."""
    out = []
    out += validate_schema(df, contract)
    out += validate_primary_key(df, contract)
    out += validate_dates(df, contract)
    out += validate_units(df, contract)
    return [f"{code}:{v}" if code else v for v in out]


# ── frozen contracts ──────────────────────────────────────────────────────

DAILY_EQUITY = DataContract(
    dataset_name="daily_equity",
    primary_key=("stock_code", "date"),
    required_columns=(
        "date", "stock_code", "open", "high", "low", "close",
        "volume", "amount", "pct_change",
    ),
    units={
        "open": "price", "high": "price", "low": "price", "close": "price",
        "volume": "shares", "amount": "CNY", "turnover": "percent",
    },
    price_basis="unadjusted",
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
    DAILY_EQUITY.dataset_name: DAILY_EQUITY,
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
