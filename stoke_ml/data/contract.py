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
* ``validate_pct_change`` — ``pct_change`` must be finite except on the first
  (listing) row; a mid-series NaN means the returns column was repaired or
  derived inconsistently.  In the qfq research frame it must also equal
  ``100*(close[t]/close[t-1]-1)`` within tolerance, so a close correction that
  left the adjacent return stale is caught (v14 §十二/§十三).
* ``validate_price_volume_amount_consistency`` — a DIAGNOSTIC economic check
  that ``amount / volume`` (the day's raw nominal VWAP) stays within a 100×
  band of the QFQ ``close``, catching a volume/amount unit corruption (手 vs 股,
  千元 vs 元).  The band is deliberately loose because raw-vs-qfq price scale
  differs across history; it is not a proof of correct units.  Two
  scale-independent pairings are hard rejections: ``volume==0 && amount>0`` and
  ``volume>0 && amount==0`` (v14 §十二).
* ``validate_source_metadata`` — a ``source`` column, when present, must be
  non-empty, and an ``adjustment_mode`` column must hold a legal value.
* ``validate_required_metadata`` — contracts may declare ``required_metadata``
  (e.g. daily K-line requires ``source`` + ``adjustment_mode``) that must be
  declared somewhere: in ``df.attrs``, as a non-empty column, or in the
  strongly-bound manifest.  Legacy files honestly record ``unknown`` — that IS
  a declaration and passes; a file with no source/adjust anywhere is not
  canonical daily.
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
    # Provenance metadata that must be declared for this dataset — either in
    # df.attrs, as a non-empty column, or in the strongly-bound manifest
    # (e.g. daily K-line requires source + adjustment_mode).
    required_metadata: tuple[str, ...] = ()
    #: Columns that MAY be present but are not required (e.g. the market_env
    #: ACCOUNT part is proxy-PIT and ablation-only — a file missing them is
    #: still schema-valid).  Enforcement: required columns must all be present;
    #: optional columns are never demanded (§v19-11).
    optional_columns: tuple[str, ...] = ()


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
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue  # all-non-finite is reported by validate_finite
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

    ``Inf``/``-Inf`` are NOT valid numbers: ``notna()`` counts them as present,
    so the finite ratio must use ``np.isfinite`` after numeric coercion.
    Suspension is represented by absent rows, not NaN/Inf OHLC rows, so a frame
    below the finite threshold is corrupt, not "suspended".
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
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        finite = int(np.isfinite(values).sum())
        ratio = finite / len(values)
        if ratio < min_ratio:
            out.append(f"{col}_finite_ratio:{ratio:.4f}<{min_ratio}")
    return out


def validate_ohlc(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """low <= open/close <= high on every bar.

    ``Inf`` is not a legal price: bars carrying any non-finite price are
    excluded from the ordering comparison and reported via ``validate_finite``
    (which rejects an all-Inf OHLC file outright).
    """
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
        finite = np.isfinite(low) & np.isfinite(high) & np.isfinite(v)
        n_low = int(((v < low - TOL) & finite).sum())
        n_high = int(((v > high + TOL) & finite).sum())
        if n_low:
            out.append(f"{c}<low:{n_low}")
        if n_high:
            out.append(f"{c}>high:{n_high}")
    return out


# Tolerance (pp) for the §十二/§十三 arithmetic identity
# ``pct_change == 100*(close[t]/close[t-1]-1)``.  The repo-wide data satisfies
# the identity to ~1e-9; the tolerance absorbs source rounding without letting
# a genuinely stale return (a close correction that did not re-derive the
# adjacent pct_change) pass.
PCT_CHANGE_TOLERANCE_PP = 0.5


def validate_pct_change(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """``pct_change`` must be finite except on the first (listing) row, and
    (v14 §十二/§十三) must equal the qfq-close arithmetic return.

    A-share ``pct_change`` is undefined for the first trading day of a stock,
    so row 0 may be NaN; every later row must be finite.  A mid-series NaN
    means the column was repaired/derived inconsistently (e.g. a gap filled
    with NaN instead of the real cross-gap return), which would corrupt the
    returns a model trains on.  A fully missing column is reported by
    ``validate_schema``; an all-NaN column is caught here by the tail check
    (every non-first row NaN), so ``pct_change`` needs no entry in
    ``required_finite_ratio`` (which would reject a legit 2-row listing file).

    The arithmetic identity check only applies inside the qfq research frame
    (the one contract that pins ``pct_change``): there ``close[t]/close[t-1]``
    is the true adjusted return, so ``pct_change[t] == 100*(ratio - 1)`` must
    hold within ``PCT_CHANGE_TOLERANCE_PP``.  A corrected close that left the
    adjacent day's return based on the pre-correction price fails here instead
    of reaching the canonical store.  The check is skipped when ``close`` is
    absent or the neighbouring closes are non-finite (reported elsewhere).
    """
    if "pct_change" not in contract.required_columns:
        return []
    if "pct_change" not in df.columns:
        return []  # validate_schema reports the missing column
    values = pd.to_numeric(df["pct_change"], errors="coerce").to_numpy(dtype="float64")
    n = len(values)
    if n <= 1:
        return []  # a lone listing-day row may legitimately be NaN
    n_nan_after = int((~np.isfinite(values[1:])).sum())
    if n_nan_after:
        return [f"pct_change_nan_after_first_row:{n_nan_after}"]
    if "close" not in df.columns:
        return []  # validate_schema reports the missing column
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
    prev, cur = close[:-1], close[1:]
    valid = np.isfinite(prev) & np.isfinite(cur) & (prev != 0)
    if not valid.any():
        return []
    expected = (cur[valid] / prev[valid] - 1.0) * 100.0
    actual = values[1:][valid]
    n_bad = int((np.abs(actual - expected) > PCT_CHANGE_TOLERANCE_PP).sum())
    return [f"pct_change_inconsistent:{n_bad}"] if n_bad else []


def validate_price_volume_amount_consistency(
    df: pd.DataFrame,
    contract: DataContract,
) -> list[str]:
    """Diagnostic economic check: ``amount / volume`` ≈ ``close`` within a
    100× band, plus hard zero/positive mutual exclusions (v14 §十二).

    ``implied_price = amount / volume`` is the day's VWAP in RAW (unadjusted)
    nominal yuan/shares, while ``close`` is the QFQ (forward-adjusted) research
    price.  These two are NOT guaranteed to share a price scale: a stock that
    split or paid dividends since a historical bar has its qfq close scaled far
    from the nominal price that day's amount/volume were recorded in.  The 100×
    band is therefore deliberately loose and this check is DIAGNOSTIC — it
    flags the classic unit corruptions (volume 手 vs 股, amount 千元 vs 元) but is
    NOT a proof of correct units: a legitimately wide qfq-vs-raw scale gap can
    fall inside the band while a modest unit error can hide outside it.

    Two pairings are scale-independent and economically impossible, so they are
    HARD rejections rather than diagnostics:
      * ``volume == 0 and amount > 0``  — a zero-volume day has no trades, so it
        cannot carry traded value (``amount_without_volume``).
      * ``volume > 0 and amount == 0``  — a positive-volume day must have traded
        value (``volume_without_amount``).
    Rows where any of the six columns is non-finite are deferred to
    ``validate_finite`` / ``validate_units`` (NaN, non-positive prices and
    negative volume/amount are reported there, not here).
    """
    cols = ("open", "high", "low", "close", "volume", "amount")
    if not set(cols).issubset(contract.required_columns):
        return []
    if not set(cols).issubset(df.columns):
        return []  # validate_schema reports the missing columns
    num = {
        c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        for c in cols
    }
    out: list[str] = []
    finite = np.ones(len(df), dtype=bool)
    for c in cols:
        finite &= np.isfinite(num[c])
    vol, amt = num["volume"], num["amount"]
    n_vol_zero_amt_pos = int(((vol == 0) & (amt > 0) & finite).sum())
    if n_vol_zero_amt_pos:
        out.append(f"amount_without_volume:{n_vol_zero_amt_pos}")
    n_vol_pos_amt_zero = int(((vol > 0) & (amt == 0) & finite).sum())
    if n_vol_pos_amt_zero:
        out.append(f"volume_without_amount:{n_vol_pos_amt_zero}")
    # Band check on the rows that actually traded (volume > 0 and amount > 0);
    # the zero/positive mismatches above are already reported.
    ok = finite & (vol > 0) & (amt > 0)
    if not ok.any():
        return out
    implied = amt[ok] / vol[ok]
    close = num["close"][ok]
    n = int(((implied < close / 100.0) | (implied > close * 100.0)).sum())
    if n:
        out.append(f"amount_volume_unit_mismatch:{n}")
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


def validate_required_metadata(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    manifest: dict | None = None,
) -> list[str]:
    """Each ``required_metadata`` key must be declared for the dataset.

    ``required_metadata`` lives in the CONTRACT (this is the schema-level
    statement of what provenance a canonical file must carry) while the value
    lives in the manifest (for on-disk daily files) or in ``df.attrs`` (for
    freshly-fetched frames).  A key counts as declared when it is present
    non-empty in any of:

    * ``df.attrs`` (the frame carries provenance, e.g. a downloader stamped it)
    * a non-empty column of the same name (raw source frames)
    * the strongly-bound ``manifest`` dict, when the caller has one

    ``unknown`` counts as declared — legacy files honestly cannot recover their
    real provenance, and the manifest records ``unknown`` truthfully.  The
    point is to reject files with NO source/adjustment declaration anywhere,
    not to demand a specific value.
    """
    # The storage manifest records the adjustment basis under "adjust"
    # (mirroring _provenance_from_attrs), while the contract names the
    # metadata column "adjustment_mode".
    _MANIFEST_ALIASES = {"adjustment_mode": "adjust"}
    declared: set[str] = set()
    for key in contract.required_metadata:
        if df.attrs.get(key):
            declared.add(key)
        elif key in df.columns and (df[key].astype("string").fillna("").str.strip() != "").any():
            declared.add(key)
        elif manifest:
            mkey = _MANIFEST_ALIASES.get(key, key)
            if manifest.get(mkey):
                declared.add(key)
    return [f"missing_metadata:{k}" for k in contract.required_metadata
            if k not in declared]


def validate_adjustment_mode(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    manifest: dict | None = None,
    formal: bool = False,
) -> list[str]:
    """The declared adjustment basis must actually match the contract's mode.

    ``RESEARCH_QFQ_DAILY`` pins ``adjustment_mode="qfq"``; a frame that
    explicitly declares ``raw`` (in ``df.attrs``, an ``adjustment_mode`` column,
    or the storage manifest's ``adjust`` field) must NOT pass the QFQ contract.
    ``unknown`` is the honest legacy declaration for pre-provenance files and is
    not treated as a concrete mismatch — the economic-semantics check only fires
    on an explicit declaration of the WRONG basis (§四.1 / P0-2).

    ``formal=True`` (v13 §五-1, canonical write/read paths) closes the legacy
    seam: when the contract pins a concrete basis, ANY declared
    ``unknown``/``n/a``/``""`` is itself a violation
    (``adjustment_mode_unknown_in_formal``) — a formal QFQ store must never
    accept files whose basis cannot be proven.  ``formal=False`` (default)
    keeps the legacy exemption for migration / audit / exploration.
    """
    if contract.adjustment_mode in ("", "n/a"):
        return []
    actual: set[str] = set()
    # Use "in attrs" (not truthiness) so an empty-string declaration is still
    # collected — formal mode must reject a declared "" basis, not skip it.
    if df.attrs.get("adjustment_mode") is not None:
        actual.add(str(df.attrs["adjustment_mode"]))
    if "adjustment_mode" in df.columns:
        actual.update(str(m) for m in df["adjustment_mode"].dropna().unique())
    if manifest and manifest.get("adjust"):
        actual.add(str(manifest["adjust"]))
    concrete = {m for m in actual if m not in ("unknown", "n/a", "")}
    bad = sorted(f"{m}!={contract.adjustment_mode}" for m in concrete
                 if m != contract.adjustment_mode)
    out = [f"adjustment_mode_mismatch:{b}" for b in bad]
    if formal:
        unknown = {m for m in actual if m in ("unknown", "n/a", "")}
        if unknown:
            out.append("adjustment_mode_unknown_in_formal")
    return out


def validate_contract(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    code: str | None = None,
    trading_days: set | None = None,
    manifest: dict | None = None,
    formal: bool = False,
) -> list[str]:
    """Run every validator; return a flat list of violation strings.

    ``manifest`` is optional; when supplied (e.g. from the storage layer's
    per-stock sidecar) it satisfies ``required_metadata`` declarations, so a
    legacy daily file whose parquet carries no provenance still passes as long
    as its manifest records source/adjust.

    ``formal=True`` enforces the strict provenance rule for the canonical
    write/read paths (v13 §五-1): an unknown adjustment basis is refused
    instead of treated as an honest legacy declaration.  ``formal=False``
    (default) keeps the legacy exemption for migration / audit.
    """
    out = []
    out += validate_schema(df, contract)
    out += validate_primary_key(df, contract)
    out += validate_dates(df, contract, trading_days=trading_days)
    out += validate_units(df, contract)
    out += validate_finite(df, contract)
    out += validate_pct_change(df, contract)
    out += validate_ohlc(df, contract)
    out += validate_price_volume_amount_consistency(df, contract)
    out += validate_source_metadata(df, contract)
    out += validate_required_metadata(df, contract, manifest=manifest)
    out += validate_adjustment_mode(df, contract, manifest=manifest, formal=formal)
    return [f"{code}:{v}" if code else v for v in out]


# ── frozen contracts ──────────────────────────────────────────────────────

# Price basis is split across distinct contracts.  On-disk daily
# K-line is the forward-adjusted (qfq) research series, so
# ``daily_equity`` aliases ``research_qfq_daily`` rather than pretending the
# same schema covers unadjusted data too.
DAILY_PRICE_COLUMNS = ("open", "high", "low", "close")
# §五-2 (v13): volume/amount joined the finite set so an all-NaN volume/amount
# column (previously let through by validate_units' all-NaN skip) is now a
# contract violation — a trading file without trade size/value is corrupt.
DAILY_REQUIRED_FINITE = {
    "open": 0.99, "high": 0.99, "low": 0.99, "close": 0.99,
    "volume": 0.99, "amount": 0.99,
}

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
            "optional column; absent for ~850 STAR-market stocks.  The feature "
            "pipeline does not synthesize a turnover proxy from amount/close "
            "any more (§五 P0: amount/qfq_close is not scale-invariant); "
            "volume_ratio / amount_ratio carry the liquidity signal instead."
        ),
        "pct_change": "NaN on the first listing day",
    },
    price_columns=DAILY_PRICE_COLUMNS,
    required_finite_ratio=DAILY_REQUIRED_FINITE,
    minimum_valid_rows=1,
    required_metadata=("source", "adjustment_mode"),
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
    required_metadata=("source", "adjustment_mode"),
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
        "date", "stock_code", "stock_name", "lhb_reason",
        "buy_amount", "sell_amount", "net_amount",
    ),
    units={"net_amount": "CNY"},
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
        "roe", "eps", "revenue_yoy", "profit_yoy",
        "debt_ratio", "gross_margin", "net_margin",
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
    # Financial institutions (banks / brokers / insurers) do not disclose
    # liquidity (流动比率) or asset-return (总资产报酬率) ratios — the source
    # emits no such rows for ~90 such issuers, so these columns are present
    # only when the issuer reports them (§v19 P1#5, #78).
    optional_columns=("roa", "current_ratio"),
)

MARKET_ENV = DataContract(
    dataset_name="market_env_daily",
    primary_key=("date",),  # DatetimeIndex-backed broadcast file
    required_columns=(
        "high_low_ratio", "market_adv_ratio", "market_turnover_z",
    ),
    # The ACCOUNT part is proxy-PIT and ablation-only (§T5): a market_env file
    # written before account_stats coverage — or by a builder that only ships
    # the PRICE part — is still schema-valid.  Never demanded.
    optional_columns=(
        "mkt_cap_total_z", "avg_account_cap_z",
        "investor_new_num", "investor_new_z",
    ),
    units={},
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    allowed_missingness={
        "mkt_cap_total_z": "account part absent before account_stats coverage",
        "avg_account_cap_z": "account part absent before account_stats coverage",
        "investor_new_num": "account part absent before account_stats coverage",
        "investor_new_z": "account part absent before account_stats coverage",
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
    MARKET_ENV.dataset_name: MARKET_ENV,
}


def get_contract(name: str) -> DataContract:
    try:
        return CONTRACTS[name]
    except KeyError:
        raise KeyError(
            f"unknown data contract: {name!r} (available: {sorted(CONTRACTS)})"
        ) from None
