"""DataContract tests.

The frozen contract is the single source of truth for schema / primary key /
units / date rules.  Validators return a flat list of violation strings (empty
== valid), and ``get_contract`` resolves the registry by dataset name.
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.contract import (
    ADJUSTMENT_FACTOR,
    CONTRACTS,
    DAILY_EQUITY,
    RAW_UNQUOTED_DAILY,
    RESEARCH_QFQ_DAILY,
    DataContract,
    get_contract,
    validate_contract,
    validate_dates,
    validate_finite,
    validate_ohlc,
    validate_primary_key,
    validate_required_metadata,
    validate_schema,
    validate_source_metadata,
    validate_units,
)


def _daily(drop=(), **over):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "stock_code": "600519",
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume": [1e6, 2e6, 3e6],
        "amount": [1e7, 2e7, 3e7],
        "turnover": [0.5, 0.6, 0.7],
        "pct_change": [1.0, 2.0, 3.0],
    })
    df = df.drop(columns=[c for c in drop if c in df.columns])
    for k, v in over.items():
        df[k] = v
    # Daily K-line requires source/adjustment_mode provenance (contract
    # required_metadata); downloaders stamp these into attrs, so mirror that.
    df.attrs["source"] = "efinance"
    df.attrs["adjustment_mode"] = "qfq"
    return df


class TestValidators:
    def test_schema_missing_column(self):
        assert validate_schema(_daily(drop=("open",)), DAILY_EQUITY) == [
            "missing_column:open"
        ]

    def test_schema_ok(self):
        assert validate_schema(_daily(), DAILY_EQUITY) == []

    def test_primary_key_dup(self):
        df = _daily()
        df.loc[1, "date"] = df.loc[0, "date"]
        assert "pk_dup:1" in validate_primary_key(df, DAILY_EQUITY)

    def test_primary_key_null(self):
        df = _daily(stock_code=[None, "600519", "600519"])
        assert any(v.startswith("pk_null") for v in validate_primary_key(df, DAILY_EQUITY))

    def test_primary_key_missing_col(self):
        assert validate_primary_key(_daily(drop=("stock_code",)), DAILY_EQUITY) == [
            "pk_missing_column:stock_code"
        ]

    def test_dates_weekend(self):
        df = _daily(date=["2024-01-06", "2024-01-07", "2024-01-08"])  # Sat/Sun/Mon
        assert "weekend_dates:2" in validate_dates(df, DAILY_EQUITY)

    def test_dates_unsorted(self):
        df = _daily(date=["2024-01-04", "2024-01-02", "2024-01-03"])
        assert "dates_not_sorted" in validate_dates(df, DAILY_EQUITY)

    def test_dates_na(self):
        df = _daily(date=["2024-01-02", None, "2024-01-04"])
        assert validate_dates(df, DAILY_EQUITY) == ["na_date:1"]

    def test_units_negative_volume(self):
        assert "volume<0:1" in validate_units(_daily(volume=[1e6, -2e6, 3e6]), DAILY_EQUITY)

    def test_units_nonpositive_price(self):
        assert "close<=0:1" in validate_units(_daily(close=[10.0, 0.0, 12.0]), DAILY_EQUITY)

    def test_units_ok(self):
        assert validate_units(_daily(), DAILY_EQUITY) == []

    def test_contract_all_clean(self):
        assert validate_contract(_daily(), DAILY_EQUITY) == []

    def test_contract_code_prefix(self):
        df = _daily(close=[10.0, 0.0, 12.0])
        assert "600519:close<=0:1" in validate_contract(df, DAILY_EQUITY, code="600519")


class TestRegistry:
    def test_get_contract(self):
        assert get_contract("daily_equity") is DAILY_EQUITY

    def test_get_contract_unknown_raises(self):
        with pytest.raises(KeyError):
            get_contract("no_such_dataset")

    def test_registry_has_major_datasets(self):
        for name in ("daily_equity", "margin", "northbound", "dragon_tiger", "fundamentals"):
            assert name in CONTRACTS

    def test_daily_contract_matches_real_schema(self):
        c = DAILY_EQUITY
        assert c.primary_key == ("stock_code", "date")
        assert "open" in c.required_columns and "pct_change" in c.required_columns
        assert c.units["volume"] == "shares"
        assert c.units["amount"] == "CNY"
        assert c.units["close"] == "price"
        # On-disk daily K-line is the forward-adjusted (qfq) research
        # series (unified qfq basis), so the contract says qfq — not unadjusted.
        assert c.price_basis == "qfq"
        assert c.adjustment_mode == "qfq"
        assert c.timezone == "Asia/Shanghai"
        assert c.source_priority == ("efinance", "akshare", "tushare", "baostock")
        # OHLC must be ~fully finite; 0-valid-row files are corrupt.
        assert c.required_finite_ratio["close"] == 0.99
        assert c.minimum_valid_rows == 1

    def test_contract_frozen(self):
        with pytest.raises(Exception):
            DAILY_EQUITY.dataset_name = "hacked"  # type: ignore[misc]

    def test_custom_contract(self):
        c = DataContract(
            dataset_name="custom",
            primary_key=("id", "ts"),
            required_columns=("id", "ts", "v"),
            units={"v": "CNY"},
            price_basis="n/a",
            timezone="Asia/Shanghai",
            calendar="x",
        )
        df = pd.DataFrame({"id": [1], "ts": ["a"], "v": [10.0]})
        assert validate_contract(df, c) == []
        df = pd.DataFrame({"id": [1], "ts": ["a"], "v": [-1.0]})
        assert validate_contract(df, c) == ["v<0:1"]


class TestFinite:
    """All-NaN key prices must fail, not silently pass."""

    def test_all_nan_close_fails(self):
        df = _daily(close=[np.nan, np.nan, np.nan])
        out = validate_finite(df, DAILY_EQUITY)
        assert any("close_finite_ratio" in v for v in out)

    def test_partial_nan_above_threshold_fails(self):
        df = _daily(close=[10.0, np.nan, 12.0])  # 2/3 finite = 0.667 < 0.99
        assert any("close_finite_ratio" in v for v in validate_finite(df, DAILY_EQUITY))

    def test_clean_passes(self):
        assert validate_finite(_daily(), DAILY_EQUITY) == []

    def test_empty_frame_fails(self):
        df = _daily().iloc[0:0]
        assert validate_finite(df, DAILY_EQUITY) != []

    def test_too_few_rows_fails(self):
        df = _daily().iloc[0:0]
        out = validate_finite(df, DAILY_EQUITY)
        assert any("too_few_rows" in v for v in out)

    def test_all_nan_close_fails_contract(self):
        df = _daily(close=[np.nan, np.nan, np.nan])
        assert any("close_finite_ratio" in v for v in validate_contract(df, DAILY_EQUITY))

    def test_inf_close_fails(self):
        """Inf is not a finite value — an all-Inf close must fail the ratio."""
        df = _daily(close=[np.inf, np.inf, np.inf])
        out = validate_finite(df, DAILY_EQUITY)
        assert any("close_finite_ratio" in v for v in out)

    def test_mixed_inf_lowers_ratio(self):
        df = _daily(close=[10.0, np.inf, 12.0])  # 2/3 finite = 0.667 < 0.99
        assert any("close_finite_ratio" in v for v in validate_finite(df, DAILY_EQUITY))

    def test_all_inf_ohlc_fails_contract(self):
        """v10 §四-1 dynamic repro: all-Inf OHLC previously returned []."""
        df = _daily(
            open=[np.inf] * 3, high=[np.inf] * 3,
            low=[np.inf] * 3, close=[np.inf] * 3,
        )
        out = validate_contract(df, DAILY_EQUITY)
        assert len(out) >= 4  # open/high/low/close each below the finite ratio
        assert any("open_finite_ratio" in v for v in out)


class TestOhlc:
    """Low <= open/close <= high on every bar."""

    def test_close_above_high_fails(self):
        df = _daily(close=[10.2, 11.2, 999.0])
        out = validate_ohlc(df, DAILY_EQUITY)
        assert any("close>high" in v for v in out)

    def test_open_below_low_fails(self):
        df = _daily(open=[10.0, 1.0, 12.0])
        out = validate_ohlc(df, DAILY_EQUITY)
        assert any("open<low" in v for v in out)

    def test_clean_passes(self):
        assert validate_ohlc(_daily(), DAILY_EQUITY) == []

    def test_nan_bar_skipped(self):
        # A bar with NaN high/low must not false-trigger the relation check.
        df = _daily(high=[10.5, np.nan, 12.5])
        assert validate_ohlc(df, DAILY_EQUITY) == []

    def test_contract_wires_ohlc(self):
        df = _daily(close=[10.2, 11.2, 999.0])
        assert any("close>high" in v for v in validate_contract(df, DAILY_EQUITY))


class TestSourceMetadata:
    """Source column non-empty, adjustment_mode legal."""

    def test_empty_source_fails(self):
        df = _daily(source=["efinance", "", None])
        out = validate_source_metadata(df, DAILY_EQUITY)
        assert any("source_empty" in v for v in out)

    def test_valid_source_passes(self):
        assert validate_source_metadata(_daily(source=["efinance"] * 3), DAILY_EQUITY) == []

    def test_invalid_adjustment_mode_fails(self):
        df = _daily(adjustment_mode=["qfq"] * 3)
        df.loc[1, "adjustment_mode"] = "bogus"
        out = validate_source_metadata(df, DAILY_EQUITY)
        assert any("adjustment_mode_invalid" in v for v in out)

    def test_valid_adjustment_mode_passes(self):
        assert validate_source_metadata(
            _daily(adjustment_mode=["qfq"] * 3), DAILY_EQUITY
        ) == []

    def test_absent_columns_are_noop(self):
        assert validate_source_metadata(_daily(), DAILY_EQUITY) == []


class TestRequiredMetadata:
    """Daily contracts require source/adjustment_mode provenance (file or
    strongly-bound manifest), so a file with no source/adjust anywhere fails."""

    def test_attrs_satisfy_requirement(self):
        # _daily() stamps attrs source/adjustment_mode, mirroring downloaders.
        assert validate_required_metadata(_daily(), DAILY_EQUITY) == []

    def test_column_satisfies_requirement(self):
        df = _daily()
        df.attrs = {}
        df["source"] = "efinance"
        df["adjustment_mode"] = "qfq"
        assert validate_required_metadata(df, DAILY_EQUITY) == []

    def test_manifest_satisfies_requirement(self):
        df = _daily()
        df.attrs = {}
        manifest = {"source": "unknown", "adjust": "unknown"}
        assert validate_required_metadata(df, DAILY_EQUITY, manifest=manifest) == []

    def test_missing_metadata_fails(self):
        df = _daily()
        df.attrs = {}
        out = validate_required_metadata(df, DAILY_EQUITY)
        assert "missing_metadata:source" in out
        assert "missing_metadata:adjustment_mode" in out

    def test_partial_missing_fails(self):
        df = _daily()
        df.attrs = {"source": "efinance"}
        out = validate_required_metadata(df, DAILY_EQUITY)
        assert "missing_metadata:adjustment_mode" in out

    def test_empty_attr_counts_as_missing(self):
        df = _daily()
        df.attrs = {"source": "", "adjustment_mode": "qfq"}
        out = validate_required_metadata(df, DAILY_EQUITY)
        assert "missing_metadata:source" in out

    def test_contract_wires_required_metadata(self):
        df = _daily()
        df.attrs = {}
        out = validate_contract(df, DAILY_EQUITY)
        assert any("missing_metadata:source" in v for v in out)

    def test_contract_manifest_satisfies_required_metadata(self):
        df = _daily()
        df.attrs = {}
        manifest = {"source": "unknown", "adjust": "unknown"}
        assert validate_contract(df, DAILY_EQUITY, manifest=manifest) == []

    def test_contract_without_required_metadata_ignores(self):
        # A contract that declares no required_metadata never flags missing keys.
        c = DataContract(
            dataset_name="custom",
            primary_key=("id", "ts"),
            required_columns=("id", "ts", "v"),
            units={"v": "CNY"},
            price_basis="n/a",
            timezone="Asia/Shanghai",
            calendar="x",
        )
        df = pd.DataFrame({"id": [1], "ts": ["a"], "v": [10.0]})
        assert validate_contract(df, c) == []


class TestCalendarMembership:
    """Optional official-calendar membership check."""

    def test_non_trading_day_fails(self):
        # 2024-01-01 is New Year's Day (exchange holiday).
        df = _daily(date=["2024-01-02", "2024-01-03", "2024-01-01"])
        trading = {
            pd.Timestamp("2024-01-02").date(),
            pd.Timestamp("2024-01-03").date(),
        }
        out = validate_dates(df, DAILY_EQUITY, trading_days=trading)
        assert any("non_trading_day:1" in v for v in out)

    def test_all_trading_days_pass(self):
        df = _daily(date=["2024-01-02", "2024-01-03", "2024-01-04"])
        trading = {d.date() for d in pd.to_datetime(df["date"])}
        assert validate_dates(df, DAILY_EQUITY, trading_days=trading) == []

    def test_string_trading_days(self):
        df = _daily(date=["2024-01-02", "2024-01-03", "2024-01-04"])
        trading = {"2024-01-02", "2024-01-03", "2024-01-04"}
        assert validate_dates(df, DAILY_EQUITY, trading_days=trading) == []


class TestSplitContracts:
    """Distinct price systems must not share one contract."""

    def test_daily_equity_aliases_qfq(self):
        assert DAILY_EQUITY is RESEARCH_QFQ_DAILY
        assert get_contract("daily_equity") is RESEARCH_QFQ_DAILY

    def test_three_price_systems_exist(self):
        assert RAW_UNQUOTED_DAILY.price_basis == "unadjusted"
        assert RAW_UNQUOTED_DAILY.adjustment_mode == "raw"
        assert RESEARCH_QFQ_DAILY.price_basis == "qfq"
        assert RESEARCH_QFQ_DAILY.adjustment_mode == "qfq"
        assert ADJUSTMENT_FACTOR.dataset_name == "adjustment_factor"

    def test_registry_has_all(self):
        for name in (
            "daily_equity", "raw_unadjusted_daily", "research_qfq_daily",
            "adjustment_factor",
        ):
            assert name in CONTRACTS

    def test_raw_contract_rejects_qfq_semantics_documented(self):
        # raw contract has no pct_change (computed only after qfq normalization)
        assert "pct_change" not in RAW_UNQUOTED_DAILY.required_columns
