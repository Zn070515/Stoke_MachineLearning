"""DataContract tests (v6 §十六).

The frozen contract is the single source of truth for schema / primary key /
units / date rules.  Validators return a flat list of violation strings (empty
== valid), and ``get_contract`` resolves the registry by dataset name.
"""
import pandas as pd
import pytest

from stoke_ml.data.contract import (
    CONTRACTS,
    DAILY_EQUITY,
    DataContract,
    get_contract,
    validate_contract,
    validate_dates,
    validate_primary_key,
    validate_schema,
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
        assert c.price_basis == "unadjusted"
        assert c.timezone == "Asia/Shanghai"
        assert c.source_priority == ("efinance", "akshare", "tushare", "baostock")

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
