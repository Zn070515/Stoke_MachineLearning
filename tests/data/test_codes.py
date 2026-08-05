"""Canonical stock-code sanitizer tests (§八-1).

A code must survive Parquet round-trips (float 600001.0), exchange prefixes
(SH600001 / 600001.SH) and numeric/nan inputs, collapsing to the same six-digit
zero-padded key — a bare ``str(x).zfill(6)`` turns 600001.0 into "600001.0".
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.codes import (
    is_valid_stock_code,
    normalize_stock_code,
    normalize_stock_code_series,
)


class TestNormalizeStockCode:
    @pytest.mark.parametrize("raw,expected", [
        (600001, "600001"),
        (1, "000001"),
        (np.int64(600519), "600519"),
        (600001.0, "600001"),        # the classic Parquet-float trap
        (1.0, "000001"),
        ("600001", "600001"),
        ("000001", "000001"),
        ("600001.0", "600001"),      # float formatted to string
        ("600001.00", None),         # not an int-like float repr
        ("SH600001", "600001"),
        ("sh600519", "600519"),
        ("600001.SH", "600001"),
        ("600519.sh", "600519"),
        (" 600001 ", "600001"),      # stray whitespace
        # §六: exchange prefix must be stripped BEFORE the integer-.0 cleanup
        # so "sh600001.0" resolves instead of failing the isdigit() gate.
        ("sh600001.0", "600001"),
        ("600001.0.SH", "600001"),   # suffix stripped, then the float .0
    ])
    def test_valid_codes(self, raw, expected):
        assert normalize_stock_code(raw) == expected

    @pytest.mark.parametrize("bad", [
        None,
        np.nan,
        float("inf"),
        "",
        "   ",
        "SH600001x",                 # illegal trailing char
        "abc",                       # non-numeric
        "600001.5",                  # non-integer float
        600001.5,
        1.5,
        True,                        # bool is not a code
        "nan",
        # §六 strict range / market-contradiction rejections:
        -1,                          # negative
        0,                           # zero → "000000"
        1000000,                     # > 999999
        "1000000",                   # 7 digits
        "000000",                    # the reserved all-zero code
        "0",                         # zero → "000000"
        "SZ600519",                  # SZ prefix contradicts Shanghai 600519
        "600519.sz",                 # SZ suffix contradicts Shanghai 600519
        "SH000001",                  # SH prefix contradicts Shenzhen 000001
        "600001.BJ",                 # BJ suffix contradicts Shanghai 600001
        "BJ600001",                  # BJ prefix contradicts Shanghai 600001
    ])
    def test_invalid_codes(self, bad):
        assert normalize_stock_code(bad) is None

    def test_five_digit_pads(self):
        assert normalize_stock_code("60000") == "060000"
        assert normalize_stock_code(60000) == "060000"


class TestNormalizeStockCodeSeries:
    def test_mixed_dtype_float_column(self):
        s = pd.Series([600001.0, 1.0, "600519", "SH600000", np.nan, "bad"])
        out = normalize_stock_code_series(s)
        assert out.tolist()[:4] == ["600001", "000001", "600519", "600000"]
        assert out.iloc[4:].isna().all()

    def test_object_column_with_prefixes(self):
        s = pd.Series(["600001.SH", "000002.SZ", "830799.BJ"])
        assert normalize_stock_code_series(s).tolist() == [
            "600001", "000002", "830799"]

    def test_all_invalid(self):
        s = pd.Series([np.nan, None, "abc", ""])
        assert normalize_stock_code_series(s).isna().all()


class TestIsValidStockCode:
    def test_valid(self):
        assert is_valid_stock_code("600001") is True
        assert is_valid_stock_code(600001.0) is True

    def test_invalid(self):
        assert is_valid_stock_code(None) is False
        assert is_valid_stock_code("junk") is False
