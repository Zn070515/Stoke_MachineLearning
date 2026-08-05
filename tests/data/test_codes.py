"""Canonical stock-code sanitizer tests (§八-1).

A code must survive Parquet round-trips (float 600001.0), exchange prefixes
(SH600001 / 600001.SH) and numeric/nan inputs, collapsing to the same six-digit
zero-padded key — a bare ``str(x).zfill(6)`` turns 600001.0 into "600001.0".
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.codes import (
    a_share_equity_segment,
    is_a_share_equity_code,
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

    def test_format_layer_accepts_non_equity(self):
        """is_valid_stock_code is the FORMAT layer only — indices/B-shares/funds
        are format-legal six-digit codes, so they pass here (§十 two layers)."""
        for c in ("100000", "200000", "500000", "700000", "900000"):
            assert is_valid_stock_code(c) is True, c


class TestAShareEquitySegment:
    def test_sh_main_and_star(self):
        for c in ("600519", "601318", "603288", "605499", "688981", "689009"):
            assert a_share_equity_segment(c) == "SH", c

    def test_sz_main_and_chinext(self):
        for c in ("000001", "001979", "002594", "003816", "300750", "301269"):
            assert a_share_equity_segment(c) == "SZ", c

    def test_bj(self):
        for c in ("430001", "830799", "871981", "889988", "920001"):
            assert a_share_equity_segment(c) == "BJ", c

    def test_non_equity_returns_none(self):
        for c in ("100000", "200000", "500000", "700000", "900000"):
            assert a_share_equity_segment(c) is None, c


class TestIsAShareEquityCode:
    """v13 §十: the canonical daily store only holds A-share common equity, so
    format-legal but non-equity codes (indices/B-shares/funds) must be False."""

    @pytest.mark.parametrize("code", [
        "600519", "000001", "688981", "300750", "430001",
        "002594", "301269", "689009", "601318", "920001",
    ])
    def test_equity_true(self, code):
        assert is_a_share_equity_code(code) is True

    @pytest.mark.parametrize("code", [
        "100000", "200000", "500000", "700000", "900000",
    ])
    def test_non_equity_false(self, code):
        assert is_a_share_equity_code(code) is False

    def test_normalizes_first(self):
        # float / exchange-prefixed inputs are normalized before the segment
        # filter — a format-legal code via any spelling is classified.
        assert is_a_share_equity_code(600519.0) is True
        assert is_a_share_equity_code("SH600519") is True
        assert is_a_share_equity_code(900001) is False  # SH B-share
        assert is_a_share_equity_code(100001) is False  # index

    def test_garbage_false(self):
        assert is_a_share_equity_code(None) is False
        assert is_a_share_equity_code("junk") is False
        assert is_a_share_equity_code(0) is False
