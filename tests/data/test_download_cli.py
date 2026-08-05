"""CLI stock-code argument normalization tests (§九-2).

``parse_stock_codes_arg`` is the single entry every download script routes its
``--stocks``/``--stock_list`` through, so an exchange-prefixed / suffix-suffixed /
numeric / bare code all collapse to the same canonical six-digit key the rest of
the codebase uses (Run Manifest, requested universe, filters, storage).
"""
import pytest

from stoke_ml.data.download_cli import parse_stock_codes_arg


def test_canonicalizes_mixed_exchange_forms():
    codes = parse_stock_codes_arg("SH600001,600001.SH,600001,abc")
    assert codes == ["600001"]


def test_drops_garbage_and_dedupes_sorts():
    codes = parse_stock_codes_arg("abc,600001,000001,SH600001,zzz,300750,abc")
    assert codes == ["000001", "300750", "600001"]


def test_empty_and_none_return_empty_list():
    assert parse_stock_codes_arg(None) == []
    assert parse_stock_codes_arg("") == []
    assert parse_stock_codes_arg("   ") == []


def test_zero_pads_bare_numeric():
    assert parse_stock_codes_arg("600000,000001,1") == ["000001", "600000"]


def test_rejects_contradictory_exchange_prefix():
    # SZ-prefix on a Shanghai leading-6 code contradicts the market → rejected.
    assert parse_stock_codes_arg("SZ600001") == []


@pytest.mark.parametrize("raw,expected", [
    ("600519", ["600519"]),
    ("sh600519", ["600519"]),
    ("600519.SH", ["600519"]),
    ("000001, 600519", ["000001", "600519"]),  # spaces tolerated
])
def test_common_input_forms(raw, expected):
    assert parse_stock_codes_arg(raw) == expected
