"""§十二 Provider Adapter 单位校验 + 真实 fixture (v14).

A-share K-line adapters must all emit ONE canonical convention at the adapter
boundary — OHLC 前复权 (qfq), volume = 股 (shares), amount = 元 (CNY) — and the
Contract (``RESEARCH_QFQ_DAILY``) hard-rejects any unit corruption.  Each
provider returns DIFFERENT raw units, so each must scale (or not) exactly once:

  Tushare (pro_bar)    ``vol`` 手 (lots of 100 shares) → ×100 股;
                       ``amount`` 千元 (1000 CNY) → ×1000 元
  Efinance (EastMoney) ``f56`` volume 手 → ×100 股; ``f57`` amount already 元
                       (no scaling)
  AKShare (Sina)       ``成交量`` already 股; ``成交额`` already 元 (no scaling)
  Baostock             ``volume`` already 股; ``amount`` already 元 (no scaling)

These tests drive each adapter with REALISTIC API-shaped fixtures — the exact
column names / formats the live API returns, NOT the already-normalized
convention — and assert the converted frame satisfies the Contract: units
correct, a legitimate zero-volume suspension day (vol=0 AND amount=0) allowed,
empty/missing values coerced to NaN and deferred by the VWAP consistency check
(the non-finite row is reported by ``validate_finite``, not the VWAP check),
and a hypothetical "vol left in 手" bug hard-rejected as
``amount_volume_unit_mismatch``.

No network: every fixture is hand-recorded from a typical live API response.
Runs in the default fast smoke suite (no slow/network markers).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.contract import (
    RESEARCH_QFQ_DAILY,
    validate_contract,
    validate_price_volume_amount_consistency,
)
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource
from stoke_ml.data.sources.a_shares.efinance_source import (
    EM_FIELD_MAP,
    EfinanceSource,
)
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource


# ── conversion helpers: mirror each fetch_daily's response→DataFrame steps ──

def _convert_tushare(raw):
    return TushareSource()._normalize(raw, "000001")


def _convert_efinance(raw):
    # EfinanceSource.fetch_daily renames the field-code columns BEFORE calling
    # _normalize — reproduce that exact step (map by code, not position).
    return EfinanceSource()._normalize(raw.rename(columns=EM_FIELD_MAP), "000001")


def _convert_akshare(raw):
    return AKShareSource()._normalize(raw, "000001")


def _convert_baostock(raw):
    # fetch_daily constructs the frame with pctChg already renamed to
    # pct_change; every value (incl. missing) is a STRING from
    # query_history_k_data_plus.
    return BaostockSource()._normalize(raw, "000001")


# ── realistic API-shaped fixtures (the RAW unit, NOT the normalized one) ──

def _tushare_raw():
    # pro_bar(adj="qfq") output: trade_date %Y%m%d, vol 手, amount 千元, pct_chg.
    # ~10 CNY stock: vol 1e6 手 = 1e8 股, amount 1e6 千元 = 1e9 元 → VWAP 10.
    return pd.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "trade_date": ["20240102", "20240103", "20240104"],
        "open": [9.9, 10.4, 10.5],
        "high": [10.2, 10.6, 10.5],
        "low": [9.8, 10.1, 10.5],
        "close": [10.0, 10.5, 10.5],
        "pct_chg": [0.0, 5.0, 0.0],
        "vol": [1_000_000.0, 1_000_000.0, 0.0],     # 手
        "amount": [1_000_000.0, 1_050_000.0, 0.0],  # 千元
    })


def _efinance_raw():
    # EastMoney kline (klt=101, fqt=1) — every value is a STRING from the
    # "f51,...,f61" fields2 kline split.  f56 volume 手, f57 amount 元.
    return pd.DataFrame({
        "f51": ["2024-01-02", "2024-01-03", "2024-01-04"],  # date
        "f52": ["9.9", "10.4", "10.5"],                     # open
        "f53": ["10.0", "10.5", "10.5"],                    # close
        "f54": ["10.2", "10.6", "10.5"],                    # high
        "f55": ["9.8", "10.1", "10.5"],                     # low
        "f56": ["1000000", "1000000", "0"],                 # volume 手
        "f57": ["1000000000", "1050000000", "0"],           # amount 元
        "f58": ["4.0", "2.0", "0.0"],                       # amplitude
        "f59": ["0.0", "5.0", "0.0"],                       # pct_change 涨跌幅 %
        "f60": ["0.0", "0.5", "0.0"],                       # change
        "f61": ["1.0", "1.1", "0.0"],                       # turnover
    })


def _akshare_raw():
    # ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq") — Chinese columns;
    # Sina qfq OMITS 涨跌幅, so _normalize derives pct_change from close.
    # 成交量 already 股, 成交额 already 元.
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "开盘": [9.9, 10.4, 10.5],
        "最高": [10.2, 10.6, 10.5],
        "最低": [9.8, 10.1, 10.5],
        "收盘": [10.0, 10.5, 10.5],
        "成交量": [100_000_000, 100_000_000, 0],     # 股 — unchanged
        "成交额": [1_000_000_000, 1_050_000_000, 0],  # 元 — unchanged
    })


def _baostock_raw():
    # query_history_k_data_plus(adjustflag="2") — EVERY value (incl. the 8th,
    # pctChg) is a STRING; fetch_daily constructs the frame with pctChg already
    # renamed to pct_change.  volume already 股, amount already 元.
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "open": ["9.9", "10.4", "10.5"],
        "high": ["10.2", "10.6", "10.5"],
        "low": ["9.8", "10.1", "10.5"],
        "close": ["10.0", "10.5", "10.5"],
        "volume": ["100000000", "100000000", "0.000000"],    # 股
        "amount": ["1000000000", "1050000000", "0.000000"],  # 元
        "pct_change": ["0.0000", "5.0000", "0.0000"],
    })


# ── (a) correct unit conversion + full Contract pass ─────────────────────

class TestTushareProviderUnits:
    def test_scales_hand_and_thousand_yuan_and_passes_contract(self):
        df = _convert_tushare(_tushare_raw())
        # 手 → 股 (×100), 千元 → 元 (×1000)
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000.0 * 1000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000.0 * 1000.0)
        # a legitimate zero-volume suspension day stays zero on BOTH axes
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        # the fully unit-correct frame passes the Contract
        assert validate_contract(df, RESEARCH_QFQ_DAILY) == []


class TestEfinanceProviderUnits:
    def test_scales_lots_to_shares_and_passes_contract(self):
        # build the frame with field-code columns + EM_FIELD_MAP rename, exactly
        # as EfinanceSource.fetch_daily does
        raw = _efinance_raw()
        df = raw.rename(columns=EM_FIELD_MAP)
        df = EfinanceSource()._normalize(df, "000001")
        # f56 手 → 股 (×100); f57 already 元 — untouched
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert validate_contract(df, RESEARCH_QFQ_DAILY) == []


class TestAkshareProviderUnits:
    def test_units_unchanged_and_pct_change_derived(self):
        df = _convert_akshare(_akshare_raw())
        # Sina already 股/元 — no scaling
        assert np.isclose(df["volume"].iloc[0], 100_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # 涨跌幅 omitted → derived from close: 100*(10.5/10.0-1) = 5.0
        assert np.isnan(df["pct_change"].iloc[0])
        assert np.isclose(df["pct_change"].iloc[1], 5.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert validate_contract(df, RESEARCH_QFQ_DAILY) == []

    def test_all_zero_zhang_die_fu_also_derives(self):
        # an older Sina frame that CARRIES 涨跌幅 but all zeros must derive
        # from close rather than persist the zeros (fillna(0).abs().sum()==0)
        raw = _akshare_raw().copy()
        raw["涨跌幅"] = [0.0, 0.0, 0.0]
        df = AKShareSource()._normalize(raw, "000001")
        assert np.isclose(df["pct_change"].iloc[1], 5.0)
        assert validate_contract(df, RESEARCH_QFQ_DAILY) == []


class TestBaostockProviderUnits:
    def test_units_unchanged_and_strings_coerced(self):
        df = _convert_baostock(_baostock_raw())
        # already 股/元 — unchanged; the string columns coerce to float
        assert df.dtypes["volume"] == np.float64
        assert df.dtypes["amount"] == np.float64
        assert np.isclose(df["volume"].iloc[0], 100_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert validate_contract(df, RESEARCH_QFQ_DAILY) == []


# ── (b) empty/missing numeric semantics ──────────────────────────────────

def _tushare_empty_raw():
    return pd.DataFrame({
        "trade_date": ["20240102", "20240103"],
        "open": [9.9, 10.4], "high": [10.2, 10.6], "low": [9.8, 10.1],
        "close": [10.0, 10.5], "pct_chg": [0.0, 5.0],
        "vol": [1_000_000.0, np.nan],      # missing volume (NaN)
        "amount": [1_000_000.0, np.nan],   # missing amount (NaN, 千元)
    })


def _efinance_empty_raw():
    # EastMoney renders a missing kline field as "-".
    return pd.DataFrame({
        "f51": ["2024-01-02", "2024-01-03"],
        "f52": ["9.9", "10.4"], "f53": ["10.0", "10.5"],
        "f54": ["10.2", "10.6"], "f55": ["9.8", "10.1"],
        "f56": ["1000000", "-"],            # missing volume → "-"
        "f57": ["1000000000", "-"],         # missing amount → "-"
        "f58": ["4.0", "0.0"], "f59": ["0.0", "5.0"],
        "f60": ["0.0", "0.5"], "f61": ["1.0", "0.0"],
    })


def _akshare_empty_raw():
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [9.9, 10.4], "最高": [10.2, 10.6], "最低": [9.8, 10.1],
        "收盘": [10.0, 10.5],
        "成交量": [100_000_000, np.nan],    # missing volume
        "成交额": [1_000_000_000, np.nan],  # missing amount
    })


def _baostock_empty_raw():
    # Baostock returns empty string "" for a missing numeric.
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": ["9.9", "10.4"], "high": ["10.2", "10.6"],
        "low": ["9.8", "10.1"], "close": ["10.0", "10.5"],
        "volume": ["100000000", ""],          # missing volume → ""
        "amount": ["1000000000", ""],         # missing amount → ""
        "pct_change": ["0.0000", "5.0000"],
    })


_EMPTY_CASES = [
    ("tushare", _tushare_empty_raw, _convert_tushare),
    ("efinance", _efinance_empty_raw, _convert_efinance),
    ("akshare", _akshare_empty_raw, _convert_akshare),
    ("baostock", _baostock_empty_raw, _convert_baostock),
]


class TestEmptyValueSemantics:
    """An empty/missing numeric from a source (Baostock ``""``, Efinance
    ``"-"``, Tushare/AKShare NaN) is coerced to NaN without raising, and the
    volume/amount consistency check DEFERS the non-finite row — it is reported
    by validate_finite, not the VWAP check."""

    @pytest.mark.parametrize(
        "label,raw,convert", _EMPTY_CASES, ids=[e[0] for e in _EMPTY_CASES]
    )
    def test_empty_value_coerced_to_nan_and_deferred(self, label, raw, convert):
        df = convert(raw())
        # every fixture's 2nd row is the empty/missing one
        assert np.isnan(df["volume"].iloc[1])
        assert np.isnan(df["amount"].iloc[1])
        # the finite row is still evaluated, the non-finite one deferred → []
        assert validate_price_volume_amount_consistency(df, RESEARCH_QFQ_DAILY) == []


# ── (c) unconverted / corrupted units are hard-rejected ──────────────────

def _daily_shape(closes, volume, amount, pct_change):
    """A normalized daily-convention frame (date/stock_code/…/attrs) for
    exercising the Contract directly, independent of any provider."""
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    df = pd.DataFrame({
        "date": dates[: len(closes)],
        "stock_code": "000001",
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": volume,
        "amount": amount,
        "pct_change": pct_change,
    })
    df.attrs["source"] = "tushare"
    df.attrs["adjustment_mode"] = "qfq"
    return df


class TestContractRejectsUnconvertedVolume:
    """A hypothetical 'vol left in 手' bug (volume = raw 手, amount still 元)
    must be HARD-rejected by the Contract, and the two scale-independent
    zero/positive exclusions fire on their own."""

    def test_vol_in_hand_is_amount_volume_unit_mismatch(self):
        # Properly converted Tushare amount (元) paired with UNconverted vol
        # (手, no ×100): implied = amount/volume = 1.1e9/1e6 = 1100 — strictly
        # beyond the 100× band (close 10.0 → [0.1, 1000]).  An exact 1.0e9
        # would land ON the band edge at 1000 and be missed, hence 1.1e9.
        df = _daily_shape(
            closes=[10.0, 10.5, 10.5],
            volume=[1_000_000.0, 1_000_000.0, 0.0],  # 手 — BUGGY, not ×100
            amount=[1.1e9, 1.155e9, 0.0],             # 元 — correctly converted
            pct_change=[0.0, 5.0, 0.0],
        )
        out = validate_contract(df, RESEARCH_QFQ_DAILY)
        assert any("amount_volume_unit_mismatch" in v for v in out)

    def test_zero_volume_positive_amount_hard_rejected(self):
        # volume==0 && amount>0 is economically impossible → hard exclusion.
        df = _daily_shape(
            closes=[10.0, 10.5, 10.5],
            volume=[0.0, 1e8, 1e8],
            amount=[1e7, 1e9, 1e9],
            pct_change=[0.0, 5.0, 0.0],
        )
        assert validate_price_volume_amount_consistency(df, RESEARCH_QFQ_DAILY) == [
            "amount_without_volume:1",
        ]
        out = validate_contract(df, RESEARCH_QFQ_DAILY)
        assert any("amount_without_volume" in v for v in out)

    def test_positive_volume_zero_amount_hard_rejected(self):
        # volume>0 && amount==0 is economically impossible → hard exclusion.
        df = _daily_shape(
            closes=[10.0, 10.5, 10.5],
            volume=[1e8, 1e8, 1e8],
            amount=[0.0, 1e9, 1e9],
            pct_change=[0.0, 5.0, 0.0],
        )
        assert validate_price_volume_amount_consistency(df, RESEARCH_QFQ_DAILY) == [
            "volume_without_amount:1",
        ]
        out = validate_contract(df, RESEARCH_QFQ_DAILY)
        assert any("volume_without_amount" in v for v in out)
