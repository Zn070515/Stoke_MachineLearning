"""§十二/§二十一 real provider response fixtures drive ``fetch_daily`` (v14).

Task 6 (``test_provider_units.py``) already drives each adapter's ``_normalize``
with raw-unit DataFrames.  The layer BETWEEN the raw API/library response and
that DataFrame — the response PARSING inside each ``fetch_daily`` — is entirely
untested here until now.  These tests drive ``fetch_daily`` END-TO-END with
mocked network/library calls whose fixtures faithfully reproduce the REAL API
response shape each adapter must parse:

  Efinance (EastMoney push2his kline)
    ``requests.get(...)`` returns a JSON body ``{"rc": 0, "data": {"klines":
    ["2024-01-02,9.9,10.0,10.2,9.8,1000000,1000000000,4.0,0.0,0.0,1.0", ...]}}``
    where each kline is ONE comma-joined string in fields2 order
    ``f51..f61`` = date,open,close,high,low,volume(手),amount(元),amplitude,
    pct_change,change,turnover.  ``fetch_daily`` splits each string and renames
    the field codes before ``_normalize`` (volume ×100 手→股; amount already 元).

  Baostock (query_history_k_data_plus)
    ``bs.login()`` -> obj with ``.error_code == "0"``; the result set exposes
    ``.next()`` (True while rows remain) and ``.get_row_data()`` (a list of
    STRINGS — every value, incl. pctChg, is a string).  volume/amount already
    股/元; the ``pctChg``→``pct_change`` rename happens when the frame is built.

  Tushare (ts.pro_bar)
    Returns a DataFrame: ``trade_date`` %Y%m%d, ``vol`` 手, ``amount`` 千元,
    ``pct_chg``.  ``_normalize`` renames, ×100 volume and ×1000 amount.
    NOTE: ``fetch_daily`` calls the MODULE-LEVEL ``ts.pro_bar`` (not
    ``self._pro.pro_bar``), so the library call is patched on the tushare module
    while ``src._pro`` is only pre-seeded to keep ``_get_pro()`` off the network.

  AKShare (ak.stock_zh_a_daily, Sina qfq)
    Returns a DataFrame with Chinese columns ``日期/开盘/最高/最低/收盘/成交量/
    成交额``; Sina qfq omits 涨跌幅 so ``_normalize`` derives ``pct_change`` from
    close.  成交量/成交额 already 股/元 (no scaling).

Every mock returns the error shapes the live APIs really produce (``rc != 0``,
missing ``klines``, ``error_code != "0"``, ``None`` frame).  All mocked — no
network, no slow/network markers; runs in the default fast smoke suite.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.contract import (
    RESEARCH_QFQ_DAILY,
    validate_contract,
)
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource


# ── realistic API-shaped response fixtures (hand-recorded) ───────────────

def _efinance_kline_rows() -> list[str]:
    """Three comma-joined kline strings in fields2 order
    ``f51..f61`` = date,open,close,high,low,volume(手),amount(元),amplitude,
    pct_change,change,turnover.  Row 3 is a legitimate zero-volume suspension
    day (vol=0 AND amount=0)."""
    return [
        "2024-01-02,9.9,10.0,10.2,9.8,1000000,1000000000,4.0,0.0,0.0,1.0",
        "2024-01-03,10.4,10.5,10.6,10.1,1000000,1050000000,2.0,5.0,0.5,1.1",
        "2024-01-04,10.5,10.5,10.5,10.5,0,0,0.0,0.0,0.0,0.0",
    ]


def _efinance_payload() -> dict:
    """The JSON body curl_cffi's ``requests.get().json()`` returns on success."""
    return {"rc": 0, "data": {"klines": _efinance_kline_rows()}}


def _baostock_row_data() -> list[list[str]]:
    """Row strings exactly as ``query_history_k_data_plus`` returns them — every
    value a STRING; volume 股 / amount 元; row 3 suspended (vol AND amount 0)."""
    return [
        ["2024-01-02", "9.9", "10.2", "9.8", "10.0",
         "100000000", "1000000000", "0.0000"],
        ["2024-01-03", "10.4", "10.6", "10.1", "10.5",
         "100000000", "1050000000", "5.0000"],
        ["2024-01-04", "10.5", "10.5", "10.5", "10.5",
         "0.000000", "0.000000", "0.0000"],
    ]


def _tushare_pro_bar_frame() -> pd.DataFrame:
    """``ts.pro_bar(adj="qfq")`` output: trade_date %Y%m%d, vol 手, amount 千元.
    Row 3 is the zero-volume suspension day (vol=0 AND amount=0)."""
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


def _akshare_daily_frame() -> pd.DataFrame:
    """``ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")`` — Chinese
    columns; Sina qfq OMITS 涨跌幅.  成交量 already 股, 成交额 already 元; row 3
    suspended (成交量 AND 成交额 0)."""
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "开盘": [9.9, 10.4, 10.5],
        "最高": [10.2, 10.6, 10.5],
        "最低": [9.8, 10.1, 10.5],
        "收盘": [10.0, 10.5, 10.5],
        "成交量": [100_000_000, 100_000_000, 0],      # 股 — unchanged
        "成交额": [1_000_000_000, 1_050_000_000, 0],  # 元 — unchanged
    })


def _assert_canonical_shape(df: pd.DataFrame) -> None:
    """Shared post-conversion assertions: 3 rows, sorted dates, stock_code,
    and a clean Contract pass (incl. the zero-volume suspension day)."""
    assert len(df) == 3
    assert list(df["date"]) == [
        dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)
    ]
    assert (df["stock_code"] == "000001").all()
    assert validate_contract(df, RESEARCH_QFQ_DAILY) == []


# ── network/library mocks ────────────────────────────────────────────────

@pytest.fixture
def efinance_api(monkeypatch):
    """Patch ``curl_cffi.requests`` (read at call time by fetch_daily's local
    ``from curl_cffi import requests``).  The test sets ``.payload`` /
    ``.status_code``; ``.calls`` records every ``requests.get`` invocation."""
    curl_cffi = pytest.importorskip("curl_cffi")

    class _FakeResponse:
        def __init__(self, payload, status_code):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeRequests:
        def __init__(self):
            self.payload = {}
            self.status_code = 200
            self.calls = []  # (url, kwargs)

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return _FakeResponse(self.payload, self.status_code)

    fake = _FakeRequests()
    monkeypatch.setattr(curl_cffi, "requests", fake)
    return fake


@pytest.fixture
def baostock_api(monkeypatch):
    """Patch ``bs.login`` / ``bs.query_history_k_data_plus`` / ``bs.logout``."""
    import baostock as bs

    state = {
        "rows": [],
        "query_error_code": "0",
        "query_error_msg": "",
        "login_error_code": "0",
        "queries": [],  # (bs_code, fields, kwargs)
        "logins": 0,
        "logouts": 0,
    }

    class _Login:
        def __init__(self, error_code, error_msg):
            self.error_code = error_code
            self.error_msg = error_msg

    class _Rowset:
        def __init__(self, rows, error_code, error_msg):
            self.error_code = error_code
            self.error_msg = error_msg
            self._rows = list(rows)
            self._cur = None

        def next(self):
            if self._rows:
                self._cur = self._rows.pop(0)
                return True
            self._cur = None
            return False

        def get_row_data(self):
            return self._cur

    def fake_login():
        state["logins"] += 1
        msg = "login failed" if state["login_error_code"] != "0" else ""
        return _Login(state["login_error_code"], msg)

    def fake_query(bs_code, fields, **kwargs):
        state["queries"].append((bs_code, fields, kwargs))
        return _Rowset(
            state["rows"], state["query_error_code"], state["query_error_msg"]
        )

    def fake_logout():
        state["logouts"] += 1

    monkeypatch.setattr(bs, "login", fake_login)
    monkeypatch.setattr(bs, "query_history_k_data_plus", fake_query)
    monkeypatch.setattr(bs, "logout", fake_logout)
    return state


@pytest.fixture
def tushare_pro_bar(monkeypatch):
    """Patch the MODULE-LEVEL ``tushare.pro_bar`` that fetch_daily calls; the
    test sets ``state["result"]`` (a DataFrame, or None for the error path)."""
    import tushare as ts

    state = {"result": None, "calls": []}

    def fake_pro_bar(**kwargs):
        state["calls"].append(kwargs)
        return state["result"]

    monkeypatch.setattr(ts, "pro_bar", fake_pro_bar)
    return state


@pytest.fixture
def tushare_source():
    """A TushareSource with a pre-seeded ``_pro`` so ``_get_pro()`` returns it
    without a token or network.  NOTE: fetch_daily calls ``ts.pro_bar`` (module
    level), NOT ``self._pro.pro_bar`` — the library call is patched by
    ``tushare_pro_bar``; ``_pro`` only satisfies the ``_get_pro()`` cache path."""
    src = TushareSource()
    src._pro = object()
    return src


@pytest.fixture
def akshare_daily(monkeypatch):
    """Patch ``ak.stock_zh_a_daily``; the test sets ``state["result"]``."""
    import akshare as ak

    state = {"result": None, "calls": []}

    def fake_daily(**kwargs):
        state["calls"].append(kwargs)
        return state["result"]

    monkeypatch.setattr(ak, "stock_zh_a_daily", fake_daily)
    return state


# ── Efinance (EastMoney) ─────────────────────────────────────────────────

class TestEfinanceFetchDaily:
    def test_happy_path_parses_kline_strings_and_scales_lots(self, efinance_api):
        efinance_api.payload = _efinance_payload()
        df = EfinanceSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        _assert_canonical_shape(df)
        # f56 volume 手 → 股 (×100); f57 amount already 元 — untouched
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert df.attrs["source"] == "efinance"
        assert df.attrs["adjustment_mode"] == "qfq"
        # glue: the request carried the correct EastMoney routing
        _url, kwargs = efinance_api.calls[0]
        assert kwargs["params"]["secid"] == "0.000001"
        assert kwargs["params"]["klt"] == "101"
        assert kwargs["params"]["fqt"] == "1"

    def test_rc_nonzero_returns_empty(self, efinance_api):
        # real API error: {"rc": 100, "msg": "..."}
        efinance_api.payload = {"rc": 100, "msg": "no data"}
        df = EfinanceSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        assert df.empty

    def test_missing_klines_returns_empty(self, efinance_api):
        # a valid rc=0 body whose data object carries no klines
        efinance_api.payload = {"rc": 0, "data": {}}
        df = EfinanceSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        assert df.empty


# ── Baostock ─────────────────────────────────────────────────────────────

class TestBaostockFetchDaily:
    def test_happy_path_parses_rows_and_keeps_units(self, baostock_api):
        baostock_api["rows"] = _baostock_row_data()
        df = BaostockSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        _assert_canonical_shape(df)
        # every value arrived as a string and was coerced to float
        assert df.dtypes["volume"] == np.float64
        assert df.dtypes["amount"] == np.float64
        # volume 股 / amount 元 — no scaling
        assert np.isclose(df["volume"].iloc[0], 100_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert df.attrs["source"] == "baostock"
        assert df.attrs["adjustment_mode"] == "qfq"
        # glue: the query asked for the qfq fields + adjustflag
        bs_code, fields, kwargs = baostock_api["queries"][0]
        assert bs_code == "sz.000001"
        assert fields == "date,open,high,low,close,volume,amount,pctChg"
        assert kwargs["frequency"] == "d"
        assert kwargs["adjustflag"] == "2"
        assert baostock_api["logouts"] == 1

    def test_query_error_code_returns_empty(self, baostock_api):
        # real API failure: rs.error_code != "0"
        baostock_api["query_error_code"] = "1"
        baostock_api["query_error_msg"] = "no such stock"
        df = BaostockSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        assert df.empty
        assert baostock_api["logouts"] == 1  # logout also on the failure path


# ── Tushare ──────────────────────────────────────────────────────────────

class TestTushareFetchDaily:
    def test_happy_path_scales_hand_and_thousand_yuan(
        self, tushare_source, tushare_pro_bar
    ):
        tushare_pro_bar["result"] = _tushare_pro_bar_frame()
        df = tushare_source.fetch_daily("000001", "2024-01-02", "2024-01-04")
        _assert_canonical_shape(df)
        # vol 手 → 股 (×100); amount 千元 → 元 (×1000)
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000.0 * 1000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000.0 * 1000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        assert df.attrs["source"] == "tushare"
        assert df.attrs["adjustment_mode"] == "qfq"
        # glue: pro_bar was called with the qfq request failover relies on
        kwargs = tushare_pro_bar["calls"][0]
        assert kwargs["ts_code"] == "000001.SZ"
        assert kwargs["adj"] == "qfq"
        assert kwargs["start_date"] == "20240102"
        assert kwargs["end_date"] == "20240104"

    def test_pro_bar_none_returns_empty(self, tushare_source, tushare_pro_bar):
        # real API failure: pro_bar returned None
        tushare_pro_bar["result"] = None
        df = tushare_source.fetch_daily("000001", "2024-01-02", "2024-01-04")
        assert df.empty


# ── AKShare (Sina) ───────────────────────────────────────────────────────

class TestAkshareFetchDaily:
    def test_happy_path_keeps_units_and_derives_pct(self, akshare_daily):
        akshare_daily["result"] = _akshare_daily_frame()
        df = AKShareSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        _assert_canonical_shape(df)
        # Sina already 股/元 — no scaling
        assert np.isclose(df["volume"].iloc[0], 100_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 1_000_000_000.0)
        assert np.isclose(df["amount"].iloc[1], 1_050_000_000.0)
        # zero-volume suspension day
        assert np.isclose(df["volume"].iloc[2], 0.0)
        assert np.isclose(df["amount"].iloc[2], 0.0)
        # 涨跌幅 omitted → derived from close: 100*(10.5/10.0-1) = 5.0
        assert np.isnan(df["pct_change"].iloc[0])
        assert np.isclose(df["pct_change"].iloc[1], 5.0)
        assert df.attrs["source"] == "akshare"
        assert df.attrs["adjustment_mode"] == "qfq"
        # glue: the request used the Sina symbol + %Y%m%d window + qfq
        kwargs = akshare_daily["calls"][0]
        assert kwargs["symbol"] == "sz000001"
        assert kwargs["start_date"] == "20240102"
        assert kwargs["end_date"] == "20240104"
        assert kwargs["adjust"] == "qfq"

    def test_none_returns_empty(self, akshare_daily):
        # real API failure: stock_zh_a_daily returned None
        akshare_daily["result"] = None
        df = AKShareSource().fetch_daily("000001", "2024-01-02", "2024-01-04")
        assert df.empty
