"""Provider market-routing matrix (§六, P0).

Every provider must derive its exchange prefix from the SINGLE authority
``market_of_code`` — never from its own leading-digit heuristic.  The 920xxx
BSE code range is the canary: ``920001`` must route as BJ everywhere —
Tushare ``920001.BJ``, Sina-family ``bj920001``, Baostock ``bj.920001``,
EastMoney ``0.920001`` — and must never leak onto Shenzhen as a bogus
``.SZ`` / ``sz920001`` request, or onto Shanghai as ``sh920001`` /
``sh.920001``.

Each provider's BJ capability was verified against the live API (2026-08-05):
EastMoney push2his kline returns data for ``0.920001`` (market 0) and errors
for ``1.920001``; Sina CN_MarketData.getKLineData returns bars for
``bj920001`` and null for ``sz920001``; Tencent qt.gtimg.cn / mkline accept
``bj920001``; Sina fund-flow and news pages accept the ``bj`` prefix.  So
every provider HERE supports BJ and must emit the correct prefix — none is
UNSUPPORTED for BJ.  ``UnsupportedMarketError`` is reserved for codes that
are not A-share common equity at all (indices / B-shares / funds).
"""
from pathlib import Path

import pandas as pd
import pytest

from stoke_ml.data.codes import (
    UnsupportedMarketError,
    market_of_code,
    normalize_stock_code,
)
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.backup_sources import (
    _sina_market_prefix,
    _tencent_market_prefix,
)
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource
from stoke_ml.data.sources.a_shares.capital_flow_source import _sina_market_code
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.sources.a_shares.minute_source import MinuteSource
from stoke_ml.data.sources.a_shares.minute_source_sina_direct import (
    SinaDirectMinuteSource,
)
from stoke_ml.data.sources.a_shares.minute_source_tencent import (
    TencentMinuteSource,
)
from stoke_ml.data.sources.a_shares.news_source import SinaNewsSource
from stoke_ml.data.sources.a_shares.sector_source import (
    _market_code as _sector_market_code,
)
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource

# BSE codes across the three real BSE ranges.
_BJ_CODES = ("430047", "830799", "920001")

# (label, fn, SH_output, SZ_output, BJ_template)
#   fn: the provider's ``_to_*`` routing callable (static method or module fn).
#   BJ_template: format string with ``{code}``, or a bare constant when the
#   provider emits only the market prefix / market id (news, sector).
_PROVIDERS = [
    ("efinance._to_secid", EfinanceSource._to_secid,
     "1.600519", "0.000001", "0.{code}"),
    ("akshare._to_sina_symbol", AKShareSource._to_sina_symbol,
     "sh600519", "sz000001", "bj{code}"),
    ("tushare._to_ts_code", TushareSource._to_ts_code,
     "600519.SH", "000001.SZ", "{code}.BJ"),
    ("baostock._to_bs_code", BaostockSource._to_bs_code,
     "sh.600519", "sz.000001", "bj.{code}"),
    ("minute._to_sina_symbol", MinuteSource._to_sina_symbol,
     "sh600519", "sz000001", "bj{code}"),
    ("tencent_minute._to_tencent_symbol", TencentMinuteSource._to_tencent_symbol,
     "sh600519", "sz000001", "bj{code}"),
    ("sina_direct_minute._to_sina_symbol", SinaDirectMinuteSource._to_sina_symbol,
     "sh600519", "sz000001", "bj{code}"),
    ("backup._sina_market_prefix", _sina_market_prefix,
     "sh600519", "sz000001", "bj{code}"),
    ("backup._tencent_market_prefix", _tencent_market_prefix,
     "sh600519", "sz000001", "bj{code}"),
    ("capital_flow._sina_market_code", _sina_market_code,
     "sh600519", "sz000001", "bj{code}"),
    ("news._to_sina_prefix", SinaNewsSource._to_sina_prefix,
     "sh", "sz", "bj"),
    ("sector._market_code", _sector_market_code,
     "1", "0", "0"),
]


class TestMarketOfCode:
    """The single authority must classify the review's matrix exactly."""

    @pytest.mark.parametrize("code,expected", [
        ("600519", "SH"),
        ("601318", "SH"),
        ("688981", "SH"),
        ("000001", "SZ"),
        ("002594", "SZ"),
        ("300750", "SZ"),
        ("430047", "BJ"),
        ("830799", "BJ"),
        ("871981", "BJ"),
        ("889988", "BJ"),
        ("920001", "BJ"),
    ])
    def test_equity_routes(self, code, expected):
        assert market_of_code(code) == expected

    def test_matrix_from_review(self):
        """600519/000001/300750/430047/830799/920001 → SH/SZ/SZ/BJ/BJ/BJ."""
        for code, expected in [
            ("600519", "SH"), ("000001", "SZ"), ("300750", "SZ"),
            ("430047", "BJ"), ("830799", "BJ"), ("920001", "BJ"),
        ]:
            assert market_of_code(code) == expected, code

    @pytest.mark.parametrize("code", [
        "100000", "200000", "500000", "900000",
    ])
    def test_non_equity_returns_none(self, code):
        assert market_of_code(code) is None

    def test_consistent_with_a_share_equity_segment(self):
        from stoke_ml.data.codes import a_share_equity_segment
        for code in ("600519", "000001", "300750", "430047", "830799", "920001"):
            assert market_of_code(code) == a_share_equity_segment(code), code


class TestProviderRoutes:
    """SH/SZ output must be unchanged from the historical behaviour."""

    @pytest.mark.parametrize(
        "label,fn,sh_exp,sz_exp,bj_tpl",
        _PROVIDERS, ids=[p[0] for p in _PROVIDERS],
    )
    def test_sh_sz_routes(self, label, fn, sh_exp, sz_exp, bj_tpl):
        assert fn("600519") == sh_exp, label
        assert fn("000001") == sz_exp, label

    @pytest.mark.parametrize(
        "label,fn,sh_exp,sz_exp,bj_tpl",
        _PROVIDERS, ids=[p[0] for p in _PROVIDERS],
    )
    def test_bj_routes_correctly(self, label, fn, sh_exp, sz_exp, bj_tpl):
        for code in _BJ_CODES:
            expected = (
                bj_tpl.format(code=code) if "{code}" in bj_tpl else bj_tpl
            )
            assert fn(code) == expected, f"{label} for {code}"


class TestNeverMasquerade:
    """No provider may turn a BSE code into a Shenzhen/Shanghai request."""

    def test_920001_canary_prefixes(self):
        for code in _BJ_CODES:
            assert EfinanceSource._to_secid(code) == f"0.{code}", code
            assert not EfinanceSource._to_secid(code).startswith("1."), code  # SH id
            assert AKShareSource._to_sina_symbol(code).startswith("bj"), code
            assert TushareSource._to_ts_code(code).endswith(".BJ"), code
            assert BaostockSource._to_bs_code(code).startswith("bj."), code
            assert MinuteSource._to_sina_symbol(code).startswith("bj"), code
            assert TencentMinuteSource._to_tencent_symbol(code).startswith("bj"), code
            assert SinaDirectMinuteSource._to_sina_symbol(code).startswith("bj"), code
            assert _sina_market_prefix(code).startswith("bj"), code
            assert _tencent_market_prefix(code).startswith("bj"), code
            assert _sina_market_code(code).startswith("bj"), code
            assert SinaNewsSource._to_sina_prefix(code) == "bj", code
            assert _sector_market_code(code) == "0", code

    def test_no_sz_or_sh_smuggled_into_bj_output(self):
        # Even a prefix-agnostic search for a Shenzhen/Shanghai marker must
        # come back empty for a BSE code (except the EastMoney market-0 id).
        for label, fn, *_ in _PROVIDERS:
            for code in _BJ_CODES:
                out = fn(code)
                assert ".SZ" not in out, (label, code, out)
                assert ".SH" not in out, (label, code, out)
                if isinstance(out, str) and out and out[0].isalpha():
                    assert not out.lower().startswith(("sz", "sh")), (label, code, out)


class TestUnsupportedMarket:
    """A non-A-share code (index / B-share / fund) is refused, not guessed."""

    @pytest.mark.parametrize(
        "label,fn,sh_exp,sz_exp,bj_tpl",
        _PROVIDERS, ids=[p[0] for p in _PROVIDERS],
    )
    def test_non_equity_raises(self, label, fn, sh_exp, sz_exp, bj_tpl):
        for code in ("100000", "500000", "900000"):
            with pytest.raises(UnsupportedMarketError):
                fn(code), label

    def test_error_is_a_value_error(self):
        # UnsupportedMarketError subclasses ValueError so existing broad
        # ``except ValueError`` handlers still catch it.
        assert issubclass(UnsupportedMarketError, ValueError)
        with pytest.raises(ValueError):
            EfinanceSource._to_secid("100000")


class _FakeSource:
    """Network-free provider stub that declares its market capability."""

    def __init__(self, name, markets, frame=None):
        self.SOURCE_NAME = name
        self._markets = set(markets)
        self._frame = frame if frame is not None else pd.DataFrame()
        self.available = True
        self.fetch_calls = 0

    def is_available(self):
        return self.available

    def supports_market(self, market):
        return market in self._markets

    def fetch_daily(self, stock_code, start_date, end_date):
        self.fetch_calls += 1
        return self._frame


class TestFailoverMarketGate:
    """failover skips a market-incapable source WITHOUT a failure count."""

    @staticmethod
    def _downloader(sources):
        dl = AShareDownloader()
        dl._sources = list(sources)
        dl._failure_counts = {}
        dl._circuit_open = {}
        return dl

    @staticmethod
    def _frame():
        return pd.DataFrame({
            "date": ["2024-01-02"], "open": [10.0], "high": [10.5],
            "low": [9.8], "close": [10.2], "volume": [1e6], "amount": [1e8],
            "pct_change": [1.0],
        })

    def test_bj_incapable_source_skipped_without_failure(self):
        shsz = _FakeSource("shsz", markets={"SH", "SZ"}, frame=self._frame())
        dl = self._downloader([shsz])
        out = dl.fetch_daily("920001", "2024-01-01", "2024-01-31")
        assert out.empty
        assert shsz.fetch_calls == 0
        assert "shsz" not in dl._failure_counts

    def test_bj_capable_source_is_asked(self):
        all_mkts = _FakeSource("all", markets={"SH", "SZ", "BJ"}, frame=self._frame())
        dl = self._downloader([all_mkts])
        out = dl.fetch_daily("920001", "2024-01-02", "2024-01-31")
        assert not out.empty
        assert all_mkts.fetch_calls == 1
        # success resets the counter to 0 — no failure was accumulated
        assert dl._failure_counts.get("all", 0) == 0

    def test_source_without_supports_market_is_not_gated(self):
        # A legacy/mock source that does not declare capability must not be
        # skipped (backward-compatible with the offline _FakeSource tests in
        # test_source_normalization.py).
        class _Legacy:
            SOURCE_NAME = "legacy"

            def __init__(self, frame):
                self._frame = frame
                self.calls = 0

            def is_available(self):
                return True

            def fetch_daily(self, stock_code, start_date, end_date):
                self.calls += 1
                return self._frame

        legacy = _Legacy(self._frame())
        dl = self._downloader([legacy])
        out = dl.fetch_daily("920001", "2024-01-02", "2024-01-31")
        assert not out.empty
        assert legacy.calls == 1
        assert dl._failure_counts.get("legacy", 0) == 0


class TestNormalizationCompat:
    """The routers still accept the same code spellings as before."""

    def test_float_and_prefixed_inputs(self):
        assert normalize_stock_code(600519.0) == "600519"
        assert EfinanceSource._to_secid(600519.0) == "1.600519"
        assert TushareSource._to_ts_code("600519.SH") == "600519.SH"
        assert BaostockSource._to_bs_code("830799.BJ") == "bj.830799"
        assert AKShareSource._to_sina_symbol("920001") == "bj920001"


@pytest.fixture(scope="module")
def _bs_code():
    import importlib.util
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "production" / "download_valuation.py"
    )
    spec = importlib.util.spec_from_file_location("download_valuation_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bs_code


class TestDownloadValuationBsCode:
    """scripts/production/download_valuation._bs_code — a Baostock provider
    route that previously raised ValueError on the 920xxx BSE range (the old
    ``6→sh; 0/3→sz; 8/4→bj; else raise`` heuristic rejected 920001)."""

    @pytest.mark.parametrize("code,expected", [
        ("600519", "sh.600519"),
        ("000001", "sz.000001"),
        ("300750", "sz.300750"),
        ("430047", "bj.430047"),
        ("830799", "bj.830799"),
        ("920001", "bj.920001"),
    ])
    def test_bs_code(self, _bs_code, code, expected):
        assert _bs_code(code) == expected

    def test_bs_code_raises_on_index(self, _bs_code):
        with pytest.raises(UnsupportedMarketError):
            _bs_code("100000")
