"""Provider normalization + cross-source seam tests.

All providers must emit a common convention at the adapter boundary:
OHLC = 前复权 (qfq), volume = 股 (shares), amount = 元 (CNY).  Cross-source
backfill must rebase the older segment's OHLC onto the primary source's
前复权 anchor so a naive concat cannot inject a fake price jump at the seam.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource


def _base_frame(volume=1_000_000.0, amount=50_000_000.0):
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "open": [10.0, 10.2, 10.4],
        "high": [10.5, 10.6, 10.7],
        "low": [9.9, 10.0, 10.1],
        "close": [10.1, 10.3, 10.5],
        "volume": [volume, volume, volume],
        "amount": [amount, amount, amount],
        "pct_change": [0.5, 1.98, 1.94],
        "turnover": [1.0, 1.1, 1.2],
        "amplitude": [2.0, 2.1, 2.2],
    })


class TestEfinanceNormalize:
    def test_volume_lots_to_shares(self):
        """EastMoney volume is 手 (×100 shares); must normalize to 股."""
        src = EfinanceSource()
        df = src._normalize(_base_frame(), "000001")
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 50_000_000.0)  # already 元

    def test_attrs_carried(self):
        src = EfinanceSource()
        df = src._normalize(_base_frame(), "000001")
        assert df.attrs["source"] == "efinance"
        assert df.attrs["adjustment_mode"] == "qfq"


def _tushare_frame():
    # Tushare pro_bar schema: trade_date %Y%m%d, vol 手, amount 千元, pct_chg
    return pd.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "trade_date": ["20240102", "20240103", "20240104"],
        "open": [10.0, 10.2, 10.4],
        "high": [10.5, 10.6, 10.7],
        "low": [9.9, 10.0, 10.1],
        "close": [10.1, 10.3, 10.5],
        "pre_close": [10.0, 10.1, 10.3],
        "pct_chg": [1.0, 1.98, 1.94],
        "vol": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        "amount": [50_000.0, 50_000.0, 50_000.0],
    })


class TestTushareNormalize:
    def test_units(self):
        """Tushare vol is 手 (×100) and amount is 千元 (×1000)."""
        src = TushareSource()
        df = src._normalize(_tushare_frame(), "000001")
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0 * 100.0)
        assert np.isclose(df["amount"].iloc[0], 50_000.0 * 1000.0)

    def test_attrs_carried(self):
        src = TushareSource()
        df = src._normalize(_tushare_frame(), "000001")
        assert df.attrs["source"] == "tushare"
        assert df.attrs["adjustment_mode"] == "qfq"


class TestAkshareNormalize:
    def test_units_already_shares_yuan(self):
        """Sina (stock_zh_a_daily) already returns 股 / 元 — no scaling."""
        src = AKShareSource()
        df = src._normalize(_base_frame(), "000001")
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 50_000_000.0)

    def test_attrs_carried(self):
        src = AKShareSource()
        df = src._normalize(_base_frame(), "000001")
        assert df.attrs["source"] == "akshare"
        assert df.attrs["adjustment_mode"] == "qfq"


class TestBaostockNormalize:
    def test_units_already_shares_yuan(self):
        src = BaostockSource()
        df = src._normalize(_base_frame(), "000001")
        assert np.isclose(df["volume"].iloc[0], 1_000_000.0)
        assert np.isclose(df["amount"].iloc[0], 50_000_000.0)

    def test_attrs_carried(self):
        src = BaostockSource()
        df = src._normalize(_base_frame(), "000001")
        assert df.attrs["source"] == "baostock"
        assert df.attrs["adjustment_mode"] == "qfq"


class TestStitchSegments:
    """Backfill (Baostock) OHLC must be rebased onto the primary 前复权 anchor."""

    def _backfill(self):
        dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
        closes = [10.0, 11.0, 12.0]  # qfq anchored to an older reference
        return pd.DataFrame({
            "date": dates, "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1e6, 1e6, 1e6], "amount": [1e8, 1e8, 1e8],
            "pct_change": [0.0, 10.0, 9.09],
        })

    def _primary(self):
        dates = [dt.date(2024, 1, 4), dt.date(2024, 1, 5), dt.date(2024, 1, 8)]
        closes = [24.0, 25.0, 26.0]  # same series anchored 2.0× higher
        return pd.DataFrame({
            "date": dates, "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1e6, 1e6, 1e6], "amount": [1e8, 1e8, 1e8],
            "pct_change": [0.0, 4.17, 4.00],
        })

    def test_rebases_onto_primary_anchor(self):
        rebased, ratio = AShareDownloader._stitch_segments(
            self._backfill(), self._primary()
        )
        assert ratio == pytest.approx(2.0)
        # overlap day (2024-01-04) is dropped from the backfill segment
        assert set(rebased["date"]) == {dt.date(2024, 1, 2), dt.date(2024, 1, 3)}
        assert np.allclose(rebased["close"], [20.0, 22.0])

    def test_stitch_produces_continuous_close(self):
        rebased, _ = AShareDownloader._stitch_segments(
            self._backfill(), self._primary()
        )
        stitched = pd.concat([rebased, self._primary()], ignore_index=True)
        stitched = stitched.sort_values("date").reset_index(drop=True)
        closes = stitched["close"].to_numpy()
        # no anchor jump at the seam: every adjacent return stays in ±30%
        rets = np.abs(np.diff(closes) / closes[:-1])
        assert (rets <= 0.30).all()

    def test_internal_returns_preserved_by_rebase(self):
        """A constant multiplicative rebase keeps each segment's daily returns.

        Only the OHLC rows that survive (non-overlap) are compared, since the
        overlap row is dropped from the backfill segment.
        """
        back = self._backfill()
        rebased, ratio = AShareDownloader._stitch_segments(back, self._primary())
        keep = back["date"].isin(rebased["date"])
        before = back.loc[keep, "close"].pct_change().dropna().to_numpy()
        after = rebased["close"].pct_change().dropna().to_numpy()
        assert ratio == pytest.approx(2.0)
        assert len(before) == len(after) > 0
        assert np.allclose(before, after)

    def test_no_overlap_returns_unchanged_and_none_ratio(self):
        back = self._backfill()
        primary = self._primary().iloc[[1, 2]].reset_index(drop=True)  # no shared day
        rebased, ratio = AShareDownloader._stitch_segments(back, primary)
        assert ratio is None
        pd.testing.assert_frame_equal(rebased.reset_index(drop=True), back)

    def test_volume_amount_not_rebased(self):
        """Only OHLC is scaled — volume/amount are already unit-consistent."""
        back = self._backfill()
        rebased, _ = AShareDownloader._stitch_segments(back, self._primary())
        keep = back["date"].isin(rebased["date"])
        assert np.allclose(rebased["volume"].to_numpy(),
                           back.loc[keep, "volume"].to_numpy())
        assert np.allclose(rebased["amount"].to_numpy(),
                           back.loc[keep, "amount"].to_numpy())


def _daily(dates, closes, volume=1e6, amount=1e8):
    """Build a normalized daily frame with the common provider convention."""
    closes = list(closes)
    return pd.DataFrame({
        "date": list(dates),
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [volume] * len(closes),
        "amount": [amount] * len(closes),
        "pct_change": [0.0] * len(closes),
    })


class _FakeSource:
    """Network-free stand-in for a provider; serves one frame per fetch call."""

    def __init__(self, name, frames=None, raise_exc=None):
        self.SOURCE_NAME = name
        self._frames = list(frames or [])
        self._raise_exc = raise_exc
        self.available = True
        self.fetch_calls = 0
        self.last_start = None
        self.last_end = None

    def is_available(self):
        return self.available

    def fetch_daily(self, stock_code, start_date, end_date):
        self.fetch_calls += 1
        self.last_start = start_date
        self.last_end = end_date
        if self._raise_exc is not None:
            exc, self._raise_exc = self._raise_exc, None
            raise exc
        if not self._frames:
            return pd.DataFrame()
        return self._frames.pop(0)


def _downloader(sources):
    d = AShareDownloader()
    d._sources = list(sources)
    d._failure_counts = {}
    d._circuit_open = {}
    return d


class TestFetchDailyExceptionBoundary:
    """A crash inside one provider must not kill the whole fetch."""

    def test_raising_source_skips_to_next(self):
        primary = _daily([dt.date(2024, 1, 2), dt.date(2024, 1, 3)], [10.0, 10.2])
        bad = _FakeSource("efinance", raise_exc=RuntimeError("boom"))
        good = _FakeSource("akshare", frames=[primary])
        empty = _FakeSource("tushare")
        dl = _downloader([bad, good, empty, _FakeSource("baostock")])
        out = dl.fetch_daily("000001", "2024-01-01", "2024-01-31")
        assert len(out) == 2
        assert out.attrs["source"] == "akshare"
        assert bad.fetch_calls == 1
        assert good.fetch_calls == 1
        assert dl._failure_counts["efinance"] == 1

    def test_all_sources_fail_returns_empty(self):
        dl = _downloader([
            _FakeSource("efinance", raise_exc=RuntimeError("x")),
            _FakeSource("akshare", frames=[pd.DataFrame()]),
            _FakeSource("tushare"),
            _FakeSource("baostock"),
        ])
        out = dl.fetch_daily("000001", "2024-01-01", "2024-01-31")
        assert out.empty

    def test_source_returning_none_treated_as_failure(self):
        """A provider returning None (not a frame) must not crash the loop."""
        none_src = _FakeSource("efinance")
        none_src.fetch_daily = lambda *a, **k: None
        primary = _daily([dt.date(2024, 1, 2)], [10.0])
        good = _FakeSource("akshare", frames=[primary])
        dl = _downloader([none_src, good, _FakeSource("tushare"),
                          _FakeSource("baostock")])
        out = dl.fetch_daily("000001", "2024-01-01", "2024-01-31")
        assert len(out) == 1
        assert out.attrs["source"] == "akshare"


class TestBackfillRejection:
    """No-overlap and extreme-ratio splices are REJECTED, not
    masked — primary data is kept as-is with the reason recorded."""

    _PRIMARY_DATES = [dt.date(2024, 1, 5), dt.date(2024, 1, 8), dt.date(2024, 1, 9)]
    _PRIMARY_CLOSES = [10.0, 10.2, 10.4]

    def _dl(self, backfill_df):
        primary = _daily(self._PRIMARY_DATES, self._PRIMARY_CLOSES)
        bs = _FakeSource("baostock", frames=[backfill_df])
        dl = _downloader([_FakeSource("efinance", frames=[primary]),
                          _FakeSource("akshare"), _FakeSource("tushare"), bs])
        return dl, bs

    def test_no_overlap_rejected_keeps_primary(self):
        # Backfill ends BEFORE primary starts — zero shared days.
        backfill = _daily([dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
                          [9.0, 9.2])
        dl, bs = self._dl(backfill)
        out = dl.fetch_daily("000001", "2000-01-01", "2024-12-31")
        assert list(out["date"]) == self._PRIMARY_DATES
        assert out.attrs["backfill_rejected"] == "no_overlap"
        assert bs.fetch_calls == 1

    def test_extreme_ratio_rejected_keeps_primary(self):
        # Shared day 2024-01-05: backfill close 2.0 vs primary 10.0 → ratio 5.0.
        backfill = _daily(
            [dt.date(2024, 1, 3), dt.date(2024, 1, 5)], [2.0, 2.0])
        dl, bs = self._dl(backfill)
        out = dl.fetch_daily("000001", "2000-01-01", "2024-12-31")
        assert list(out["date"]) == self._PRIMARY_DATES
        assert out.attrs["backfill_rejected"] == "ratio=5.00"
        assert bs.fetch_calls == 1

    def test_valid_overlap_splices_and_stamps_segments(self):
        # Shared day 2024-01-05 has backfill close == primary close → ratio 1.0.
        backfill = _daily(
            [dt.date(2000, 1, 3), dt.date(2000, 1, 4), dt.date(2024, 1, 5)],
            [5.0, 5.2, 10.0])
        dl, bs = self._dl(backfill)
        out = dl.fetch_daily("000001", "2000-01-01", "2024-12-31")
        # Backfill rows (minus overlap) prepend the primary rows.
        assert len(out) == 2 + 3
        assert out["date"].iloc[0] == dt.date(2000, 1, 3)
        assert out["date"].iloc[2] == dt.date(2024, 1, 5)
        assert out.attrs["backfill_rejected"] is None
        assert out.attrs["backfilled_from"] == "baostock"
        segs = out.attrs["source_segments"]
        assert len(segs) == 2
        assert segs[0]["source"] == "baostock"
        assert segs[0]["rows"] == 2
        assert segs[0]["end"] == "2000-01-04"
        assert segs[1]["source"] == "efinance"
        assert segs[1]["rows"] == 3

    def test_backfill_window_requests_overlap(self):
        """§十-2: the backfill request must extend ~45d past the primary start
        so a legitimately adjacent series gets an overlap day to calibrate on."""
        primary = _daily(self._PRIMARY_DATES, self._PRIMARY_CLOSES)
        bs = _FakeSource("baostock", frames=[_daily([dt.date(2024, 1, 5)], [10.0])])
        dl = _downloader([_FakeSource("efinance", frames=[primary]),
                          _FakeSource("akshare"), _FakeSource("tushare"), bs])
        out = dl.fetch_daily("000001", "2000-01-01", "2024-12-31")
        assert bs.fetch_calls == 1
        # The backfill request must extend ~45 calendar days past the primary
        # start (2024-01-05) so the two series overlap by at least one day.
        assert bs.last_start == "2000-01-01"
        got_end = pd.Timestamp(bs.last_end)
        assert got_end >= pd.Timestamp("2024-02-19")  # 2024-01-05 + 45d
        assert got_end <= pd.Timestamp("2024-02-21")
        assert len(out) == 3  # spliced (backfill overlap row dropped)
