"""Provider normalization + cross-source seam tests (review v6 §六).

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
