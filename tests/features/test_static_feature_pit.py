"""Review v8 §三-2: static-feature Point-In-Time audit.

Static features must use values known at the decision day, not values
present-backfilled from today.  Three guarantees are tested here:

1. ``industry_code`` is NOT a static feature — the only stock→industry maps
   available (sector_map.json / stock_sector_cache.csv) are current-snapshot
   mappings with no point-in-time membership history, so a static industry
   code would backfill today's classification onto historical rows.
2. The four remaining statics are truncation-invariant: appending future rows
   (a "build features over 2000-2026 then train a 2015 window" mistake) must
   not change the static value at any earlier column.
3. Fundamental/valuation merges are PIT-lagged 1 day: a figure disclosed on
   day ``d`` first reaches the model at ``d+1``, never the same day.
"""

import numpy as np
import pandas as pd

from stoke_ml.features.pipeline import FeaturePipeline, _PIT_STATIC_COLS

SEQ_LEN = 20
HORIZON = 5


def _make_synthetic_panel(n_stocks=8, n_days=200, seed=7):
    rng = np.random.RandomState(seed)
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for i, code in enumerate(codes):
        drift = 0.0005 * (i % 3 - 1)
        close = 10.0 * np.cumprod(1 + rng.normal(drift, 0.02, n_days))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = np.abs(rng.normal(1e6, 2e5, n_days))
        amount = volume * close
        for t in range(n_days):
            rows.append({
                "date": dates[t], "stock_code": code,
                "open": float(open_[t]), "high": float(high[t]),
                "low": float(low[t]), "close": float(close[t]),
                "volume": float(volume[t]), "amount": float(amount[t]),
            })
    return pd.DataFrame(rows)


def _pipeline(**kw):
    base = dict(
        seq_len=SEQ_LEN, minute_mode=False,
        use_board=False, use_sector=False, use_concept=False,
    )
    base.update(kw)
    return FeaturePipeline(**base)


class TestStaticColumns:
    def test_industry_code_excluded(self):
        """industry_code must not be a static feature (no PIT membership source)."""
        assert "industry_code" not in _PIT_STATIC_COLS

    def test_all_statics_pit_derivable(self):
        assert set(_PIT_STATIC_COLS) == {
            "price_60d_q", "amt_60d_q", "listing_days", "board_code",
        }


class TestStaticTruncationInvariance:
    """Appending future rows must not change earlier static columns.

    Mirrors the §三-1 scaler truncation test: the 2000-2026 full-history build
    feeding a 2015 training window must produce the same 2015 static values as
    a build that stops at the end of 2015.
    """

    def test_truncated_panel_equals_full_panel_on_overlap(self):
        panel = _make_synthetic_panel()
        cut = 150  # columns [0, cut) overlap between the two builds

        full = _pipeline().build_panel_features(panel, horizon=HORIZON)
        trunc = _pipeline().build_panel_features(
            _truncate_by_date(panel, cut), horizon=HORIZON
        )

        assert full["static_features"].shape[2] == len(_PIT_STATIC_COLS)
        assert trunc["static_features"].shape[2] == len(_PIT_STATIC_COLS)

        # Overlap columns [0, cut) must be bit-identical — no future row may
        # influence an earlier static value.
        a = full["static_features"][:, :cut, :]
        b = trunc["static_features"][:, :cut, :]
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)

    def test_listing_days_is_chronological(self):
        """listing_days must increase with the calendar column, never decrease."""
        panel = _make_synthetic_panel()
        data = _pipeline().build_panel_features(panel, horizon=HORIZON)
        listing = data["static_features"][:, :, _PIT_STATIC_COLS.index("listing_days")]
        diffs = np.diff(listing, axis=1)
        assert (diffs >= 0).all(), "listing_days must be non-decreasing over time"


class TestFundamentalPitLag:
    """PE/PB & financials must be PIT-lagged: disclosed at d, known at d+1."""

    def test_fundamental_merge_lags_by_one_day(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "stock_code": ["000001"] * 5,
            "open": [10.0] * 5, "high": [10.5] * 5, "low": [9.5] * 5,
            "close": [10.0] * 5, "volume": [1e6] * 5,
        })
        # Raw quarterly fundamental row: report disclosed on day 0.
        fd = pd.DataFrame({
            "disclose_date": [dates[0]],
            "report_date": [dates[0]],
            "roe": [15.0],
        })

        out = _pipeline()._merge_fundamental(df, fd)
        roe = out["roe"].to_numpy()

        # Disclosed on day 0 → forward-filled, then shifted 1: day 0 holds 0
        # (a same-day value would leak today's disclosure into today's signal),
        # day 1+ carries the value.
        assert roe[0] == 0.0, f"roe must not be known on disclose day, got {roe}"
        assert np.allclose(roe[1:], 15.0), f"roe must appear at day 1, got {roe}"

    def test_valuation_merge_lags_by_one_day(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        df = pd.DataFrame({
            "date": dates,
            "stock_code": ["000001"] * 4,
            "open": [10.0] * 4, "high": [10.5] * 4, "low": [9.5] * 4,
            "close": [10.0] * 4, "volume": [1e6] * 4,
        })
        vd = pd.DataFrame({
            "date": dates, "stock_code": ["000001"] * 4,
            "pe_ttm": [12.0, 12.5, 13.0, 13.5],
        })
        out = _pipeline()._merge_valuation(df, vd)
        pe = out["pe_ttm"].to_numpy()
        # Same-day value must NOT appear at its own row (would be a leak):
        # day t must carry the value known at t-1.
        assert pe[0] == 0.0
        assert np.allclose(pe[1:], [12.0, 12.5, 13.0])


def _truncate_by_date(panel: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Keep only the first *n_days* trading days for every stock."""
    keep_dates = sorted(panel["date"].unique())[:n_days]
    return panel[panel["date"].isin(keep_dates)].reset_index(drop=True)
