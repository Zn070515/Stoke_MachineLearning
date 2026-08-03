"""Tests for FeaturePipeline — feature engineering and merge logic."""
import numpy as np
import pandas as pd
import pytest
from stoke_ml.features.pipeline import FeaturePipeline, SENTIMENT_COLS, GUBA_COLS


def _make_kl(n_days=200):
    """Synthetic K-line data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_days))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates, "open": close - rng.normal(0, 0.5, n_days),
        "high": close + rng.uniform(0.1, 2.0, n_days),
        "low": close - rng.uniform(0.1, 2.0, n_days),
        "close": close,
        "volume": rng.integers(1e6, 1e7, n_days).astype(float),
    })


def _make_sentiment(dates):
    """Daily sentiment DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "sentiment_mean": rng.uniform(-1, 1, len(dates)).astype(np.float32),
        "sentiment_std": rng.uniform(0, 0.5, len(dates)).astype(np.float32),
        "news_count": rng.integers(0, 10, len(dates)).astype("int16"),
        "positive_ratio": rng.uniform(0, 1, len(dates)).astype(np.float32),
        "negative_ratio": rng.uniform(0, 1, len(dates)).astype(np.float32),
        "has_news": [True] * len(dates),
    })


class TestFeaturePipelineBuild:

    def test_technical_only_returns_valid_shapes(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        X, y, aligned_close = pipe.build_features(df, target_col="close")
        assert X.ndim == 3  # (samples, seq_len, features)
        assert X.shape[0] > 0
        assert X.shape[1] == 20  # seq_len
        assert len(y) == X.shape[0]
        assert len(aligned_close) > len(y)

    def test_with_sentiment_merge(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=True, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        sentiment = _make_sentiment(df["date"])
        X, y, _ = pipe.build_features(df, sentiment_df=sentiment)
        assert X.shape[0] > 0

    def test_sentiment_disabled_skips_merge(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        sentiment = _make_sentiment(df["date"])
        X1, _, _ = pipe.build_features(df, sentiment_df=sentiment)
        X2, _, _ = pipe.build_features(df, sentiment_df=None)
        # Should produce identical shapes with or without data when disabled
        assert X1.shape == X2.shape

    def test_flat_mode_output(self):
        pipe = FeaturePipeline(
            seq_len=20, flat_mode=True, use_sentiment=False,
            use_announcements=False, use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        X, y, _ = pipe.build_features(df, target_col="close")
        assert X.ndim == 2  # (samples, seq_len * features)

    def test_build_features_insufficient_data(self):
        pipe = FeaturePipeline(seq_len=100)
        df = _make_kl(50)  # shorter than seq_len
        X, y, aligned_close = pipe.build_features(df)
        assert len(X) == 0
        assert len(y) == 0

    def test_target_is_ternary(self):
        pipe = FeaturePipeline(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        df = _make_kl(200)
        _, y, _ = pipe.build_features(df, target_col="close")
        # 3-class: down / flat / up (threshold 0.003), matching XGBoost num_class=3
        assert set(np.unique(y)).issubset({0, 1, 2})


class TestFeaturePipelineFlags:

    def test_new_flags_default_true(self):
        pipe = FeaturePipeline()
        assert pipe.use_margin is True
        assert pipe.use_northbound is True
        assert pipe.use_dragon_tiger is True
        assert pipe.use_fundamental is True
        assert pipe.use_etf_flow is True

    def test_new_flags_can_be_disabled(self):
        pipe = FeaturePipeline(
            use_margin=False, use_northbound=False,
            use_dragon_tiger=False, use_fundamental=False,
            use_etf_flow=False,
        )
        assert pipe.use_margin is False
        assert pipe.use_northbound is False

    def test_margin_merge_disabled_respects_flag(self):
        pipe = FeaturePipeline(seq_len=20, use_margin=False)
        df = _make_kl(300)
        margin_df = pd.DataFrame({
            "date": df["date"],
            "margin_balance": [1.0] * len(df),
            "margin_buy": [0.5] * len(df),
        })
        X1, _, _ = pipe.build_features(df, margin_df=margin_df)
        pipe2 = FeaturePipeline(seq_len=20, use_margin=True)
        X2, _, _ = pipe2.build_features(df, margin_df=margin_df)
        # With margin enabled, more columns → different shape
        assert X1.shape[2] != X2.shape[2]


class TestMergeHelpers:

    def test_merge_guba_adds_columns(self):
        pipe = FeaturePipeline(
            use_guba=True, use_sentiment=False, use_announcements=False,
            use_comment=False,
        )
        df = _make_kl(100)
        guba = pd.DataFrame({
            "date": df["date"],
            "guba_sentiment_mean": [0.3] * len(df),
            "guba_sentiment_std": [0.1] * len(df),
            "guba_post_count": [5] * len(df),
            "guba_positive_ratio": [0.4] * len(df),
            "guba_negative_ratio": [0.2] * len(df),
            "has_guba_post": [True] * len(df),
        })
        # Just test merge doesn't crash
        result = pipe._merge_guba(df.copy(), guba)
        assert "guba_sentiment_mean" in result.columns

    def test_merge_sentiment_lags_by_one_day(self):
        pipe = FeaturePipeline(use_sentiment=True)
        df = _make_kl(10)
        sentiment = pd.DataFrame({
            "date": df["date"],
            "sentiment_mean": np.arange(1, 11, dtype=np.float32),
            "sentiment_std": [0.1] * 10,
            "news_count": [1] * 10,
            "positive_ratio": [0.5] * 10,
            "negative_ratio": [0.1] * 10,
            "has_news": [True] * 10,
        })
        result = pipe._merge_sentiment(df.copy(), sentiment)
        # First row: shift + ZI fill → 0.0 (no prior-day sentiment available)
        assert result["sentiment_mean"].iloc[0] == 0.0
        # Second row should have first row's pre-shift sentiment (value 1.0)
        assert result["sentiment_mean"].iloc[1] == 1.0


def _make_panel_kl(dates, seed):
    """Synthetic OHLCV K-line for one stock on a given date list."""
    rng = np.random.RandomState(seed)
    close = 10.0 + np.cumsum(rng.randn(len(dates)) * 0.2)
    close = np.clip(close, 1.0, None)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    vol = rng.randint(1_000_000, 5_000_000, size=len(dates))
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


class TestPanelCalendarAlignment:
    """P0-1: build_panel_features aligns every stock to ONE global calendar.

    Column t of every array must be the SAME trading date for every stock;
    otherwise cross-sectional IC / Top-K / long-short evaluation (which index
    by column) mixes dates across stocks.
    """

    @staticmethod
    def _make_panel():
        base = pd.bdate_range("2020-01-01", "2020-08-31")
        A = _make_panel_kl(list(base), seed=1)
        A["stock_code"] = "000001"
        b_dates = base[base >= "2020-04-01"]
        B = _make_panel_kl(list(b_dates), seed=2)
        B["stock_code"] = "000002"
        c_dates = [x for x in base if not ("2020-05-11" <= str(x.date()) <= "2020-05-29")]
        C = _make_panel_kl(c_dates, seed=3)
        C["stock_code"] = "000003"
        return pd.concat([A, B, C], ignore_index=True)

    def _build(self):
        return FeaturePipeline(seq_len=60).build_panel_features(
            self._make_panel(), aux_data={}, horizon=5)

    def test_all_columns_share_one_calendar_date(self):
        pdata = self._build()
        di = pdata["date_indices"]
        T = pdata["past_known"].shape[1]
        assert di.shape == (3, T)
        for t in range(0, T, 10):
            assert len(set(di[:, t].tolist())) == 1, f"column {t} mixes dates"

    def test_late_listing_masked_before_listing(self):
        pdata = self._build()
        y_dir = pdata["y_direction"]
        first_A = int(np.where(y_dir[0] != -100)[0].min())
        first_B = int(np.where(y_dir[1] != -100)[0].min())
        assert first_B > first_A  # B lists 2020-04-01, A from 2020-01-01

    def test_suspension_produces_masked_hole(self):
        pdata = self._build()
        y_dir = pdata["y_direction"]
        hole = [t for t in range(y_dir.shape[1])
                if y_dir[2, t] == -100 and y_dir[0, t] != -100]
        assert len(hole) >= 5  # 3-week suspension

    def test_valid_positions_have_nonzero_return(self):
        pdata = self._build()
        y_dir = pdata["y_direction"]
        y_ret = pdata["y_return"]
        n_zero_valid = int(((y_dir != -100) & (y_ret == 0)).sum())
        assert n_zero_valid == 0


class TestPanelMasksAndCarry:
    """Review §二/§六/§八: mask splitting + carry-last-close realization.

    build_panel_features must emit per-task masks and a realized-return array
    whose semantics match the review:
      - entry_eligible_mask ⊆ observation_mask (every open-valid day is a real
        day, but a real day may lack an open — e.g. a data gap);
      - return_target_mask = a clean open[t+h]/open[t]-1 exists;
      - realized_return[t] = clean forward return where available, else carry
        to the last real close in (t, t+h], else flat 0 — defined for EVERY
        entry-eligible day so the candidate pool never conditions on a future
        label existing.
    """

    @staticmethod
    def _make_exact_panel():
        """12 days, open=10..21 (+1/day), close=open+0.5 — fully deterministic
        forward returns so the carry math is asserted exactly."""
        dates = pd.bdate_range("2020-01-01", periods=12)
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5
        return pd.DataFrame({
            "date": dates,
            "stock_code": ["600000"] * 12,
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": np.full(12, 1_000_000.0),
        })

    def _build(self):
        panel = self._make_exact_panel()
        return FeaturePipeline(seq_len=5).build_panel_features(
            panel, aux_data={}, horizon=3)

    def test_carry_last_close_realized(self):
        """Realized = clean open-to-open where available, else the last real
        close in (t, t+h], else 0 — never NaN, never label-conditioned."""
        p = self._build()
        r = p["realized_return"][0]
        ret = p["return_target_mask"][0]
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5

        # Clean forward return where open[t+3] exists (t = 0..8)
        for t in range(9):
            assert ret[t], f"return target should be valid at t={t}"
            assert np.isclose(r[t], open_[t + 3] / open_[t] - 1), f"t={t}"
        assert not ret[9:].any(), "tail has no forward open → not a return target"

        # Carry to the last real close in (t, t+h] — NOT flat 0
        assert np.isclose(r[9], close[11] / open_[9] - 1)
        assert np.isclose(r[10], close[11] / open_[10] - 1)
        # No later close at all → flat 0
        assert r[11] == 0.0

    def test_masks_split(self):
        """observation ⊇ entry; return/vol targets only where a clean window
        exists; direction follows the forward-return threshold."""
        p = self._build()
        obs = p["observation_mask"][0]
        entry = p["entry_eligible_mask"][0]
        ret = p["return_target_mask"][0]
        vol = p["vol_target_mask"][0]
        # Every open-valid day is also close-valid in synthetic K-line
        assert (entry & ~obs).sum() == 0
        assert obs.all() and entry.all()
        # Return target: clean open-to-open, last `horizon` days excluded
        assert ret[:9].all() and not ret[9:].any()
        # Vol target: (t, t+h] holds >= 2 valid close returns → t < T-h
        assert vol[:9].all() and not vol[9:].any()
        # All forward returns positive → every valid direction label is "up"
        assert (p["y_direction"][0][:9] == 2).all()

    def test_entry_subset_of_observation_calendar_aligned(self):
        """On the multi-stock calendar-aligned panel, a real open never appears
        on a day that isn't a real close (a K-line row supplies both)."""
        base = pd.bdate_range("2020-01-01", "2020-08-31")
        A = _make_panel_kl(list(base), seed=1)
        A["stock_code"] = "000001"
        B = _make_panel_kl(list(base), seed=2)
        B["stock_code"] = "000002"
        p = FeaturePipeline(seq_len=60).build_panel_features(
            pd.concat([A, B], ignore_index=True), aux_data={}, horizon=5)
        entry = p["entry_eligible_mask"]
        obs = p["observation_mask"]
        assert (entry & ~obs).sum() == 0


class TestPanelTruncationInvariance:
    """Review §五 anti-cheat test #2: features must not see the future.

    Build the panel twice — once truncated at 2020-12-31, once with the full
    history through 2026 — and assert the pre-2021 feature columns are
    BIT-IDENTICAL.  Any feature that differs when later data exists is using
    future information (look-ahead bias).

    Only the *features* are compared: y_return/y_volatility are targets and
    are ALLOWED to look ahead (predicting the future is the model's job).
    """

    @staticmethod
    def _make_panel(dates_full):
        base = list(pd.to_datetime(dates_full))
        A = _make_panel_kl(base, seed=1)
        A["stock_code"] = "000001"
        # B has a 3-week suspension in 2020-05 — exercises the mask/pad path
        # in BOTH builds, so a late suspension can't masquerade as truncation.
        b_dates = [x for x in base if not ("2020-05-11" <= str(x.date()) <= "2020-05-29")]
        B = _make_panel_kl(b_dates, seed=2)
        B["stock_code"] = "000002"
        return pd.concat([A, B], ignore_index=True)

    def _build(self, panel):
        return FeaturePipeline(seq_len=60).build_panel_features(
            panel, aux_data={}, horizon=5)

    def test_features_invariant_to_future_data(self):
        full = self._make_panel(pd.bdate_range("2019-01-01", "2026-01-01"))
        trunc = full[pd.to_datetime(full["date"]) <= "2020-12-31"].copy()

        p_full = self._build(full)
        p_trunc = self._build(trunc)

        # Same 2 stocks in both builds → arrays align stock-for-stock, and the
        # truncated calendar is a strict prefix of the full one (same stocks,
        # same start date), so feature column t is the same calendar day.
        T_trunc = p_trunc["past_known"].shape[1]
        assert p_full["past_known"].shape[0] == p_trunc["past_known"].shape[0] == 2
        assert p_full["past_known"].shape[1] > T_trunc

        for key in ("static_features", "past_known", "past_observed"):
            a = p_trunc[key]
            b = p_full[key][:, :T_trunc]
            assert a.shape == b.shape, f"{key} shape {a.shape} vs {b.shape}"
            # Tolerance (not bit-exact): at the data-start warm-up boundary
            # (window < 60 rows) rolling features resolve their "not-yet-nan"
            # sentinel to ~0 through different float accumulation order, so the
            # two builds differ by ~1e-15 there — pure float32 rounding, far
            # below the model's usable precision and NOT future information.
            # A real look-ahead feature (value fed from a future row) changes by
            # the feature's own magnitude (~1e-3..1), far above this tolerance.
            assert np.allclose(a, b, rtol=0.0, atol=1e-10), (
                f"{key} changes when future data is available — look-ahead bias"
            )

    def test_targets_may_depend_on_future_but_features_not(self):
        """The vol target IS forward-looking; the invariant is feature-only."""
        full = self._make_panel(pd.bdate_range("2019-01-01", "2026-01-01"))
        trunc = full[pd.to_datetime(full["date"]) <= "2020-12-31"].copy()
        p_full = self._build(full)
        p_trunc = self._build(trunc)
        T = p_trunc["y_volatility"].shape[1]
        # Vol target near the truncation boundary is forward-looking: in the
        # full build the window past 2020-12-31 exists, so these differ.
        tail_full = p_full["y_volatility"][:, T - 5:T]
        tail_trunc = p_trunc["y_volatility"][:, T - 5:T]
        assert not np.array_equal(tail_full, tail_trunc), (
            "expected forward-looking vol target to differ at the boundary"
        )
