"""Tests for FeaturePipeline — feature engineering and merge logic."""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from stoke_ml.config import load_config as _real_load_config
from stoke_ml.features.pipeline import (
    FeaturePipeline, SENTIMENT_COLS, GUBA_COLS, _PIT_STATIC_COLS,
    _min_vol_nobs, _not_long_suspended, fold_dead_feature_columns,
)


def _make_kl(n_days=200):
    """Synthetic K-line data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_days))
    close = np.maximum(close, 1.0)
    volume = rng.integers(1e6, 1e7, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "open": close - rng.normal(0, 0.5, n_days),
        "high": close + rng.uniform(0.1, 2.0, n_days),
        "low": close - rng.uniform(0.1, 2.0, n_days),
        "close": close,
        "volume": volume,
        "amount": volume * close,
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
        result = pipe._aux._merge_guba(df.copy(), guba)
        assert "guba_sentiment_mean" in result.columns

    def test_merge_sentiment_keeps_effective_date(self):
        # NewsStorage maps post-close → next trading day at the storage layer,
        # so the feature layer must NOT shift again: the
        # value on its effective date stays put and is usable at next open.
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
        result = pipe._aux._merge_sentiment(df.copy(), sentiment)
        # No double lag: each row keeps its own effective-date value.
        assert result["sentiment_mean"].iloc[0] == 1.0
        assert result["sentiment_mean"].iloc[1] == 2.0

    def test_merge_guba_keeps_effective_date(self):
        pipe = FeaturePipeline(
            use_guba=True, use_sentiment=False, use_announcements=False,
            use_comment=False,
        )
        df = _make_kl(10)
        guba = pd.DataFrame({
            "date": df["date"],
            "guba_sentiment_mean": np.arange(1, 11, dtype=np.float32),
            "guba_sentiment_std": [0.1] * 10,
            "guba_post_count": [5] * 10,
            "guba_positive_ratio": [0.4] * 10,
            "guba_negative_ratio": [0.2] * 10,
            "has_guba_post": [True] * 10,
        })
        result = pipe._aux._merge_guba(df.copy(), guba)
        assert result["guba_sentiment_mean"].iloc[0] == 1.0
        assert result["guba_sentiment_mean"].iloc[1] == 2.0

    def test_merge_industry_never_emits_per_stock_relative_cols(self):
        # ind_matched_return / stock_vs_industry mapped a
        # stock onto its industry via the current-snapshot sector_map.json,
        # backfilling today's classification onto historical rows.  They must
        # not appear in the merged output, whatever the input carries.
        pipe = FeaturePipeline(use_industry=True)
        df = _make_kl(10)
        ind = pd.DataFrame({
            "date": df["date"],
            "ind_pct_up": [0.5] * 10,
            "ind_return_mean": [0.01] * 10,
            "ind_return_std": [0.02] * 10,
            "ind_return_max": [0.03] * 10,
            "ind_return_min": [-0.02] * 10,
            "ind_return_skew": [0.1] * 10,
            "ind_dispersion_20d": [0.005] * 10,
            "ind_matched_return": [0.0] * 10,   # must be dropped
            "stock_vs_industry": [0.0] * 10,   # must be dropped
        })
        result = pipe._aux._merge_industry(df.copy(), ind)
        assert "ind_matched_return" not in result.columns
        assert "stock_vs_industry" not in result.columns
        assert "ind_pct_up" in result.columns
        assert "ind_dispersion_20d" in result.columns

    def test_merge_earnings_keeps_effective_date(self):
        pipe = FeaturePipeline(
            use_earnings=True, use_sentiment=False, use_announcements=False,
            use_comment=False, use_guba=False,
        )
        df = _make_kl(10)
        edf = pd.DataFrame({
            "date": df["date"],
            "net_profit_yoy_low": np.arange(1, 11, dtype=np.float32),
            "net_profit_yoy_high": np.arange(2, 12, dtype=np.float32),
            "net_profit_low": np.arange(3, 13, dtype=np.float32),
            "net_profit_high": np.arange(4, 14, dtype=np.float32),
            "has_forecast": [True] * 10,
        })
        result = pipe._aux._merge_earnings(df.copy(), edf)
        assert result["net_profit_yoy_low"].iloc[0] == 1.0
        assert result["net_profit_yoy_low"].iloc[1] == 2.0


class TestMarketEnvAccountSplit:
    """§T5: the market_env ACCOUNT sub-part (PROXY-PIT, monthly investor/mkt-cap
    stats) is consumed ONLY via the explicit ablation opt-in
    (``use_market_env_account``) or once it is declared VERIFIED — never by a
    default revision-safe run.  The PRICE part (verified, same-day trade data)
    is always consumed.  Exercises the REAL consumption path
    (``_engineer_features`` → ``_merge_market_env``), not just the manifest."""

    _ACCOUNT_COLS = (
        "mkt_cap_total_z", "avg_account_cap_z",
        "investor_new_num", "investor_new_z",
    )
    _PRICE_COLS = ("high_low_ratio", "market_adv_ratio", "market_turnover_z")

    def _me_df(self, df):
        return pd.DataFrame({
            "date": df["date"],
            "high_low_ratio": 0.5,
            "market_adv_ratio": 0.6,
            "market_turnover_z": 0.1,
            "mkt_cap_total_z": 0.2,
            "avg_account_cap_z": 0.3,
            "investor_new_num": 100.0,
            "investor_new_z": 0.4,
        })

    def _pipe(self, **kw):
        base = dict(
            seq_len=20, use_sentiment=False, use_announcements=False,
            use_guba=False, use_comment=False,
        )
        base.update(kw)
        return FeaturePipeline(**base)

    def test_default_proxy_run_excludes_account_cols(self):
        """(a) The default formal pipeline consumes ONLY the verified PRICE
        columns — the 4 PROXY ACCOUNT columns are absent from the engineered
        feature matrix."""
        pipe = self._pipe()
        assert pipe.use_market_env_account is False
        df = _make_kl(60)
        feats = pipe._engineer_features(df.copy(), market_env_df=self._me_df(df))
        for c in self._PRICE_COLS:
            assert c in feats.columns, f"price col {c!r} missing"
        for c in self._ACCOUNT_COLS:
            assert c not in feats.columns, f"account col {c!r} leaked in proxy default"

    def test_ablation_flag_on_includes_account_cols(self):
        """(b) The explicit ablation opt-in (use_market_env_account=True) adds
        the ACCOUNT columns — they are consumed on that flag, and only on it."""
        pipe = self._pipe(use_market_env_account=True)
        assert pipe.use_market_env_account is True
        df = _make_kl(60)
        feats = pipe._engineer_features(df.copy(), market_env_df=self._me_df(df))
        for c in self._PRICE_COLS:
            assert c in feats.columns
        for c in self._ACCOUNT_COLS:
            assert c in feats.columns, f"account col {c!r} missing with flag on"

    def test_verified_account_included_by_default(self, monkeypatch):
        """(c) Once the account part is declared VERIFIED (builder upgrade), the
        account columns join the verified set automatically — included on a
        default run with the ablation flag OFF."""
        import stoke_ml.config.feature_profile as _fp
        monkeypatch.setattr(_fp, "MARKET_ENV_ACCOUNT_PIT", "verified")
        pipe = self._pipe()  # use_market_env_account defaults False
        df = _make_kl(60)
        feats = pipe._engineer_features(df.copy(), market_env_df=self._me_df(df))
        for c in self._PRICE_COLS:
            assert c in feats.columns
        for c in self._ACCOUNT_COLS:
            assert c in feats.columns, f"verified account col {c!r} missing by default"


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
        "amount": vol.astype(np.float64) * close,
    })


def test_amount_missing_raises():
    """§十一-5: the formal daily contract REQUIRES `amount` — a panel without
    it must fail loudly, never silently fall back to volume×close / price."""
    df = _make_panel_kl(pd.bdate_range("2020-01-02", periods=20), seed=1)
    df["stock_code"] = "000001"
    df = df.drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        FeaturePipeline(seq_len=5).build_panel_features(
            df, aux_data={}, horizon=1)


class TestPanelCalendarAlignment:
    """build_panel_features aligns every stock to ONE global calendar.

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


class TestCleanCalendarDates:
    """Unit tests for the per-stock date-axis cleaner."""

    def test_drops_weekend_and_dedupes_keep_last(self):
        fp = FeaturePipeline()
        df = pd.DataFrame({
            "date": ["2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19",
                     "2020-03-20", "2020-03-21", "2020-03-16"],
            "x": [1, 2, 3, 4, 5, 99, 6],
        })
        out = fp._clean_calendar_dates(df, "000001")
        assert out is not None
        dates = pd.to_datetime(out["date"])
        assert "2020-03-21" not in dates.dt.date.astype(str).tolist()
        assert len(dates) == len(dates.unique())
        # keep-last: the later duplicate (x=6) survives
        assert out.loc[out["date"] == "2020-03-16", "x"].iloc[0] == 6
        assert dates.is_monotonic_increasing

    def test_holiday_bar_dropped(self):
        fp = FeaturePipeline()
        # 2020-04-06 (清明) is a Monday holiday; neighbours are trading days.
        df = pd.DataFrame({
            "date": ["2020-04-03", "2020-04-06", "2020-04-07"],
            "x": [1, 2, 3],
        })
        out = fp._clean_calendar_dates(df, "000001")
        assert out["date"].tolist() == ["2020-04-03", "2020-04-07"]

    def test_all_rows_bad_returns_none(self):
        fp = FeaturePipeline()
        df = pd.DataFrame({"date": ["2020-03-21", "2020-03-22"], "x": [1, 2]})
        assert fp._clean_calendar_dates(df, "000001") is None

    def test_clean_calendar_dates_forwards_data_dir(self, tmp_path, monkeypatch):
        """§九: _clean_calendar_dates must thread the data_dir it is given into
        _get_panel_calendar — the strict calendar follows the frozen
        exchange_calendar artifact at the data root the caller actually reads."""
        from stoke_ml.features.panel_helpers import _get_panel_calendar as _real
        captured = {}

        def fake_get_panel_calendar(data_dir=None):
            captured["data_dir"] = data_dir
            return _real(data_dir)

        monkeypatch.setattr(
            "stoke_ml.features.pipeline._get_panel_calendar", fake_get_panel_calendar)
        fp = FeaturePipeline()
        df = pd.DataFrame({"date": ["2020-03-16", "2020-03-17"], "x": [1, 2]})
        out = fp._clean_calendar_dates(df, "000001", data_dir=str(tmp_path))
        assert out is not None
        assert captured["data_dir"] == str(tmp_path)


class TestGetPanelCalendarDataDir:
    """§九: _get_panel_calendar must honor an explicit data_dir — a formal flow
    passes the data root it actually reads, so the frozen exchange_calendar
    artifact at THAT root is authoritative (and hash-bindable), never the
    process config default."""

    def test_explicit_data_dir_reads_that_artifact(self, tmp_path):
        import datetime as dt

        from stoke_ml.data.calendar import load_calendar, save_calendar
        from stoke_ml.features.panel_helpers import _get_panel_calendar
        # Write a calendar artifact at tmp_path whose verified_until DIFFERS
        # from the code default — the explicit-data_dir call must read IT.
        save_calendar(str(tmp_path), "a_shares")
        frame = load_calendar(str(tmp_path), "a_shares")
        frame["verified_until"] = pd.Timestamp("2025-12-31")
        frame.to_parquet(str(tmp_path / "exchange_calendar" / "a_shares.parquet"))
        cal = _get_panel_calendar(str(tmp_path))
        assert cal.verified_until == dt.date(2025, 12, 31)
        assert cal.verified_until != dt.date(2026, 12, 31)

    def test_no_arg_resolves_config_default(self):
        import datetime as dt

        from stoke_ml.features.panel_helpers import _get_panel_calendar
        # No arg → the config default data root, whose artifact (or the code
        # fallback) carries the verified 2026-12-31 bound — NOT the modified
        # artifact written at the explicit test root above.
        cal = _get_panel_calendar()
        assert cal.verified_until == dt.date(2026, 12, 31)


class TestPanelRowIdentity:
    """§v12-P0 regression: build_panel_features row i MUST map to the
    returned stock_codes[i] — never to the position of the original code in
    the raw panel.  When a stock is cleaned out, every subsequent row would
    otherwise be mislabelled (board one-hot, universe mask, OOS artifact
    codes) without any error being raised."""

    @staticmethod
    def _make_panel():
        base = pd.bdate_range("2020-01-01", "2020-08-31")
        A = _make_panel_kl(list(base), seed=1)
        A["stock_code"] = "000001"  # SZ main board (starts with "00")
        # B: EVERY row is an off-calendar day (both Saturdays) →
        # _clean_calendar_dates returns None → B drops out of the feature
        # stack entirely.
        B = _make_panel_kl(["2020-03-14", "2020-03-21"], seed=2)
        B["stock_code"] = "000002"
        C = _make_panel_kl(list(base), seed=3)
        C["stock_code"] = "600519"  # SH main board (starts with "60")
        return pd.concat([A, B, C], ignore_index=True)

    def _build(self):
        return FeaturePipeline(seq_len=60).build_panel_features(
            self._make_panel(), aux_data={}, horizon=5)

    def test_dropped_stock_excluded_from_stock_codes(self):
        p = self._build()
        assert p["stock_codes"] == ["000001", "600519"]
        assert p["static_features"].shape[0] == 2
        assert p["past_observed"].shape[0] == 2

    def test_row_one_maps_to_original_third_stock(self):
        # Regression: before the fix, row 1 carried B's (dropped) position, so
        # its board one-hot came from code "000002" instead of "600519".
        p = self._build()
        cols = list(_PIT_STATIC_COLS)
        j_sz = cols.index("board_sz_main")
        j_sh = cols.index("board_sh_main")
        sf = p["static_features"]
        # Row 0 = 000001 (SZ main): sz_main fires, sh_main stays cold.
        assert sf[0, :, j_sz].max() == 1.0
        assert sf[0, :, j_sh].max() == 0.0
        # Row 1 = 600519 (SH main): sh_main fires, sz_main stays cold.
        assert sf[1, :, j_sh].max() == 1.0
        assert sf[1, :, j_sz].max() == 0.0


class TestPanelCalendarDateValidity:
    """build_panel_features keeps every stock's date axis on the
    official A-share calendar before the UNION date axis is built, so a wrong
    weekend/closed-day bar cannot expand the global panel time dimension."""

    @staticmethod
    def _panel_with_bad_rows():
        base = pd.bdate_range("2020-01-02", "2020-06-30")
        A = _make_panel_kl(list(base), seed=1)
        A["stock_code"] = "000001"
        B = _make_panel_kl(list(base), seed=2)
        B["stock_code"] = "000002"
        dup = B[B["date"] == "2020-03-16"].copy()          # duplicate date
        wk = _make_panel_kl([pd.Timestamp("2020-03-21")], seed=9)  # Saturday
        wk["stock_code"] = "000002"
        B = pd.concat([B, dup, wk], ignore_index=True)
        return pd.concat([A, B], ignore_index=True)

    def _build(self):
        return FeaturePipeline(seq_len=60).build_panel_features(
            self._panel_with_bad_rows(), aux_data={}, horizon=5)

    def test_global_calendar_is_all_trading_days(self):
        pdata = self._build()
        gd = pd.DatetimeIndex(pdata["global_dates"])
        assert not (gd.dayofweek >= 5).any()            # no weekend columns
        assert len(set(gd.tolist())) == len(gd)          # no duplicate columns
        assert pd.Timestamp("2020-03-21") not in gd      # injected weekend dropped
        assert pd.Timestamp("2020-04-06") not in gd      # 清明 holiday excluded
        assert pd.Timestamp("2020-03-16") in gd          # valid trading day kept

    def test_union_invariant_fires_when_cleaning_bypassed(self, monkeypatch):
        """§九-1: if per-stock calendar cleaning were (regression) skipped, a
        weekend bar that reaches the UNION must raise instead of silently
        widening the global panel time axis."""
        base = pd.bdate_range("2020-01-02", "2020-06-30")
        A = _make_panel_kl(list(base), seed=1)
        A["stock_code"] = "000001"
        wk = _make_panel_kl([pd.Timestamp("2020-03-21")], seed=9)  # Saturday
        wk["stock_code"] = "000001"
        panel = pd.concat([A, wk], ignore_index=True)
        fp = FeaturePipeline(seq_len=60)
        # Simulate an upstream regression that bypasses the per-stock cleaner.
        monkeypatch.setattr(
            fp, "_clean_calendar_dates",
            lambda df, code, data_dir=None: df.sort_values("date").reset_index(drop=True))
        with pytest.raises(ValueError, match="official a_shares"):
            fp.build_panel_features(panel, aux_data={}, horizon=5)


class TestCrossSectionCleanup:
    """Cross-sectional statistics must not be polluted by an
    inf, quantile ties must rank equally, and the size proxy must read the real
    `amount` column instead of re-estimating volume×qfq-close."""

    @staticmethod
    def _dates():
        return list(pd.bdate_range("2020-03-02", "2020-08-31"))

    def _build(self, panel):
        return FeaturePipeline(seq_len=60).build_panel_features(
            panel, aux_data={}, horizon=5)

    def test_inf_row_does_not_corrupt_the_date_cross_section(self):
        """§十二-2: a single inf in one stock's feature must be filtered before
        the per-date mean/std, so the OTHER stocks' z-scores on that date stay
        intact instead of being zeroed by the final nan_to_num."""
        dirty = self._build(self._panel(with_inf=True))
        # No NaN/Inf may leak into the emitted feature arrays.
        for key in ("past_known", "past_observed", "static_features"):
            assert np.isfinite(dirty[key]).all(), f"{key} leaked non-finite"
        gd = pd.DatetimeIndex(dirty["global_dates"])
        idx = int(np.where(gd == pd.Timestamp("2020-06-01"))[0][0])
        # Locate the injected raw column via the emitted column manifest.
        cf = dirty["past_observed_cols"].index("custom_factor")
        # Its own inf cell is sanitized to a finite 0...
        assert dirty["past_observed"][1, idx, cf] == 0.0
        # ...while the OTHER stocks keep live z-scores on the inf date — not
        # zeroed by a polluted mean/std (without the finite pre-filter their
        # z-scores would be NaN→0 because the date's mean/std becomes inf).
        others = dirty["past_observed"][[0, 2, 3], idx, cf]
        assert np.isfinite(others).all()
        assert (others != 0).all(), \
            "an inf must not zero the other stocks' z-scores on that date"

    def test_bool_state_column_survives_cross_section_norm(self):
        """§P1-7: a bool state flag (has_ever_observed / is_stale) reaching the
        cross-sectional path must not crash np.isfinite — numpy 2.x raises
        TypeError on bool — and must be emitted as finite z-scores, not zeroed."""
        frames = []
        for i, code in enumerate(["000001", "000002", "000003", "000004"]):
            df = _make_panel_kl(TestCrossSectionCleanup._dates(), seed=200 + i)
            df["stock_code"] = code
            df["state_known"] = np.array([i % 2 == 0] * len(df))
            frames.append(df)
        p = self._build(pd.concat(frames, ignore_index=True))
        po_cols = list(p["past_observed_cols"])
        assert "state_known" in po_cols
        vals = p["past_observed"][:, :, po_cols.index("state_known")]
        assert np.isfinite(vals).all()
        # The bool survives z-scoring to ±1 (or 0 outside a stock's history) —
        # a crashed/polluted cross-section would zero the whole channel.
        assert np.unique(vals).size >= 2

    @staticmethod
    def _panel(with_inf=False, inf_date="2020-06-01"):
        dates = TestCrossSectionCleanup._dates()
        frames = []
        for i, code in enumerate(["000001", "000002", "000003", "000004"]):
            df = _make_panel_kl(dates, seed=100 + i)
            df["stock_code"] = code
            # Deterministic per-stock PO feature — normalized cross-sectionally,
            # with no downstream derivation (isolates the z-score path).
            df["custom_factor"] = 1.0 + 0.01 * i + np.arange(len(df)) / 1000.0
            if with_inf and code == "000002":
                df.loc[df["date"] == pd.Timestamp(inf_date), "custom_factor"] = np.inf
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def test_quantile_ties_get_equal_rank(self):
        """§十二-3: stocks with identical trailing-60d amounts must get the SAME
        amt_60d_q quantile (rank method='average'), independent of array order."""
        dates = self._dates()
        frames = []
        for code in ["000001", "000002", "000003"]:
            df = _make_panel_kl(dates, seed=7)   # identical OHLCV for all 3
            df["stock_code"] = code
            df["amount"] = df["volume"] * df["close"]   # identical across stocks
            frames.append(df)
        p = self._build(pd.concat(frames, ignore_index=True))
        amt_q = p["static_features"][:, :, _PIT_STATIC_COLS.index("amt_60d_q")]
        ready = p["observation_mask"].all(axis=0) & (amt_q > 0).all(axis=0)
        assert ready.any(), "need a cross-section where all 3 stocks are listed"
        assert np.allclose(amt_q[0, ready], amt_q[1, ready])
        assert np.allclose(amt_q[0, ready], amt_q[2, ready])

    def test_amt_uses_real_amount_column(self):
        """§十二-4: amt_60d_q must track the canonical `amount` column.  Stock B
        has identical OHLCV to A but HALF the real turnover, so it must rank
        below A — volume×qfq-close (the old proxy) would make them a tie."""
        dates = self._dates()
        A = _make_panel_kl(dates, seed=7)
        A["stock_code"] = "000001"
        B = _make_panel_kl(dates, seed=7)        # identical OHLCV to A
        B["stock_code"] = "000002"
        A["amount"] = A["volume"] * A["close"]
        B["amount"] = 0.5 * B["volume"] * B["close"]  # real turnover is HALF
        p = self._build(pd.concat([A, B], ignore_index=True))
        amt_q = p["static_features"][:, :, _PIT_STATIC_COLS.index("amt_60d_q")]
        a_q, b_q = amt_q[0], amt_q[1]
        assert np.all(b_q[60:] < a_q[60:]), \
            "B's lower real turnover must rank below A's"


class TestPanelMasksAndCarry:
    """Mask splitting + carry-last-close realization.

    build_panel_features must emit per-task masks and a realized-return array
    with these semantics:
      - entry_eligible_mask ⊆ observation_mask (every open-valid day is a real
        day, but a real day may lack an open — e.g. a data gap);
      - return_target_mask = a usable forward return exists (§T13 decision 3):
        clean open[t+h]/open[t]-1 where a real exit open exists, else carry to
        the last real close in (t, t+h], else no label (NaN);
      - realized_return[t] = clean forward return where available, else carry
        to the last real close in (t, t+h], else flat 0 — defined for EVERY
        entry-eligible day so the candidate pool never conditions on a future
        label existing.  For carried days it is bit-identical to the training
        label (return_target_mask True).
    """

    @staticmethod
    def _make_exact_panel():
        """12 days, open=10..21 (+1/day), close=open+0.5 — fully deterministic
        forward returns so the carry math is asserted exactly.  Starts on
        2020-01-02 (NOT 01-01, which is 元旦) so every bdate is an official
        trading day — the §十二-1 calendar-clean pass drops nothing."""
        dates = pd.bdate_range("2020-01-02", periods=12)
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
            "amount": np.full(12, 1_000_000.0) * close,
        })

    def _build(self):
        panel = self._make_exact_panel()
        return FeaturePipeline(seq_len=5).build_panel_features(
            panel, aux_data={}, horizon=3)

    def test_carry_last_close_realized(self):
        """Realized = clean open-to-open where available, else the last real
        close in (t, t+h], else 0 — never NaN, never label-conditioned.  §T13:
        the TRAINING label (y_return) now carries non-fillable exits with the
        SAME value, so the tail becomes a label too (return_target_mask True)."""
        p = self._build()
        r = p["realized_return"][0]
        y_ret = p["y_return"][0]
        ret = p["return_target_mask"][0]
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5

        # Clean forward return where open[t+3] exists (t = 0..8)
        for t in range(9):
            assert ret[t], f"return target should be valid at t={t}"
            assert np.isclose(r[t], open_[t + 3] / open_[t] - 1), f"t={t}"

        # §T13 carried tail: last real close in (t, t+h] — training label and
        # eval realized agree exactly.
        for t in (9, 10):
            assert ret[t], f"carried tail should be a label at t={t}"
            expect = close[11] / open_[t] - 1
            assert np.isclose(y_ret[t], expect), f"training label t={t}"
            assert np.isclose(r[t], expect), f"realized t={t}"
        # No close at all in the exit window → no label (NaN), realized flat 0
        assert not ret[11], "no close in the exit window → not a return target"
        assert y_ret[11] == 0.0
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
        # Return target: clean open-to-open plus §T13 carried tail labels;
        # only the final day (no close in its exit window) has no label.
        assert ret[:11].all() and not ret[11]
        # Vol target: (t, t+h] holds >= 2 valid close returns → t < T-h
        assert vol[:9].all() and not vol[9:].any()
        # All forward returns (clean + carried) positive → "up" everywhere
        assert (p["y_direction"][0][:11] == 2).all()

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


class TestPanelCarriedReturnLabel:
    """§T13 decision 3: the training return label carries non-fillable exits to
    the last real close in (t, t+h], aligned bit-identically with the evaluation
    realized path, and the panel emits a per-date exit-fill probability
    (fill_prob)."""

    @staticmethod
    def _exact_panel():
        """12 days, open=10..21 (+1/day), close=open+0.5 — deterministic so the
        carry math is asserted exactly.  Business days from 2020-01-02 (trading
        days; §十二-1 calendar-clean drops nothing)."""
        dates = pd.bdate_range("2020-01-02", periods=12)
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
            "amount": np.full(12, 1_000_000.0) * close,
        })

    def _build(self, panel):
        return FeaturePipeline(seq_len=5).build_panel_features(
            panel, aux_data={}, horizon=3)

    def test_clean_exit_label_unchanged(self):
        """A stock with a real open[t+h] keeps the clean open-to-open label."""
        p = self._build(self._exact_panel())
        ret = p["return_target_mask"][0]
        y_ret = p["y_return"][0]
        open_ = np.arange(10.0, 22.0)
        for t in range(9):
            assert ret[t], f"clean exit label missing at t={t}"
            assert np.isclose(y_ret[t], open_[t + 3] / open_[t] - 1.0), f"t={t}"

    def test_nonfillable_tail_carries_to_last_close(self):
        """No open[t+h] (tail) but a real close in (t, t+h] → the training label
        carries to the last real close, EXACTLY the evaluation realized value
        (alignment guarantee)."""
        p = self._build(self._exact_panel())
        y_ret = p["y_return"][0]
        r = p["realized_return"][0]
        ret = p["return_target_mask"][0]
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5
        # t=9: window (9, min(12, 11)] = (9, 11] → last real close = close[11].
        expect = close[11] / open_[9] - 1.0
        assert ret[9]
        assert np.isclose(y_ret[9], expect)
        assert np.isclose(r[9], expect)
        # t=10: window (10, 11] → close[11].
        expect = close[11] / open_[10] - 1.0
        assert ret[10]
        assert np.isclose(y_ret[10], expect)
        assert np.isclose(r[10], expect)

    def test_no_close_in_window_gives_no_label(self):
        """No real close in the exit window → ret_fwd NaN / return_target False
        (realized still flat 0, but the day is not a training label)."""
        p = self._build(self._exact_panel())
        y_ret = p["y_return"][0]
        ret = p["return_target_mask"][0]
        r = p["realized_return"][0]
        assert not ret[11], "final day has no exit close → not a label"
        assert y_ret[11] == 0.0, "no label → return zeroed"
        assert r[11] == 0.0, "realized flat (no exit)"

    def test_suspension_before_exit_carries_to_last_close(self):
        """A mid-panel suspension (open missing at t+h) with closes continuing
        in the window → carry to the last real close in (t, t+h]."""
        df = self._exact_panel()
        # Simulate a suspension: day 8 has no real open (the exit open for
        # t=5) but keeps a close, so the carry window (5, 8] is non-empty.
        df.loc[8, "open"] = 0.0
        p = self._build(df)
        y_ret = p["y_return"][0]
        r = p["realized_return"][0]
        ret = p["return_target_mask"][0]
        entry = p["entry_eligible_mask"][0]
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5
        # open[8]=0 → day 8 is not entry-eligible and not a valid exit open.
        assert not entry[8]
        # Entry t=5: open[8] missing → carry to last real close in (5, 8] = close[8].
        expect = close[8] / open_[5] - 1.0
        assert ret[5]
        assert np.isclose(y_ret[5], expect)
        assert np.isclose(r[5], expect)

    def test_fill_prob_per_date_fraction(self):
        """fill_prob[t] = fraction of entry-eligible stocks at t with a real
        open[t+horizon]; NaN for the tail columns (no exit window)."""
        dates = pd.bdate_range("2020-01-02", periods=12)
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5
        base = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": np.full(12, 1_000_000.0),
            "amount": np.full(12, 1_000_000.0) * close,
        })
        A = base.copy(); A["stock_code"] = "600001"
        B = base.copy(); B["stock_code"] = "600002"
        B.loc[4, "open"] = 0.0  # B suspended at day 4 (horizon=2 exit for t=2)
        p = FeaturePipeline(seq_len=5).build_panel_features(
            pd.concat([A, B], ignore_index=True), aux_data={}, horizon=2)
        fill = p["fill_prob"]
        assert "fill_prob" in p, "fill_prob must be in the panel payload"
        assert fill.shape == (12,)
        # t=0,1: both open-valid AND open[t+2] valid → 2/2 = 1.0
        assert np.isclose(fill[0], 1.0)
        assert np.isclose(fill[1], 1.0)
        # t=2: B's exit open[4] is missing → filled 1/2
        assert np.isclose(fill[2], 0.5)
        # t=3,4: both valid at t and t+2 → 1.0
        assert np.isclose(fill[3], 1.0)
        assert np.isclose(fill[4], 1.0)
        # tail columns (t+horizon >= max_T) → NaN
        assert np.isnan(fill[10]) and np.isnan(fill[11])

    def test_entry_fill_prob_per_date_fraction(self):
        """entry_fill_prob[t] = fraction of DECISION-eligible stocks at t with
        a fillable entry open at t (§十八); NaN where no stock is
        decision-eligible (t=0: no close[t-1] yet).  Unlike fill_prob it has
        NO horizon pairing, so the full [:max_T] grid is populated."""
        dates = pd.bdate_range("2020-01-02", periods=12)
        open_ = np.arange(10.0, 22.0)
        close = open_ + 0.5
        base = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": np.full(12, 1_000_000.0),
            "amount": np.full(12, 1_000_000.0) * close,
        })
        A = base.copy(); A["stock_code"] = "600001"
        B = base.copy(); B["stock_code"] = "600002"
        B.loc[4, "open"] = 0.0  # B has a real close[3] but no entry open at day 4
        p = FeaturePipeline(seq_len=5).build_panel_features(
            pd.concat([A, B], ignore_index=True), aux_data={}, horizon=2)
        efp = p["entry_fill_prob"]
        assert "entry_fill_prob" in p, "entry_fill_prob must be in the panel payload"
        assert efp.shape == (12,)
        # t=0: no decision (no real close at t-1) → NaN
        assert np.isnan(efp[0])
        # t=1,2,3: both decision-eligible AND fillable → 2/2
        assert np.isclose(efp[1], 1.0)
        assert np.isclose(efp[2], 1.0)
        assert np.isclose(efp[3], 1.0)
        # t=4: B is decision-eligible (real close[3]) but has NO entry open
        # (open[4]=0) → only A fills → 1/2.  This is exactly the §十八
        # decision-eligible-but-unfillable execution-risk scenario.
        assert np.isclose(efp[4], 0.5)
        # t=5..11: both decision-eligible and fillable → 1.0
        assert np.isclose(efp[5:], 1.0).all()


class TestPanelTruncationInvariance:
    """Anti-cheat test #2: features must not see the future.

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

    @pytest.mark.slow
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

    @pytest.mark.slow
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


def _vol_panel(closes, drop_idxs, code):
    """One stock's panel frame with rows *drop_idxs* removed (suspension)."""
    from stoke_ml.data.calendar import TradingCalendar
    base = pd.to_datetime(
        TradingCalendar().get_trading_days("2024-01-01", "2024-03-01")[:30]
    )
    keep = np.array([i for i in range(30) if i not in set(drop_idxs)])
    c = np.asarray(closes, dtype=np.float64)
    return pd.DataFrame({
        "date": base[keep], "open": c[keep], "high": c[keep] + 0.5,
        "low": c[keep] - 0.5, "close": c[keep], "volume": 1e6,
        "amount": c[keep] * 1e6, "stock_code": code,
    })


def test_min_vol_nobs_threshold():
    """§十四-3: min valid returns for a vol label = max(1, ceil(horizon/2))."""
    assert _min_vol_nobs(1) == 1
    assert _min_vol_nobs(5) == 3
    assert _min_vol_nobs(20) == 10


def test_forward_vol_nobs_three_tiers():
    """§十四-3: forward_vol_nobs records the per-label count of valid daily
    returns in each forward window; vol_target_mask requires >=
    _min_vol_nobs(horizon) (hard floor 2).  Full-h window vs partial-window vs
    <2-value: only the first is labelable for horizon=5."""
    closes = 100.0 + np.arange(30, dtype=np.float64)
    a = _vol_panel(closes, [], "000001")                 # full 5/5 valid
    b = _vol_panel(closes, [10, 11, 12], "000002")       # 2/5 valid (partial)
    c = _vol_panel(closes, [10, 11, 12, 13], "000003")   # 1/5 valid (<2 floor)
    panel = pd.concat([a, b, c], ignore_index=True)
    p = FeaturePipeline(seq_len=10).build_panel_features(
        panel, aux_data={}, horizon=5)
    # Column 8 → forward window [9, 10, 11, 12, 13].
    # Tier 1: full window, all 5 closes valid → label.
    assert p["forward_vol_nobs"][0, 8] == 5
    assert p["vol_target_mask"][0, 8]
    # Tier 2: 2 valid closes < ceil(5/2)=3 → no label, count still recorded.
    assert p["forward_vol_nobs"][1, 8] == 2
    assert not p["vol_target_mask"][1, 8]
    # Tier 3: 1 valid close < hard floor 2 → no label.
    assert p["forward_vol_nobs"][2, 8] == 1
    assert not p["vol_target_mask"][2, 8]
    # Tail: no full forward window → nobs stays 0.
    assert p["forward_vol_nobs"][0, 29] == 0


def test_volatility_target_spans_full_horizon_with_suspension():
    """A '5-day vol' label must span the FULL horizon —
    suspended days get a 0 return and the resumption day records the close
    gap — instead of collapsing to whichever days actually traded."""
    from stoke_ml.data.calendar import TradingCalendar
    # Use the OFFICIAL A-share calendar, not bdate_range: bdate_range includes
    # non-trading days (e.g. 2024-01-01, 2024-02-09) that build_panel_features
    # drops, which would break the column-index <-> date mapping.
    base = pd.to_datetime(
        TradingCalendar().get_trading_days("2024-01-01", "2024-03-01")[:30]
    )
    closes = 100.0 + np.arange(30, dtype=np.float64)
    a = pd.DataFrame({
        "date": base, "open": closes, "high": closes + 0.5,
        "low": closes - 0.5, "close": closes, "volume": 1e6,
        "amount": closes * 1e6, "stock_code": "000001",
    })
    # ONE suspended day (index 11) so the forward window [9..13] still holds 4
    # valid closes (>= _min_vol_nobs(5)=3) and stays labelable — the §十四-3
    # threshold would reject a 3-day suspension here (2/5 valid).
    b_idx = np.array([i for i in range(30) if i != 11])
    b_close = closes.copy()
    b_close[12] = b_close[10] * 1.10  # +10% gap over the 1-day suspension
    b = pd.DataFrame({
        "date": base[b_idx], "open": b_close[b_idx],
        "high": b_close[b_idx] + 0.5, "low": b_close[b_idx] - 0.5,
        "close": b_close[b_idx], "volume": 1e6,
        "amount": b_close[b_idx] * 1e6, "stock_code": "000002",
    })
    panel = pd.concat([a, b], ignore_index=True)
    p = FeaturePipeline(seq_len=10).build_panel_features(
        panel, aux_data={}, horizon=5)

    # Column 8 → forward window [9, 10, 11, 12, 13] for stock B (row 1):
    # day 11 suspended → 0 return; day 12 resumption → the +10% gap.
    win = np.array([
        b_close[9] / b_close[8] - 1.0,   # day 9: normal daily return
        b_close[10] / b_close[9] - 1.0,  # day 10: normal daily return
        0.0,                             # day 11: suspended (zero return)
        b_close[12] / b_close[10] - 1.0, # day 12: resumption gap
        b_close[13] / b_close[12] - 1.0, # day 13: normal daily return
    ])
    expected = float(np.std(win))
    assert p["vol_target_mask"][1, 8]
    assert p["forward_vol_nobs"][1, 8] == 4  # 4 valid closes in the window
    assert np.isclose(p["y_volatility"][1, 8], expected)
    # Guard: skipping the suspended day (0 return) and the resumption gap would
    # give a different, biased std — the label must span the FULL horizon.
    skip_zero = np.array([
        b_close[9] / b_close[8] - 1.0,
        b_close[10] / b_close[9] - 1.0,
        b_close[12] / b_close[10] - 1.0,
        b_close[13] / b_close[12] - 1.0,
    ])
    assert not np.isclose(expected, float(np.std(skip_zero)))


class TestNotLongSuspended:
    """§七-3 long-suspension gate: disqualified from the first column whose
    consecutive missing-close run reaches the threshold, through `lookback`
    trading columns after the run resumes.  Pre-listing columns are never
    "missing"."""

    def test_never_suspended_all_eligible(self):
        obs = np.ones((1, 10), dtype=bool)
        mask = _not_long_suspended(obs, np.array([0]), 10, 3, 1)
        assert mask.tolist() == [[True] * 10]

    def test_trigger_and_lookback_window(self):
        # cols 3-5 missing → run reaches 3 at col 5; resumes col 6; lookback=1
        # → disqualified on the trigger day AND `lookback` trading columns after
        # it: cols 5, 6.
        obs = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1]], dtype=bool)
        mask = _not_long_suspended(obs, np.array([0]), 10, 3, 1)
        assert mask.tolist() == [[1, 1, 1, 1, 1, 0, 0, 1, 1, 1]]

    def test_pre_listing_missing_not_counted(self):
        # Stock first lists at col 4; the F's before listing are not a halt.
        obs = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=bool)
        mask = _not_long_suspended(obs, np.array([4]), 10, 3, 1)
        assert mask.tolist() == [[True] * 10]

    def test_run_below_threshold_never_triggers(self):
        # Max run of 2 < threshold 3 → no disqualification.
        obs = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=bool)
        mask = _not_long_suspended(obs, np.array([0]), 10, 3, 1)
        assert mask.tolist() == [[True] * 10]

    def test_multiple_disjoint_runs_each_trigger(self):
        # threshold=2, lookback=0: cols 1-2 run reaches 2 at col 2; cols 5-6
        # reaches 2 at col 6 → disqualified on cols 2 and 6.
        obs = np.array([[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]], dtype=bool)
        mask = _not_long_suspended(obs, np.array([0]), 10, 2, 0)
        assert mask.tolist() == [[1, 1, 0, 1, 1, 1, 0, 1, 1, 1]]

    def test_never_listed_all_eligible(self):
        obs = np.zeros((1, 10), dtype=bool)
        mask = _not_long_suspended(obs, np.array([-1]), 10, 3, 1)
        assert mask.tolist() == [[True] * 10]

    def test_empty_returns_zero_width(self):
        mask = _not_long_suspended(np.zeros((0, 10), dtype=bool),
                                   np.array([], dtype=np.int32), 10, 3, 1)
        assert mask.shape == (0, 10)


class TestUniverseGateIntegration:
    """§七-3 end-to-end: `universe_eligible_mask` is produced and merged into
    `decision_eligible_mask`, so a long-suspended stock can never be ranked as a
    fresh candidate."""

    def test_suspension_excluded_from_decision_pool(self, monkeypatch):
        def _patched_load_config():
            cfg = _real_load_config()
            # Small thresholds so a synthetic 3-day gap triggers the gate;
            # min_amount=0 disables the turnover floor (synthetic panel has no
            # canonical `amount` column anyway).
            cfg.universe = OmegaConf.create({
                "long_suspension_days": 3,
                "suspension_lookback": 1,
                "min_amount_60d": 0,
            })
            return cfg

        monkeypatch.setattr("stoke_ml.config.load_config", _patched_load_config)
        base = _make_kl(200)
        gap_start, gap_len = 100, 3
        a = base.copy()
        a["stock_code"] = "000001"
        b = base.drop(index=range(gap_start, gap_start + gap_len)).copy()
        b["stock_code"] = "000002"
        panel = pd.concat([a, b], ignore_index=True)

        pipe = FeaturePipeline(seq_len=20, use_sentiment=False,
                               use_announcements=False, use_guba=False,
                               use_comment=False)
        p = pipe.build_panel_features(panel, aux_data={}, horizon=1)
        uni = p["universe_eligible_mask"]   # (2, T)
        dec = p["decision_eligible_mask"]   # (2, T)
        assert uni.shape == dec.shape
        assert uni.shape[0] == 2
        # decision ⊆ universe: nothing ranked tradable outside the gate.
        assert not (dec & ~uni).any()
        # The never-suspended stock is eligible everywhere.
        assert uni[0].all()
        # Stock 000002: exactly ONE suspension trigger → a contiguous
        # disqualified window [trigger, trigger+1] (lookback=1).
        false_cols = np.where(~uni[1])[0]
        assert len(false_cols) == 2
        assert np.all(np.diff(false_cols) == 1)
        # The trigger column is the LAST missing column — the last gap date on
        # the union calendar (calendar cleaning may shift raw indices).
        gap_dates = [pd.Timestamp(d).date() for d in base["date"].iloc[gap_start:gap_start + gap_len]]
        grid_dates = [pd.Timestamp(d).date() for d in p["global_dates"]]
        assert grid_dates[false_cols[0]] == gap_dates[-1]


class TestFoldDeadFeatureColumns:
    """fold_dead_feature_columns must judge constancy only on observed rows of
    the training window, ignore zero-padding, and return axis-2 index lists."""

    def _train_data(self, pk: np.ndarray, po: np.ndarray, obs: np.ndarray) -> dict:
        return {
            "past_known": pk.astype(np.float32),
            "past_observed": po.astype(np.float32),
            "observation_mask": obs.astype(bool),
        }

    def test_drops_constant_columns_and_keeps_varying(self):
        # 3 stocks × 4 days × 4 features, all rows observed, every stock identical.
        pk = np.array([
            [[1, 2, 5, 10], [1, 3, 5, 11], [1, 4, 5, 12], [1, 5, 5, 13]],
            [[1, 2, 5, 10], [1, 3, 5, 11], [1, 4, 5, 12], [1, 5, 5, 13]],
            [[1, 2, 5, 10], [1, 3, 5, 11], [1, 4, 5, 12], [1, 5, 5, 13]],
        ], dtype=np.float32)  # features 0,2 constant; 1,3 vary
        po = np.array([
            [[9, 7, 1, 3], [9, 7, 2, 3], [9, 7, 3, 3], [9, 7, 4, 3]],
            [[9, 7, 1, 3], [9, 7, 2, 3], [9, 7, 3, 3], [9, 7, 4, 3]],
            [[9, 7, 1, 3], [9, 7, 2, 3], [9, 7, 3, 3], [9, 7, 4, 3]],
        ], dtype=np.float32)  # features 0,1,3 constant; 2 varies
        obs = np.ones((3, 4), dtype=bool)
        cols = ["a", "b", "c", "d"]
        pk_idx, po_idx = fold_dead_feature_columns(
            self._train_data(pk, po, obs), cols, cols,
        )
        assert pk_idx == [0, 2]
        assert po_idx == [0, 1, 3]

    def test_zero_padding_is_not_evidence(self):
        # Only stock 0 is listed; stocks 1-2 are all-padding (obs False).
        # Padding must not count: feature 0 varies on the listed stock → kept,
        # feature 2 is constant on the listed stock → dropped, even though its
        # padded rows read 0 elsewhere (a varying feature isn't saved by fake
        # padding values, and a constant one isn't killed by them either).
        pk = np.array([
            [[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 5, 5]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ], dtype=np.float32)
        obs = np.array([
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ])
        cols = ["a", "b", "c"]
        pk_idx, po_idx = fold_dead_feature_columns(
            self._train_data(pk, pk.copy(), obs), cols, cols, ratio=0.9,
        )
        assert pk_idx == [2]
        assert po_idx == [2]

    def test_sparse_keep_prefix_families_are_exempt(self):
        # A rare-event column (market_state_) constant on all stocks except one
        # brief activation: 9/10 stocks time-constant → >= 0.9 would drop it,
        # but the SPARSE_KEEP_PREFIXES exemption keeps it.  A same-shaped plain
        # column without the prefix IS dropped.
        rows = [[1, 0, 0], [2, 0, 0], [1, 0, 0], [2, 0, 0]]
        pk = np.array([rows] * 10, dtype=np.float32)
        # Stock 0 has one activation day (t=2) for features 1 and 2.
        pk[0, 2, 1] = 1.0
        pk[0, 2, 2] = 1.0
        # feature 0 varies in time for every stock; feature 1 fires once for
        # stock 0 (market_state family); feature 2 fires once for stock 0 (plain).
        pk_cols = ["plain_a", "market_state_up", "plain_b"]
        po_cols = pk_cols
        obs = np.ones((10, 4), dtype=bool)
        pk_idx, po_idx = fold_dead_feature_columns(
            self._train_data(pk, pk.copy(), obs), pk_cols, po_cols,
        )
        # 9/10 stocks are time-constant on features 1 and 2 (0.9 threshold) —
        # the prefixed one is exempt, the plain one is dropped.
        assert pk_idx == [2]
        assert po_idx == [2]

    def test_ratio_threshold_governs_drop(self):
        # 4 stocks; feature 0 varies on exactly one stock, constant on 3 → 0.75.
        pk = np.array([
            [[1, 9], [2, 9], [3, 9], [4, 9]],
            [[5, 9], [5, 9], [5, 9], [5, 9]],
            [[5, 9], [5, 9], [5, 9], [5, 9]],
            [[5, 9], [5, 9], [5, 9], [5, 9]],
        ], dtype=np.float32)  # feature 0 constant on 3/4, feature 1 on 4/4
        po = np.ones_like(pk)
        obs = np.ones((4, 4), dtype=bool)
        cols = ["a", "b"]
        lo = fold_dead_feature_columns(self._train_data(pk, po, obs), cols, cols, ratio=0.9)
        hi = fold_dead_feature_columns(self._train_data(pk, po, obs), cols, cols, ratio=0.7)
        # 0.75-constant feature dropped only under the looser ratio; the
        # 1.0-constant feature is dropped under both.
        assert lo == ([1], [0, 1])
        assert hi == ([0, 1], [0, 1])

    def test_all_zero_column_is_dead(self):
        # Feature 1 is all-zero (a ZI-filled channel with no data anywhere) —
        # constant for every observed stock → dropped; features 0, 2 vary.
        pk = np.array([
            [[1, 0, 3], [2, 0, 4], [3, 0, 5], [4, 0, 6]],
            [[1, 0, 3], [2, 0, 4], [3, 0, 5], [4, 0, 6]],
        ], dtype=np.float32)
        po = np.ones_like(pk)
        obs = np.ones((2, 4), dtype=bool)
        cols = ["a", "b", "c"]
        pk_idx, po_idx = fold_dead_feature_columns(self._train_data(pk, po, obs), cols, cols)
        assert pk_idx == [1]
        assert po_idx == [0, 1, 2]


# ---------------------------------------------------------------------------
# §T18: _build_preprocessing narrows the OmegaConf conversion catch
# ---------------------------------------------------------------------------

class TestBuildPreprocessingNarrow:
    def test_omegaconf_failure_falls_back(self, monkeypatch):
        """An OmegaConfBaseException (e.g. a failed interpolation resolve) is
        handled: the config degrades to {} and an empty pipeline is built."""
        from omegaconf import OmegaConf
        from omegaconf.errors import InterpolationResolutionError
        pipe = FeaturePipeline(
            preprocessing_config=OmegaConf.create({"preprocessing": {}}),
            use_new_preprocessing=False,
        )

        def _raise(cfg, **kw):
            raise InterpolationResolutionError("unresolved interpolation")

        monkeypatch.setattr(OmegaConf, "to_container", _raise)
        out = pipe._build_preprocessing()
        assert out is not None  # built from the fallback {} config

    def test_non_omegaconf_error_propagates(self, monkeypatch):
        """§T18: a failure outside (ValueError, TypeError, OmegaConfBaseException)
        must propagate instead of being silently swallowed."""
        from omegaconf import OmegaConf
        pipe = FeaturePipeline(
            preprocessing_config=OmegaConf.create({"preprocessing": {}}),
            use_new_preprocessing=False,
        )

        def _raise(cfg, **kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(OmegaConf, "to_container", _raise)
        with pytest.raises(RuntimeError, match="unexpected"):
            pipe._build_preprocessing()
