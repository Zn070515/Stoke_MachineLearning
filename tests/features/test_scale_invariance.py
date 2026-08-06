"""OHLC-scale-invariance for price-derived technical features (§五 P0).

Forward-adjusted (qfq) historical prices re-anchor with each future corporate
action, so an absolute qfq level leaks future corporate behaviour into
historical cross-sections.  Every price-DERIVED feature must therefore be a
scale-invariant ratio: scaling a stock's OHLC by any positive constant k must
leave the feature grid unchanged.

This test drives TechnicalIndicators and TrendScorer on a synthetic A-share
series, rescales OHLC by k ∈ {0.5, 2, 10}, and asserts every price-derived
column is unchanged within float tolerance, with identical NaN placement.
"""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.technical import TechnicalIndicators
from stoke_ml.features.scoring import TrendScorer

_SCALES = [0.5, 2.0, 10.0]

# Price-derived features that must be invariant under OHLC × k.  Non-price
# series (volume / amount, which qfq adjustment does not touch) must be
# unchanged as well — asserted separately for amount_ma5.
_FIXED = [
    "ma_5", "ma_10", "ma_20", "ma_60", "ma_120",
    "ema_12", "ema_26",
    "macd_dif", "macd_dea", "macd_hist",
    "boll_mid", "boll_upper", "boll_lower", "boll_pct",
    "atr_14",
    "kmid", "klen", "kmid2", "kup", "kup2", "klow", "klow2", "ksft", "ksft2",
    "open0", "high0", "low0",
    "adx", "adxr", "pdi", "mdi",
    "mfi_14", "cmo_14", "trix",
    "volume_ratio", "obv", "vol_up_ratio_20",
    "amount_ratio",
]
_FIXED += [f"rsi_{p}" for p in (6, 12, 24)]
_FIXED += [f"kdj_{s}_{p}" for p in (9, 14) for s in ("k", "d", "j")]
_FIXED += [f"roc_{p}" for p in (6, 12, 20)]
_FIXED += [f"wr_{p}" for p in (10, 20)]
_FIXED += [f"cci_{p}" for p in (14, 20)]
_FIXED += [f"vol_{p}" for p in (5, 20)]
for d in (5, 10, 20, 30, 60):
    _FIXED += [f"{pre}{d}d" for pre in (
        "max_", "min_", "qtlu_", "qtld_", "rank_", "rsv_",
        "corr_", "cord_",
        "beta_", "rsqr_", "resi_", "vma_", "vstd_",
        "cntp_", "cntn_", "cntd_", "sump_", "sumn_", "sumd_",
        "imax_", "imin_", "imxd_", "wvma_", "vsump_", "vsumn_", "vsumd_",
    )]


def _make_ohlcv(n_days=250):
    """Plausible A-share OHLCV+amount series with an upward-trending price."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_days))
    close = np.maximum(close, 1.0)
    open_ = close * (1 + rng.normal(0, 0.008, n_days))
    high = np.maximum(close, open_) * (1 + rng.uniform(0.001, 0.02, n_days))
    low = np.minimum(close, open_) * (1 - rng.uniform(0.001, 0.02, n_days))
    volume = rng.integers(1_000_000, 20_000_000, n_days).astype(float)
    amount = volume * close
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume, "amount": amount,
    })


def _scale_ohlc(df, k):
    scl = df.copy()
    for col in ("open", "high", "low", "close"):
        scl[col] = scl[col] * k
    return scl


def _assert_same(a: np.ndarray, b: np.ndarray, col: str, k: float):
    """Values match within tolerance AND NaN placement is identical."""
    assert np.array_equal(np.isnan(a), np.isnan(b)), (
        f"{col}: NaN placement changed under OHLC×{k}"
    )
    assert np.allclose(a, b, equal_nan=True, atol=1e-8), (
        f"feature {col} changed under OHLC×{k}"
    )


class TestPriceDerivedFeaturesScaleInvariance:

    @pytest.mark.parametrize("k", _SCALES)
    def test_scale_invariant_features_unchanged(self, k):
        df = _make_ohlcv()
        orig = TechnicalIndicators().compute_all(df)
        scl = TechnicalIndicators().compute_all(_scale_ohlc(df, k))

        for col in _FIXED:
            assert col in orig.columns, f"missing {col} in technical output"
            _assert_same(orig[col].to_numpy(dtype=float),
                         scl[col].to_numpy(dtype=float), col, k)

    @pytest.mark.parametrize("k", _SCALES)
    def test_amount_series_unchanged(self, k):
        """amount is real CNY (not re-anchored by qfq) → must not change."""
        df = _make_ohlcv()
        orig = TechnicalIndicators().compute_all(df)
        scl = TechnicalIndicators().compute_all(_scale_ohlc(df, k))

        for col in ("amount", "amount_ma5", "amount_ratio"):
            _assert_same(orig[col].to_numpy(dtype=float),
                         scl[col].to_numpy(dtype=float), col, k)

    @pytest.mark.parametrize("k", _SCALES)
    def test_trend_scorer_scale_invariant(self, k):
        """trend_level / bias_ma{5,10,20,60} must be unchanged under OHLC×k."""
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        scorer = TrendScorer()
        ro = scorer.score(ti.compute_all(df))
        rs = scorer.score(ti.compute_all(_scale_ohlc(df, k)))

        for col in ("trend_level", "buy_signal",
                    "bias_ma5", "bias_ma10", "bias_ma20", "bias_ma60",
                    "volume_shrink", "volume_heavy"):
            _assert_same(ro[col].to_numpy(dtype=float),
                         rs[col].to_numpy(dtype=float), col, k)

    def test_turnover_proxy_removed(self):
        """§五 P0: amount/qfq_close is not scale-invariant → column dropped."""
        df = _make_ohlcv()
        result = TechnicalIndicators().compute_all(df)
        assert "turnover_proxy" not in result.columns
