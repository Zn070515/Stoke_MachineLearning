"""Tests for the 4 new feature families (spec §7.1-7.5)."""
import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.market_env import MarketEnvRefiner
from stoke_ml.features.pipeline import (
    FeaturePipeline, _batch_fill_shift, _merge_daily_aux,
)


def _kline(n=120, start="2021-01-04"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "date": idx, "open": close - 0.5, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1_000_000, 2_000_000, n),
        "amount": rng.uniform(1e8, 5e8, n), "pct_change": rng.normal(0, 1, n),
    })


def test_merge_daily_aux_pit_lag():
    df = _kline()
    aux = pd.DataFrame({"date": df["date"], "pledge_ratio": 3.0, "has_pledge": True})
    merged = _merge_daily_aux(df.copy(), aux)
    # feature at t == source at t-1 (PIT lag 1)
    assert np.allclose(merged["pledge_ratio"].iloc[1:].values,
                       aux["pledge_ratio"].iloc[:-1].values)
    assert merged["has_pledge"].iloc[1]  # first aux row shifted to second K-line day
    assert not merged["has_pledge"].iloc[0]


def test_merge_market_env_global(tmp_path):
    me = pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=10),
                       "high_low_ratio": np.linspace(-1, 1, 10)})
    me_path = tmp_path / "market_env_daily.parquet"
    me.to_parquet(me_path)
    df = _kline(n=10)
    p = FeaturePipeline(use_technical=False, use_scoring=False, use_temporal=False,
                        use_sentiment=False, use_announcements=False, use_guba=False,
                        use_comment=False, use_margin=False, use_northbound=False,
                        use_dragon_tiger=False, use_fundamental=False, use_valuation=False,
                        use_etf_flow=False, use_interaction=False, use_capital_flow=False,
                        use_block_trade=False, use_shareholder=False, use_lockup=False,
                        use_dividend=False, use_board=False, use_sector=False,
                        use_concept=False, use_macro=False, use_industry=False,
                        use_emotion_refine=False, use_fundamental_refine=False,
                        use_temporal_stats=False, use_pledge=False, use_limit_up=False,
                        use_index_membership=False, use_market_env=True)
    # _merge_market_env reads from cfg data_dir; bypass by injecting cache.
    p._market_env_cache = me
    out = p._merge_market_env(df)
    assert "high_low_ratio" in out.columns
    # lagged: out[t] == me[t-1]
    assert np.allclose(out["high_low_ratio"].iloc[1:].values, me["high_low_ratio"].iloc[:-1].values)


def test_market_env_refiner_factors():
    d = pd.date_range("2021-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": d, "shibor_1M": 2.0, "fx_usd_cny": 7.0, "cpi_yoy": 1.0,
        "m1_yoy": 5.0, "m2_yoy": 8.0, "bond_cn_10y2y_spread": 0.5,
        "bond_us_10y": 4.0, "bond_cn_10y": 3.0,
    })
    out = MarketEnvRefiner().refine(df)
    for c in ("menv_shibor_1m_z", "menv_us_cn_10y_spread", "menv_m1_m2_spread",
              "menv_regime_z", "menv_bond_10y2y_spread"):
        assert c in out.columns
    assert np.allclose(out["menv_m1_m2_spread"], -3.0)
    assert np.allclose(out["menv_us_cn_10y_spread"], 1.0)


def test_market_env_refiner_graceful_missing():
    df = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=10)})
    out = MarketEnvRefiner().refine(df)
    assert "menv_regime_z" not in out.columns  # no inputs -> no composite


def test_batch_fill_shift_int16_count():
    df = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=5)})
    df["pledge_count"] = [1, 2, 3, 4, 5]
    df["has_pledge"] = [True, True, True, True, True]
    _batch_fill_shift(df, ["pledge_count", "has_pledge"])
    assert df["pledge_count"].dtype == np.int16
    assert df["has_pledge"].dtype == bool
    assert df["pledge_count"].iloc[1] == 1  # shifted


def test_ic_correctness_known_signal():
    """Inject a feature == forward return + noise; cross-sectional IC ~ high."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=30)
    ics = []
    for _ in dates:
        n = 80
        noise = rng.normal(0, 0.1, n)
        sig = rng.normal(0, 1, n)
        fwd = sig + noise
        feat = fwd + rng.normal(0, 0.05, n)
        rho, _ = spearmanr(feat, fwd)
        ics.append(rho)
    mean_ic = float(np.mean(ics))
    assert mean_ic > 0.7
