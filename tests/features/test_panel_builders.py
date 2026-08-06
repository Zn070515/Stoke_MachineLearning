"""Unit tests for the panel_builders subpackage (§二十一 T17 review).

Focused regression net for the T8 memmap swap: ``PanelArrays`` allocation,
per-stock write via ``TargetBuilder``, ``sanitize``/``assemble`` round-trip,
and a small pure-function check for ``EligibilityBuilder``.  These do NOT
exercise ``build_panel_features`` end-to-end (that is covered elsewhere) —
they pin the builder/container seams directly.
"""

import numpy as np
import pandas as pd
import pytest

from stoke_ml.features.panel_builders._arrays import PanelArrays
from stoke_ml.features.panel_builders._targets import TargetBuilder


def _tiny_panel():
    """Two stocks, 8 trading days, monotone prices -> predictable targets."""
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
    ])
    dfs = []
    for code, base in [("000001", 10.0), ("000002", 20.0)]:
        px = base + np.arange(len(dates)) * 0.1
        dfs.append(pd.DataFrame({
            "date": dates,
            "stock_code": code,
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": np.full(len(dates), 1e6),
            "amount": np.full(len(dates), 1e8),
        }))
    all_dates = sorted({d for df in dfs for d in pd.to_datetime(df["date"])})
    date_to_pos = {str(d.date()): i for i, d in enumerate(all_dates)}
    return dfs, ["000001", "000002"], len(all_dates), date_to_pos


def test_panel_arrays_round_trip():
    """TargetBuilder writes into PanelArrays; sanitize+assemble round-trips."""
    dfs, valid_codes, max_T, date_to_pos = _tiny_panel()
    N = len(dfs)

    arrays = PanelArrays(N, max_T)
    TargetBuilder(horizon=1).compute(dfs, valid_codes, max_T, date_to_pos, arrays)

    # Target arrays have the right shape; every row is a real observation.
    assert arrays.obs.shape == (N, max_T)
    assert arrays.obs.all()
    assert arrays.entry.all()
    assert arrays.y_dir.shape == (N, max_T)
    assert arrays.y_ret.shape == (N, max_T)
    # Strictly increasing prices -> 1-day forward return > dir_threshold -> up
    # (2) for every column except the last (no exit window there -> -100).
    assert (arrays.y_dir[0] == 2).sum() == max_T - 1
    assert arrays.y_dir[0, -1] == -100

    # Feature-grid round trip: write a sentinel into past_known (within the
    # [-10, 10] sanitize clip window so it survives), drop a NaN into static
    # (sanitize must zero it), then confirm both round-trip through assemble.
    arrays.alloc_features(static_dim=1, pk_dim=1, po_dim=1)
    arrays.pk[:, :, 0] = 3.5
    arrays.static[:, :, 0] = np.nan
    arrays.sanitize()

    all_dates = sorted({d for df in dfs for d in pd.to_datetime(df["date"])})
    out = arrays.assemble(
        global_dates=np.array(all_dates, dtype="datetime64[ns]"),
        decision_arr=np.ones((N, max_T), dtype=bool),
        history_arr=np.ones((N, max_T), dtype=bool),
        universe_eligible_arr=np.ones((N, max_T), dtype=bool),
        fill_prob_arr=np.zeros(max_T, dtype=np.float64),
        pk_cols=["pk0"],
        po_cols=["po0"],
        valid_codes=valid_codes,
    )

    expected_keys = {
        "static_features", "past_known", "past_observed",
        "y_direction", "y_return", "y_volatility",
        "date_indices", "global_dates",
        "observation_mask", "entry_eligible_mask",
        "return_target_mask", "vol_target_mask",
        "forward_vol_nobs", "realized_return", "fill_prob",
        "decision_eligible_mask", "history_eligible_mask",
        "universe_eligible_mask",
        "close_price", "open_price",
        "past_known_cols", "past_observed_cols", "stock_codes",
    }
    assert set(out.keys()) == expected_keys
    assert out["past_known"].shape == (N, max_T, 1)
    assert out["past_known"][0, 3, 0] == 3.5  # sentinel survived sanitize
    assert out["static_features"][0, 3, 0] == 0.0  # NaN zeroed by sanitize
    assert out["past_known_cols"] == ["pk0"]
    assert out["stock_codes"] == valid_codes
    assert out["global_dates"].dtype == "datetime64[ns]"
    assert out["date_indices"].shape == (N, max_T)


def test_eligibility_builder_universe_mask(monkeypatch):
    """EligibilityBuilder produces a sane decision/history/universe mask."""
    from stoke_ml.features.panel_builders._eligibility import EligibilityBuilder

    # Deterministic universe config — don't depend on the repo's config.yaml.
    monkeypatch.setattr(
        "stoke_ml.config.load_config",
        lambda: {"universe": {
            "long_suspension_days": 60,
            "suspension_lookback": 60,
            "min_amount_60d": 5_000_000,
        }},
    )

    N, T = 2, 5
    obs = np.ones((N, T), dtype=bool)
    first_col = np.zeros(N, dtype=np.int32)  # listed from day 0
    amt60 = np.full((N, T), 1e8, dtype=np.float32)  # very liquid
    has_amount = np.ones(N, dtype=bool)

    decision, history, universe = EligibilityBuilder(
        seq_len=2, min_history=1,
    ).compute(obs, first_col, amt60, has_amount)

    assert decision.shape == (N, T)
    assert history.shape == (N, T)
    assert universe.shape == (N, T)
    # decision[0] is False (no close[t-1] yet), decision[1:] True.
    assert not decision[:, 0].any()
    assert decision[:, 1:].all()
    # Universe eligible on columns 1+: listed from day 0, never long-suspended,
    # amt60 well above the liquidity floor.  Column 0 is legitimately False —
    # the causal 60d-turnover shift leaves no turnover known at close[-1] for
    # the entry at day 0, so the liquidity floor fails there.
    assert not universe[:, 0].any()
    assert universe[:, 1:].all()
