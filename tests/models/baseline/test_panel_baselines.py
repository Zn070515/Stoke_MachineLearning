"""Panel baselines tests.

The baseline benchmark must satisfy the same PIT / alignment contracts as the
xLSTM panel model:

- ``build_flat_samples`` extracts the feature cross-section at column ``e-1``
  (the decision day before entry at ``open[e]``) and the target at column ``e``,
  filtered by the label + eligibility gates at ``e``.
- ``build_momentum_grid`` scores an entry day with ONLY data before it.
- A precomputed (N, n_windows) grid is handed to ``evaluate_portfolio`` via
  ``precomputed_preds`` and consumed directly — no sequential flat-grid adapter,
  which would misalign on a sparse eval_mask (suspended stocks leave a window
  with fewer than N stocks).
- ``FittedScoreAdapter`` exposes a ``reset()`` so the benchmark loop can treat
  every adapter uniformly, and returns the (dir, ret, vol) triple the evaluator
  consumes.
"""
import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from stoke_ml.models.baseline.panel_baselines import (
    FittedScoreAdapter,
    _stratified_quotas,
    build_flat_samples,
    build_momentum_grid,
    entry_column_features,
    sequence_summary_dim,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import evaluate_portfolio


class _LinearModel:
    """Minimal sklearn-like surface: predict(X) = X @ w."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=np.float32)

    def predict(self, X):
        return X @ self.w


def _make_deterministic_panel(n_stocks=3, n_timesteps=10):
    """PIT panel with a verifiable feature contract.

    static/pk/po column ``t`` are deterministic scalars so an extracted sample
    at column ``e-1`` is exactly checkable; y_return at ``t`` is ``t + 100``.
    """
    n = n_stocks
    T = n_timesteps
    static = np.zeros((n, T, 1), dtype=np.float32)
    pk = np.zeros((n, T, 1), dtype=np.float32)
    po = np.zeros((n, T, 1), dtype=np.float32)
    for t in range(T):
        static[:, t, 0] = t + 1.0
        pk[:, t, 0] = t + 1000.0
        po[:, t, 0] = t + 100000.0
    y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        y[:, t] = t + 100.0
    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_return": y,
        "return_target_mask": np.ones((n, T), dtype=bool),
        "decision_eligible_mask": np.ones((n, T), dtype=bool),
        "close_price": np.ones((n, T), dtype=np.float32),
    }


class TestBuildFlatSamples:
    def test_pit_feature_at_e_minus_1_target_at_e(self):
        """For entry column e, X is the cross-section at e-1 and y is the label
        at e — the exact PIT contract the xLSTM also sees."""
        data = _make_deterministic_panel()
        X, y = build_flat_samples(data, entry_start=5, entry_end=9)
        # e in {5,6,7,8} × 3 stocks, all rows within a column identical.
        assert X.shape == (12, 3)
        assert y.shape == (12,)
        # e=5 (first column processed): static=5, pk=1004, po=100004, y=105.
        np.testing.assert_allclose(X[0], [5.0, 1004.0, 100004.0], atol=1e-6)
        assert y[0] == pytest.approx(105.0)
        # e=8 (last): static=8, pk=1007, po=100007, y=108.
        np.testing.assert_allclose(X[-1], [8.0, 1007.0, 100007.0], atol=1e-6)
        assert y[-1] == pytest.approx(108.0)

    def test_label_gate_filters_rows_at_entry_column(self):
        data = _make_deterministic_panel()
        # Stock 1 is not a valid label at e=6 → its row at that column drops.
        data["return_target_mask"][1, 6] = False
        X, y = build_flat_samples(data, entry_start=5, entry_end=9)
        # e=5: 3 rows; e=6: 2 rows; e=7: 3; e=8: 3 → 11 total.
        assert X.shape == (11, 3)
        # The e=6 block now has exactly two rows ([6,1005,100005], y=106).
        mid = [r.tolist() for r in X]
        e6_rows = [row for row, yy in zip(mid, y) if yy == pytest.approx(106.0)]
        assert len(e6_rows) == 2, f"expected 2 surviving rows at e=6, got {e6_rows}"

    def test_extra_gate_filters_at_entry_column(self):
        data = _make_deterministic_panel()
        # decision-ineligible at e=7 → that column drops the masked stock.
        data["decision_eligible_mask"][2, 7] = False
        X, y = build_flat_samples(data, entry_start=5, entry_end=9)
        assert X.shape == (11, 3)
        e7_rows = [yy for yy in y if yy == pytest.approx(107.0)]
        assert len(e7_rows) == 2, f"expected 2 rows at e=7, got {len(e7_rows)}"

    def test_all_zero_feature_rows_dropped(self):
        data = _make_deterministic_panel()
        # Zero the feature cross-section for stock 2 at fcol=4 → its e=5 row is
        # all-padding and must not enter training.
        data["static_features"][2, 4, :] = 0.0
        data["past_known"][2, 4, :] = 0.0
        data["past_observed"][2, 4, :] = 0.0
        X, y = build_flat_samples(data, entry_start=5, entry_end=9)
        # e=5 loses stock 2 → 2 rows, plus 3+3+3 → 11.
        assert X.shape == (11, 3)
        e5_rows = [yy for yy in y if yy == pytest.approx(105.0)]
        assert len(e5_rows) == 2, f"zero-padding row survived at e=5: {e5_rows}"

    def test_max_rows_caps_column_collection(self):
        data = _make_deterministic_panel()
        X, y = build_flat_samples(
            data, entry_start=5, entry_end=9, max_rows=5)
        # The cap is allocated date-stratified (largest
        # remainder over counts [3,3,3,3], budget 5 → quotas [2,1,1,1]) so ALL
        # entry columns stay represented — the old greedy collection stopped
        # after 2 columns (6 rows) and only ever saw the earliest dates.
        assert len(X) == 5, f"expected exactly max_rows rows, got {len(X)}"
        assert len(set(y.tolist())) == 4, (
            f"expected all 4 entry columns represented, got {sorted(set(y.tolist()))}")

    def test_stratified_quotas_proportional_largest_remainder(self):
        q = _stratified_quotas(np.array([3, 3, 3, 3]), 5)
        np.testing.assert_array_equal(q, [2, 1, 1, 1])
        q2 = _stratified_quotas(np.array([10, 90]), 20)
        np.testing.assert_array_equal(q2, [2, 18])
        # Zero-count columns must never receive budget.
        q3 = _stratified_quotas(np.array([0, 5]), 3)
        assert q3[0] == 0 and q3[1] == 3

    def test_seq_features_train_eval_consistency(self):
        """The seq summary a baseline sees at FIT time must equal what the
        eval-time FittedScoreAdapter builds for the same decision day — so
        train/eval vectors match by construction."""
        data = _make_deterministic_panel()
        seq_len = 5
        e = 7  # entry column; decision day = e-1 = 6
        X, y = build_flat_samples(
            data, entry_start=e, entry_end=e + 1,
            seq_features=True, seq_len=seq_len)
        assert X.shape == (3, 3 + sequence_summary_dim(1, 1))
        fcol = e - 1
        static_2d = torch.from_numpy(data["static_features"][:, fcol])
        pk_win = torch.from_numpy(data["past_known"][:, e - seq_len:e])
        po_win = torch.from_numpy(data["past_observed"][:, e - seq_len:e])
        Xe = entry_column_features(static_2d, pk_win, po_win, with_seq=True)
        np.testing.assert_allclose(X, Xe.numpy(), atol=1e-6)

    def test_empty_when_no_eligible_columns(self):
        data = _make_deterministic_panel()
        data["return_target_mask"][:, 5:] = False
        X, y = build_flat_samples(data, entry_start=5, entry_end=9)
        assert X.shape == (0, 3)
        assert y.shape == (0,)


class TestBuildMomentumGrid:
    def test_trailing_mean_daily_return_pit(self):
        close = np.array([[10.0, 10.0, 11.0, 12.0, 12.0, 11.0]],
                         dtype=np.float32)
        grid = build_momentum_grid(
            {"close_price": close}, entry_start=2, entry_end=5, lookback=2)
        # daily: d0=0, d1=0, d2=0.1, d3=1/11, d4=0, d5=-1/12
        assert grid.shape == (1, 3)
        expected = [
            0.0,                                  # e=2: daily[0:2] = (0+0)/2
            (0.0 + 0.1) / 2.0,                    # e=3: daily[1:3]
            (0.1 + 1.0 / 11.0) / 2.0,             # e=4: daily[2:4]
        ]
        np.testing.assert_allclose(grid[0], expected, atol=1e-6)
        # Only data BEFORE entry is used: e=3 uses daily[1:3], never d3..d5.
        assert grid[0, 1] == pytest.approx(0.05)

    def test_gaps_forward_filled_to_zero_return(self):
        close = np.array([[10.0, np.nan, 11.0, 12.0]], dtype=np.float32)
        grid = build_momentum_grid(
            {"close_price": close}, entry_start=2, entry_end=4, lookback=2)
        # filled = [10,10,11,12] → daily = [0, 0, 0.1, 1/11]
        np.testing.assert_allclose(
            grid[0], [0.0, (0.0 + 0.1) / 2.0], atol=1e-6)

    def test_leading_gap_yields_zero(self):
        close = np.array([[np.nan, 10.0, 11.0]], dtype=np.float32)
        grid = build_momentum_grid(
            {"close_price": close}, entry_start=2, entry_end=3, lookback=2)
        # filled = [0,10,11] (pre-first-valid zeroed) → daily[1]=0 (denom 0)
        np.testing.assert_allclose(grid[0], [0.0], atol=1e-6)


class TestEntryColumnFeatures:
    def test_3d_static_takes_last_column(self):
        static = torch.randn(4, 5, 2)
        pk = torch.randn(4, 5, 3)
        po = torch.randn(4, 5, 1)
        X = entry_column_features(static, pk, po)
        assert X.shape == (4, 6)
        torch.testing.assert_close(
            X, torch.cat([static[:, -1], pk[:, -1], po[:, -1]], dim=1))

    def test_2d_legacy_static_used_as_is(self):
        static = torch.randn(4, 2)
        pk = torch.randn(4, 5, 3)
        po = torch.randn(4, 5, 1)
        X = entry_column_features(static, pk, po)
        assert X.shape == (4, 6)
        torch.testing.assert_close(
            X, torch.cat([static, pk[:, -1], po[:, -1]], dim=1))


class TestFittedScoreAdapter:
    def test_returns_model_prediction_triple(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        adapter = FittedScoreAdapter(_LinearModel(weights))
        static = torch.randn(4, 5, 2)
        pk = torch.randn(4, 5, 3)
        po = torch.randn(4, 5, 1)
        d, ret, v = adapter.forward(static, pk, po)
        assert d.shape == (4, 1) and ret.shape == (4, 1) and v.shape == (4, 1)
        X = torch.cat([static[:, -1], pk[:, -1], po[:, -1]], dim=1).numpy()
        np.testing.assert_allclose(ret.numpy().reshape(-1), X @ weights,
                                   atol=1e-5)
        # dir/vol heads are unused placeholders the evaluator ignores.
        assert np.all(d.numpy() == 0.0) and np.all(v.numpy() == 0.0)

    def test_reset_is_noop(self):
        adapter = FittedScoreAdapter(_LinearModel([1.0]))
        assert adapter.reset() is adapter  # stateless — API parity only


def _make_sparse_marker_panel(n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
                              up_stocks=(1,), down_stocks=(2,), seed=0,
                              suspend_stock=1, suspend_len=80,
                              suspend_start=None):
    """Priced marker panel with a mid-panel suspension (sparse eval_mask).

    Same construction as ``_make_3d_marker_panel`` but NaN's out one stock's
    open/close over a contiguous mid-panel range.  That flips its
    decision/entry masks False inside the range (close[t-1] / open[t] no longer
    real), so the evaluator's candidate pool loses that stock for a stretch of
    windows — exactly the sparse eval_mask that a sequential flat-grid adapter
    cannot replay without drifting alignment.  The strongest-signal stock
    (default ``suspend_stock=1``, marker=2.0) is the one suspended, so a
    scrambled grid collapses the IC.
    """
    rng = np.random.RandomState(seed)
    drift = np.zeros(n_stocks)
    marker = np.zeros(n_stocks)
    for i in up_stocks:
        drift[i] = 0.008
        marker[i] = 2.0
    for i in down_stocks:
        drift[i] = -0.008
        marker[i] = -2.0
    rets = rng.randn(n_stocks, n_timesteps) * 0.01 + drift[:, None]
    close = 10.0 * np.exp(np.cumsum(rets, axis=1))
    open_ = close * (1.0 + 0.001 * rng.randn(n_stocks, n_timesteps))

    if suspend_start is None:
        # Inside the eval window grid, not touching the price sleeve pad.
        suspend_start = seq_len + 30
    if suspend_len and suspend_len > 0:
        close[suspend_stock, suspend_start:suspend_start + suspend_len] = np.nan
        open_[suspend_stock, suspend_start:suspend_start + suspend_len] = np.nan

    ret_fwd = np.full((n_stocks, n_timesteps), np.nan, dtype=np.float32)
    ret_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    if n_timesteps > horizon:
        both = np.isfinite(open_[:, :-horizon]) & np.isfinite(open_[:, horizon:])
        ret_fwd[:, :n_timesteps - horizon][both] = (
            open_[:, horizon:][both] / open_[:, :-horizon][both] - 1.0)
        ret_tgt[:, :n_timesteps - horizon] = both
    y_ret = np.nan_to_num(ret_fwd, nan=0.0).astype(np.float32)
    y_dir = np.full((n_stocks, n_timesteps), -100, dtype=np.int64)
    y_dir[ret_tgt] = np.where(ret_fwd[ret_tgt] > 0, 2,
                              np.where(ret_fwd[ret_tgt] < 0, 0, 1))

    obs = np.isfinite(close)
    entry = np.isfinite(open_)
    decision = np.zeros((n_stocks, n_timesteps), dtype=bool)
    decision[:, 1:] = obs[:, :-1]
    history = np.ones((n_stocks, n_timesteps), dtype=bool)

    static = np.zeros((n_stocks, n_timesteps, 1), dtype=np.float32)
    static[:, :, 0] = marker[:, None]

    return {
        "static_features": static,
        "past_known": rng.randn(n_stocks, n_timesteps, 12).astype(np.float32),
        "past_observed": rng.randn(n_stocks, n_timesteps, 10).astype(np.float32),
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": np.abs(y_ret).astype(np.float32),
        "date_indices": np.tile(np.arange(n_timesteps), (n_stocks, 1)),
        "observation_mask": obs,
        "entry_eligible_mask": entry,
        "return_target_mask": ret_tgt,
        "vol_target_mask": ret_tgt,
        "close_price": close.astype(np.float32),
        "open_price": open_.astype(np.float32),
        "decision_eligible_mask": decision,
        "history_eligible_mask": history,
    }


def _make_3d_marker_panel(n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
                          up_stocks=(1,), down_stocks=(2,), seed=0):
    """Priced synthetic panel with PIT 3D (N,T,1) static carrying a per-stock
    marker (mirrors test_evaluate._make_priced_panel but with a time axis so
    build_flat_samples can read the marker at any decision column e-1)."""
    rng = np.random.RandomState(seed)
    drift = np.zeros(n_stocks)
    marker = np.zeros(n_stocks)
    for i in up_stocks:
        drift[i] = 0.008
        marker[i] = 2.0
    for i in down_stocks:
        drift[i] = -0.008
        marker[i] = -2.0
    rets = rng.randn(n_stocks, n_timesteps) * 0.01 + drift[:, None]
    close = 10.0 * np.exp(np.cumsum(rets, axis=1))
    open_ = close * (1.0 + 0.001 * rng.randn(n_stocks, n_timesteps))

    ret_fwd = np.full((n_stocks, n_timesteps), np.nan, dtype=np.float32)
    ret_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    if n_timesteps > horizon:
        both = np.isfinite(open_[:, :-horizon]) & np.isfinite(open_[:, horizon:])
        ret_fwd[:, :n_timesteps - horizon][both] = (
            open_[:, horizon:][both] / open_[:, :-horizon][both] - 1.0)
        ret_tgt[:, :n_timesteps - horizon] = both
    y_ret = np.nan_to_num(ret_fwd, nan=0.0).astype(np.float32)
    y_dir = np.full((n_stocks, n_timesteps), -100, dtype=np.int64)
    y_dir[ret_tgt] = np.where(ret_fwd[ret_tgt] > 0, 2,
                              np.where(ret_fwd[ret_tgt] < 0, 0, 1))

    obs = np.isfinite(close)
    entry = np.isfinite(open_)
    decision = np.zeros((n_stocks, n_timesteps), dtype=bool)
    decision[:, 1:] = obs[:, :-1]
    history = np.ones((n_stocks, n_timesteps), dtype=bool)

    static = np.zeros((n_stocks, n_timesteps, 1), dtype=np.float32)
    static[:, :, 0] = marker[:, None]

    return {
        "static_features": static,
        "past_known": rng.randn(n_stocks, n_timesteps, 12).astype(np.float32),
        "past_observed": rng.randn(n_stocks, n_timesteps, 10).astype(np.float32),
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": np.abs(y_ret).astype(np.float32),
        "date_indices": np.tile(np.arange(n_timesteps), (n_stocks, 1)),
        "observation_mask": obs,
        "entry_eligible_mask": entry,
        "return_target_mask": ret_tgt,
        "vol_target_mask": ret_tgt,
        "close_price": close.astype(np.float32),
        "open_price": open_.astype(np.float32),
        "decision_eligible_mask": decision,
        "history_eligible_mask": history,
    }


class TestBaselineThroughEvaluator:
    """End-to-end: a baseline scored by the REAL evaluate_portfolio must rank
    correctly.  A precomputed grid is consumed via ``precomputed_preds`` (no
    sequential adapter), and FittedScoreAdapter must satisfy the PIT feature
    contract — a misalignment scrambles the scores and collapses the IC to ~0."""

    SEQ_LEN, HORIZON = 60, 5

    def test_precomputed_grid_aligns_in_evaluator_order(self):
        panel = _make_3d_marker_panel()
        n_windows = 320 - self.SEQ_LEN
        grid = np.tile(panel["static_features"][:, -1, :], (1, n_windows))
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON,
                          min_stocks_per_day=5)  # 12-stock panel
        m = evaluate_portfolio(None, panel, cfg, torch.device("cpu"),
                               horizon=self.HORIZON, precomputed_preds=grid)
        assert m["ic_mean"] > 0.3, (
            f"precomputed grid misaligned to evaluator order: IC={m['ic_mean']:.3f}")

    def test_precomputed_grid_survives_sparse_eval_mask(self):
        # Suspension-style sparse pool: the strongest-signal stock (marker=2.0)
        # is NaN-priced for ~1/3 of the window grid, so the candidate pool has
        # fewer than N stocks there.  A sequential flat-grid replay drifts on
        # the first sparse window and reads every later window's scores off the
        # wrong stocks (empirically IC 0.53 → 0.084).  precomputed_preds skips
        # replay entirely: the (N, W) grid is consumed directly, so the marker
        # signal must survive the sparse pool.
        panel = _make_sparse_marker_panel(suspend_stock=1, suspend_len=80)
        n_windows = 320 - self.SEQ_LEN
        grid = np.tile(panel["static_features"][:, -1, :], (1, n_windows))
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON,
                          min_stocks_per_day=5, min_history=1)  # 12-stock panel
        m = evaluate_portfolio(None, panel, cfg, torch.device("cpu"),
                               horizon=self.HORIZON, precomputed_preds=grid)
        assert m["ic_mean"] > 0.3, (
            f"sparse eval_mask scrambled precomputed grid scores: "
            f"IC={m['ic_mean']:.3f}")

    def test_fitted_baseline_ranks_via_clean_ic(self):
        panel = _make_3d_marker_panel()
        Xtr, ytr = build_flat_samples(
            panel, entry_start=self.SEQ_LEN, entry_end=self.SEQ_LEN + 60)
        assert Xtr.shape[1] == 1 + 12 + 10
        model = Ridge(alpha=1.0)
        model.fit(Xtr, ytr)
        adapter = FittedScoreAdapter(model)
        adapter.reset()
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON,
                          min_stocks_per_day=5)  # 12-stock panel
        m = evaluate_portfolio(adapter, panel, cfg, torch.device("cpu"),
                               horizon=self.HORIZON)
        assert m["ic_mean"] > 0.1, (
            f"fitted baseline lost the marker signal: IC={m['ic_mean']:.3f}")
