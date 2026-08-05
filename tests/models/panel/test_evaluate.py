import io

import numpy as np
import pytest
import torch
import torch.nn as nn

from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.evaluate import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_daily_return_profit_factor,
    compute_equity_curve,
    compute_bootstrap_sharpe_ci,
    compute_ic_summary,
    evaluate_portfolio,
    _candidate_pool,
    _combine_book_daily,
    _ls_exposure_ledger,
    _run_sleeve_sim,
    _simulate_sleeve_account,
)


def _ledger_pnl(ledger):
    """Total per-position P&L of a ledger: unresolved rows carry
    `unrealized_pnl`, all others `net_pnl` (§十三-3)."""
    return sum(r["unrealized_pnl"] if r["exit_status"] == "unresolved"
               else r["net_pnl"] for r in ledger)


class TestSharpe:
    def test_positive_returns(self):
        daily_returns = torch.tensor([0.01, 0.02, 0.015, 0.005, 0.01])
        sharpe = compute_sharpe(daily_returns)
        assert sharpe > 0

    def test_zero_returns(self):
        daily_returns = torch.zeros(20)
        sharpe = compute_sharpe(daily_returns)
        assert sharpe == 0.0

    def test_negative_returns(self):
        daily_returns = torch.tensor([-0.01, -0.02, -0.005, -0.015])
        sharpe = compute_sharpe(daily_returns)
        assert sharpe < 0

    def test_annualization(self):
        """Sharpe with 252-day annualization should be in reasonable range."""
        daily_returns = torch.randn(252) * 0.01 + 0.0005
        sharpe = compute_sharpe(daily_returns)
        assert -5.0 < sharpe < 5.0


class TestSortino:
    def test_only_downside(self):
        """Sortino > Sharpe when returns have upside skew — only penalizes downside."""
        ret = torch.tensor([0.02, 0.03, -0.01, 0.01, -0.005])
        sortino = compute_sortino(ret)
        sharpe = compute_sharpe(ret)
        assert sortino > sharpe  # upside fluctuation not penalized

    def test_all_positive(self):
        ret = torch.tensor([0.01, 0.02, 0.015, 0.01])
        sortino = compute_sortino(ret, annualize=False)
        # No downside (< 2 downside samples) → Sortino is undefined → NaN
        # so summaries/JSON don't show an unbounded inf.
        assert np.isnan(sortino)


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        equity = torch.tensor([1.0, 1.1, 0.9, 0.95, 1.05])
        mdd = compute_max_drawdown(equity)
        # Peak=1.1, trough=0.9 → drawdown=(1.1-0.9)/1.1 ≈ 0.182
        assert 0.18 < mdd < 0.19

    def test_no_drawdown(self):
        equity = torch.tensor([1.0, 1.1, 1.2, 1.3])
        mdd = compute_max_drawdown(equity)
        assert mdd == 0.0


class TestCalmar:
    def test_positive_returns(self):
        ret = torch.tensor([0.01, 0.02, 0.005, -0.005, 0.015])
        calmar = compute_calmar(ret, horizon=1)
        assert calmar > 0


class TestDailyReturnProfitFactor:
    def test_profitable(self):
        ret = torch.tensor([0.02, -0.01, 0.03, -0.005, 0.01])
        pf = compute_daily_return_profit_factor(ret)
        # profits = 0.02+0.03+0.01=0.06, losses = 0.01+0.005=0.015 → PF=4.0
        assert pf > 1.0

    def test_losing(self):
        ret = torch.tensor([-0.02, 0.01, -0.03, -0.01])
        pf = compute_daily_return_profit_factor(ret)
        # profits = 0.01, losses = 0.06 → PF ≈ 0.167
        assert pf < 1.0

    def test_all_positive(self):
        ret = torch.tensor([0.01, 0.02, 0.005])
        pf = compute_daily_return_profit_factor(ret)
        assert pf == float("inf")


class TestEquityCurve:
    def test_starts_at_one(self):
        ret = torch.tensor([0.01, -0.02, 0.03])
        eq = compute_equity_curve(ret)
        assert eq[0].item() == 1.0
        assert len(eq) == 4  # 3 returns + initial

    def test_cumulative(self):
        ret = torch.tensor([0.1, -0.1])
        eq = compute_equity_curve(ret)
        # 1.0 → 1.1 → 0.99
        assert abs(eq[-1].item() - 0.99) < 1e-6


class TestBootstrapSharpeCI:
    """IID resampling of autocorrelated returns understates the
    CI.  The bootstrap must preserve within-block time structure."""

    def test_finite_ci_for_iid_returns(self):
        rng = np.random.RandomState(0)
        rets = rng.randn(120) * 0.01 + 0.0005
        lo, hi = compute_bootstrap_sharpe_ci(rets, horizon=1)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi

    def test_block_bootstrap_wider_than_iid_on_autocorrelated(self):
        # AR(1) with strong persistence: single-return resampling destroys the
        # autocorrelation and reports a too-narrow CI; block resampling must
        # widen it (block_len=1 recovers the old iid behaviour).
        rng = np.random.RandomState(7)
        n = 200
        rets = np.zeros(n)
        for t in range(1, n):
            rets[t] = 0.6 * rets[t - 1] + rng.randn() * 0.01 + 0.0003
        lo_b, hi_b = compute_bootstrap_sharpe_ci(rets, horizon=1, n_boot=600)
        lo_i, hi_i = compute_bootstrap_sharpe_ci(
            rets, horizon=1, n_boot=600, block_len=1)
        assert (hi_b - lo_b) > (hi_i - lo_i), (
            f"block CI {hi_b - lo_b:.4f} should exceed iid CI {hi_i - lo_i:.4f}"
        )

    def test_horizon_floored_block_not_narrower(self):
        # §十五-3: with horizon>1 the adjacent daily returns share overlapping
        # holdings.  A block floored at the horizon must NOT give a narrower CI
        # than a tiny block on an autocorrelated series (the floor preserves the
        # overlap structure the sleeve actually has).
        rng = np.random.RandomState(3)
        n = 240
        rets = np.zeros(n)
        for t in range(1, n):
            rets[t] = 0.7 * rets[t - 1] + rng.randn() * 0.01 + 0.0002
        lo_f, hi_f = compute_bootstrap_sharpe_ci(
            rets, horizon=1, n_boot=600, block_len=20)
        lo_s, hi_s = compute_bootstrap_sharpe_ci(
            rets, horizon=1, n_boot=600, block_len=2)
        assert (hi_f - lo_f) >= (hi_s - lo_s), (
            f"horizon-floored CI {hi_f - lo_f:.4f} must not be narrower "
            f"than small-block CI {hi_s - lo_s:.4f}"
        )


class TestICSummaryNeweyWest:
    """The plain mean/std IR ignores serial correlation in the
    per-day IC series.  ic_newey_west_t must be smaller than the iid t-stat
    when the ICs are positively autocorrelated."""

    def test_empty_returns_zero(self):
        s = compute_ic_summary([])
        assert s["ic_newey_west_t"] == 0.0
        assert s["ic_ir"] == 0.0

    def test_constant_predictions_no_daily_ics(self):
        # Constant predictions → every day's Spearman is degenerate → empty
        # IC series → no claimed signal.
        s = compute_ic_summary([])
        assert s["ic_newey_west_t"] == 0.0

    def test_nw_t_shrinks_under_positive_autocorrelation(self):
        rng = np.random.RandomState(1)
        n = 300
        ics = np.zeros(n)
        for t in range(1, n):
            ics[t] = 0.9 * ics[t - 1] + rng.randn() * 0.02 + 0.0005
        s = compute_ic_summary(ics.tolist(), horizon=1)
        naive_t = s["ic_mean"] / (s["ic_std"] / np.sqrt(n))
        assert s["ic_newey_west_t"] > 0
        assert s["ic_newey_west_t"] < naive_t, (
            f"NW t {s['ic_newey_west_t']:.2f} must be < iid t {naive_t:.2f}"
        )

    def test_nw_t_positive_for_iid_signal(self):
        rng = np.random.RandomState(0)
        ics = (rng.randn(200) * 0.02 + 0.005).tolist()
        s = compute_ic_summary(ics, horizon=1)
        assert np.isfinite(s["ic_newey_west_t"])
        assert s["ic_newey_west_t"] > 0


def _make_synthetic_panel(n_stocks=24, n_timesteps=320, seed=0):
    """Noise-only synthetic panel: returns are i.i.d. Gaussians, independent of
    every feature.  A model cannot learn signal from it — exactly the state a
    model reaches after label-shuffle training."""
    rng = np.random.RandomState(seed)
    static = rng.randn(n_stocks, 6).astype(np.float32)
    pk = rng.randn(n_stocks, n_timesteps, 12).astype(np.float32)
    po = rng.randn(n_stocks, n_timesteps, 10).astype(np.float32)
    y_dir = np.zeros((n_stocks, n_timesteps), dtype=np.int64)
    y_ret = (rng.randn(n_stocks, n_timesteps) * 0.02).astype(np.float32)
    y_vol = np.abs(rng.randn(n_stocks, n_timesteps) * 0.01).astype(np.float32)
    date_indices = np.tile(np.arange(n_timesteps), (n_stocks, 1))
    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": y_vol,
        "date_indices": date_indices,
    }


class _ConstantReturnModel(nn.Module):
    """#4: every prediction identical → the evaluation pipeline must
    not manufacture alpha (zero IC, no systematic long-short / quintile spread)."""

    def __init__(self, ret_val: float = 0.0):
        super().__init__()
        self.ret_val = ret_val

    def forward(self, static, pk, po):
        b = pk.shape[0]
        return (
            torch.zeros(b, 3, dtype=torch.float32),          # direction: flat
            torch.full((b, 1), self.ret_val, dtype=torch.float32),  # return
            torch.full((b, 1), 0.02, dtype=torch.float32),   # vol: typical
        )


class _RandomReturnModel(nn.Module):
    """#3 proxy: predictions independent of the inputs, standing in
    for a model that learned nothing from globally-shuffled labels.  If the
    evaluation pipeline is clean, this must show IC≈0 and no alpha.

    seed draws from a per-instance generator so results are reproducible — the
    global torch RNG state varies with whatever tests ran before, which made
    the |LS|<1.0 anti-cheat bound trip intermittently."""

    def __init__(self, seed=None):
        super().__init__()
        self._gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def forward(self, static, pk, po):
        b = pk.shape[0]
        return (
            torch.zeros(b, 3, dtype=torch.float32),
            torch.randn(b, 1, dtype=torch.float32, generator=self._gen) * 0.01,
            torch.full((b, 1), 0.02, dtype=torch.float32),
        )


class _StaticMarkerModel(nn.Module):
    """Return prediction = the static marker column, so each stock gets a
    DISTINCT, per-window-identical prediction.  Lets a test steer which stock
    the top-K / bottom-K selection picks, to verify the candidate pool honors
    entry eligibility rather than future-label validity."""

    def forward(self, static, pk, po):
        b = pk.shape[0]
        return (
            torch.zeros(b, 3, dtype=torch.float32),
            static[:, :1].float(),
            torch.full((b, 1), 0.02, dtype=torch.float32),
        )


class TestEvaluateCandidatePool:
    """Survivor bias: the candidate pool must be ENTRY-ELIGIBLE
    stocks (real open at the entry day), NOT stocks that happen to have a
    future label.  With carry-realized returns every entry-eligible stock has
    a tradeable outcome, so selection never conditions on survival."""

    SEQ_LEN = 60
    HORIZON = 1
    N_TIMESTEPS = 200

    def _pool_data(self, with_mask: bool):
        n_stocks, n_timesteps, seq_len = 3, self.N_TIMESTEPS, self.SEQ_LEN
        rng = np.random.RandomState(0)
        # static[0] = per-stock prediction marker: stock 0 low, stock 1 mid,
        # stock 2 high — but stock 2 is NOT entry-eligible.
        static = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
        pk = rng.randn(n_stocks, n_timesteps, 12).astype(np.float32)
        po = rng.randn(n_stocks, n_timesteps, 10).astype(np.float32)
        # Every stock has a VALID future label (y_dir never -100) — the old
        # survivor pool (y_dir != -100) would wrongly include stock 2.
        y_dir = np.full((n_stocks, n_timesteps), 2, dtype=np.int64)
        y_ret = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
        y_vol = np.full((n_stocks, n_timesteps), 0.02, dtype=np.float32)

        t = np.arange(n_timesteps)
        realized = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
        # stock 0/1: small, oscillating, always positive spread (top>bottom);
        # stock 2: strongly NEGATIVE outcome but the model predicts it highest —
        # if it leaked into the pool, long-short would flip sign.
        realized[0, seq_len:] = (-0.001 - 0.0005 * np.sin(t[seq_len:] / 3)).astype(np.float32)
        realized[1, seq_len:] = (0.002 + 0.0005 * np.sin(t[seq_len:] / 3)).astype(np.float32)
        realized[2, seq_len:] = (-0.10 + 0.0005 * np.cos(t[seq_len:] / 3)).astype(np.float32)

        data = {
            "static_features": static,
            "past_known": pk,
            "past_observed": po,
            "y_direction": y_dir,
            "y_return": y_ret,
            "y_volatility": y_vol,
            "realized_return": realized,
        }
        if with_mask:
            entry = np.ones((n_stocks, n_timesteps), dtype=bool)
            entry[2] = False  # stock 2: no real open at entry → not eligible
            data["entry_eligible_mask"] = entry
        return data

    def _eval(self, data):
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        return evaluate_portfolio(
            _StaticMarkerModel(), data, cfg, torch.device("cpu"),
            horizon=self.HORIZON,
        )

    def test_entry_eligible_pool_excludes_future_label_stock(self):
        """With the mask present, stock 2 (valid label but no real open) is
        excluded → long-short is driven only by eligible stocks 0/1 → POSITIVE
        spread.  The old future-label pool would include stock 2 (predicted
        top) and flip long-short negative — the survivor bias."""
        m = self._eval(self._pool_data(with_mask=True))
        assert m["n_periods"] > 100
        assert m["long_sharpe"] > 0, f"ineligible stock leaked in: {m['long_sharpe']}"
        assert m["ls_sharpe"] > 0, f"ineligible stock leaked in: {m['ls_sharpe']}"

    def test_without_mask_survivor_bias_flips_spread(self):
        """Control: with NO entry mask, the fallback pool is y_dir != -100 which
        includes stock 2 → its big negative outcome in the top-K LONG flips the
        spread negative.  This proves the entry-mask (not something else) is
        what suppresses the survivor bias above."""
        m = self._eval(self._pool_data(with_mask=False))
        assert m["n_periods"] > 100
        assert m["ls_sharpe"] < 0, f"fallback pool did not show survivor bias: {m['ls_sharpe']}"


class TestEvaluateRealizedPreference:
    """evaluate_portfolio must use val_data['realized_return'] (carry-last-close,
    defined for every entry-eligible day) and IGNORE the raw_returns argument
    when both are present — otherwise evaluation would silently fall back to a
    label-conditioned return."""

    def test_realized_return_preferred_over_raw_returns(self):
        n_stocks, n_timesteps, seq_len = 2, 200, 60
        rng = np.random.RandomState(1)
        static = rng.randn(n_stocks, 1).astype(np.float32)
        pk = rng.randn(n_stocks, n_timesteps, 12).astype(np.float32)
        po = rng.randn(n_stocks, n_timesteps, 10).astype(np.float32)
        y_dir = np.zeros((n_stocks, n_timesteps), dtype=np.int64)
        y_ret = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
        y_vol = np.full((n_stocks, n_timesteps), 0.02, dtype=np.float32)
        t = np.arange(n_timesteps)
        realized = np.tile(
            (0.02 + 0.01 * np.sin(t / 3)).astype(np.float32), (n_stocks, 1))
        raw = np.tile(
            (-0.02 + 0.01 * np.cos(t / 3)).astype(np.float32), (n_stocks, 1))
        data = {
            "static_features": static,
            "past_known": pk,
            "past_observed": po,
            "y_direction": y_dir,
            "y_return": y_ret,
            "y_volatility": y_vol,
            "entry_eligible_mask": np.ones((n_stocks, n_timesteps), dtype=bool),
            "realized_return": realized,
        }
        cfg = PanelConfig(seq_len=seq_len, horizon=1)
        m = evaluate_portfolio(
            _ConstantReturnModel(), data, cfg, torch.device("cpu"),
            horizon=1, raw_returns=raw,
        )
        # EW baseline is the cross-sectional mean of the EVALUATED actuals.
        # realized has positive mean (+0.02) → positive EW sharpe; raw has
        # negative mean → if wrongly preferred, EW sharpe would be negative.
        assert m["eligible_ew_sharpe"] > 0, (
            f"raw_returns leaked into eval: {m['eligible_ew_sharpe']}")


class TestEvaluateAllHorizonPhases:
    """Every entry phase (offset 0..horizon-1) is evaluated and
    concatenated, so no in-sample entry days are wasted.  The old single-phase
    loop discarded (horizon-1)/horizon of the data."""

    def test_all_entry_phases_used(self):
        n_stocks, n_timesteps, seq_len, horizon = 12, 260, 60, 5
        n_windows = n_timesteps - seq_len
        data = _make_synthetic_panel(n_stocks=n_stocks, n_timesteps=n_timesteps)
        data["entry_eligible_mask"] = np.ones((n_stocks, n_timesteps), dtype=bool)
        # Materialize a writable copy (broadcast_to yields a read-only view
        # that trips torch.as_tensor's writability warning).
        data["realized_return"] = np.tile(
            (0.001 * np.sin(np.arange(n_timesteps) / 7)).astype(np.float32),
            (n_stocks, 1))
        cfg = PanelConfig(seq_len=seq_len, horizon=horizon)
        m = evaluate_portfolio(
            _ConstantReturnModel(), data, cfg, torch.device("cpu"), horizon=horizon)
        # ~1 period per window day, NOT n_windows/horizon
        assert m["n_periods"] >= n_windows - 2, (
            f"only {m['n_periods']}/{n_windows} periods — entry phases wasted"
        )


class TestEvaluateAntiCheat:
    """Anti-cheat tests #3 (label shuffle) & #4 (constant preds).

    Both assert the evaluation pipeline produces no alpha when there is no
    signal.  They are fast and training-free: instead of actually running
    label-shuffled training (a ~75 min CPU job that on synthetic data cannot
    even learn the *un*-shuffled signal — see probe), a model whose outputs are
    noise / constant plays the role of the null hypothesis.
    """

    SEQ_LEN = 60
    HORIZON = 5

    @staticmethod
    def _raw_returns(data):
        n_stocks = data["y_return"].shape[0]
        n_windows = data["y_return"].shape[1] - TestEvaluateAntiCheat.SEQ_LEN
        raw = np.zeros((n_stocks, TestEvaluateAntiCheat.SEQ_LEN + n_windows),
                       dtype=np.float32)
        raw[:, TestEvaluateAntiCheat.SEQ_LEN:] = data["y_return"][:, :n_windows]
        return raw

    def _eval(self, model, seed):
        data = _make_synthetic_panel(seed=seed)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        return evaluate_portfolio(
            model, data, cfg, torch.device("cpu"),
            horizon=self.HORIZON, raw_returns=self._raw_returns(data),
        )

    def test_constant_predictions_yield_zero_ic(self):
        """#4: constant predictions → spearman is degenerate → IC must be 0."""
        m = self._eval(_ConstantReturnModel(), seed=0)
        assert m["ic_mean"] == 0.0
        assert m["ic_ir"] == 0.0

    def test_constant_predictions_produce_no_systematic_spread(self):
        """#4: averaged over seeds, constant predictions show no long-short
        or quintile alpha (selection degenerates to stock order = random).

        NOTE on seed count: the annualized Sharpe of a zero-true-signal book
        is an estimate with std ≈ sqrt(252/n_periods) ≈ 1.0 per seed (here the
        book is a single top/bottom stock on 12 names → std ~0.9).  3 seeds
        give a mean std of ~0.5, so `< 1.0` is only ~2σ — flaky.  10 seeds cut
        it to ~0.28 → ~3.5σ, a genuine no-alpha assertion."""
        lss, q5s = [], []
        for seed in range(10):
            m = self._eval(_ConstantReturnModel(), seed=seed)
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert abs(np.mean(lss)) < 1.0, f"constant preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"constant preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"

    def test_random_predictions_yield_near_zero_ic(self):
        """#3: label-shuffle proxy — noise predictions must show |IC|≈0."""
        ics = [self._eval(_RandomReturnModel(seed=s), s)["ic_mean"] for s in range(10)]
        assert abs(np.mean(ics)) < 0.03, f"random preds mean IC={np.mean(ics):+.4f}"

    def test_random_predictions_produce_no_systematic_alpha(self):
        """#3: noise predictions must not generate portfolio alpha (10 seeds —
        see the power note on test_constant_predictions_produce_no_systematic_spread)."""
        lss, q5s = [], []
        for seed in range(10):
            m = self._eval(_RandomReturnModel(seed=seed), seed=seed)
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert abs(np.mean(lss)) < 1.0, f"random preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"random preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"


def _make_priced_panel(
    n_stocks=12,
    n_timesteps=320,
    seq_len=60,
    horizon=5,
    seed=0,
    up_stocks=(),
    down_stocks=(),
    no_decision_stocks=(),
    no_open_stocks=(),
    with_decision=True,
):
    """Synthetic panel with REAL price paths for the sleeve-account evaluation
    path.  close/open are random walks with a per-stock daily
    drift so forward returns are rankable.  static[:, 0] holds a per-stock
    marker that _StaticMarkerModel uses as the return prediction:

      up_stocks           drift +0.8%/day, marker +2.0
      down_stocks         drift -0.8%/day, marker -2.0
      no_decision_stocks  drift -0.8%/day, marker +3.0 — decision_eligible
                          forced False in the eval window (a LEAKER a broken
                          pool that ranked on future labels would pick)
      no_open_stocks      drift 0, marker +4.0, open missing on the first 5
                          eval days (selected but unfilled → cash, no backfill)

    with_decision=False drops decision/history masks so _candidate_pool falls
    back to entry-eligibility — the control where the leaker leaks in.
    """
    rng = np.random.RandomState(seed)
    drift = np.zeros(n_stocks)
    for i in up_stocks:
        drift[i] = 0.008
    for i in down_stocks:
        drift[i] = -0.008
    for i in no_decision_stocks:
        drift[i] = -0.008
    rets = rng.randn(n_stocks, n_timesteps) * 0.01 + drift[:, None]
    close = 10.0 * np.exp(np.cumsum(rets, axis=1))
    open_ = close * (1.0 + 0.001 * rng.randn(n_stocks, n_timesteps))
    for i in no_open_stocks:
        open_[i, seq_len:seq_len + 5] = np.nan  # unfillable entry days

    # Clean forward open->open return (training label), as in the pipeline.
    ret_fwd = np.full((n_stocks, n_timesteps), np.nan, dtype=np.float32)
    ret_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    if n_timesteps > horizon:
        both = np.isfinite(open_[:, :-horizon]) & np.isfinite(open_[:, horizon:])
        ret_fwd[:, :n_timesteps - horizon][both] = (
            open_[:, horizon:][both] / open_[:, :-horizon][both] - 1.0)
        ret_tgt[:, :n_timesteps - horizon] = both
    y_ret = np.nan_to_num(ret_fwd, nan=0.0).astype(np.float32)
    y_dir = np.full((n_stocks, n_timesteps), -100, dtype=np.int64)
    y_dir[ret_tgt] = np.where(
        ret_fwd[ret_tgt] > 0, 2, np.where(ret_fwd[ret_tgt] < 0, 0, 1))

    obs = np.isfinite(close)
    entry = np.isfinite(open_)
    decision = np.zeros((n_stocks, n_timesteps), dtype=bool)
    decision[:, 1:] = obs[:, :-1]
    for i in no_decision_stocks:
        decision[i, seq_len:] = False
    history = np.ones((n_stocks, n_timesteps), dtype=bool)

    static = np.zeros((n_stocks, 6), dtype=np.float32)
    for i in up_stocks:
        static[i, 0] = 2.0
    for i in down_stocks:
        static[i, 0] = -2.0
    for i in no_decision_stocks:
        static[i, 0] = 3.0
    for i in no_open_stocks:
        static[i, 0] = 4.0

    data = {
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
    }
    if with_decision:
        data["decision_eligible_mask"] = decision
        data["history_eligible_mask"] = history
    return data


class TestSleeveTrueDailySeries:
    """The priced evaluation returns a TRUE
    daily return series over the whole price path (n_periods == W+horizon,
    day 0 included), NOT the phase-concatenated W/h — the sleeve account marks
    to close every calendar day."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def _eval(self, **kw):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), **kw)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        return evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)

    def test_sleeve_series_is_true_daily(self):
        m = self._eval()
        n_windows = self.N_TS - self.SEQ_LEN
        # The series is TRUE daily over the available price path (each sleeve
        # marks to close every calendar day incl. day 0's open->close), NOT
        # the phase-concatenated W/h.  The synthetic panel holds
        # exactly n_windows price columns after the seq_len slice (no trailing
        # horizon pad), so the sleeve series spans n_windows periods here.
        assert m["n_periods"] == n_windows, (
            f"n_periods={m['n_periods']} != {n_windows} — sleeve "
            f"series must be TRUE daily over the whole available price path")
        assert m["n_periods"] > self.HORIZON * 40, (
            f"series too short: {m['n_periods']}")


class TestSleeveDecisionHistoryPool:
    """The selection pool is DECISION & HISTORY eligible
    (close[t-1] real + window covered).  A stock whose signal yesterday was
    missing (decision-ineligible) must NOT be selectable even though its
    prediction is the strongest — otherwise selection would leak."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def _pool_panel(self, with_decision):
        return _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON,
            up_stocks=(1,), down_stocks=(2,), no_decision_stocks=(3,),
            with_decision=with_decision, seed=1)

    def test_decision_pool_excludes_leaker(self):
        panel = self._pool_panel(with_decision=True)
        n_windows = self.N_TS - self.SEQ_LEN
        pool = _candidate_pool(panel, n_windows, self.SEQ_LEN).numpy()
        assert not pool[3].any(), (
            "decision-ineligible leaker (marker +3, -0.8%/day drift) leaked "
            "into the selection pool")
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)
        assert m["long_sharpe"] > 0.5, (
            f"leaker leaked into top-K long: long_sharpe={m['long_sharpe']:.3f}")

    def test_without_decision_pool_leaker_flips_sign(self):
        """Control: drop decision/history masks → fallback pool includes the
        leaker → its strongest prediction + negative outcome flips the long
        book negative.  Proves the mask (not something else) suppresses it."""
        panel = self._pool_panel(with_decision=False)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)
        assert m["long_sharpe"] < 0, (
            f"fallback pool did not admit the leaker: "
            f"long_sharpe={m['long_sharpe']:.3f}")


class TestSleeveNoBackfill:
    """A selected-but-unfillable stock (no real open at entry)
    keeps its weight CASH — the allocation is NOT redistributed to fillable
    stocks (递补).  An unfilled sleeve contributes zero P&L even if another
    stock moves big that day."""

    def test_unfilled_sleeve_stays_cash(self):
        preds = np.array([[0.9, 0.9, 0.9],
                          [0.1, 0.1, 0.1]], dtype=np.float32)  # top-1 long = stock 0
        close = np.array([[10.0, 11.0, 12.0],
                          [20.0, 26.0, 27.0]], dtype=np.float32)  # stock 1 +30%
        open_ = np.array([[np.nan, 10.5, 11.5],   # stock 0 unfillable on day 0
                          [20.0, 25.5, 26.5]], dtype=np.float32)
        pool = np.ones((2, 3), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=1, top_fraction=0.5,
            cost=0.0, mode="long")
        counts = res["exit_stats"]["counts"]
        assert counts["unfilled"] == 1, f"unfilled={counts['unfilled']}"
        # Day 0 is now part of the series: the day-0 sleeve is
        # unfilled, so its return is exactly 0 — the slot was NOT handed to
        # stock 1, whose +30% move never shows up.  The day-1 return is only
        # stock 0's open->close (10.5 -> 11).
        assert np.isclose(res["daily"][0], 0.0, atol=1e-6), (
            f"unfilled day-0 sleeve must return 0: daily[0]={res['daily'][0]:.4f}")
        assert np.isclose(res["daily"][1], 11.0 / 10.5 - 1.0, atol=1e-4), (
            f"backfilled stock 1 into the day-1 sleeve: "
            f"daily[1]={res['daily'][1]:.4f}")


class TestSleeveCosts:
    """Transaction costs drag net below gross, and the sleeve
    turns over real notional every day."""

    def test_gross_above_net_and_turnover_positive(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), seed=0)
        cfg = PanelConfig(seq_len=60, horizon=5)  # txn_cost = 5 bps/side
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5)
        assert m["long_turnover"] > 0.0, "sleeve account should turn over notional"
        assert m["long_gross_sharpe"] > m["long_sharpe"], (
            f"costs must drag net below gross: "
            f"gross={m['long_gross_sharpe']:.3f} net={m['long_sharpe']:.3f}")
        assert m["ls_gross_sharpe"] > m["ls_sharpe"], (
            f"ls costs: gross={m['ls_gross_sharpe']:.3f} net={m['ls_sharpe']:.3f}")


class TestSleeveExitStatus:
    """Exits are classified
    clean/delayed/delisted/unresolved/unfilled with P&L shares; a selected
    stock with a missing entry open is counted unfilled, never silently traded.
    The old "carry" is a true DELAYED exit; delist (close<=0) is split
    from unresolved (price path ends with the position still open)."""

    def test_exit_status_keys_and_unfilled(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), no_open_stocks=(4,), seed=2)
        cfg = PanelConfig(seq_len=60, horizon=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5)
        es = m["exit_status"]
        assert set(es["counts"].keys()) == {
            "clean", "delayed", "delisted", "unresolved", "unfilled"}
        assert set(es["pnl_share"].keys()) == {
            "clean", "delayed", "delisted", "unresolved", "unfilled"}
        assert es["counts"]["clean"] > 0, "fully-trading stocks should exit clean"
        assert es["counts"]["unfilled"] > 0, (
            "top-marker no-open stock is selected but unfilled → must be counted")
        assert np.isclose(sum(es["pnl_share"].values()), 1.0, atol=1e-6)

    def test_three_leg_exit_status(self):
        """The priced report must expose per-leg exit status
        for long / short / benchmark (equal-weight), so the short book's
        unfillable/trailing positions can be audited independently of the long
        leg.  Each leg carries the full stable schema."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), no_open_stocks=(4,), seed=3)
        cfg = PanelConfig(seq_len=60, horizon=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5)
        keys = {"clean", "delayed", "delisted", "unresolved", "unfilled"}
        for leg in ("long_exit_status", "short_exit_status",
                    "eligible_ew_exit_status"):
            assert leg in m, f"missing per-leg exit status {leg}"
            es = m[leg]
            assert set(es["counts"].keys()) == keys
            assert set(es["pnl"].keys()) == keys
            assert set(es["pnl_share"].keys()) == keys
            assert set(es["abs_pnl_share"].keys()) == keys
            assert set(es["capital_days"].keys()) == keys
            assert "avg_delayed_days" in es
        # backward-compat alias still points at the long leg.
        assert m["exit_status"] == m["long_exit_status"]

    def test_require_price_path_fails_without_prices(self):
        """Formal training must fail loudly when price paths
        are missing, instead of silently downgrading to the legacy estimator."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5, seed=4)
        del panel["close_price"]
        del panel["open_price"]
        cfg = PanelConfig(seq_len=60, horizon=5)
        with pytest.raises(ValueError, match="require_price_path"):
            evaluate_portfolio(
                _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
                horizon=5, require_price_path=True)


class TestSleeveDelayedDelist:
    """A missing exit open is a TRUE delayed exit — the position
    keeps holding and retries on later days, NOT a same-day "carry" sale at the
    stale close.  A zero close at exit is a delist; missing data with no resume
    is unresolved, never a delist."""

    def test_delayed_on_missing_exit_open(self):
        # W=4 signal days, horizon=2, price path padded to Wp=6 (= W+horizon) so
        # every sleeve reaches its scheduled exit and none is left unresolved.
        preds = np.array([[0.9, 0.9, 0.9, 0.9],
                          [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                          [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]], dtype=np.float32)
        open_ = np.array([[10.5, 11.5, np.nan, 12.5, 13.5, 14.5],
                          [20.5, 21.5, 22.5, 23.5, 24.5, 25.5]], dtype=np.float32)
        pool = np.ones((2, 4), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=2, top_fraction=0.5,
            cost=0.0, mode="long")
        counts = res["exit_stats"]["counts"]
        # Stock 0's scheduled exit at d=2 has no open → retried and sold at the
        # d=3 open → DELAYED.  Stock 1 exits on schedule → CLEAN.
        assert counts["delayed"] >= 1, f"expected a delayed exit, got {counts}"
        assert counts["clean"] >= 1, f"expected a clean exit, got {counts}"
        assert counts["unresolved"] == 0, f"no position should dangle: {counts}"

    def test_zero_close_is_unresolved_not_delist(self):
        """A carried close <= 0 after ffill cannot
        distinguish a real delisting from a dead data row, so the position is
        NEVER force-sold at 0 — it stays held and books as UNRESOLVED at the
        path end."""
        preds = np.array([[0.9, 0.9, 0.9],
                          [0.1, 0.1, 0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, 0.0],   # stock 0 price data dies at d=2
                          [20.0, 21.0, 22.0]], dtype=np.float32)
        open_ = np.array([[10.5, 11.5, np.nan],
                          [20.5, 21.5, 22.5]], dtype=np.float32)
        pool = np.ones((2, 3), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=2, top_fraction=0.5,
            cost=0.0, mode="long")
        counts = res["exit_stats"]["counts"]
        assert counts["delisted"] == 0, (
            f"price<=0 must NOT auto-classify as a delist: {counts}")
        assert counts["unresolved"] >= 1, (
            f"expected DATA-MISSING positions booked unresolved: {counts}")

    def test_delayed_exit_holds_and_captures_resumed_open(self):
        """Scheduled exit day has no open, trading resumes 2 days
        later.  The position is NOT liquidated on the scheduled day, keeps being
        marked to close (funds occupied), and is actually sold at the resumed
        open → DELAYED."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)  # top = stock 0
        close = np.array([[10.0, 11.0, 12.0, 13.0, 14.0],
                          [20.0, 21.0, 22.0, 23.0, 24.0]], dtype=np.float32)
        # stock 0: open missing on the scheduled exit day (d=1) and d=2, resumes d=3
        open_ = np.array([[10.5, np.nan, np.nan, 13.5, 14.5],
                          [20.5, 21.5, 22.5, 23.5, 24.5]], dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=1, top_fraction=0.5,
            cost=0.0, mode="long")
        counts = res["exit_stats"]["counts"]
        assert counts["delayed"] == 1, f"expected exactly one delayed exit, got {counts}"
        assert counts["clean"] == 0 and counts["unresolved"] == 0
        daily = np.asarray(res["daily"])
        # Day 0's open->close (10.5 -> 10.0) is now part of the
        # series, so the entry-day return is a loss.
        assert daily[0] < 0, f"day-0 open->close (10.5→10.0) should lose: {daily}"
        # Held through the suspension (d=1, d=2): NAV tracked close 11 then 12
        # — funds stayed occupied instead of being dumped at the stale close on
        # the scheduled day.
        assert daily[1] > 0 and daily[2] > 0, (
            f"position was liquidated on the scheduled day: {daily}")
        # The delayed liquidation at the resumed open 13.5 captured the gap
        # 12 → 13.5.
        assert np.isclose(daily[3], 13.5 / 12.0 - 1.0, atol=1e-4), (
            f"delayed exit did not capture the resumed open: {daily[3]:.4f}")

    def test_missing_data_is_unresolved_not_delisted(self):
        """A stock whose price path just ENDS (NaN, no resume)
        is UNRESOLVED at the end — never auto-interpreted as a same-price sale
        and never a delist.  Only a carried close <= 0 is a delist."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, np.nan],
                          [20.0, 21.0, 22.0]], dtype=np.float32)
        open_ = np.array([[10.5, np.nan, np.nan],
                          [20.5, 21.5, 22.5]], dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=1, top_fraction=0.5,
            cost=0.0, mode="long")
        counts = res["exit_stats"]["counts"]
        assert counts["unresolved"] == 1, f"expected unresolved, got {counts}"
        assert counts["delisted"] == 0, f"missing data must not be a delist: {counts}"
        assert counts["clean"] == 0


class TestSleeveDelistForceSell:
    """§七-1: a stock with a KNOWN delisting record (delist_day >= 0) is
    FORCE-sold at its last carried close on the delisting day — even before
    the sleeve's scheduled exit — and classified DELISTED.  A stock without
    such a record (delist_day == -1) keeps the old unresolved semantics."""

    def test_known_delist_force_sold_before_scheduled_exit(self):
        """Stock 0 is delisted at sim column 1 (delist_day=1), before its
        scheduled exit at d=2.  It is force-sold at the carried close 11.0 and
        booked DELISTED with actual_exit_day=1."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, 12.0, 13.0],
                          [20.0, 21.0, 22.0, 23.0]], dtype=np.float32)
        open_ = np.array([[10.5, 11.5, 12.5, 13.5],
                          [20.5, 21.5, 22.5, 23.5]], dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        delist_day = np.array([1, -1])
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=2, top_fraction=0.5,
            cost=0.0, mode="long", return_ledger=True, delist_day=delist_day)
        counts = res["exit_stats"]["counts"]
        assert counts["delisted"] == 1, f"expected exactly one delisted, got {counts}"
        assert counts["clean"] == 0, f"delist must not be clean: {counts}"
        assert counts["delayed"] == 0, f"delist must not be delayed: {counts}"
        assert counts["unresolved"] == 0, f"delist must not dangle unresolved: {counts}"
        ledger = res["ledger"]
        assert len(ledger) == 1
        rec = ledger[0]
        assert rec["exit_status"] == "delisted"
        assert rec["actual_exit_day"] == 1, f"force-sell day wrong: {rec}"
        assert rec["exit_price"] == 11.0, f"delisting close wrong: {rec}"

    def test_known_delist_without_data_dies_never_delists(self):
        """A stock whose data just ENDS (like before) but has NO delist record
        (delist_day == -1) stays UNRESOLVED — the §七-1 force-sell is gated on
        the universe record, not on the price path ending."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, np.nan],
                          [20.0, 21.0, 22.0]], dtype=np.float32)
        open_ = np.array([[10.5, np.nan, np.nan],
                          [20.5, 21.5, 22.5]], dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=1, top_fraction=0.5,
            cost=0.0, mode="long", delist_day=np.array([-1, -1]))
        counts = res["exit_stats"]["counts"]
        assert counts["unresolved"] == 1, f"expected unresolved, got {counts}"
        assert counts["delisted"] == 0, f"no delist record → no force-sell: {counts}"

    def test_delist_day_outside_window_never_fires(self):
        """A delist day pointing past the price path (>= Wp) never fires within
        the sim: the position exits at its scheduled open like normal."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)
        close = np.array([[10.0, 11.0, 12.0],
                          [20.0, 21.0, 22.0]], dtype=np.float32)
        open_ = np.array([[10.5, 11.5, 12.5],
                          [20.5, 21.5, 22.5]], dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=2, top_fraction=0.5,
            cost=0.0, mode="long", delist_day=np.array([99, -1]))
        counts = res["exit_stats"]["counts"]
        assert counts["delisted"] == 0, f"out-of-window delist must not fire: {counts}"
        assert counts["clean"] == 1, f"scheduled exit should be clean: {counts}"
        assert counts["unresolved"] == 0

    def test_delist_day_wrong_shape_raises(self):
        preds = np.ones((2, 1), dtype=np.float32)
        close = np.ones((2, 3), dtype=np.float32)
        open_ = np.ones((2, 3), dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        with pytest.raises(ValueError):
            _run_sleeve_sim(preds, close, open_, pool, horizon=1,
                            top_fraction=0.5, cost=0.0, mode="long",
                            delist_day=np.array([0, 0, 0]))


class TestLsAlgebra:
    """`short_d` is ALREADY the short account's real
    daily return (side=-1 applied inside the simulator), so the long-short book
    is a SUM of the legs, NOT `long - short`.  When both legs profit the LS
    book must profit too — the old formula cancelled simultaneous success to
    ~0."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def test_both_legs_profit_does_not_cancel(self):
        """End-to-end: the up stock (top marker) makes the long leg money and
        the down stock (bottom marker) makes the short leg money → long_sharpe
        AND ls_sharpe must both be clearly positive.  `long - short` would
        cancel them to ≈0."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), seed=8)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)
        assert m["long_sharpe"] > 0.3, f"long leg should profit: {m['long_sharpe']}"
        assert m["ls_sharpe"] > 0.3, (
            f"both legs profit → LS must be positive, got {m['ls_sharpe']}")
        # The exposure metadata keeps the leverage assumption explicit
        # `target` is the nominal 100% gross / 50-50 / net-0
        # book construction, while `realized` reports the daily MEASURED
        # exposure from the sleeve simulation (unfilled slots → below target,
        # delayed exits → above), so the two are never conflated.
        tgt = m["exposure"]["target"]
        assert tgt["gross_exposure"] == 1.0
        assert tgt["net_exposure"] == 0.0
        assert tgt["long_exposure"] == 0.5
        assert tgt["short_exposure"] == 0.5
        r = m["exposure"]["realized"]
        assert 0.0 < r["mean_gross_exposure"] <= 1.0, (
            f"realized gross must be a measured fraction of NAV: "
            f"{r['mean_gross_exposure']:.3f}")
        daily = r["daily"]
        assert len(daily["gross_exposure"]) == len(daily["net_exposure"]) > 0
        assert len(daily["active_long_sleeves"]) == len(daily["active_short_sleeves"])

    def test_short_leg_rises_when_bottom_falls(self):
        """§十七.2: the short-only NAV must RISE when the bottom stock falls —
        `short_d` carries the + sign into the LS book, it is not re-subtracted."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, down_stocks=(2,), seed=9)
        n_windows = self.N_TS - self.SEQ_LEN
        p0 = self.SEQ_LEN
        close = panel["close_price"][:, p0:p0 + n_windows + self.HORIZON]
        open_ = panel["open_price"][:, p0:p0 + n_windows + self.HORIZON]
        pool = _candidate_pool(panel, n_windows, self.SEQ_LEN).numpy()
        preds = np.zeros((12, n_windows), dtype=np.float32)
        preds[2] = -2.0  # bottom marker → short picks the falling stock
        short_a = _simulate_sleeve_account(
            preds, close, open_, pool, self.HORIZON, 0.1, 0.0, "short")
        short_d = np.asarray(short_a["daily"])
        assert short_d.mean() > 0.003, (
            f"short NAV must rise when the bottom falls: {short_d.mean():+.4f}")


class TestExposureLedger:
    """§十三-1 / §十三-2: the LS exposure ledger normalizes by close NAV
    (value shares, signed short) so the accounting identity holds on every
    solvent day, and a bankrupt leg/book (NAV <= 0) is NaN in the daily ledger
    and excluded from the summary instead of divided by a fake 1.0."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def test_exposure_identities_hold_every_day(self):
        """End-to-end: on every solvent day, long + short + cash == 1 with
        gross == |long| + |short| and net == long + short (signed short)."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), seed=10)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)
        daily = m["exposure"]["realized"]["daily"]
        long_mv = np.asarray(daily["long_market_value"], dtype=np.float64)
        short_mv = np.asarray(daily["short_market_value"], dtype=np.float64)
        cash = np.asarray(daily["cash"], dtype=np.float64)
        gross = np.asarray(daily["gross_exposure"], dtype=np.float64)
        net = np.asarray(daily["net_exposure"], dtype=np.float64)
        finite = np.isfinite(long_mv) & np.isfinite(short_mv) & np.isfinite(cash)
        assert finite.any(), "expected at least one solvent day with a ledger"
        assert np.max(np.abs(long_mv[finite] + short_mv[finite] + cash[finite]
                             - 1.0)) <= 1e-6, (
            "long + short + cash must equal the unit account on solvent days")
        assert np.max(np.abs(gross[finite] - (np.abs(long_mv[finite])
                                              + np.abs(short_mv[finite])))) <= 1e-6
        assert np.max(np.abs(net[finite] - (long_mv[finite] + short_mv[finite]))) <= 1e-6
        assert (short_mv[finite] <= 0.0).all(), (
            "the short leg's value share must be signed negative")

    def test_exposure_stable_under_large_gap(self):
        """§十六: a large (but not bankrupting) intraday gap must not break the
        exposure 口径 — the close-NAV-relative value shares still satisfy
        long + short + cash == 1 and stay finite on every day, no fabricated
        normalization denominator."""
        long_d = np.array([0.30, -0.25, 0.15, 0.10])   # ±30% swings
        short_d = np.array([0.20, -0.30, 0.10, 0.05])
        comb_nav = 0.5 * np.cumprod(1.0 + long_d) + 0.5 * np.cumprod(1.0 + short_d)
        assert (comb_nav > 0.0).all(), "scenario must stay solvent every day"
        long_a = {
            "daily": long_d,
            "net_exposure": np.ones(4),        # long-only → net == gross
            "cash_ratio": np.zeros(4),
            "delayed_capital": np.zeros(4),
            "active_sleeves": np.zeros(4, dtype=int),
            "bankrupt_day": None,
        }
        short_a = {
            "daily": short_d,
            "net_exposure": -np.ones(4),       # short value is negative
            "cash_ratio": np.full(4, 2.0),     # 1.0 proceeds + 1.0 collateral
            "delayed_capital": np.zeros(4),
            "active_sleeves": np.zeros(4, dtype=int),
            "bankrupt_day": None,
        }
        led = _ls_exposure_ledger(long_a, short_a)
        daily = led["realized"]["daily"]
        long_mv = np.asarray(daily["long_market_value"], dtype=np.float64)
        short_mv = np.asarray(daily["short_market_value"], dtype=np.float64)
        cash = np.asarray(daily["cash"], dtype=np.float64)
        gross = np.asarray(daily["gross_exposure"], dtype=np.float64)
        net = np.asarray(daily["net_exposure"], dtype=np.float64)
        assert np.isfinite(long_mv).all() and np.isfinite(short_mv).all()
        assert np.isfinite(cash).all() and np.isfinite(gross).all()
        assert np.max(np.abs(long_mv + short_mv + cash - 1.0)) <= 1e-6
        assert np.max(np.abs(gross - (np.abs(long_mv) + np.abs(short_mv)))) <= 1e-6
        assert np.max(np.abs(net - (long_mv + short_mv))) <= 1e-6
        assert (short_mv <= 0.0).all()

    def test_combine_book_daily_freezes_after_bankruptcy(self):
        """§十三-2: once the blended NAV goes <= 0 the return series records
        the blow-up ratio and then freezes — no fake-1.0 post-bankruptcy
        returns."""
        long_d = np.array([-0.4, -0.4, 0.1])
        short_d = np.array([-0.4, -0.4, 0.1])
        out = _combine_book_daily(long_d, short_d, 1.0, 1.0, subtract=1.0)
        a = np.cumprod(1.0 + long_d)
        b = np.cumprod(1.0 + short_d)
        nav = a + b - 1.0
        assert nav[0] > 0 and nav[1] <= 0, f"scenario must blow up: nav={nav}"
        assert np.isclose(out[0], nav[0] - 1.0)
        assert np.isclose(out[1], nav[1] / nav[0] - 1.0), (
            "the blow-up day still reports the true NAV ratio")
        assert (out[2:] == 0.0).all(), "post-bankruptcy days must freeze to 0"

    def test_short_leg_bankrupts_and_freezes(self):
        """A shorted stock that more than doubles makes the short leg's NAV go
        negative; the account latches bankrupt_day, records the blow-up return
        once, and stops marking afterwards."""
        preds = np.array([[0.5]], dtype=np.float32)
        close = np.array([[10.0, 30.0, 31.0, 32.0]], dtype=np.float32)
        open_ = np.array([[10.0, 30.0, 31.0, 32.0]], dtype=np.float32)
        pool = np.ones((1, 1), dtype=bool)
        res = _run_sleeve_sim(preds, close, open_, pool, horizon=1,
                              top_fraction=0.5, cost=0.0, mode="short")
        assert res["bankrupt_day"] == 1, (
            f"blow-up on the exit day, got {res['bankrupt_day']}")
        daily = np.asarray(res["daily"])
        assert daily[1] <= -1.0, f"blow-up day return must be <= -1: {daily[1]}"
        assert (daily[2:] == 0.0).all(), "post-bankruptcy days must freeze"
        assert np.isclose(res["final_nav"], -1.0, atol=1e-6), (
            f"final_nav pins at the blow-up mark: {res['final_nav']}")
        counts = res["exit_stats"]["counts"]
        assert counts["clean"] == 1 and counts["unresolved"] == 0

    def test_ledger_nans_bankrupt_day(self):
        """A bankrupt leg (short NAV <= 0 on day 2) makes that day NaN in the
        ledger and surfaces `bankrupt.short_day`; the summary only uses the
        remaining solvent days."""
        long_a = {
            "daily": np.zeros(4),
            "net_exposure": np.zeros(4),
            "cash_ratio": np.ones(4),
            "delayed_capital": np.zeros(4),
            "active_sleeves": np.zeros(4, dtype=int),
            "bankrupt_day": None,
        }
        short_a = {
            "daily": np.array([0.0, 0.0, -2.0, 0.1]),
            "net_exposure": np.array([0.0, 0.0, -2.0, -2.0]),
            "cash_ratio": np.array([1.0, 1.0, 3.0, 3.0]),
            "delayed_capital": np.zeros(4),
            "active_sleeves": np.zeros(4, dtype=int),
            "bankrupt_day": 2,
        }
        led = _ls_exposure_ledger(long_a, short_a)
        long_mv = np.asarray(led["realized"]["daily"]["long_market_value"])
        assert np.isnan(long_mv[2]) and np.isnan(long_mv[3]), (
            "bankrupt days must be NaN in the daily ledger")
        assert np.isfinite(long_mv[0]) and np.isfinite(long_mv[1])
        assert led["bankrupt"] == {"long_day": None, "short_day": 2}
        # Only the 2 solvent days enter the summary.
        assert led["realized"]["mean_gross_exposure"] == 0.0


class TestLsNoRebalance:
    """The LS metric and the ledger must be ONE account policy.
    Both legs start at their target weight and are never rebalanced, so the
    combined daily return is the NAV ratio of the weighted NAV blend (方案 A)
    — NOT the arithmetic mean of the legs' daily returns (which silently
    assumes a daily 50/50 reallocation and drifts from the ledger after day 1).
    """

    def test_combine_book_daily_matches_no_rebalance_nav(self):
        long_d = np.array([0.10, 0.20, -0.05, 0.03])
        short_d = np.array([0.00, -0.10, 0.15, 0.02])
        ls = _combine_book_daily(long_d, short_d, 0.5, 0.5)
        long_nav = np.cumprod(1.0 + long_d)
        short_nav = np.cumprod(1.0 + short_d)
        nav = 0.5 * long_nav + 0.5 * short_nav
        expected = np.empty_like(nav)
        expected[0] = nav[0] - 1.0
        expected[1:] = nav[1:] / nav[:-1] - 1.0
        np.testing.assert_allclose(ls, expected, rtol=1e-12)

    def test_diverges_from_arithmetic_mean_after_day_one(self):
        # The two policies coincide on day 1 (50/50 of the first-day returns)
        # but diverge once the legs' NAVs drift apart.
        long_d = np.array([0.10, 0.20, -0.05])
        short_d = np.array([0.00, -0.10, 0.15])
        ls = _combine_book_daily(long_d, short_d, 0.5, 0.5)
        assert np.isclose(ls[0], 0.5 * (long_d[0] + short_d[0]))
        assert not np.isclose(ls[1], 0.5 * (long_d[1] + short_d[1])), (
            "daily-rebalance mean must NOT equal the no-rebalance NAV return")
        assert not np.isclose(ls[2], 0.5 * (long_d[2] + short_d[2]))

    def test_combine_book_daily_2x_leverage(self):
        # 2x gross = full unit long + full unit short; the short leg's margin
        # loan makes the book NAV = long_nav + short_nav - 1.0.
        long_d = np.array([0.05, 0.10, -0.02])
        short_d = np.array([0.02, -0.04, 0.03])
        ls2x = _combine_book_daily(long_d, short_d, 1.0, 1.0, subtract=1.0)
        long_nav = np.cumprod(1.0 + long_d)
        short_nav = np.cumprod(1.0 + short_d)
        nav = long_nav + short_nav - 1.0
        expected = np.empty_like(nav)
        expected[0] = nav[0] - 1.0
        expected[1:] = nav[1:] / nav[:-1] - 1.0
        np.testing.assert_allclose(ls2x, expected, rtol=1e-12)


class TestLastSleeveExit:
    """The simulation runs to the END of the price path
    (Wp columns), so the sleeve entered on the last signal day W-1 liquidates at
    open[W-1+horizon] with its exit cost booked and no position left active."""

    def test_last_sleeve_exits_with_cost_booked(self):
        # W=3 signal days, horizon=2 → the d=2 sleeve exits at d=4 (W-1+h).
        # Price path padded to Wp=5 so the exit is real, not unresolved.
        preds = np.array([[0.9, 0.9, 0.9],
                          [0.1, 0.1, 0.1]], dtype=np.float32)  # top = stock 0
        close = np.array([[10.0, 11.0, 12.0, 13.0, 14.0],
                          [20.0, 21.0, 22.0, 23.0, 24.0]], dtype=np.float32)
        open_ = np.array([[10.5, 11.5, 12.5, 13.5, 14.5],
                          [20.5, 21.5, 22.5, 23.5, 24.5]], dtype=np.float32)
        pool = np.ones((2, 3), dtype=bool)
        res = _simulate_sleeve_account(
            preds, close, open_, pool, horizon=2, top_fraction=0.5,
            cost=0.001, mode="long")
        # Simulated through the last exit day W-1+h = Wp-1 → 5 daily periods,
        # day 0's open->close now included.
        assert len(res["daily"]) == 5, f"len={len(res['daily'])}"
        counts = res["exit_stats"]["counts"]
        # 3 sleeves × 1 filled stock each, all exiting exactly on schedule.
        assert counts["clean"] == 3, f"clean={counts['clean']}"
        assert counts["unresolved"] == 0, "no sleeve may dangle at the end"
        assert counts["delayed"] == 0 and counts["delisted"] == 0
        # The final exit (d=4) booked a sell cost → gross return exceeds net.
        gross = np.asarray(res["gross_daily"])
        net = np.asarray(res["daily"])
        assert gross[-1] > net[-1], "final sleeve exit cost was not booked"


class TestSleeveAccountIdentity:
    """The strongest consistency check for any backtest — the
    returned daily-return series must compound to EXACTLY the account's final
    NAV, and the exit ledger must reconcile to it.  The most dangerous failure
    mode in a quant backtest is not an error but a strategy whose equity curve
    looks normal while the reported stats derive from a DIFFERENT path; this
    pins the identity across every mode and cost level."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def _inputs(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), seed=0)
        n_w = self.N_TS - self.SEQ_LEN
        p0 = self.SEQ_LEN
        preds = np.tile(panel["static_features"][:, :1], (1, n_w)).astype(np.float32)
        pool = _candidate_pool(panel, n_w, self.SEQ_LEN).numpy()
        close = panel["close_price"][:, p0:p0 + n_w + self.HORIZON]
        open_ = panel["open_price"][:, p0:p0 + n_w + self.HORIZON]
        return preds, close, open_, pool

    @pytest.mark.parametrize("mode", ["long", "short", "ew"])
    @pytest.mark.parametrize("cost", [0.0, 0.0005, 0.002])
    def test_daily_compounds_to_final_nav(self, mode, cost):
        preds, close, open_, pool = self._inputs()
        res = _run_sleeve_sim(preds, close, open_, pool, self.HORIZON,
                              top_fraction=0.1, cost=cost, mode=mode)
        daily = np.asarray(res["daily"], dtype=np.float64)
        assert np.all(np.isfinite(daily)), f"{mode}: daily has non-finite entries"
        compound = float(np.prod(1.0 + daily))
        assert compound > 0.0
        assert np.isclose(compound, res["final_nav"], rtol=1e-9, atol=1e-9), (
            f"{mode}@{cost}: prod(1+daily)={compound:.12f} != "
            f"final_nav={res['final_nav']:.12f}")

    def test_net_and_gross_series_both_reconcile(self):
        """The net and the cost=0 gross runs are INDEPENDENT accounts
        — each must satisfy the identity internally, and the net book carries a
        cost drag so it never compounds above the gross one."""
        preds, close, open_, pool = self._inputs()
        acc = _simulate_sleeve_account(preds, close, open_, pool, self.HORIZON,
                                       0.1, 0.001, "long")
        net_final = float(np.prod(1.0 + np.asarray(acc["daily"], dtype=np.float64)))
        gross_final = float(np.prod(1.0 + np.asarray(acc["gross_daily"], dtype=np.float64)))
        assert np.isfinite(net_final) and np.isfinite(gross_final)
        assert net_final > 0.0 and gross_final > 0.0
        assert net_final <= gross_final + 1e-9, (
            f"net final {net_final:.8f} above gross {gross_final:.8f} — "
            f"costs must not inflate the account")

    def test_ledger_reconciles_to_final_nav_across_modes(self):
        preds, close, open_, pool = self._inputs()
        for mode in ("long", "short", "ew"):
            res = _run_sleeve_sim(preds, close, open_, pool, self.HORIZON,
                                  0.1, 0.001, mode, return_ledger=True)
            total = _ledger_pnl(res["ledger"])
            assert np.isclose(total, res["final_nav"] - 1.0, atol=1e-8), (
                f"{mode}: sum(ledger pnl)={total:.10f} != "
                f"final_nav-1={res['final_nav'] - 1.0:.10f}")


class TestLastHorizonSuspendedExit:
    """A signal fired on the last signal day whose ENTIRE exit
    window is suspended must NOT silently vanish at test end.  The position
    holds (pending) and either exits at the resumed open with cost booked
    (DELAYED) or books UNRESOLVED at the final carried mark — never
    "signal → entry → test ends" with the position unaccounted for.

    Exact reviewer scenario: stock A day0 open 100, day1 close 110,
    days 2-5 suspended, day6 open 120.
    """

    def _run(self, close_path, open_path, horizon=5, cost=0.001,
             return_ledger=False):
        preds = np.array([[0.9], [0.1]], dtype=np.float32)  # top-1 = stock 0
        close = np.array(close_path, dtype=np.float32)
        open_ = np.array(open_path, dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        return _run_sleeve_sim(preds, close, open_, pool, horizon=horizon,
                               top_fraction=0.5, cost=cost, mode="long",
                               return_ledger=return_ledger)

    def test_resume_in_padded_window_is_delayed_exit_with_cost(self):
        # W=1 signal day (d0 = the last signal day), horizon=5, price path
        # padded to Wp=7 so the day-6 resume sits inside the walked window.
        close = [[105.0, 110.0, 110.0, 110.0, 110.0, 110.0, 120.0],
                 [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]]
        open_ = [[100.0, np.nan, np.nan, np.nan, np.nan, np.nan, 120.0],
                 [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]]
        res = self._run(close, open_, return_ledger=True)
        counts = res["exit_stats"]["counts"]
        assert counts["delayed"] == 1, f"expected a delayed exit, got {counts}"
        assert counts["clean"] == 0 and counts["unresolved"] == 0
        # The pending exit resolved at the resumed open — capital not dumped at
        # the stale close on the scheduled exit day.
        assert res["exit_stats"]["pnl"]["delayed"] > 0.0, (
            "delayed exit bought 100 / sold 120 → P&L must be positive")
        # The final-day exit cost is booked in the NET run only.
        net = np.asarray(res["daily"])
        gross = np.asarray(res["gross_daily"]) if "gross_daily" in res else None
        # (gross_daily is only present via _simulate_sleeve_account)
        assert res["ledger"][0]["exit_status"] == "delayed"
        assert res["ledger"][0]["exit_price"] == 120.0
        assert res["ledger"][0]["exit_cost"] > 0.0, (
            "delayed exit must book a sell cost")

    def test_simulate_account_books_exit_cost_on_delayed_day(self):
        """The gross/net separation (independent cost=0 account) shows the
        resumed exit day's cost drag: gross return on the final day exceeds the
        net one."""
        preds = np.array([[0.9], [0.1]], dtype=np.float32)
        close = np.array([[105.0, 110.0, 110.0, 110.0, 110.0, 110.0, 120.0],
                          [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]],
                         dtype=np.float32)
        open_ = np.array([[100.0, np.nan, np.nan, np.nan, np.nan, np.nan, 120.0],
                          [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]],
                         dtype=np.float32)
        pool = np.ones((2, 1), dtype=bool)
        res = _simulate_sleeve_account(preds, close, open_, pool, horizon=5,
                                       top_fraction=0.5, cost=0.001, mode="long")
        net = np.asarray(res["daily"])
        gross = np.asarray(res["gross_daily"])
        assert gross[-1] > net[-1], (
            "final-day delayed exit cost was not booked into the net account")

    def test_no_resume_books_unresolved_at_final_mark(self):
        # Same entry but the price path ends INSIDE the suspension (Wp=6 = W+h,
        # exit day d5 has no open and no resume) → UNRESOLVED at the carried
        # close, entering the ledger with entry cost charged and the held mark
        # still reflected in NAV.
        close = [[105.0, 110.0, 110.0, 110.0, 110.0, 110.0],
                 [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]]
        open_ = [[100.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                 [20.5, 21.5, 22.5, 23.5, 24.5, 25.5]]
        res = self._run(close, open_, return_ledger=True)
        counts = res["exit_stats"]["counts"]
        assert counts["unresolved"] == 1, f"expected unresolved, got {counts}"
        assert counts["delayed"] == 0 and counts["clean"] == 0
        # NAV was affected by the held position: it gained the carried mark
        # 105→110 on 1/5 weight minus the entry cost → final NAV > 1.
        assert res["final_nav"] > 1.0, f"NAV untouched? final_nav={res['final_nav']:.6f}"
        # The audit trail records the locked capital: entry cost charged, no
        # fictitious sell cost, and the held position MARKED at the final
        # carried close (§十三-3) — not fabricated as a realized exit.
        led = res["ledger"]
        assert len(led) == 1, f"expected the single filled sleeve, got {len(led)}"
        r = led[0]
        assert r["exit_status"] == "unresolved"
        assert r["entry_price"] == 100.0
        assert r["actual_exit_day"] is None
        assert r["mark_day"] == 5
        assert r["mark_price"] == 110.0, "unresolved marks the final carried close"
        assert r["exit_price"] is None and r["realized_return"] is None
        assert r["net_pnl"] is None
        assert r["entry_cost"] > 0.0 and r["exit_cost"] == 0.0
        # Ledger reconciles to the account identity even with capital locked.
        assert np.isclose(r["unrealized_pnl"], res["final_nav"] - 1.0, atol=1e-9)


class TestRawReturnUnits:
    """Q5−Q1 "bp" and clean IC must be in RAW return
    units even when the training label y_return is z-scored + clipped per fold.
    y_return_raw (saved before normalization) must be preferred for both."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def test_quintile_and_ic_use_raw_returns(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), seed=4)
        raw = panel["y_return"].astype(np.float32).copy()
        panel["y_return_raw"] = raw
        # Simulate train_panel's per-fold normalization of the LABEL only; the
        # raw copy above must survive untouched for evaluation.
        z = np.zeros_like(raw)
        for t in range(raw.shape[1]):
            col = raw[:, t]
            s = float(col.std())
            z[:, t] = (col - float(col.mean())) / s if s > 1e-8 else 0.0
        panel["y_return"] = np.clip(z, -5.0, 5.0).astype(np.float32)
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON,
                          min_stocks_per_day=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"),
            horizon=self.HORIZON)
        # Clean IC computed on the RAW actuals → strong, as in the
        # un-normalized control test.
        assert m["ic_mean"] > 0.3, f"clean IC lost on raw path: {m['ic_mean']}"
        # q5-q1 in raw units (~±0.8% legs → tens of bp), far below the z-score
        # scale — if the normalized label leaked, this would be ~1-2.
        assert 0.001 < m["q5mq1_ret"] < 0.10, (
            f"quintile spread in wrong units: {m['q5mq1_ret']:.4f}")


class TestTurnoverScaleInvariant:
    """Turnover is traded notional normalized by the
    day's opening NAV — a ratio, so a 1M account reports the same turnover as
    a 1 account.  Scaling the whole price level (equivalent to a bigger account,
    since sleeve weights are NAV fractions) must leave daily returns and the
    turnover ratio unchanged."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def _run(self, price_scale):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), seed=6)
        n_windows = self.N_TS - self.SEQ_LEN
        p0 = self.SEQ_LEN
        close = panel["close_price"][:, p0:p0 + n_windows + self.HORIZON]
        open_ = panel["open_price"][:, p0:p0 + n_windows + self.HORIZON]
        pool = _candidate_pool(panel, n_windows, self.SEQ_LEN).numpy()
        preds = np.zeros((12, n_windows), dtype=np.float32)
        preds[1] = 2.0
        return _simulate_sleeve_account(
            preds, close * price_scale, open_ * price_scale, pool,
            self.HORIZON, 0.1, cost=0.001, mode="long")

    def test_turnover_and_returns_invariant_to_capital_scale(self):
        small = self._run(1.0)
        big = self._run(1e6)
        # Mathematically identical; float32 shares = w/(price*1e6) leave only
        # ~1e-8 rounding, so compare at float32 epsilon scale.
        assert np.allclose(small["daily"], big["daily"], atol=1e-6)
        assert np.isclose(small["turnover"]["daily_avg"],
                          big["turnover"]["daily_avg"], rtol=1e-6)


class TestCleanICSeparation:
    """IC is computed over the CLEAN y_return (open->open) ×
    return_target_mask, separated from the exec P&L that uses fills/costs/
    carry.  A rankable market must show strong positive clean IC even though
    the exec path deducts costs."""

    def test_ic_uses_clean_return(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), seed=3)
        cfg = PanelConfig(seq_len=60, horizon=5, min_stocks_per_day=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5)
        assert m["ic_mean"] > 0.3, f"clean IC not positive: {m['ic_mean']:.3f}"
        assert m["q5mq1_ret"] > 0.0, (
            f"clean quintile spread wrong: {m['q5mq1_ret']:.4f}")


class TestSleeveAntiCheat:
    """review §五 anti-cheat on the PRICED path: constant / random predictions
    must show no IC, no long-short, no quintile spread even though the market
    contains real rankable drift — the evaluation must not manufacture alpha."""

    SEQ_LEN, HORIZON, N_TS = 60, 5, 320

    def _eval(self, model, seed):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=self.N_TS, seq_len=self.SEQ_LEN,
            horizon=self.HORIZON, up_stocks=(1,), down_stocks=(2,), seed=seed)
        # txn_cost=0.0: this class tests "no manufactured SELECTION alpha".
        # Since the sleeve account deducts entry costs from cash,
        # a zero-alpha book honestly shows a systematic negative LS drag that
        # would trip the abs()<1.0 bound calibrated when costs were
        # under-counted.  Cost drag is covered separately by TestSleeveCosts.
        cfg = PanelConfig(seq_len=self.SEQ_LEN, horizon=self.HORIZON,
                          txn_cost=0.0, min_stocks_per_day=5)
        return evaluate_portfolio(
            model, panel, cfg, torch.device("cpu"), horizon=self.HORIZON)

    def test_constant_preds_no_alpha(self):
        # 10 seeds — the annualized Sharpe of a zero-signal book has per-seed
        # std ≈ sqrt(252/n_periods) ≈ 1.0; 3 seeds left the mean at ~2σ and the
        # test flaky.  See TestEvaluateAntiCheat for the power note.
        ics, lss, q5s = [], [], []
        for seed in range(10):
            m = self._eval(_ConstantReturnModel(), seed)
            ics.append(m["ic_mean"])
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert all(ic == 0.0 for ic in ics), "constant preds IC must be 0"
        assert abs(np.mean(lss)) < 1.0, f"constant preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"constant preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"

    def test_random_preds_no_alpha(self):
        ics, lss, q5s = [], [], []
        for seed in range(10):
            m = self._eval(_RandomReturnModel(seed=seed), seed)
            ics.append(m["ic_mean"])
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert abs(np.mean(ics)) < 0.05, f"random preds mean IC={np.mean(ics):+.4f}"
        assert abs(np.mean(lss)) < 1.0, f"random preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"random preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"


class TestSleeveLedger:
    """The per-position ledger — one record per FILLED
    position with entry/exit price, scheduled/actual exit day, exit status,
    gross/net PnL and attributed costs — must reconcile to the account identity
    (sum(net_pnl) == final_nav - 1) and be fully recoverable from a saved tape
    (preds + price paths + pool + metadata) replayed through _run_sleeve_sim."""

    _SCHEMA = {
        "entry_day", "stock", "mode", "entry_price", "entry_notional",
        "target_weight", "executed_weight", "entry_nav", "shares",
        "scheduled_exit_day", "actual_exit_day", "exit_status", "exit_price",
        "realized_return", "mark_day", "mark_price", "gross_pnl",
        "entry_cost", "exit_cost", "net_pnl", "unrealized_pnl",
    }

    def test_ledger_reconciles_to_final_nav(self):
        preds = np.array([[0.9, 0.9, 0.9, 0.9],
                          [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)  # top-1 long = stock 0
        close = np.array([[10.0, 11.0, 12.0, 13.0, 14.0],
                          [20.0, 26.0, 27.0, 28.0, 29.0]], dtype=np.float32)
        open_ = np.array([[10.0, 10.5, 11.5, 12.5, 13.5],
                          [20.0, 25.5, 26.5, 27.5, 28.5]], dtype=np.float32)
        pool = np.ones((2, 4), dtype=bool)
        res = _run_sleeve_sim(
            preds, close, open_, pool, horizon=1, top_fraction=0.5,
            cost=0.0005, mode="long", return_ledger=True)
        led = res["ledger"]
        assert len(led) == 4, f"one clean position per signal day, got {len(led)}"
        assert set(led[0]) == self._SCHEMA, set(led[0])
        for r in led:
            assert r["exit_status"] in {"clean", "delayed", "unresolved"}
            # net = gross minus BOTH attributed costs (entry cost always, exit
            # cost only when a real exit was executed — unresolved has none).
            if r["exit_status"] == "unresolved":
                assert r["exit_cost"] == 0.0
                # §十三-3: unresolved holds are marked, not exited — no exit
                # price/return/realized net; P&L is unrealized at the mark.
                assert r["actual_exit_day"] is None
                assert r["mark_day"] is not None and r["mark_price"] is not None
                assert r["exit_price"] is None and r["realized_return"] is None
                assert r["net_pnl"] is None
                pnl = r["unrealized_pnl"]
            else:
                assert r["mark_day"] is None and r["mark_price"] is None
                assert r["unrealized_pnl"] is None
                pnl = r["net_pnl"]
            assert np.isclose(pnl, r["gross_pnl"] - r["entry_cost"] - r["exit_cost"],
                              atol=1e-12)
            assert np.isclose(r["entry_notional"], r["shares"] * r["entry_price"],
                              rtol=1e-6)
            # §十四-2: executed_weight is the REAL fraction of entry NAV, never a
            # nominal amount; the cash cap can only push it below target_weight.
            assert np.isclose(r["executed_weight"],
                              r["entry_notional"] / r["entry_nav"], rtol=1e-6)
            assert r["target_weight"] > 0
            assert r["executed_weight"] <= r["target_weight"] + 1e-9
        total = _ledger_pnl(led)
        final_nav = float(np.prod(1.0 + res["daily"]))
        assert np.isclose(total, final_nav - 1.0, atol=1e-9), (
            f"sum(ledger pnl)={total:.8f} != final_nav - 1 = {final_nav - 1:.8f}")

    def test_ledger_exposed_via_evaluate_portfolio(self):
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), seed=0)
        cfg = PanelConfig(seq_len=60, horizon=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5,
            return_ledger=True)
        led = m.get("long_ledger")
        assert led is not None and len(led) > 0, "return_ledger must emit fills"
        assert set(led[0]) == self._SCHEMA, set(led[0])
        assert {r["exit_status"] for r in led} <= {"clean", "delayed", "unresolved"}
        # Filled positions have a real entry price and positive value.
        assert all(np.isfinite(r["entry_price"]) and r["entry_notional"] > 0
                   for r in led)
        # §十四-1: every leg's ledger is exposed, not just the long one.
        short_led = m.get("short_ledger")
        assert short_led is not None and len(short_led) > 0, "short ledger missing"
        assert {r["mode"] for r in short_led} == {"short"}
        assert set(short_led[0]) == self._SCHEMA, set(short_led[0])
        ew_led = m.get("ew_ledger")
        assert ew_led is not None and len(ew_led) > 0, "ew ledger missing"
        assert {r["mode"] for r in ew_led} == {"ew"}
        sel_led = m.get("selected_universe_ew_ledger")
        assert sel_led is not None and len(sel_led) > 0, "selected-universe ledger missing"
        assert {r["mode"] for r in sel_led} == {"ew"}
        ls_led = m.get("ls_ledger")
        assert ls_led is not None and len(ls_led) > 0, "ls ledger missing"
        assert {r["mode"] for r in ls_led} == {"long", "short"}
        # LS tape is long+short concatenated and sorted by (entry_day, stock, mode).
        assert len(ls_led) == len(led) + len(short_led)
        keys = [(r["entry_day"], r["stock"], r["mode"]) for r in ls_led]
        assert keys == sorted(keys)
        assert set(ls_led[0]) == self._SCHEMA, set(ls_led[0])

    def test_tape_roundtrip_replays_long_account(self):
        """A tape holding preds + close/open price paths + pool + metadata (the
        exact arrays train_panel.py saves per fold) must replay
        through _run_sleeve_sim to the SAME final NAV the sleeve account built,
        so an offline consumer reconstructs the backtest without the model."""
        panel = _make_priced_panel(
            n_stocks=12, n_timesteps=320, seq_len=60, horizon=5,
            up_stocks=(1,), down_stocks=(2,), seed=0)
        cfg = PanelConfig(seq_len=60, horizon=5)
        m = evaluate_portfolio(
            _StaticMarkerModel(), panel, cfg, torch.device("cpu"), horizon=5,
            return_ledger=True)
        orig_ledger = m["long_ledger"]
        assert orig_ledger, "tape test needs a non-empty original account"

        # _make_priced_panel stores static as (N, 6) per-stock markers, not a
        # time axis — the window-day grid is defined by the price path.
        n_timesteps = panel["close_price"].shape[1]
        n_w = n_timesteps - cfg.seq_len
        p0 = cfg.seq_len
        # _StaticMarkerModel predicts static[:, :1] for every window, so the
        # per-window pred matrix is the marker tiled across W.
        preds_np = np.tile(panel["static_features"][:, :1], (1, n_w)).astype(np.float32)
        pool_np = _candidate_pool(panel, n_w, cfg.seq_len).numpy()
        close_grid = panel["close_price"][:, p0:p0 + n_w + cfg.horizon]
        open_grid = panel["open_price"][:, p0:p0 + n_w + cfg.horizon]

        buf = io.BytesIO()
        np.savez(buf, preds=preds_np, close_price=close_grid, open_price=open_grid,
                 pool=pool_np, horizon=cfg.horizon, top_fraction=0.1,
                 cost=cfg.txn_cost)
        buf.seek(0)
        z = np.load(buf)
        replay = _run_sleeve_sim(
            z["preds"], z["close_price"], z["open_price"], z["pool"],
            int(z["horizon"]), float(z["top_fraction"]), float(z["cost"]),
            mode="long", return_ledger=True)

        orig_final = 1.0 + _ledger_pnl(orig_ledger)
        replay_final = float(np.prod(1.0 + replay["daily"]))
        assert np.isclose(replay_final, orig_final, rtol=1e-6, atol=1e-9), (
            f"tape replay final NAV {replay_final:.8f} != original {orig_final:.8f}")
        replay_net = _ledger_pnl(replay["ledger"])
        assert np.isclose(replay_net, replay_final - 1.0, atol=1e-9)


class TestContinuousOosReplay:
    """§十四-4: ONE continuous long sleeve account replayed across all fold
    tapes.  A fold's NAV restarts at 1 and is aggregated by mean Sharpe — that
    is a set of disjoint OOS signal windows, not a continuous strategy.  The
    replay keeps a single account whose NAV carries over fold boundaries (the
    previous fold's sleeves keep settling while the next fold's model signals),
    and the final Sharpe/MDD/CAGR come from that account only."""

    def _write_tape(self, path, *, stocks, dates, price_dates, preds, pool,
                    close, open, horizon, seq_len, top_fraction, cost,
                    delist_day=None, data_version="test",
                    universe_status_hash="u", membership_hash="m",
                    calendar_hash=None, evaluator_version=None,
                    price_convention=None, exit_policy=None,
                    strategy_mode=None,
                    model_source_hash=None, model_config_hash=None,
                    feature_schema_hash=None, weight_hash="test"):
        kw = dict(decision_eligible=pool, history_eligible=pool, pool=pool,
                  entry_eligible=np.ones_like(pool),
                  return_target_mask=np.ones_like(pool),
                  return_target=np.zeros_like(pool),
                  close_price=close, open_price=open,
                  price_dates=np.array(price_dates),
                  horizon=horizon, seq_len=seq_len,
                  top_fraction=top_fraction, cost=cost,
                  data_version=data_version,
                  universe_status_hash=universe_status_hash,
                  membership_hash=membership_hash,
                  model_hash="test", weight_hash=weight_hash)
        # §十五-1/§十五-2/§十六: the policy + full metadata + model-identity
        # keys a FORMAL replay requires.  Omitted by default so legacy-behaviour
        # tests (missing keys → tolerated in non-formal mode) keep exercising
        # that path.
        for k, v in [("calendar_hash", calendar_hash),
                     ("evaluator_version", evaluator_version),
                     ("price_convention", price_convention),
                     ("exit_policy", exit_policy),
                     ("strategy_mode", strategy_mode),
                     ("model_source_hash", model_source_hash),
                     ("model_config_hash", model_config_hash),
                     ("feature_schema_hash", feature_schema_hash)]:
            if v is not None:
                kw[k] = v
        if delist_day is not None:
            kw["delist_day"] = np.asarray(delist_day, dtype=int)
        np.savez(path, preds=preds, dates=np.array(dates),
                 stocks=np.array(stocks), **kw)

    def test_account_carries_across_fold_boundary(self, tmp_path):
        from scripts.production.train_panel import _replay_continuous_oos

        # Global price axis: 10 days.  stock0 rises monotonically, stock1 falls.
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        close0 = np.arange(10, 20, dtype=np.float32)     # 10 .. 19
        close1 = np.arange(20, 10, -1, dtype=np.float32)  # 20 .. 11
        stocks = ["000001", "000002"]
        # Fold 1 (OLDER): signal days dates[0:4], price tail dates[0:6].
        # Fold 0 (NEWER): signal days dates[4:8], price tail dates[4:10].
        # Lexicographic tape order fold_000/fold_001 = NEWEST/OLDEST.
        fold_new = {
            "stocks": stocks,
            "dates": dates[4:8],
            "price_dates": dates[4:10],
            # picks stock1 (row 1 = 000002) — the falling name.
            "preds": np.array([[0.1, 0.1, 0.1, 0.1],
                               [0.9, 0.9, 0.9, 0.9]], dtype=np.float32),
            "pool": np.ones((2, 4), dtype=bool),
            "close": np.stack([close0[4:10], close1[4:10]]),
            "open": np.stack([close0[4:10], close1[4:10]]),
        }
        fold_old = {
            "stocks": stocks,
            "dates": dates[0:4],
            "price_dates": dates[0:6],
            # picks stock0 (row 0 = 000001) — the rising name.
            "preds": np.array([[0.9, 0.9, 0.9, 0.9],
                               [0.1, 0.1, 0.1, 0.1]], dtype=np.float32),
            "pool": np.ones((2, 4), dtype=bool),
            "close": np.stack([close0[0:6], close1[0:6]]),
            "open": np.stack([close0[0:6], close1[0:6]]),
        }
        horizon, seq_len, top_fraction, cost = 2, 60, 0.5, 0.0
        self._write_tape(str(tmp_path / "fold_000.npz"), horizon=horizon,
                         seq_len=seq_len, top_fraction=top_fraction, cost=cost,
                         **fold_new)
        self._write_tape(str(tmp_path / "fold_001.npz"), horizon=horizon,
                         seq_len=seq_len, top_fraction=top_fraction, cost=cost,
                         **fold_old)

        cont = _replay_continuous_oos(str(tmp_path))
        assert cont is not None
        acc = cont["account"]
        daily = np.asarray(acc["daily"], dtype=np.float64)
        # ONE account ran the FULL union price path — not per-fold NAV restarts.
        assert len(daily) == len(dates) == 10
        assert cont["price_dates"] == dates
        assert cont["stocks"] == stocks
        final_nav = acc["final_nav"]
        assert final_nav is not None and final_nav > 0
        assert np.isclose(float(np.prod(1.0 + daily)), final_nav, rtol=1e-5)

        # Model switch at the fold boundary: the ledger carries fills from BOTH
        # folds (old signals days 0-3, new signals days 4-7) in one account.
        led = cont["ledger"]
        assert led is not None and len(led) > 0
        entry_days = set(int(d) for d in led["entry_day"])
        assert entry_days & set(range(0, 4)), "expected fold-OLD signals"
        assert entry_days & set(range(4, 8)), "expected fold-NEW signals"
        # A position entered in the OLD fold settles INSIDE the new fold's
        # window in the same account: entry at signal day 3 schedules exit at
        # day 5 (horizon=2), which is a real price day in the union grid.
        d3 = led[led["entry_day"] == 3]
        assert len(d3) == 1
        assert int(d3.iloc[0]["actual_exit_day"]) == 5
        assert d3.iloc[0]["exit_status"] == "clean"
        assert d3.iloc[0]["stock_code"] == "000001"
        assert d3.iloc[0]["entry_date"] == dates[3]

        # Ledger rows map to union stocks / dates and the metrics derive from
        # the continuous account.
        assert set(led["stock_code"]) <= set(cont["stocks"])
        assert all(d in cont["price_dates"] for d in led["entry_date"])
        m = cont["metrics"]
        assert m["n_days"] == 10 and m["n_stocks"] == 2
        assert np.isfinite(m["sharpe"])
        assert m["maxdd"] >= 0.0
        assert m["final_nav"] == final_nav

    def test_merged_tape_dates_strictly_increasing_with_scrambled_fold(
            self, tmp_path):
        """§十六: the merged continuous-OOS price axis must be strictly
        increasing even when a fold tape stores its price_dates out of
        chronological order.  The union grid is keyed by date, so the account
        stays monotonic and column d ↔ date holds without assuming the producer
        wrote the rows in order."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 6)]  # 01 .. 05
        stocks = ["000001"]
        close0 = np.arange(10, 15, dtype=np.float32)  # 10 .. 14, monotonic
        # Fold tape stores price rows REVERSED (dates and prices stay aligned
        # with each other, but the date column itself is out of order).
        scrambled = list(reversed(dates))
        scrambled_close = np.stack([close0[::-1]])  # aligned to scrambled dates
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks,
            dates=dates[:3],
            price_dates=scrambled,
            preds=np.array([[0.9, 0.9, 0.9]], dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=scrambled_close,
            open=scrambled_close,
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
        )

        cont = _replay_continuous_oos(str(tmp_path))
        assert cont is not None
        merged = cont["price_dates"]
        assert merged == dates  # the union grid is re-sorted chronologically
        assert all(a < b for a, b in zip(merged, merged[1:])), (
            "merged tape dates must be strictly increasing")
        # One account column per merged price day — the monotonic grid is the
        # account's time axis.
        assert len(np.asarray(cont["account"]["daily"])) == len(dates)
        final_nav = cont["account"]["final_nav"]
        assert final_nav is not None and final_nav > 0

    def test_no_tapes_returns_none(self, tmp_path):
        from scripts.production.train_panel import _replay_continuous_oos

        assert _replay_continuous_oos(str(tmp_path)) is None

    def test_replay_force_sells_delisted_position(self, tmp_path):
        """§十二.1: a tape's delist_day (per-stock, in the fold's sim column
        space) is mapped onto the union price axis and forwarded to the sleeve
        simulator, so the continuous account books a DELISTED force-sell exactly
        as the live per-fold run did — no silent UNRESOLVED degradation."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 8)]
        stocks = ["000001"]
        close = np.arange(10, 17, dtype=np.float32)  # 10..16, monotonic
        # delist_day=2 → force-sell at price_dates[2] = 2025-01-03.
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks,
            dates=dates[0:2],
            price_dates=dates,
            preds=np.array([[0.9, 0.9]], dtype=np.float32),
            pool=np.ones((1, 2), dtype=bool),
            close=np.stack([close]),
            open=np.stack([close]),
            horizon=3, seq_len=60, top_fraction=0.5, cost=0.0,
            delist_day=np.array([2], dtype=int),
        )

        cont = _replay_continuous_oos(str(tmp_path))
        assert cont is not None
        led = cont["ledger"]
        assert led is not None and len(led) > 0
        assert set(led["exit_status"]) == {"delisted"}, (
            "delisted force-sell must fire in the continuous replay")

    def test_overlapping_close_mismatch_fails(self, tmp_path):
        """§十二.2: two folds owning the same (stock, day) MUST carry identical
        prices; a later write must not silently overwrite an earlier one."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 7)]
        stocks = ["000001"]
        a_close = np.arange(10, 16, dtype=np.float32)   # 10..15
        b_close = a_close.copy()
        b_close[3] = 999.0                               # disagree on 01-04
        # fold_000 = NEWEST (later window), fold_001 = OLDEST (covers 0..5).
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks, dates=dates[3:6], price_dates=dates[3:6],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([a_close[3:6]]), open=np.stack([a_close[3:6]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0)
        self._write_tape(
            str(tmp_path / "fold_001.npz"),
            stocks=stocks, dates=dates[0:3], price_dates=dates[0:6],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([b_close[0:6]]), open=np.stack([b_close[0:6]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0)

        with pytest.raises(ValueError, match="overlapping fold close prices"):
            _replay_continuous_oos(str(tmp_path))

    def test_tape_version_or_universe_hash_mismatch_fails(self, tmp_path):
        """§十二.2: all tapes must describe the same data_version and the same
        universe/membership records — a fold re-downloaded mid-run or a
        delist-file edit between runs invalidates the continuous account."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 7)]
        stocks = ["000001"]
        close = np.arange(10, 16, dtype=np.float32)
        # Non-overlapping windows (dates[0:3] vs dates[3:6]) so the ONLY
        # inconsistency is the version/hash — the price assertion must not fire.
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks, dates=dates[3:6], price_dates=dates[3:6],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[3:6]]), open=np.stack([close[3:6]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
            data_version="v1", universe_status_hash="u1")
        self._write_tape(
            str(tmp_path / "fold_001.npz"),
            stocks=stocks, dates=dates[0:3], price_dates=dates[0:3],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[0:3]]), open=np.stack([close[0:3]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
            data_version="v2", universe_status_hash="u2")

        with pytest.raises(ValueError, match="data_version"):
            _replay_continuous_oos(str(tmp_path))

    def test_mixed_policy_horizon_fails(self, tmp_path):
        """§十五-1: all tapes must share the same STRATEGY POLICY.  A horizon=5
        fold mixed with a horizon=20 fold must not be explained by the first
        tape's horizon — the account would silently misread every later fold."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 7)]
        stocks = ["000001"]
        close = np.arange(10, 16, dtype=np.float32)
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks, dates=dates[3:6], price_dates=dates[3:6],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[3:6]]), open=np.stack([close[3:6]]),
            horizon=5, seq_len=60, top_fraction=0.5, cost=0.0)
        self._write_tape(
            str(tmp_path / "fold_001.npz"),
            stocks=stocks, dates=dates[0:3], price_dates=dates[0:3],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[0:3]]), open=np.stack([close[0:3]]),
            horizon=20, seq_len=60, top_fraction=0.5, cost=0.0)

        with pytest.raises(ValueError, match="horizon"):
            _replay_continuous_oos(str(tmp_path))

    def test_mixed_policy_cost_fails(self, tmp_path):
        """§十五-1: the transaction-cost policy must match across folds too."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 7)]
        stocks = ["000001"]
        close = np.arange(10, 16, dtype=np.float32)
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks, dates=dates[3:6], price_dates=dates[3:6],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[3:6]]), open=np.stack([close[3:6]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.001)
        self._write_tape(
            str(tmp_path / "fold_001.npz"),
            stocks=stocks, dates=dates[0:3], price_dates=dates[0:3],
            preds=np.ones((1, 3), dtype=np.float32),
            pool=np.ones((1, 3), dtype=bool),
            close=np.stack([close[0:3]]), open=np.stack([close[0:3]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.003)

        with pytest.raises(ValueError, match="cost"):
            _replay_continuous_oos(str(tmp_path))

    def test_formal_replay_refuses_legacy_tape(self, tmp_path):
        """§十五-2: a FORMAL replay (the production headline account) must
        refuse a tape that lacks required metadata (calendar_hash here) instead
        of replaying it under legacy semantics — the final headline can never
        silently blend a legacy tape."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 6)]
        stocks = ["000001"]
        close = np.arange(10, 15, dtype=np.float32)
        # fold_000 is complete (all metadata); fold_001 lacks calendar_hash.
        self._write_tape(
            str(tmp_path / "fold_000.npz"),
            stocks=stocks, dates=dates[0:2], price_dates=dates[0:2],
            preds=np.ones((1, 2), dtype=np.float32),
            pool=np.ones((1, 2), dtype=bool),
            close=np.stack([close[0:2]]), open=np.stack([close[0:2]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
            calendar_hash="c1")
        self._write_tape(
            str(tmp_path / "fold_001.npz"),
            stocks=stocks, dates=dates[2:4], price_dates=dates[2:4],
            preds=np.ones((1, 2), dtype=np.float32),
            pool=np.ones((1, 2), dtype=bool),
            close=np.stack([close[2:4]]), open=np.stack([close[2:4]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0)

        # Non-formal: legacy tolerance keeps replaying.
        assert _replay_continuous_oos(str(tmp_path)) is not None
        # Formal: missing calendar_hash on any tape → hard failure.
        with pytest.raises(ValueError, match="calendar_hash"):
            _replay_continuous_oos(str(tmp_path), formal=True)

    def test_formal_replay_accepts_full_metadata(self, tmp_path):
        """§十五-2: a FORMAL replay passes when every tape carries the full
        required metadata + a consistent policy."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 6)]
        stocks = ["000001"]
        close = np.arange(10, 15, dtype=np.float32)
        for fname, sl in [("fold_000.npz", (0, 2)), ("fold_001.npz", (2, 4))]:
            self._write_tape(
                str(tmp_path / fname),
                stocks=stocks, dates=dates[sl[0]:sl[1]],
                price_dates=dates[sl[0]:sl[1]],
                preds=np.ones((1, sl[1] - sl[0]), dtype=np.float32),
                pool=np.ones((1, sl[1] - sl[0]), dtype=bool),
                close=np.stack([close[sl[0]:sl[1]]]),
                open=np.stack([close[sl[0]:sl[1]]]),
                horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
                data_version="test", universe_status_hash="u",
                membership_hash="m", calendar_hash="c1",
                evaluator_version="ev1", price_convention="open_to_open",
                exit_policy="scheduled_horizon_delayed_delist_force_sell",
                strategy_mode="long_top_fraction",
                delist_day=np.array([-1], dtype=int),
                model_source_hash="src1", model_config_hash="cfg1",
                feature_schema_hash="feat1")

        cont = _replay_continuous_oos(str(tmp_path), formal=True)
        assert cont is not None
        assert cont["account"]["final_nav"] > 0

    def test_model_name_filter_isolates_baseline_tapes(self, tmp_path):
        """§十五-3: the continuous replay filters tapes by model.  A baseline
        tape (fold_000_lgbm.npz) sharing an oos_dir must never join the deep
        replay (default scan matches only digit-stem fold_NNN.npz), and
        model_name="lgbm" must replay ONLY the lgbm tapes — never the deep tape
        nor another model's."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 7)]
        close_up = np.arange(10, 16, dtype=np.float32)      # 000001 rises
        close_dn = np.arange(16, 10, -1, dtype=np.float32)  # 000002 falls
        kw_common = dict(horizon=1, seq_len=60, top_fraction=0.5, cost=0.0)
        # Deep tape: 000001 only.
        self._write_tape(
            str(tmp_path / "fold_000.npz"), stocks=["000001"],
            dates=dates[0:2], price_dates=dates[0:2],
            preds=np.ones((1, 2), dtype=np.float32),
            pool=np.ones((1, 2), dtype=bool),
            close=np.stack([close_up[0:2]]), open=np.stack([close_up[0:2]]),
            **kw_common)
        # Two lgbm folds: 000002 only.
        for fname, sl in [("fold_000_lgbm.npz", (0, 2)),
                          ("fold_001_lgbm.npz", (2, 4))]:
            self._write_tape(
                str(tmp_path / fname), stocks=["000002"],
                dates=dates[sl[0]:sl[1]], price_dates=dates[sl[0]:sl[1]],
                preds=np.ones((1, sl[1] - sl[0]), dtype=np.float32),
                pool=np.ones((1, sl[1] - sl[0]), dtype=bool),
                close=np.stack([close_dn[sl[0]:sl[1]]]),
                open=np.stack([close_dn[sl[0]:sl[1]]]),
                **kw_common)

        # Default scan: digit-stem deep tape only — baseline tapes excluded.
        deep = _replay_continuous_oos(str(tmp_path))
        assert deep is not None
        assert deep["stocks"] == ["000001"]
        # model_name filter: ONLY the lgbm tapes — deep tape excluded.
        lgbm = _replay_continuous_oos(str(tmp_path), model_name="lgbm")
        assert lgbm is not None
        assert lgbm["stocks"] == ["000002"]
        assert len(lgbm["account"]["daily"]) == 4
        # A model with no tapes in the dir yields None.
        assert _replay_continuous_oos(str(tmp_path), model_name="ridge") is None

    def test_baseline_formal_replay_under_model_name(self, tmp_path):
        """§十五-3: a baseline's named fold tapes (fold_000_lgbm.npz) replay as
        ONE continuous account under model_name with formal=True when every
        tape carries the full metadata — mirroring what train_baselines_panel.py
        writes for each baseline model."""
        from scripts.production.train_panel import _replay_continuous_oos

        dates = [f"2025-01-{d:02d}" for d in range(1, 6)]
        stocks = ["000001"]
        close = np.arange(10, 15, dtype=np.float32)
        for fname, sl in [("fold_000_lgbm.npz", (0, 2)),
                          ("fold_001_lgbm.npz", (2, 4))]:
            self._write_tape(
                str(tmp_path / fname),
                stocks=stocks, dates=dates[sl[0]:sl[1]],
                price_dates=dates[sl[0]:sl[1]],
                preds=np.ones((1, sl[1] - sl[0]), dtype=np.float32),
                pool=np.ones((1, sl[1] - sl[0]), dtype=bool),
                close=np.stack([close[sl[0]:sl[1]]]),
                open=np.stack([close[sl[0]:sl[1]]]),
                horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
                data_version="test", universe_status_hash="u",
                membership_hash="m", calendar_hash="c1",
                evaluator_version="ev1", price_convention="open_to_open",
                exit_policy="scheduled_horizon_delayed_delist_force_sell",
                strategy_mode="long_top_fraction",
                delist_day=np.array([-1], dtype=int),
                model_source_hash="src1", model_config_hash="cfg1",
                feature_schema_hash="feat1")
        cont = _replay_continuous_oos(
            str(tmp_path), model_name="lgbm", formal=True)
        assert cont is not None
        assert cont["account"]["final_nav"] > 0
        # Without the filter the default scan must NOT pick up these tapes.
        assert _replay_continuous_oos(str(tmp_path), formal=True) is None

    # §十六: the MODEL identity (source / config / feature-schema hashes) must
    # be identical across folds in formal replay; only `weight_hash` (the actual
    # trained parameters) is allowed to differ.  A "first two folds VSN+xLSTM,
    # last three plain LSTM" switch must be refused, not silently blended.

    _MODEL_META = dict(
        data_version="test", universe_status_hash="u", membership_hash="m",
        calendar_hash="c1", evaluator_version="ev1",
        price_convention="open_to_open",
        exit_policy="scheduled_horizon_delayed_delist_force_sell",
        strategy_mode="long_top_fraction",
        delist_day=np.array([-1], dtype=int),
    )

    def _model_meta_tape(self, tmp_path, sl, meta, **overrides):
        """One fold tape over dates[sl] with full formal metadata + model hashes."""
        dates = [f"2025-01-{d:02d}" for d in range(1, 5)]
        close = np.arange(10, 14, dtype=np.float32)
        self._write_tape(
            str(tmp_path / sl[0]),
            stocks=["000001"], dates=dates[sl[1]:sl[2]],
            price_dates=dates[sl[1]:sl[2]],
            preds=np.ones((1, sl[2] - sl[1]), dtype=np.float32),
            pool=np.ones((1, sl[2] - sl[1]), dtype=bool),
            close=np.stack([close[sl[1]:sl[2]]]),
            open=np.stack([close[sl[1]:sl[2]]]),
            horizon=1, seq_len=60, top_fraction=0.5, cost=0.0,
            **meta, **overrides)

    def test_mixed_model_config_hash_fails(self, tmp_path):
        """§十六: folds trained under DIFFERENT hyper-parameter configs (e.g.
        backbone VSN+xLSTM vs plain LSTM) must be refused in formal replay."""
        from scripts.production.train_panel import _replay_continuous_oos

        self._model_meta_tape(tmp_path, ("fold_000.npz", 0, 2),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg-vsn",
                              feature_schema_hash="feat1")
        self._model_meta_tape(tmp_path, ("fold_001.npz", 2, 4),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg-lstm",
                              feature_schema_hash="feat1")
        with pytest.raises(ValueError, match="model_config_hash"):
            _replay_continuous_oos(str(tmp_path), formal=True)

    def test_mixed_model_source_hash_fails(self, tmp_path):
        """§十六: folds produced by DIFFERENT architecture/training code must be
        refused in formal replay (a mid-run code edit is a model switch)."""
        from scripts.production.train_panel import _replay_continuous_oos

        self._model_meta_tape(tmp_path, ("fold_000.npz", 0, 2),
                              dict(self._MODEL_META),
                              model_source_hash="src-a", model_config_hash="cfg1",
                              feature_schema_hash="feat1")
        self._model_meta_tape(tmp_path, ("fold_001.npz", 2, 4),
                              dict(self._MODEL_META),
                              model_source_hash="src-b", model_config_hash="cfg1",
                              feature_schema_hash="feat1")
        with pytest.raises(ValueError, match="model_source_hash"):
            _replay_continuous_oos(str(tmp_path), formal=True)

    def test_mixed_feature_schema_hash_fails(self, tmp_path):
        """§十六: folds built on DIFFERENT feature schemas must be refused in
        formal replay (a feature-column edit between runs changes the input)."""
        from scripts.production.train_panel import _replay_continuous_oos

        self._model_meta_tape(tmp_path, ("fold_000.npz", 0, 2),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg1",
                              feature_schema_hash="feat-a")
        self._model_meta_tape(tmp_path, ("fold_001.npz", 2, 4),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg1",
                              feature_schema_hash="feat-b")
        with pytest.raises(ValueError, match="feature_schema_hash"):
            _replay_continuous_oos(str(tmp_path), formal=True)

    def test_formal_replay_refuses_missing_model_identity_field(self, tmp_path):
        """§十六: a tape missing ANY model-identity key fails FORMAL replay (a
        legacy tape must never be silently blended), while non-formal keeps the
        legacy tolerance."""
        from scripts.production.train_panel import _replay_continuous_oos

        self._model_meta_tape(tmp_path, ("fold_000.npz", 0, 2),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg1",
                              feature_schema_hash="feat1")
        self._model_meta_tape(tmp_path, ("fold_001.npz", 2, 4),
                              dict(self._MODEL_META),
                              model_source_hash="src1", feature_schema_hash="feat1")
        # Non-formal: the missing model_config_hash key is tolerated.
        assert _replay_continuous_oos(str(tmp_path)) is not None
        # Formal: missing required model_config_hash → hard failure.
        with pytest.raises(ValueError, match="model_config_hash"):
            _replay_continuous_oos(str(tmp_path), formal=True)

    def test_weight_hash_may_differ_across_folds(self, tmp_path):
        """§十六: `weight_hash` (the actual trained parameters) is ALLOWED to
        differ per fold — every fold's tape maps to its own fold_XXX_model.pt —
        so a formal replay accepts folds with distinct weight hashes."""
        from scripts.production.train_panel import _replay_continuous_oos

        self._model_meta_tape(tmp_path, ("fold_000.npz", 0, 2),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg1",
                              feature_schema_hash="feat1", weight_hash="w-a")
        self._model_meta_tape(tmp_path, ("fold_001.npz", 2, 4),
                              dict(self._MODEL_META),
                              model_source_hash="src1", model_config_hash="cfg1",
                              feature_schema_hash="feat1", weight_hash="w-b")
        cont = _replay_continuous_oos(str(tmp_path), formal=True)
        assert cont is not None
        assert cont["account"]["final_nav"] > 0
