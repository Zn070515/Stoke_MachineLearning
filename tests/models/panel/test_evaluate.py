import numpy as np
import torch
import torch.nn as nn

from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.evaluate import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_profit_factor,
    compute_equity_curve,
    compute_bootstrap_sharpe_ci,
    compute_ic_summary,
    evaluate_portfolio,
)


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
        # No downside → infinite Sortino (all returns above target)
        assert sortino == float("inf")


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


class TestProfitFactor:
    def test_profitable(self):
        ret = torch.tensor([0.02, -0.01, 0.03, -0.005, 0.01])
        pf = compute_profit_factor(ret)
        # profits = 0.02+0.03+0.01=0.06, losses = 0.01+0.005=0.015 → PF=4.0
        assert pf > 1.0

    def test_losing(self):
        ret = torch.tensor([-0.02, 0.01, -0.03, -0.01])
        pf = compute_profit_factor(ret)
        # profits = 0.01, losses = 0.06 → PF ≈ 0.167
        assert pf < 1.0

    def test_all_positive(self):
        ret = torch.tensor([0.01, 0.02, 0.005])
        pf = compute_profit_factor(ret)
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
    """Review §十三: iid resampling of autocorrelated returns understates the
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


class TestICSummaryNeweyWest:
    """Review §十三: the plain mean/std IR ignores serial correlation in the
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
    """Review §五 #4: every prediction identical → the evaluation pipeline must
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
    """Review §五 #3 proxy: predictions independent of the inputs, standing in
    for a model that learned nothing from globally-shuffled labels.  If the
    evaluation pipeline is clean, this must show IC≈0 and no alpha."""

    def forward(self, static, pk, po):
        b = pk.shape[0]
        return (
            torch.zeros(b, 3, dtype=torch.float32),
            torch.randn(b, 1, dtype=torch.float32) * 0.01,
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
    """Review §二 survivor bias: the candidate pool must be ENTRY-ELIGIBLE
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
        assert m["ew_sharpe"] > 0, f"raw_returns leaked into eval: {m['ew_sharpe']}"


class TestEvaluateAllHorizonPhases:
    """Review §五: every entry phase (offset 0..horizon-1) is evaluated and
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
    """Review §五 anti-cheat tests #3 (label shuffle) & #4 (constant preds).

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
        or quintile alpha (selection degenerates to stock order = random)."""
        lss, q5s = [], []
        for seed in range(3):
            m = self._eval(_ConstantReturnModel(), seed=seed)
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert abs(np.mean(lss)) < 1.0, f"constant preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"constant preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"

    def test_random_predictions_yield_near_zero_ic(self):
        """#3: label-shuffle proxy — noise predictions must show |IC|≈0."""
        ics = [self._eval(_RandomReturnModel(), s)["ic_mean"] for s in range(3)]
        assert abs(np.mean(ics)) < 0.03, f"random preds mean IC={np.mean(ics):+.4f}"

    def test_random_predictions_produce_no_systematic_alpha(self):
        """#3: noise predictions must not generate portfolio alpha."""
        lss, q5s = [], []
        for seed in range(3):
            m = self._eval(_RandomReturnModel(), seed=seed)
            lss.append(m["ls_sharpe"])
            q5s.append(m["q5mq1_ret"])
        assert abs(np.mean(lss)) < 1.0, f"random preds long-short={np.mean(lss):+.2f}"
        assert abs(np.mean(q5s)) < 0.01, f"random preds q5-q1={np.mean(q5s) * 1e4:+.0f}bp"
