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
