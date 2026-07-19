import torch
from stoke_ml.models.panel.evaluate import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_profit_factor,
    compute_equity_curve,
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
