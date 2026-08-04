import torch
from stoke_ml.models.panel.loss import UncertaintyLoss, AdjMSELoss, PairwiseRankingLoss


class TestUncertaintyLoss:
    def test_output_is_scalar(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        losses = [torch.tensor(0.5), torch.tensor(0.01), torch.tensor(0.02)]
        total = loss_fn(losses)
        assert total.ndim == 0  # scalar

    def test_learnable_params(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        assert loss_fn.log_vars.numel() == 3  # 3 log-variance values

    def test_variance_positive(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        sigma = torch.exp(loss_fn.log_vars)
        assert (sigma > 0).all()

    def test_forward_pass_works(self):
        loss_fn = UncertaintyLoss(num_tasks=3)
        losses = [torch.tensor(0.7, requires_grad=False),
                  torch.tensor(0.05, requires_grad=False),
                  torch.tensor(0.03, requires_grad=False)]
        total = loss_fn(losses)
        total.backward()
        for p in loss_fn.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    def test_inactive_task_excluded(self):
        """Review v4 §八: an inactive task contributes neither loss nor log_var."""
        loss_fn = UncertaintyLoss(num_tasks=3)
        l1 = torch.tensor(0.5)
        l2 = torch.tensor(0.2)
        l3 = torch.tensor(0.1)
        # With task 1 inactive, only tasks 0 and 2 enter the total.
        two_active = loss_fn([l1, torch.zeros(()), l3],
                             task_active_mask=[True, False, True])
        log_vars = torch.clamp(loss_fn.log_vars, -2.0, 10.0)
        expected = (0.5 * (torch.exp(-log_vars[0]) * l1 + log_vars[0])
                    + 0.5 * (torch.exp(-log_vars[2]) * l3 + log_vars[2]))
        assert torch.allclose(two_active, expected, atol=1e-6)
        # Dropping task 1's log_var regularizer strictly lowers the total.
        assert two_active < loss_fn([l1, l2, l3])

    def test_inactive_task_no_grad(self):
        """Review v4 §八: an inactive task's log_var must not receive gradients."""
        loss_fn = UncertaintyLoss(num_tasks=2)
        loss = loss_fn([torch.tensor(0.5), torch.tensor(0.2)],
                       task_active_mask=[True, False])
        loss.backward()
        assert loss_fn.log_vars.grad is not None
        assert loss_fn.log_vars.grad[1] == 0.0  # inactive task untouched
        assert loss_fn.log_vars.grad[0] != 0.0


class TestAdjMSELoss:
    def test_same_sign_uses_gamma_weight(self):
        """Same-sign prediction: loss ≈ gamma * (pred-target)²."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05, -0.03])
        target = torch.tensor([0.02, -0.01])
        loss = loss_fn(pred, target)
        expected_mse = ((pred - target) ** 2).mean()
        assert torch.allclose(loss, 0.1 * expected_mse, atol=1e-6)

    def test_wrong_sign_penalty(self):
        """Wrong-sign prediction: loss ≈ (1+gamma) * (pred-target)² = 11× penalty."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05, -0.03])
        target = torch.tensor([-0.02, 0.01])  # opposite signs
        loss = loss_fn(pred, target)
        expected_mse = ((pred - target) ** 2).mean()
        assert torch.allclose(loss, 1.1 * expected_mse, atol=1e-6)

    def test_wrong_sign_penalty_ratio(self):
        """Wrong sign costs (1+gamma)/gamma = 11× same-sign for equal |error|."""
        loss_fn = AdjMSELoss(gamma=0.1)
        # |pred-target|=0.02 in both cases, squared_error=0.0004
        pred = torch.tensor([0.01])
        same_target = torch.tensor([0.03])   # 0.01*0.03>0 → same sign → γ=0.1
        wrong_target = torch.tensor([-0.01])  # 0.01*(-0.01)<0 → wrong sign → 1.1
        loss_same = loss_fn(pred, same_target)
        loss_wrong = loss_fn(pred, wrong_target)
        assert torch.allclose(loss_wrong / loss_same, torch.tensor(11.0), atol=1e-4)

    def test_zero_pred_boundary(self):
        """pred=0, target≠0: 0*target=0 ≥ 0 → same-sign path (gamma weight)."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.0])
        target = torch.tensor([0.05])
        loss = loss_fn(pred, target)
        expected = torch.tensor(0.1 * (0.05 ** 2))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_zero_target_boundary(self):
        """pred≠0, target=0: pred*0=0 ≥ 0 → same-sign path (gamma weight)."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05])
        target = torch.tensor([0.0])
        loss = loss_fn(pred, target)
        expected = torch.tensor(0.1 * (0.05 ** 2))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_backward_works(self):
        """Gradients flow through both same-sign and wrong-sign paths."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05, -0.03], requires_grad=True)
        target = torch.tensor([-0.02, 0.01])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_reduction_none(self):
        """reduction='none' returns elementwise losses (review v4 §九 val accumulation)."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05, -0.03])
        target = torch.tensor([0.02, -0.01])
        elem = loss_fn(pred, target, reduction="none")
        assert elem.shape == pred.shape
        assert torch.allclose(elem.mean(), loss_fn(pred, target), atol=1e-6)


class TestPairwiseRankingLoss:
    def test_per_date_spread_normalization(self):
        """Review v4 §十: spread penalty is computed per date, not over the mixed batch.

        Date 1 has 10× the target dispersion of date 0.  Whole-batch
        normalization would blend them; per-date must penalize date 1's
        collapsed predictions by its own target scale.
        """
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=1.0,
                                      spread_target=1.0, spread_weight=1.0)
        date_idx = torch.tensor([0, 0, 0, 1, 1, 1])
        target = torch.tensor([-0.1, 0.0, 0.1, -1.0, 0.0, 1.0], dtype=torch.float32)
        pred = torch.full((6,), 0.5, dtype=torch.float32)  # constant → hinge 0
        mask = torch.ones(6, dtype=torch.float32)
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        # Date 1 is penalised by its own std (1.0), date 0 by its own (0.1).
        target_std_0 = target[:3].std()
        target_std_1 = target[3:].std()
        expected = (target_std_0 * 3 + target_std_1 * 3) / 6
        assert torch.allclose(loss, expected, atol=1e-5)
        assert stats[0]["n_dates"] == 2
        assert stats[0]["stocks_per_date"] == [3, 3]
        assert stats[0]["n_pairs"] == 6

    def test_single_stock_date_skipped(self):
        """A date with a single valid stock contributes no pairs."""
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=1.0)
        date_idx = torch.tensor([0, 0, 0, 1])
        target = torch.tensor([-0.1, 0.0, 0.1, 0.5], dtype=torch.float32)
        pred = torch.tensor([0.3, 0.2, 0.1, 0.4], dtype=torch.float32)
        mask = torch.ones(4, dtype=torch.float32)
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert not torch.isnan(loss)
        assert stats[0]["stocks_per_date"] == [3]
        assert stats[0]["n_pairs"] == 3

    def test_perfect_ordering_low_loss(self):
        """Perfect within-date ordering → hinge 0; only spread penalty remains."""
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=1.0,
                                      spread_target=1.0, spread_weight=0.0)
        date_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        target = torch.tensor([-0.2, -0.1, 0.1, 0.2, -2.0, -1.0, 1.0, 2.0],
                              dtype=torch.float32)
        pred = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0],
                            dtype=torch.float32)
        mask = torch.ones(8, dtype=torch.float32)
        loss = loss_fn(pred, target, mask, date_idx)
        assert torch.allclose(loss, torch.zeros(()), atol=1e-5)
