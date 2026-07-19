import torch
from stoke_ml.models.panel.loss import UncertaintyLoss, AdjMSELoss


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
