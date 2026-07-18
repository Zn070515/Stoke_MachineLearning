import torch
from stoke_ml.models.panel.loss import UncertaintyLoss


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
