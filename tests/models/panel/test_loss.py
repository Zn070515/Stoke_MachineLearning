import inspect

import numpy as np
import torch
from stoke_ml.models.panel.loss import (
    UncertaintyLoss, FixedTaskWeights, AdjMSELoss, PairwiseRankingLoss,
)


def _make_rank_panel(n_stocks=8, n_timesteps=90, seq_len=60, horizon=5):
    """Small priced panel for the date-centric rank-loss integration tests.

    Mirrors tests/models/panel/test_train.py::_make_masked_panel: real price
    paths, per-task masks, and global date_indices — the exact shapes the
    production ``PanelDataset`` consumes.
    """
    rng = np.random.RandomState(0)
    close = np.full((n_stocks, n_timesteps), 10.0, dtype=np.float32)
    shocks = rng.randn(n_stocks, n_timesteps - 1).astype(np.float32) * 0.02
    close[:, 1:] = close[:, :1] * np.exp(np.cumsum(shocks, axis=1))
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0]
    gaps = 1.0 + rng.randn(n_stocks, n_timesteps - 1).astype(np.float32) * 0.005
    open_[:, 1:] = close[:, :-1] * gaps

    obs = np.ones((n_stocks, n_timesteps), dtype=bool)
    entry = np.ones((n_stocks, n_timesteps), dtype=bool)
    decision = np.ones((n_stocks, n_timesteps), dtype=bool)
    decision[:, 0] = False  # no close[t-1]

    ret_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    vol_tgt = np.zeros((n_stocks, n_timesteps), dtype=bool)
    for t in range(n_timesteps - horizon):
        have_exit = np.isfinite(open_[:, t + horizon])
        ret_tgt[:, t] = have_exit
        if t + horizon < n_timesteps:
            fwd = close[:, t + 1:t + horizon + 1]
            vol_tgt[:, t] = have_exit & np.isfinite(fwd).all(axis=1)

    y_ret = np.zeros((n_stocks, n_timesteps), dtype=np.float32)
    for t in range(n_timesteps - horizon):
        y_ret[:, t] = open_[:, t + horizon] / open_[:, t] - 1.0

    y_dir = np.full((n_stocks, n_timesteps), -100, dtype=np.int64)
    sel = y_ret[ret_tgt]
    y_dir[ret_tgt] = np.where(sel > 0.001, 2, np.where(sel < -0.001, 0, 1))

    y_vol = np.zeros((n_stocks, n_timesteps), dtype=np.float32)

    static = rng.randn(n_stocks, n_timesteps, 4).astype(np.float32)
    pk = rng.randn(n_stocks, n_timesteps, 12).astype(np.float32)
    po = rng.randn(n_stocks, n_timesteps, 6).astype(np.float32)
    date_indices = np.tile(np.arange(n_timesteps, dtype=np.int64)[None, :],
                           (n_stocks, 1))

    return {
        "static_features": static,
        "past_known": pk,
        "past_observed": po,
        "y_direction": y_dir,
        "y_return": y_ret,
        "y_volatility": y_vol,
        "observation_mask": obs,
        "entry_eligible_mask": entry,
        "decision_eligible_mask": decision,
        "return_target_mask": ret_tgt,
        "vol_target_mask": vol_tgt,
        "date_indices": date_indices,
        "close_price": close,
        "open_price": open_,
    }


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
        """An inactive task contributes neither loss nor log_var."""
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
        """An inactive task's log_var must not receive gradients."""
        loss_fn = UncertaintyLoss(num_tasks=2)
        loss = loss_fn([torch.tensor(0.5), torch.tensor(0.2)],
                       task_active_mask=[True, False])
        loss.backward()
        assert loss_fn.log_vars.grad is not None
        assert loss_fn.log_vars.grad[1] == 0.0  # inactive task untouched
        assert loss_fn.log_vars.grad[0] != 0.0


class TestFixedTaskWeights:
    """§十一.3: the UncertaintyLoss ablation — equal-weight multi-task loss.

    Carries no learnable parameters (nothing in the optimizer's loss group)
    and matches UncertaintyLoss's forward(losses, task_active_mask) signature
    so train.py swaps one for the other without branching.
    """

    def test_equal_weight_mean_over_active_tasks(self):
        loss_fn = FixedTaskWeights(num_tasks=3)
        total = loss_fn([torch.tensor(0.4), torch.tensor(0.1), torch.tensor(0.1)],
                        task_active_mask=[True, True, False])
        assert torch.allclose(total, torch.tensor(0.25), atol=1e-6)  # (0.4+0.1)/2

    def test_scale_independent_of_task_count(self):
        """Mean over active tasks, not their sum — enabling/disabling a task
        must not rescale the combined loss."""
        loss_fn = FixedTaskWeights(num_tasks=2)
        total2 = loss_fn([torch.tensor(0.4), torch.tensor(0.2)],
                         task_active_mask=[True, True])
        total1 = loss_fn([torch.tensor(0.4), torch.tensor(0.2)],
                         task_active_mask=[True, False])
        assert torch.allclose(total2, torch.tensor(0.3), atol=1e-6)
        assert torch.allclose(total1, torch.tensor(0.4), atol=1e-6)

    def test_all_inactive_returns_zero(self):
        loss_fn = FixedTaskWeights(num_tasks=3)
        total = loss_fn([torch.tensor(0.4), torch.tensor(0.1), torch.tensor(0.1)],
                        task_active_mask=[False, False, False])
        assert torch.allclose(total, torch.zeros(()), atol=1e-6)

    def test_no_learnable_params(self):
        loss_fn = FixedTaskWeights(num_tasks=3)
        assert list(loss_fn.parameters()) == []

    def test_forward_signature_matches_uncertainty(self):
        sig_a = inspect.signature(FixedTaskWeights.forward)
        sig_b = inspect.signature(UncertaintyLoss.forward)
        assert list(sig_a.parameters) == list(sig_b.parameters)


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
        """reduction='none' returns elementwise losses."""
        loss_fn = AdjMSELoss(gamma=0.1)
        pred = torch.tensor([0.05, -0.03])
        target = torch.tensor([0.02, -0.01])
        elem = loss_fn(pred, target, reduction="none")
        assert elem.shape == pred.shape
        assert torch.allclose(elem.mean(), loss_fn(pred, target), atol=1e-6)


class TestPairwiseRankingLoss:
    def test_per_date_spread_normalization(self):
        """Spread penalty is computed per date, not over the mixed batch.

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


class TestPairwiseRankingLossDateCentric:
    """§七/§十六 cross-sectional semantics of PairwiseRankingLoss on the
    date-centric contract.  With the production DataLoader (batch_size=1 +
    DateSampler) a batch IS one date's complete (or cap-sampled) cross-section;
    batch_size>1 concatenates dates via panel_collate.  These tests pin the
    pair-counting and per-date normalization either way."""

    def test_complete_cross_section_single_date(self):
        """One batch with N stocks on a single date produces exactly C(N,2)
        pairs — the full cross-section, none lost to batch boundaries."""
        loss_fn = PairwiseRankingLoss()
        N = 7
        date_idx = torch.zeros(N, dtype=torch.long)
        target = torch.linspace(-1.0, 1.0, N)
        pred = torch.linspace(-1.0, 1.0, N)  # perfect ordering → hinge 0
        mask = torch.ones(N, dtype=torch.float32)
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert stats[0]["n_dates"] == 1
        assert stats[0]["stocks_per_date"] == [N]
        assert stats[0]["n_pairs"] == N * (N - 1) // 2
        assert torch.isfinite(loss)

    def test_mixed_date_grouping_exact_pair_total(self):
        """2 dates × M stocks in one batch → n_pairs == 2 × C(M,2); pairs only
        ever form within a date (no cross-date pairs)."""
        loss_fn = PairwiseRankingLoss()
        M = 4
        date_idx = torch.tensor([0] * M + [1] * M)
        target = torch.tensor([-0.2, -0.1, 0.1, 0.2, -2.0, -1.0, 1.0, 2.0],
                              dtype=torch.float32)
        pred = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0],
                            dtype=torch.float32)
        mask = torch.ones(2 * M, dtype=torch.float32)
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert stats[0]["n_dates"] == 2
        assert stats[0]["stocks_per_date"] == [M, M]
        assert stats[0]["n_pairs"] == 2 * (M * (M - 1) // 2)
        assert torch.isfinite(loss)

    def test_per_date_scale_invariance_and_no_cross_date_pairs(self):
        """Scale-invariance per date: multiplying ONE date's predictions by a
        positive constant leaves the hinge unchanged (predictions are
        normalized by their own date's std).  Also pins n_pairs == 2×C(3,2)
        with no cross-date pairs."""
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=1.0,
                                      spread_target=1.0, spread_weight=0.0)
        date_idx = torch.tensor([0, 0, 0, 1, 1, 1])
        target = torch.tensor([-0.1, 0.0, 0.1, -1.0, 0.0, 1.0],
                              dtype=torch.float32)
        pred = torch.tensor([0.2, 0.5, 0.8, 1.0, 2.0, 3.0], dtype=torch.float32)
        mask = torch.ones(6, dtype=torch.float32)
        stats: list[dict] = []
        base = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert stats[0]["n_dates"] == 2
        assert stats[0]["stocks_per_date"] == [3, 3]
        assert stats[0]["n_pairs"] == 6  # 2 × C(3,2), no cross-date pairs

        pred_scaled = pred.clone()
        pred_scaled[3:] = pred_scaled[3:] * 10.0  # scale date 1 only
        scaled = loss_fn(pred_scaled, target, mask, date_idx)
        assert torch.allclose(scaled, base, atol=1e-6)

    def test_date_with_two_stocks_forms_one_pair(self):
        """Boundary: a date with exactly 2 ret-valid stocks forms exactly 1 pair."""
        loss_fn = PairwiseRankingLoss()
        date_idx = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([-0.1, 0.1, -0.2, 0.2], dtype=torch.float32)
        pred = torch.tensor([0.3, 0.1, 0.4, 0.2], dtype=torch.float32)
        mask = torch.ones(4, dtype=torch.float32)
        stats: list[dict] = []
        loss_fn(pred, target, mask, date_idx, stats=stats)
        assert stats[0]["stocks_per_date"] == [2, 2]
        assert stats[0]["n_pairs"] == 2  # 1 pair per date

    def test_small_date_group_reported_honestly(self):
        """A date with a single ret-valid stock appears in n_dates but NOT in
        stocks_per_date / n_pairs — the stats report it honestly."""
        loss_fn = PairwiseRankingLoss()
        date_idx = torch.tensor([0, 0, 0, 1])
        target = torch.tensor([-0.1, 0.0, 0.1, 0.5], dtype=torch.float32)
        pred = torch.tensor([0.3, 0.2, 0.1, 0.4], dtype=torch.float32)
        mask = torch.ones(4, dtype=torch.float32)
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert not torch.isnan(loss)
        assert stats[0]["n_dates"] == 2
        assert stats[0]["stocks_per_date"] == [3]
        assert stats[0]["n_pairs"] == 3

    def test_all_dates_below_two_ret_valid_zero_pairs_no_crash(self):
        """A batch where NO date has >= 2 ret-valid stocks returns 0 loss and
        does not crash; no stats entry is appended (there are no pairs)."""
        loss_fn = PairwiseRankingLoss()
        date_idx = torch.tensor([0, 0, 1])
        target = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        pred = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0, 1.0])  # date 0: 1 valid, date 1: 1 valid
        stats: list[dict] = []
        loss = loss_fn(pred, target, mask, date_idx, stats=stats)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
        assert stats == []


class TestPairwiseRankingLossPanelDatasetIntegration:
    """Feed a REAL PanelDataset window (the date-centric 11-tuple) into
    PairwiseRankingLoss — proves the loss consumes the contract end-to-end."""

    SEQ_LEN = 60

    def _first_big_window(self, ds):
        counts = ds.valid_mask.sum(dim=0)
        return int(torch.where(counts >= 3)[0][0])

    def test_capped_cross_section_forms_pairs_over_sample(self):
        from stoke_ml.models.panel.dataset import PanelDataset
        data = _make_rank_panel(n_stocks=8, n_timesteps=90, seq_len=self.SEQ_LEN,
                                horizon=5)
        cap = 4
        ds = PanelDataset(data, seq_len=self.SEQ_LEN, min_history=50,
                          max_stocks_per_date=cap, training=True)
        window = self._first_big_window(ds)
        (static, pk, po, y_dir, y_ret, y_vol,
         date_idx, dir_mask, ret_mask, vol_mask, stock_indices) = ds[window]
        assert ret_mask.numel() <= cap, (
            f"capped batch exceeded cap: {ret_mask.numel()} > {cap}")
        n_ret = int(ret_mask.sum().item())
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=0.1,
                                      spread_target=1.0, spread_weight=0.5)
        stats: list[dict] = []
        # pred uses the 11-tuple's return target as a stand-in prediction — the
        # point is the loss consumes the date-centric shapes/masks correctly.
        loss = loss_fn(y_ret, y_ret, ret_mask.float(), date_idx, stats=stats)
        assert torch.isfinite(loss)
        assert stats[0]["n_dates"] == 1
        assert stats[0]["n_pairs"] == n_ret * (n_ret - 1) // 2
        assert stats[0]["stocks_per_date"] == [n_ret]

    def test_uncapped_complete_cross_section(self):
        from stoke_ml.models.panel.dataset import PanelDataset
        data = _make_rank_panel(n_stocks=8, n_timesteps=90, seq_len=self.SEQ_LEN,
                                horizon=5)
        ds = PanelDataset(data, seq_len=self.SEQ_LEN, min_history=50,
                          max_stocks_per_date=None, training=True)
        window = self._first_big_window(ds)
        (static, pk, po, y_dir, y_ret, y_vol,
         date_idx, dir_mask, ret_mask, vol_mask, stock_indices) = ds[window]
        n_valid = int(ds.valid_mask[:, window].sum().item())
        assert stock_indices.numel() == n_valid  # complete cross-section
        n_ret = int(ret_mask.sum().item())
        loss_fn = PairwiseRankingLoss(margin=0.0, tau=0.1,
                                      spread_target=1.0, spread_weight=0.5)
        stats: list[dict] = []
        loss = loss_fn(y_ret, y_ret, ret_mask.float(), date_idx, stats=stats)
        assert torch.isfinite(loss)
        assert stats[0]["n_dates"] == 1
        assert stats[0]["n_pairs"] == n_ret * (n_ret - 1) // 2


class TestRankPoolStatsExpectedPairs:
    """§七/§十六 honest pair-coverage denominator (train._rank_pool_stats).

    The old C(min(entry, cap), 2) over-counted the achievable pair space
    whenever ret-invalid stocks sat in the entry pool (return targets require a
    realized exit), so pair_coverage could structurally never reach 1.0 — the
    code-review finding this sub-task fixes."""

    def test_expected_pairs_no_cap_or_superset_cap(self):
        from stoke_ml.models.panel.train import _expected_pairs
        assert _expected_pairs(10, 10, None) == 45.0  # C(10,2)
        assert _expected_pairs(10, 10, 20) == 45.0    # cap >= n → full cross-section
        assert _expected_pairs(10, 10, 4) == 6.0      # sample 4 of 10, all ret-valid

    def test_expected_pairs_degenerate(self):
        from stoke_ml.models.panel.train import _expected_pairs
        assert _expected_pairs(10, 0, 4) == 0.0   # no ret-valid → no pairs
        assert _expected_pairs(4, 1, 4) == 0.0    # <2 ret-valid → no pairs
        assert _expected_pairs(0, 0, 4) == 0.0    # empty date

    def test_ret_invalid_stocks_shrink_achievable_pair_space(self):
        """The reviewer's bias, exactly: with 10 entry-valid but only 5
        ret-valid, capping the sample at 4 yields FEWER expected pairs than
        C(min(10,4),2)=6 — the old denominator could not be reached."""
        from stoke_ml.models.panel.train import _expected_pairs
        naive = 4 * 3 // 2
        expected = _expected_pairs(10, 5, 4)
        assert 0.0 < expected < naive
        # Exact hypergeometric expectation E[C(X,2)], X ~ Hypergeo(N=10,K=5,n=4).
        e_x = 4 * 5 / 10
        var_x = e_x * (1 - 5 / 10) * (10 - 4) / (10 - 1)
        e_x2 = var_x + e_x * e_x
        assert abs(expected - (e_x2 - e_x) / 2) < 1e-9

    def test_rank_pool_stats_no_cap_is_exact(self):
        from stoke_ml.models.panel.train import _rank_pool_stats
        from stoke_ml.models.panel.dataset import PanelDataset
        data = _make_rank_panel(n_stocks=8, n_timesteps=90, seq_len=60, horizon=5)
        ds = PanelDataset(data, seq_len=60, min_history=50,
                          max_stocks_per_date=None, training=True)
        stocks_per_date, expected = _rank_pool_stats(ds)
        ret_counts = (ds.valid_mask & ds.ret_target[:, ds.seq_len:]).sum(dim=0)
        ret_counts = ret_counts.tolist()
        assert stocks_per_date == [n for n in ret_counts if n > 0]
        exact = sum(n * (n - 1) // 2 for n in ret_counts if n > 0)
        assert abs(expected - exact) < 1e-6

    def test_rank_pool_stats_capped_below_naive(self):
        """With a cap the expectation never exceeds the naive C(min(entry,
        cap), 2) total the old metric used."""
        from stoke_ml.models.panel.train import _rank_pool_stats
        from stoke_ml.models.panel.dataset import PanelDataset
        data = _make_rank_panel(n_stocks=8, n_timesteps=90, seq_len=60, horizon=5)
        ds = PanelDataset(data, seq_len=60, min_history=50,
                          max_stocks_per_date=3, training=True)
        _, expected = _rank_pool_stats(ds)
        entry_counts = ds.valid_mask.sum(dim=0).tolist()
        naive = sum(min(e, 3) * (min(e, 3) - 1) // 2
                    for e in entry_counts if e > 0)
        assert expected <= naive
