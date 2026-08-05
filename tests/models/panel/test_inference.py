"""Unit tests for the §十五-1 multiple-testing corrections.

The tests use deterministic return series (no RNG inside the fixtures) so the
expected direction of each statistic is asserted exactly, and random draws
inside the bootstrap-based functions are seeded.
"""

import numpy as np
import pytest

from stoke_ml.models.panel.inference import (
    compute_deflated_sharpe,
    compute_psr,
    deflated_sharpe_benchmark,
    spa_test,
)

# A deterministic, strongly positive, low-volatility daily series (mean 0.002,
# sd ~0.0098) — Sharpe far above zero for any reasonable sample length.
_POS = np.array([0.01] * 600 + [-0.01] * 400, dtype=np.float64)


def test_psr_zero_sharpe_is_half():
    # Alternating ±1 → sample mean exactly 0 → SR̂=0 → PSR(0)=Φ(0)=0.5, exactly.
    rets = np.tile([-1.0, 1.0], 50)
    assert abs(compute_psr(rets, 0.0) - 0.5) < 1e-9


def test_psr_strong_positive_is_near_one():
    assert compute_psr(_POS, 0.0) > 0.999


def test_psr_higher_benchmark_lowers():
    low = compute_psr(_POS, 0.0)
    high = compute_psr(_POS, 2.0)
    assert low > high


def test_psr_degenerate_and_short_returns_nan():
    const = np.full(50, 0.001)
    assert np.isnan(compute_psr(const, 0.0))
    assert np.isnan(compute_psr(np.array([0.001, 0.002]), 0.0))
    # NaN inside the series are dropped, not propagated.
    with_nan = np.concatenate([_POS, [np.nan]])
    assert compute_psr(with_nan, 0.0) > 0.999


def test_dsr_single_trial_nan():
    trial_sharpes = [0.5, 1.0, 1.5]
    assert np.isnan(compute_deflated_sharpe(_POS, 1, trial_sharpes))
    assert np.isnan(compute_deflated_sharpe(_POS, None, trial_sharpes))


def test_dsr_deflation_increases_with_trials():
    trial_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]
    dsr_few = compute_deflated_sharpe(_POS, 2, trial_sharpes)
    dsr_many = compute_deflated_sharpe(_POS, 200, trial_sharpes)
    dsr_enormous = compute_deflated_sharpe(_POS, 1_000_000, trial_sharpes)
    assert dsr_few > dsr_many > dsr_enormous
    # With enough trials the expected-max benchmark overtakes the observed SR.
    assert dsr_enormous < 0.5


def test_dsr_below_psr():
    trial_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]
    psr = compute_psr(_POS, 0.0)
    dsr = compute_deflated_sharpe(_POS, 200, trial_sharpes)
    assert dsr < psr


def test_dsr_bootstrap_fallback_without_trial_sharpes():
    # No observed trial distribution → documented block-bootstrap proxy.
    dsr = compute_deflated_sharpe(_POS, 50, None)
    assert np.isfinite(dsr)
    assert 0.0 < dsr <= 1.0


def test_dsr_constant_returns_nan():
    assert np.isnan(compute_deflated_sharpe(np.full(50, 0.001), 20, [0.5, 1.0]))


def test_deflated_benchmark_properties():
    assert deflated_sharpe_benchmark(2, 1.0) > 0.0
    # More trials → higher expected maximum (deeper deflation).
    assert deflated_sharpe_benchmark(10, 1.0) > deflated_sharpe_benchmark(2, 1.0)
    # More dispersion among trials → higher expected maximum.
    assert deflated_sharpe_benchmark(5, 2.0) > deflated_sharpe_benchmark(5, 0.5)
    # Degenerate inputs → NaN, not a fabricated number.
    assert np.isnan(deflated_sharpe_benchmark(1, 1.0))
    assert np.isnan(deflated_sharpe_benchmark(2, -1.0))


def _noise(seed: int, n: int = 500, scale: float = 0.01) -> np.ndarray:
    return np.random.RandomState(seed).normal(0.0, scale, n)


def test_spa_no_edge():
    strategy = _noise(123)
    benchmark = strategy  # identical → excess is exactly zero.
    out = spa_test(strategy, benchmark, n_boot=500)
    assert out["stat"] == 0.0
    assert out["p_value"] == 1.0
    assert out["n_strategies"] == 1


def test_spa_genuine_edge_rejected_null():
    benchmark = _noise(7)
    strategy = benchmark + 0.01  # constant, clearly positive edge.
    out = spa_test(strategy, benchmark, n_boot=500)
    assert out["p_value"] < 0.05


def test_spa_losing_strategy_not_superior():
    benchmark = _noise(7)
    strategy = benchmark - 0.01  # clearly worse.
    out = spa_test(strategy, benchmark, n_boot=500)
    assert out["p_value"] > 0.3


def test_spa_best_of_many_drives_result():
    benchmark = _noise(7)
    loser = benchmark - 0.02
    winner = benchmark + 0.02
    out_good = spa_test(np.stack([loser, winner]), benchmark, n_boot=500)
    assert out_good["n_strategies"] == 2
    assert out_good["p_value"] < 0.05
    out_bad = spa_test(np.stack([loser, loser - 0.005]), benchmark, n_boot=500)
    assert out_bad["p_value"] > 0.3


def test_spa_length_mismatch_nan():
    out = spa_test(_POS, np.ones(10), n_boot=100)
    assert np.isnan(out["stat"])
    assert np.isnan(out["p_value"])


def test_spa_accepts_horizon_block_floor():
    # §十五-3: `horizon` floors the resampled block length (overlapping
    # holdings); the interface must accept it and still detect a genuine edge.
    benchmark = _noise(7)
    strategy = benchmark + 0.01
    out = spa_test(strategy, benchmark, n_boot=500, horizon=20)
    assert out["p_value"] < 0.05
    assert out["stat"] > 0.0
    # horizon=1 (default) is unchanged from the old behavior.
    out1 = spa_test(strategy, benchmark, n_boot=500, horizon=1)
    assert out1["p_value"] < 0.05
