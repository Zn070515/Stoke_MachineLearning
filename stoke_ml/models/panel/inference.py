"""Statistical inference for strategy evaluation — multiple-testing corrections.

Implements the three §十五-1 adjustments on top of the existing per-day IC /
Newey-West / bootstrap machinery in evaluate.py:

* **Probabilistic Sharpe Ratio (PSR)** — Bailey & López de Prado (2012).
  Is an observed Sharpe statistically distinguishable from a benchmark when the
  returns are NOT normally distributed?  Skewness and kurtosis widen or narrow
  the Sharpe's standard error, so a Sharpe that looks high under a normal
  assumption can be noise once fat tails are accounted for.

* **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014).
  PSR evaluated against the *expected maximum Sharpe of N trials*.  Iterating
  over many models / features / horizons / losses and reporting the best
  inflates the apparent Sharpe (data snooping).  DSR deflates the benchmark by
  the multiplicity N and the cross-trial dispersion of the tried Sharpes.

* **SPA / Reality Check (RC)** — Hansen (2005), a recentered, more powerful
  variant of White's (2000) Reality Check.  Is the *best of K* strategies
  genuinely better than a benchmark, after accounting for the fact that the
  best one was picked?  A block bootstrap over the (strategy − benchmark)
  differences produces a p-value for the max-mean statistic, recentered by the
  positive part of each sample mean so the null (no edge) is actually imposed
  — White's un-recentered RC re-centers the bootstrap at the observed mean and
  is severely underpowered for a genuine edge.

All Sharpe math is on the *per-period* (daily) scale internally — skewness /
kurtosis are computed from the same daily returns the Sharpe comes from, so the
standard error formula stays dimensionally consistent.  The benchmark Sharpes
are accepted annualized (matching the codebase convention, `√(252/horizon)`)
and converted back to per-period inside.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, skew, kurtosis

# Euler–Mascheroni constant — appears in the expected-maximum-SR formula.
_EULER_GAMMA = 0.5772156649015328606


def _as_1d(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _moments(
    returns: np.ndarray,
) -> tuple[float, float, float, int] | None:
    """Per-period Sharpe, sample skewness, sample raw kurtosis, and n.

    `ku` is the RAW (Pearson) kurtosis — 3 for a normal — because the PSR/DSR
    standard-error formula `(κ−1)/4` recovers the classical `1 + SR²/2` normal
    variance only when κ is raw kurtosis, not excess.
    """
    n = int(returns.size)
    if n < 3:
        return None
    s = float(returns.std(ddof=1))
    if s < 1e-12:
        return None
    sr = float(returns.mean()) / s
    sk = float(skew(returns, bias=False))
    ku = float(kurtosis(returns, fisher=False, bias=False))
    return sr, sk, ku, n


def _sharpe_se2(sr_daily: float, sk: float, ku: float, n: int) -> float:
    """Variance of the per-period Sharpe estimator (Bailey & LP 2012).

    Falls back to the iid-normal baseline `(1 + SR²/2)/(n−1)` if the
    skew/kurtosis estimate goes degenerate (negative variance), rather than
    manufacturing a confidence bound from noise.
    """
    se2 = (1.0 - sk * sr_daily + ((ku - 1.0) / 4.0) * sr_daily ** 2) / (n - 1.0)
    if se2 <= 0.0:
        se2 = (1.0 + sr_daily ** 2 / 2.0) / (n - 1.0)
    return se2


def compute_psr(
    returns: np.ndarray,
    sr_benchmark_ann: float = 0.0,
    horizon: int = 1,
    n_obs: int | None = None,
) -> float:
    """Probabilistic Sharpe Ratio: P( SR̂ > SR* ) under non-normal returns.

    returns: daily (or per-period) return series.
    sr_benchmark_ann: benchmark Sharpe, annualized (default 0 → "is the Sharpe
        positive at all?").
    horizon: annualization factor used for the benchmark conversion.
    n_obs: override the sample count (e.g. to use the effective sample size
        after overlap adjustment) — defaults to len(returns).
    """
    arr = _as_1d(returns)
    mom = _moments(arr)
    if mom is None:
        return float("nan")
    sr_daily, sk, ku, n = mom
    if n_obs is not None:
        n = int(n_obs)
        if n < 3:
            return float("nan")
    sr_star_daily = float(sr_benchmark_ann) / math.sqrt(252.0 / horizon)
    se2 = _sharpe_se2(sr_daily, sk, ku, n)
    z = (sr_daily - sr_star_daily) / math.sqrt(se2)
    return float(norm.cdf(z))


def deflated_sharpe_benchmark(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe of N iid trials under the null (Bailey & LP 2014).

    `sr_variance` is the cross-trial variance of the (annualized) Sharpe
    ratios, and the returned benchmark is in the same annualized units.
    """
    if n_trials is None or n_trials < 2 or not np.isfinite(sr_variance) or sr_variance < 0.0:
        return float("nan")
    sqrt_var = math.sqrt(float(sr_variance))
    a = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
    b = _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sqrt_var * (a + b)


def _block_bootstrap_sharpes(
    returns: np.ndarray,
    horizon: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap distribution of the ANNUALIZED Sharpe."""
    n = int(returns.size)
    if n < 5:
        return np.array([float("nan")], dtype=np.float64)
    L = max(2, int(np.ceil(n ** (1 / 3))), int(horizon))
    L = min(L, n)
    n_blocks = int(np.ceil(n / L))
    rng = np.random.RandomState(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.randint(0, n - L + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + L] for s in starts])[:n]
        m = sample.mean()
        s = sample.std(ddof=1)
        out[i] = (m / s) * math.sqrt(252.0 / horizon) if s > 1e-8 else 0.0
    return out


def compute_deflated_sharpe(
    returns: np.ndarray,
    n_trials: int | None,
    trial_sharpes: np.ndarray | list[float] | None = None,
    horizon: int = 1,
    n_obs: int | None = None,
    n_boot: int = 500,
    seed: int = 42,
) -> float:
    """Deflated Sharpe Ratio: P( SR̂ > SR_0 ), SR_0 the expected max of N trials.

    n_trials: number of research trials iterated to arrive at this strategy
        (registry count).  <2 → NaN (deflation is undefined for a single trial).
    trial_sharpes: the observed Sharpes of the tried strategies, whose
        cross-trial variance estimates V(SR_n).  When fewer than two finite
        values are supplied, the block-bootstrap variance of `returns` is used
        as a documented proxy for the trial dispersion.
    """
    arr = _as_1d(returns)
    mom = _moments(arr)
    if mom is None or n_trials is None or int(n_trials) < 2:
        return float("nan")
    sr_daily, sk, ku, n = mom
    if n_obs is not None:
        n = int(n_obs)
        if n < 3:
            return float("nan")
    sr_ann = sr_daily * math.sqrt(252.0 / horizon)
    vals = [float(x) for x in (trial_sharpes or []) if np.isfinite(x)]
    if len(vals) >= 2:
        sr_var = float(np.var(np.array(vals), ddof=1))
    else:
        sr_var = float(np.var(_block_bootstrap_sharpes(arr, horizon, n_boot, seed)))
    sr0_ann = deflated_sharpe_benchmark(int(n_trials), sr_var)
    if not np.isfinite(sr0_ann):
        return float("nan")
    sr0_daily = sr0_ann / math.sqrt(252.0 / horizon)
    se2 = _sharpe_se2(sr_daily, sk, ku, n)
    z = (sr_daily - sr0_daily) / math.sqrt(se2)
    return float(norm.cdf(z))


def spa_test(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_boot: int = 2000,
    block_len: int | None = None,
    horizon: int = 1,
    seed: int = 42,
) -> dict:
    """Hansen's (2005) SPA p-value for the best-of-K strategies vs a benchmark.

    The K strategies are evaluated on the SAME calendar days, so the excess
    (strategy − benchmark) rows are resampled as BLOCKS — all K together — to
    preserve both cross-sectional correlation and time-series autocorrelation.

    Observed statistic `stat = max_k √T · mean_k`; each bootstrap draw computes
    the same max on a resampled block, recentered by the POSITIVE part of the
    sample mean (`p_pos`) so that a genuinely positive edge is compared against
    the null rather than against a bootstrap that re-absorbs the edge (White's
    un-recentered RC degenerates to p≈0.5 for any constant edge).  A low
    p-value says the best strategy still beats the benchmark after paying for
    the selection.

    `horizon` floors the block length: with horizon>1 the adjacent daily
    returns share overlapping holdings, so the resampled block must be at least
    as long as that overlap (§十五-3).
    """
    S = np.asarray(strategy_returns, dtype=np.float64)
    B = np.asarray(benchmark_returns, dtype=np.float64).ravel()
    if S.ndim == 1:
        S = S[None, :]
    n_strat, T = S.shape
    if T != B.size or T < 5:
        return {"stat": float("nan"), "p_value": float("nan"),
                "n_strategies": int(n_strat)}
    excess = S - B[None, :]
    means = np.nanmean(excess, axis=1)
    stat = float(np.max(np.sqrt(T) * means))
    p_pos = np.where(means > 0.0, means, 0.0)
    L = block_len or max(2, int(np.ceil(T ** (1 / 3))), int(horizon))
    L = min(L, T)
    n_blocks = int(np.ceil(T / L))
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_boot):
        starts = rng.randint(0, T - L + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:T]
        boot_means = np.nanmean(excess[:, idx], axis=1)
        boot_stat = float(np.max(np.sqrt(T) * (boot_means - p_pos)))
        if boot_stat >= stat:
            count += 1
    return {"stat": stat, "p_value": count / n_boot, "n_strategies": int(n_strat)}
