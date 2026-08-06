"""Panel evaluator — metrics / IC / sleeve-account / portfolio evaluation (§二十一).

Split from a single 1748-line module into sibling ``evaluate_*`` modules by
responsibility.  This module is the thin re-export layer: every public and
private name the pre-split ``stoke_ml.models.panel.evaluate`` exposed is
re-exported here, so ``from stoke_ml.models.panel.evaluate import X`` keeps
working for every pre-split ``X``.

  * ``evaluate_metrics``   — pure risk/return metrics (Sharpe / Sortino / ...)
  * ``evaluate_ic``        — per-day Spearman IC, clean-IC, candidate pool
  * ``evaluate_account``   — chronological sleeve-account simulation + ledger
  * ``evaluate_portfolio`` — the portfolio module (``evaluate_portfolio`` /
    ``evaluate_sharpe`` + helpers)
"""
# ruff: noqa: F401  (pure re-export layer — every name below is a re-export)
import logging

from stoke_ml.models.panel.evaluate_metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_daily_return_profit_factor,
    compute_equity_curve,
    compute_bootstrap_sharpe_ci,
)
from stoke_ml.models.panel.evaluate_ic import (
    _compute_daily_ic,
    _raw_clean_rank_ic,
    _newey_west_t,
    compute_ic_summary,
    _candidate_pool,
)
from stoke_ml.models.panel.evaluate_account import (
    _build_portfolio_returns,
    _ffill_last_np,
    _run_sleeve_sim,
    _simulate_sleeve_account,
    _ls_exposure_ledger,
    _combine_book_daily,
    _sleeve_account_metrics,
)
from stoke_ml.models.panel.evaluate_portfolio import (
    evaluate_sharpe,
    evaluate_portfolio,
    _quintile_analysis,
    _empty_result,
    compute_prediction_diversity,
    _build_raw_actuals,
    _EMPTY_EXPOSURE,
)

logger = logging.getLogger(__name__)

# Version stamp for the panel evaluator — experiments freeze the evaluator
# that produced their numbers.  Bump on any behavioral change to
# the sleeve-account / IC / quintile logic.
EVALUATOR_VERSION = "2026-08-05"
