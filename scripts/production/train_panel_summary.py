"""OOS backtest + summary + registry finalization for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — the combined-OOS series /
per-position ledger write, the continuous-OOS replay, the fold summary dict,
the experiment artifacts writer, and the project-wide registry append.  This is
the tail of the fold loop's write path, extracted so ``train_panel``'s ``main``
stays a thin orchestration of satellites.  ``train_panel`` re-exports
``_finalize_summary`` for backward compatibility.
"""
import argparse
import dataclasses
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.models.panel import PanelConfig

from scripts.production.train_panel_oos import _replay_continuous_oos
from scripts.production.train_panel_folds import (
    _FoldLoopResult, _fmt_date, _save_artifacts,
)
from scripts.production.train_panel_registry import (
    _EXPERIMENT_REGISTRY_PATH,
    _ablation_desc,
    _append_experiment_registry,
    _objective_desc,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _SummaryInputs:
    """All ``main()`` locals the OOS/summary finalization consumes.

    One frozen dataclass keeps the input boundary explicit and documented
    instead of a ~30-position call at the call site.  The fold loop's
    accumulated output travels here as a single ``fold_result``
    (:class:`_FoldLoopResult`) so its nine lists are defined once, not
    duplicated between two dataclasses (a drift bug source); the remaining
    fields are run context main() built before and after the loop.
    """
    fold_result: _FoldLoopResult
    version_info: dict
    outdir: str
    oos_dir: str
    n_trials: int
    experiment_registry: list[dict]
    experiment_signature: str
    universe_desc: str
    args: argparse.Namespace
    profile_name: str
    train_gate_desc: str
    eval_gate_desc: str
    universe_hashes: dict
    universe_membership: dict | None
    global_dates: np.ndarray
    lockbox_start: int
    lockbox_len: int
    n_timesteps: int
    channel_manifest: dict | None
    universe_resolved: list[str]
    universe_used: list[str]
    config: PanelConfig


def _finalize_summary(inp: _SummaryInputs) -> None:
    """Write the OOS backtest artifacts, fold summary, and registry tail.

    §十四-3/§十四-4: persists the combined daily OOS series (date-ordered), the
    combined per-position ledger, and ONE continuous long sleeve account
    replayed across all fold tapes — the FINAL Sharpe/MDD/CAGR come from that
    account only, not a mean of fold-restart NAVs.  Then writes the experiment
    artifacts (args / universe / summary via ``_save_artifacts``) and appends
    this run to the project-wide experiment registry so the NEXT run's DSR
    multiplicity counts it (written even when no fold completed — an aborted /
    short run is still a research trial).
    """
    oos_preds_all = inp.fold_result.oos_preds_all
    oos_dates_all = inp.fold_result.oos_dates_all
    oos_stocks_all = inp.fold_result.oos_stocks_all
    oos_pool_all = inp.fold_result.oos_pool_all
    oos_fold_all = inp.fold_result.oos_fold_all
    oos_weight_hash_all = inp.fold_result.oos_weight_hash_all
    oos_ledgers = inp.fold_result.oos_ledgers
    version_info = inp.version_info
    outdir = inp.outdir
    oos_dir = inp.oos_dir
    n_trials = inp.n_trials
    experiment_registry = inp.experiment_registry
    experiment_signature = inp.experiment_signature
    all_sharpes = inp.fold_result.all_sharpes
    fold_histories = inp.fold_result.fold_histories
    universe_desc = inp.universe_desc
    args = inp.args
    profile_name = inp.profile_name
    train_gate_desc = inp.train_gate_desc
    eval_gate_desc = inp.eval_gate_desc
    universe_hashes = inp.universe_hashes
    universe_membership = inp.universe_membership
    global_dates = inp.global_dates
    lockbox_start = inp.lockbox_start
    lockbox_len = inp.lockbox_len
    n_timesteps = inp.n_timesteps
    channel_manifest = inp.channel_manifest
    universe_resolved = inp.universe_resolved
    universe_used = inp.universe_used
    config = inp.config

    # Combined daily OOS series: one row per (stock, entry
    # day) across all non-overlapping folds — the input to the sleeve-account
    # backtest, kept separate from fold-level aggregates.
    if oos_preds_all:
        oos_series = pd.DataFrame({
            "entry_date": oos_dates_all,
            "stock_code": oos_stocks_all,
            "pred": np.concatenate(oos_preds_all),
            # The exact select pool the sleeve account ranked over (decision &
            # history) — a tape must expose the candidate set it was built
            # from, not only the selected fills.
            "candidate_eligible": np.concatenate(oos_pool_all),
            # Provenance: data + model versions so every tape
            # row is traceable to the exact experiment it was produced by.
            "data_version": version_info["data_manifest_hash"],
            "model_hash": version_info["model_hash"],
            # fold + trained-parameter hash per row so the tape
            # maps to the exact weights (fold_XXX_model.pt) that produced it.
            "fold": oos_fold_all,
            "weight_hash": oos_weight_hash_all,
        })
        # §十四-3: folds are walked most-recent-first, so the concatenated rows
        # are reverse-chronological chunk by chunk.  Sort before persisting so
        # the tape is date-ordered regardless of fold iteration order — a naive
        # consumer must not need to re-sort to feed the series chronologically.
        oos_series = oos_series.sort_values(
            ["entry_date", "stock_code"]).reset_index(drop=True)
        oos_series_path = os.path.join(outdir, "oos_series.parquet")
        oos_series.to_parquet(oos_series_path)
        logger.info("OOS series: %d rows -> %s", len(oos_series), oos_series_path)

    # Combined per-position ledger across all folds — the
    # single file a consumer reads to reproduce every fill of the backtest.
    # Same reverse-chunk problem as the series: sort by entry date then fold /
    # entry_day so the combined tape is chronological.
    if oos_ledgers:
        combined_ledger = pd.concat(oos_ledgers, ignore_index=True)
        combined_ledger = combined_ledger.sort_values(
            ["entry_date", "fold", "entry_day", "stock", "mode"]
        ).reset_index(drop=True)
        oos_ledger_path = os.path.join(outdir, "oos_ledger.parquet")
        combined_ledger.to_parquet(oos_ledger_path)
        logger.info("Combined OOS ledger: %d rows -> %s",
                    len(combined_ledger), oos_ledger_path)

    # §十四-4: ONE continuous long sleeve account replayed across ALL fold
    # tapes.  Each fold restarts NAV at 1 and is aggregated by mean Sharpe —
    # that is a set of disjoint OOS signal windows, not a continuous strategy.
    # This replay keeps a single account whose NAV carries over fold
    # boundaries (the previous fold's sleeves keep settling while the next
    # fold's model signals), and the FINAL Sharpe/MDD/CAGR come from THIS
    # account only.
    # §十二.3: the DSR trial-Sharpe dispersion is the HISTORICAL OOS Sharpe
    # distribution from the registry (prior rows only — this run appends after),
    # so the deflation reflects real past research trials, not this account.
    historical_sharpes = [
        e.get("oos_continuous_sharpe")
        for e in experiment_registry
        if isinstance(e.get("oos_continuous_sharpe"), (int, float))
    ]
    cont = (_replay_continuous_oos(oos_dir, n_trials=n_trials,
                                   trial_sharpes=historical_sharpes,
                                   formal=True)
            if oos_preds_all else None)
    if cont is not None:
        daily = np.asarray(cont["account"]["daily"], dtype=np.float64)
        # The account starts at NAV 1.0 on the close BEFORE day 0, so the NAV
        # after day c's close is the cumulative product through c (one row per
        # price date — final entry == final_nav by the simulator's identity).
        nav = (1.0 + daily).cumprod()
        pd.DataFrame({
            "price_date": cont["price_dates"],
            "nav": nav,
            "daily_return": daily,
        }).to_parquet(os.path.join(outdir, "oos_continuous.parquet"))
        if cont["ledger"] is not None:
            cont["ledger"].to_parquet(
                os.path.join(outdir, "oos_continuous_ledger.parquet"))
        logger.info(
            "Continuous OOS account: %d days across %d stocks | "
            "Sharpe=%.2f MaxDD=%.2f CAGR=%.2f final_nav=%.3f",
            len(daily), len(cont["stocks"]),
            cont["metrics"]["sharpe"], cont["metrics"]["maxdd"],
            cont["metrics"]["cagr"] if cont["metrics"]["cagr"] is not None
            else float("nan"),
            cont["metrics"]["final_nav"] if cont["metrics"]["final_nav"]
            is not None else float("nan"),
        )

    summary_data = None
    if all_sharpes:
        logger.info("=== %d-Fold Summary ===", len(all_sharpes))
        logger.info("LS_Sharpe mean: %.2f ± %.2f", np.mean(all_sharpes), np.std(all_sharpes))
        # IC comes from the outer-test evaluation of the exact deployed
        # checkpoint (outer_metrics) — never an in-loop proxy.
        all_ics = [
            f["outer_metrics"].get("ic_mean", float("nan"))
            for f in fold_histories if f.get("outer_metrics")
        ]
        all_ics = [x for x in all_ics if not np.isnan(x)]
        if all_ics:
            logger.info("IC mean: %.4f ± %.4f", np.mean(all_ics), np.std(all_ics))
        summary_data = {
            # Freeze the data/code/feature versions so the run
            # stays explainable days later (same info also in version.json).
            "version": version_info,
            # §十五-1: how many research trials (incl. this one) the DSR
            # multiplicity was computed against.
            "n_trials": n_trials,
            "experiment_signature": experiment_signature,
            "n_folds": len(all_sharpes),
            "ls_sharpe_mean": float(np.mean(all_sharpes)),
            "ls_sharpe_std": float(np.std(all_sharpes)),
            "ic_mean": float(np.mean(all_ics)) if all_ics else None,
            "ic_std": float(np.std(all_ics)) if all_ics else None,
            "universe": universe_desc,
            # §八.3: which universe gates applied to which split — the summary
            # is self-describing about the train/eval gate asymmetry.  The
            # default trains on the broad historical-member union (ungated);
            # --strict-index-training additionally gates the inner-train loss
            # by per-day membership so training matches the eval candidate
            # pool.
            "strict_index_training": bool(args.strict_index_training),
            # §十四: the frozen feature profile whose required-channel /
            # minimum-coverage gates this run enforced (T19 hashes it into the
            # experiment signature).  profile_name is the RESOLVED value ("none"
            # when the gate is inactive), so a no-formal/no-gate run does not
            # claim a gate it never enforced.
            "feature_profile": profile_name,
            "train_gate": train_gate_desc,
            "eval_gate": eval_gate_desc,
            # §P0-6: content hashes of the universe records the whole-run gates
            # consumed (delist status + index membership), so the summary is
            # provably tied to those artifact files.
            "universe_status_hash": universe_hashes["universe_status_hash"],
            "membership_hash": universe_hashes["membership_hash"],
            # §T6/§十四: the feature vintage policy (what CHANNELS this run
            # admitted) AND the universe-membership provenance (the CSI
            # membership the universe gate consumed) are declared side by side
            # — feature-vintage revision-safe NEVER implies the universe gate
            # avoided latest-reconstructed membership data.
            "feature_vintage": args.vintage_policy,
            "universe_membership": universe_membership,
            # Non-overlapping folds (step == val_len) — each
            # fold's SIGNAL windows are strictly non-overlapping (the last
            # batch's position exits may extend past a fold boundary), so
            # mean±std is the dispersion of disjoint signal windows.
            "folds_overlap": False,
            "fold_note": (
                "disjoint signal windows (step == val_len; strictly "
                "non-overlapping signal/entry days — the last batch's exits may "
                "extend past a fold boundary); per-fold metrics come from "
                "separate trainings, not repeated experiments on one model"
            ),
            "lockbox": {
                "months": args.lockbox_months,
                "start": _fmt_date(global_dates, lockbox_start),
                "end": _fmt_date(global_dates, n_timesteps - 1),
                "n_steps": lockbox_len,
                "note": "Reserved for a single final run once the design "
                        "freezes — no fold trains on or evaluates it.  The "
                        "horizon steps before it are a settlement buffer so "
                        "the last fold's exits stop before the lockbox opens.",
            },
            "oos_series": "oos_series.parquet",
            # The per-position fill ledger written above.
            "oos_ledger": "oos_ledger.parquet" if oos_ledgers else None,
            # §十四-4: headline comes from ONE continuous long sleeve account
            # replayed across all fold tapes — not the mean of fold-restart
            # NAVs.  Sharpe/MDD/CAGR are that account's, annualized at 252.
            "oos_continuous": (
                {
                    "file": "oos_continuous.parquet",
                    "ledger": ("oos_continuous_ledger.parquet"
                               if cont["ledger"] is not None else None),
                    "sharpe": cont["metrics"]["sharpe"],
                    "maxdd": cont["metrics"]["maxdd"],
                    "cagr": cont["metrics"]["cagr"],
                    "final_nav": cont["metrics"]["final_nav"],
                    "n_days": cont["metrics"]["n_days"],
                    "n_stocks": cont["metrics"]["n_stocks"],
                    # §十五-1: the continuous Sharpe read against data-snooping
                    # (PSR vs zero; DSR vs the expected max of n_trials).
                    "psr": cont["metrics"]["psr"],
                    "dsr": cont["metrics"]["dsr"],
                    "dsr_n_trials": cont["metrics"]["dsr_n_trials"],
                    "note": "One continuous long sleeve account replayed across "
                            "all fold tapes; NAV carries over fold boundaries.  "
                            "Final Sharpe/MDD/CAGR come from this account only, "
                            "not a mean of fold-restart NAVs.",
                }
                if cont is not None else None),
            "folds": [],
        }
        for i, f in enumerate(fold_histories):
            m = f["outer_metrics"]
            summary_data["folds"].append({
                "fold": i + 1,
                "best_epoch": f["best_epoch"],
                "eval_epoch": f["best_epoch"],
                "inner_eval_epoch": f.get("inner_eval_epoch"),
                "inner_eval_ls_sharpe": f.get("inner_eval_ls_sharpe"),
                "inner_eval_ic": f.get("inner_eval_ic"),
                "weight_hash": f.get("weight_hash"),
                "model_path": f.get("model_path"),
                "train_start": f.get("train_start"),
                "train_end": f.get("train_end"),
                "inner_val_start": f.get("inner_val_start"),
                "inner_val_end": f.get("inner_val_end"),
                "context_start": f.get("context_start"),
                "signal_start": f.get("signal_start"),
                "entry_start": f.get("entry_start"),
                "entry_end": f.get("entry_end"),
                "exit_end": f.get("exit_end"),
                "test_start": f.get("test_start"),
                "test_end": f.get("test_end"),
                "ls_sharpe": m.get("ls_sharpe"),
                "ic_mean": m.get("ic_mean"),
                "ic_ir": m.get("ic_ir"),
                "long_sharpe": m.get("long_sharpe"),
                "q5mq1_ret": m.get("q5mq1_ret"),
                "eligible_ew_sharpe": m.get("eligible_ew_sharpe"),
                "selected_universe_ew_sharpe": m.get("selected_universe_ew_sharpe"),
            })
    else:
        logger.warning("No valid folds completed")

    _save_artifacts(
        outdir, args, universe_resolved, universe_used, universe_desc, summary_data,
        channel_manifest=channel_manifest,
        version=version_info,
    )

    # §十五-1: register this run in the project-wide experiment ledger so the
    # NEXT run's DSR multiplicity counts it.  Written even when no fold
    # completed — an aborted / short run is still a research trial.
    registry_entry = {
        # §十二.6: signature for dedup + distinct-trial counting.
        "experiment_signature": experiment_signature,
        "outdir": outdir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": version_info.get("git_commit"),
        "data_manifest_hash": version_info.get("data_manifest_hash"),
        "feature_schema_hash": version_info.get("feature_schema_hash"),
        "model_hash": version_info.get("model_hash"),
        "universe_hash": version_info.get("universe_hash"),
        "horizon": config.horizon,
        "objective": _objective_desc(config),
        "ablation": _ablation_desc(config),
        "lockbox": {
            "months": args.lockbox_months,
            "start": _fmt_date(global_dates, lockbox_start),
            "end": _fmt_date(global_dates, n_timesteps - 1),
        },
        "n_folds": len(all_sharpes) if all_sharpes else 0,
        # §十二.6: an aborted run (no completed fold / no continuous account)
        # is still a registered trial for the DSR N — made explicit here.
        "aborted": not (all_sharpes and cont is not None),
        "ls_sharpe_mean": float(np.mean(all_sharpes)) if all_sharpes else None,
        "oos_continuous_sharpe": (
            cont["metrics"]["sharpe"] if cont is not None else None),
        "psr": cont["metrics"]["psr"] if cont is not None else None,
        "dsr": cont["metrics"]["dsr"] if cont is not None else None,
        "dsr_n_trials": n_trials,
    }
    _append_experiment_registry(_EXPERIMENT_REGISTRY_PATH, registry_entry)
    logger.info(
        "Experiment registry: %d prior distinct trials -> this is trial #%d "
        "(signature=%s, %s)",
        len(experiment_registry), n_trials, experiment_signature, outdir,
    )
