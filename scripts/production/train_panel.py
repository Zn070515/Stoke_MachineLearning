"""Train VSN+xLSTM panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --stocks 500 --epochs 30 --max-folds 1
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --universe csi300 --stocks 300 --outdir reports/exp/csi300
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --stock-list 600519,000001,000858
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --no-aux  # skip auxiliary data for quick test

Universe modes (--universe): first / random / stratified / all / csi300 / csi500 / csi800.
Artifacts (args.json, universe_resolved.txt, universe_used.txt, summary.json)
are saved to --outdir (default reports/experiments/<timestamp>).
"""
import argparse
import dataclasses
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stoke_ml.config import get_project_root, load_config
from stoke_ml.data.calendar import (
    TradingCalendar,  # noqa: F401  re-exported for import-compat
)
from stoke_ml.data.vintage_policy import universe_membership_provenance
from stoke_ml.features.pipeline import (
    _PIT_STATIC_COLS, fold_dead_feature_columns,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.panel_store import (
    panel_store_complete,
)
from stoke_ml.models.panel.train import train_panel
from stoke_ml.models.panel.evaluate import (
    _run_sleeve_sim,  # noqa: F401  re-exported for import-compat
    compute_equity_curve,  # noqa: F401  re-exported for import-compat
    compute_max_drawdown,  # noqa: F401  re-exported for import-compat
    compute_sharpe,  # noqa: F401  re-exported for import-compat
    evaluate_portfolio,
)
from scripts.production.data_quality_gate import (
    QUALITY_GATE_VERSION, contract_version, dataset_fingerprint,
)
from scripts.production.train_panel_oos import (
    _file_sha256,  # noqa: F401  re-exported for import-compat
    _state_dict_hash,
    _verify_tape_weight_hash,  # noqa: F401  re-exported for import-compat
    _replay_continuous_oos,
)
from scripts.production.train_panel_registry import (
    _calendar_freeze,
    _experiment_version,
    _EXPERIMENT_REGISTRY_PATH,
    _LOCKBOX_MARKER_PATH,  # noqa: F401  re-exported for import-compat
    _read_lockbox_marker,  # noqa: F401  re-exported for import-compat
    _mark_lockbox_used,  # noqa: F401  re-exported for import-compat
    _require_single_use_lockbox,
    _ablation_desc,
    _objective_desc,
    _experiment_signature,
    _distinct_trial_count,
    _registry_lock,  # noqa: F401  re-exported for import-compat
    _load_experiment_registry,
    _append_experiment_registry,
)

from scripts.production.train_panel_universe import (
    _discover_stocks,
    _exchange_group,  # noqa: F401  re-exported for import-compat
    _is_csi_universe,  # noqa: F401  re-exported for import-compat
    _strict_index_training_effective,
    _load_index_universe,  # noqa: F401  re-exported for import-compat
    _resolve_universe,
    _require_all_universe_prebuilt,
)
from scripts.production.train_panel_gates import (
    _assess_universe_reconciliation,  # noqa: F401  re-exported for import-compat
    _gate_enforced,
    _formal_mode,
    _resolve_required_set,
    _enforce_channel_coverage,
    _require_quality_gate,
    _check_verified_until_scope,
)
from scripts.production.train_panel_folds import (
    _quality_fail_reason,  # noqa: F401  re-exported for import-compat
    _fold_eligible_stocks,
    _mask_stocks,
    _require_universe_artifacts,  # noqa: F401  re-exported for import-compat
    _fold_universe_gates,
    _apply_candidate_gates,
    _gate_inner_train_membership,
    _gate_descriptions,
    _fold_delist_day,
    _universe_artifact_hashes,
    _cross_sectional_normalize,
    _slice_panel,
    _fmt_date,
    _augment_sequence,
    _save_artifacts,
)
from scripts.production.train_panel_panel import (
    _panel_pipeline_kwargs,  # noqa: F401  re-exported for import-compat
    _panel_store_meta,  # noqa: F401  re-exported for import-compat
    _validate_panel_store_path,
    _resolve_panel,
    _panel_memory_gb,  # noqa: F401  re-exported for import-compat
    _streaming_peak_memory_gb,  # noqa: F401  re-exported for import-compat
    _enforce_universe_memory,
    _host_available_gb,
    _estimate_panel_memory,  # noqa: F401  re-exported for import-compat
    _early_panel_memory_guard,
    _new_channel_entry,  # noqa: F401  re-exported for import-compat
    _finalize_channel,  # noqa: F401  re-exported for import-compat
    _load_channel_aux,  # noqa: F401  re-exported for import-compat
    load_aux_data,  # noqa: F401  re-exported for import-compat
    _prebuilt_channel_coverage,  # noqa: F401  re-exported for import-compat
    _trading_day_count,  # noqa: F401  re-exported for import-compat
    _date_coverage_fraction,  # noqa: F401  re-exported for import-compat
    _probe_broadcast_dates,  # noqa: F401  re-exported for import-compat
    _BASE_DIM_PREFERENCE,  # noqa: F401  re-exported for import-compat
    _SWITCH_KEY,  # noqa: F401  re-exported for import-compat
    _UNIVERSE_MEMORY_WARN_GB,  # noqa: F401  re-exported for import-compat
    _UNIVERSE_MEMORY_REFUSE_GB,  # noqa: F401  re-exported for import-compat
    _UNIVERSE_MEMORY_HARD_GB,  # noqa: F401  re-exported for import-compat
    _HAS_FLAG_CHANNELS,  # noqa: F401  re-exported for import-compat
)
from scripts.production.train_panel_model import (
    _weight_hash,
    _predict_outer,
    _best_eval_metrics,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# §十一.3 architecture-ablation switchboard.  Each entry maps a human name to
# PanelConfig field overrides that switch OFF one component of the production
# architecture, isolating where the model's edge comes from.  All default to
# the production config, so a run WITHOUT --ablation is the formal baseline.
_ABLATIONS: dict[str, dict] = {
    "plain_lstm": {"backbone": "lstm"},
    "vsn_lstm": {"backbone": "lstm", "use_vsn": True},
    "xlstm_no_vsn": {"use_vsn": False},
    "return_only": {"use_dir_head": False, "use_vol_head": False},
    "no_vol_head": {"use_vol_head": False},
    "no_dir_head": {"use_dir_head": False},
    "fixed_task_weights": {"fixed_task_weights": True},
    "no_ranking": {"use_ranking_loss": False},
    "no_pit_static": {"use_pit_static": False},
}


def main():
    parser = argparse.ArgumentParser(description="Train VSN+xLSTM panel model")
    parser.add_argument("--stocks", type=int, default=500,
                        help="Universe size / cap (default: 500; with "
                             "--universe first: first N sorted; random/stratified: "
                             "N sampled; csi*: N cap)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"],
                        help="Stock universe selection (default: random; "
                             "csi* = index constituents, PIT ever-held union)")
    parser.add_argument("--allow-high-risk-universe", action="store_true",
                        help="§七-P0 escape hatch: an explicit override for the "
                             "universe memory guard — a high-memory universe "
                             "(all / large csi800) proceeds with a prominent "
                             "warning instead of being refused.  Default: "
                             "refused.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for universe sampling and data "
                             "augmentation (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Experiment artifact dir (default: "
                             "reports/experiments/<timestamp>)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--lockbox-months", type=int, default=0,
                        help="Reserve the last N months as an untouched lockbox "
                             "— no fold trains on or evaluates it; kept for a "
                             "single final run once the design freezes.  The "
                             "lockbox is single-use: the first FORMAL run that "
                             "opens it records the marker and a later formal run "
                             "is refused.  Default 0 = lockbox OFF (opt in for "
                             "the one final run with --lockbox-months 12).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in days (1/5/20)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Model hidden dimension (default: 128)")
    parser.add_argument("--xlstm-blocks", type=int, default=2,
                        help="Number of xLSTM blocks (default: 2)")
    parser.add_argument("--rank-weight", type=float, default=0.1,
                        help="Ranking loss weight (0=disable, default: 0.1)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=sorted(_ABLATIONS),
                        help="§十一.3: switch OFF ONE architecture component to "
                             "isolate where performance comes from.  Choices: "
                             + ", ".join(sorted(_ABLATIONS))
                             + ".  Default: full production architecture "
                             "(the formal baseline).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="§十一.1: apply the fixed per-fold corruption pass "
                             "(Gaussian noise + one global time mask + one global "
                             "feature dropout, generated once and reused across "
                             "all epochs).  OFF by default — this is a fixed "
                             "data-corruption, not online per-sample augmentation, "
                             "so it is opt-in ablation only.")
    parser.add_argument("--log-gradient-flow", action="store_true",
                        help="Log per-parameter-group gradient norms each epoch "
                             "(after optimizer.step, before zero_grad)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--require-aux-channels", type=str, default="",
                        help="Comma-separated aux channels that must have "
                             "loaded_stocks>0; experiment "
                             "FAILS otherwise. Default: none required")
    parser.add_argument("--feature-profile", type=str, default="headline_v1",
                        help="Frozen feature profile (stoke_ml/config/"
                             "feature_profile.py).  A FORMAL, gate-enforced run "
                             "adds the profile's required_channels to "
                             "--require-aux-channels and enforces its "
                             "minimum-coverage thresholds on probeable "
                             "channels.  'none' disables the required-channel "
                             "coverage gate (§十四). Default: headline_v1")
    parser.add_argument("--prebuilt", type=str, default=None,
                        help="Load panel-mode prebuilt features from this dir "
                             "(built via build_features.py --panel-mode). "
                             "Skips aux data loading and live feature "
                             "engineering — the panel is built from the "
                             "prebuilt parquets")
    parser.add_argument("--require-feature-manifest",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Require every prebuilt feature parquet to carry a "
                             "matching sidecar manifest (missing / stale / "
                             "schema-drift / different-git-commit FAILS the run "
                             "instead of warning). Default: on. Use "
                             "--no-require-feature-manifest for legacy prebuilt "
                             "dirs built without manifests (§十一-1)")
    parser.add_argument("--panel-store", type=str, default=None,
                        help="§十六 memmap lazy-storage dir for the built panel. "
                             "When DIR already holds a complete store it is loaded "
                             "instead of loading K-line + re-stacking the panel, so "
                             "a large-universe re-run never materializes the whole "
                             "dense (N,T,D) feature grid in RAM (arrays are mmap'd "
                             "and read lazily by the panel dataset / _slice_panel).  A "
                             "store's meta.json config fingerprint is checked on "
                             "load — a mismatch (horizon/seq_len/start/end/"
                             "universe/feature switches) REFUSES the run so a stale "
                             "store can't silently train on wrong targets.  "
                             "Otherwise the panel built this run is persisted there "
                             "for future runs.  Default: off — build in memory as "
                             "always.")
    parser.add_argument("--scratch-dir", type=str, default=None,
                        help="§T7 scratch dir for the STREAMING panel build's "
                             "per-stock Pass-1 pickles.  Default: derived as "
                             "<panel-store>/scratch/<run_id>/; with no "
                             "--panel-store the build is not streaming (dense "
                             "in-memory) so this is unused.  An explicit "
                             "location is never swept by the startup stale "
                             "cleanup (only tool-owned dirs are).")
    parser.add_argument("--no-require-quality-gate", action="store_true",
                        help="Skip the required quality-gate report check "
                             "(dev smoke only; §六-2 wants a matching report "
                             "before any real training run)")
    parser.add_argument("--no-formal", action="store_true",
                        help="Exploratory mode: allow degraded universe gates "
                             "when a required PIT artifact is missing, with a "
                             "prominent warning, instead of refusing to start "
                             "(§P0-7; formal is the default)")
    parser.add_argument("--strict-index-training",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="§八.3 + §T6 decision 2: gate the inner-TRAIN loss "
                             "by per-day index membership for csi300/csi500/"
                             "csi800.  Default: None = decide from the universe "
                             "— ON for the strict-CSI universes "
                             "(csi300/csi500/csi800), OFF otherwise.  An "
                             "explicit --strict-index-training / "
                             "--no-strict-index-training forces the value "
                             "regardless of universe.  When OFF, inner_train "
                             "learns from the broad historical-member union and "
                             "only the RANKED candidate pools "
                             "(inner_val/outer_test) are membership-gated.")
    parser.add_argument("--vintage-policy", type=str, default="revision-safe",
                        choices=["revision-safe", "allow-revised",
                                 "headline-strict"],
                        help="§T2/§T7/§T3: vintage-admission policy for the "
                             "feature set.  revision-safe (default) admits "
                             "channels whose source_vintage is "
                             "immutable_snapshot and DENIES "
                             "latest_revised-sourced ones (fundamental/macro/"
                             "earnings/valuation/pledge/shareholder/"
                             "index_membership/market_env_refine/sector/"
                             "concept); allow-revised additionally admits "
                             "latest_revised-sourced channels (legacy / "
                             "ablation use); headline-strict (new) is the "
                             "strictest tier — it additionally requires "
                             "pit_alignment == \"verified\" (with an explicit "
                             "scale-invariant waiver for daily_qfq/market_env), "
                             "so proxy-aligned channels are denied unless "
                             "waived.  The legacy name \"safe-only\" is the "
                             "pre-T3 alias for revision-safe.")
    parser.add_argument("--allow-fundamental-ablation", action="store_true",
                        help="T3 research decision #1: ABLATION ONLY — force the "
                             "fundamental channel ON even under revision-safe.  "
                             "This is the ONLY way fundamental enters a "
                             "revision-safe run; never use it for formal "
                             "headline/lockbox runs.  Under allow-revised "
                             "fundamental is already on, so this is a no-op "
                             "there.  Only the fundamental channel is affected "
                             "— the other policy-denied channels stay off.")
    parser.add_argument("--quality-gate-report", type=str, default=None,
                        help="Path to the quality-gate report to verify "
                             "(default: <repo>/reports/data_quality_gate.json)")
    parser.add_argument("--allow-missing-universe", action="store_true",
                        help="§八-2 escape hatch: proceed when the gate's "
                             "universe reconciliation reports requested stocks "
                             "missing from disk.  The missing list is still "
                             "recorded (universe_missing.txt in the outdir) — "
                             "the gap is surfaced, never silent.")
    parser.add_argument("--minute", action="store_true",
                        help="Use minute-frequency K-line data instead of daily")
    parser.add_argument("--minute-frequency", type=str, default="60",
                        choices=["5", "15", "30", "60"],
                        help="Bar frequency for minute mode (default: 60)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60 daily, 64 minute)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    # §T6 decision 2: the strict inner-train membership gate defaults ON for
    # the strict-CSI universes, OFF otherwise — an explicit CLI flag wins.
    args.strict_index_training = _strict_index_training_effective(
        args.strict_index_training, args.universe)

    # §十六: decide the store-load path up front — BEFORE universe resolution
    # and the K-line load — so a complete-store re-run never reads the
    # multi-GB input panel only to discard it.  meta.json staleness is checked
    # at load (after universe resolution, when n_stocks is known).
    _store_load = bool(args.panel_store) and panel_store_complete(args.panel_store)
    if args.panel_store:
        _validate_panel_store_path(args.panel_store)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if _gate_enforced(args):
        _report_path = args.quality_gate_report or str(
            get_project_root() / "reports" / "data_quality_gate.json"
        )
        gate_report = _require_quality_gate(
            data_dir, args.prebuilt, _report_path,
            allow_missing=args.allow_missing_universe,
        )
        logger.info(
            "Quality-gate report verified: %s (scope=%s scanned=%s/%s "
            "manifest_contract_full_scan=%s)",
            _report_path,
            gate_report.get("scope"),
            gate_report.get("scanned_files"),
            gate_report.get("total_files"),
            gate_report.get("manifest_contract_full_scan"),
        )

    # §七-P0: the full market cannot be feature-engineered in RAM.  `--universe
    # all` must read prebuilt panel features (build_features.py --panel-mode)
    # rather than live-engineering 5530 stocks (~225GB of feature arrays on a
    # ~96GB host).  Without --prebuilt this is refused outright; with it the
    # post-build memory estimate below still warns when the panel is too big.
    _require_all_universe_prebuilt(
        args.universe, args.prebuilt, store_complete=_store_load)

    if args.stock_list:
        stock_list = [c.strip() for c in args.stock_list.split(",")]
        universe_desc = f"stock-list (explicit, n={len(stock_list)})"
    elif args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        stock_list = MinuteStorage(data_dir).list_stocks(args.minute_frequency)
        if args.stocks:
            stock_list = stock_list[:args.stocks]
        universe_desc = f"minute-mode (n={len(stock_list)})"
    else:
        all_stocks = _discover_stocks(data_dir, None)
        stock_list, universe_desc = _resolve_universe(
            all_stocks, args.universe, args.stocks, args.seed, data_dir,
            formal=_gate_enforced(args),
        )

    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)

    universe_resolved = list(stock_list)

    # §七-P0: refuse an oversized universe BEFORE the panel build allocates the
    # dense (N, T, D) grids — the post-build check below is too late (the build
    # itself can OOM first).  Skipped on a store-load re-run (no build, lazy
    # mmap, and the store's surviving subset may be far smaller than the
    # requested universe) and whenever the feature dim cannot be estimated
    # without building (live builds — the post-build check covers those).
    _early_panel_memory_guard(args, stock_list, data_dir, _store_load)

    # Stock-level quality is judged per-fold, point-in-time, inside the fold
    # loop (_fold_eligible_stocks uses only columns before train_end) — no
    # full-history ejection up front.  Row-level badness is
    # handled as masks in the pipeline, not stock ejection.
    universe_used = list(stock_list)

    logger.info("Universe: %s", universe_desc)

    # §十六: a complete --panel-store skips the K-line load AND the feature
    # build entirely — the panel arrays are mmap'd and read lazily downstream.
    # The decision was made up front (before universe resolution, so a store
    # re-run never reads 5530 stocks' OHLCV only to discard it); _resolve_panel
    # loads the store under its meta.json config guard, or else engineers the
    # panel live (and persists it when --panel-store is set).
    # §十四: resolved required set = explicit --require-aux-channels ∪ the
    # active frozen feature profile's required_channels (formal + gate-enforced
    # + named profile); coverage_contracts carries the profile's per-channel
    # coverage contracts — each channel's (metric, threshold) — ({} when the
    # profile is inactive / 'none').
    required_set, coverage_contracts, profile_name = _resolve_required_set(args)
    logger.info(
        "Feature profile: %s (required=%s, coverage_contracts=%s)",
        profile_name, sorted(required_set),
        sorted({ch: f"{c.metric}:{c.threshold}"
                for ch, c in coverage_contracts.items()}.items()),
    )
    seq_len = args.seq_len or (64 if args.minute else 60)

    panel_data, channel_manifest = _resolve_panel(
        args, stock_list, seq_len, data_dir, required_set, _store_load)

    # §v12-P0: panel row identity — stock_codes comes from the pipeline's
    # valid_codes (only stocks whose features survived cleaning), NEVER
    # re-derived from the raw panel: a stock whose features were cleaned out
    # would otherwise shift every subsequent array row's stock label (board
    # one-hot, universe/delist mask, OOS artifact codes) with no error raised.
    panel_stocks = list(panel_data["stock_codes"])
    assert len(panel_stocks) == panel_data["past_observed"].shape[0], (
        "panel stock_codes length != past_observed rows (row identity broken)")
    assert len(panel_stocks) == panel_data["static_features"].shape[0], (
        "panel stock_codes length != static_features rows (row identity broken)")
    assert len(set(panel_stocks)) == len(panel_stocks), (
        "duplicate stock codes in panel (row identity broken)")

    # §十四 required-channel + coverage-contract gate: a required channel with
    # ZERO coverage, an UNPROBEABLE required channel in formal mode, or a
    # probeable required channel below its profile contract minimum aborts the
    # experiment instead of silently training on air.
    _enforce_channel_coverage(
        required_set, channel_manifest, coverage_contracts,
        formal=_formal_mode(args))
    if channel_manifest:
        summary_bits = ", ".join(
            f"{k}={v.get('status')}({v.get('coverage')})"
            for k, v in sorted(channel_manifest.items()) if not k.startswith("_")
        )
        logger.info("Channel coverage manifest: %s", summary_bits)

    # Union trading calendar (datetime64[ns]) — fold boundaries in index space
    # map back to real dates for the summary.
    global_dates = panel_data.get("global_dates")

    # §九-3: strict formal run — the panel must stay within the VERIFIED
    # calendar window (see _check_verified_until_scope).  Exploratory runs can
    # pass --no-require-quality-gate to opt out.
    refusal = _check_verified_until_scope(
        global_dates, enforce=_gate_enforced(args), data_dir=data_dir)
    if refusal:
        raise SystemExit(refusal)

    # §七-1/§七-3: the whole-run universe gates, computed ONCE via the shared
    # helper so the baselines (train_baselines_panel.py) consume the SAME
    # candidate-pool gates (§P0-5).  nd_mask blocks ENTRY from a known
    # delisting column on; mem_mask enforces per-day index membership for
    # csi300/csi500/csi800; delist_global feeds each fold's delist_day so the
    # sleeve simulator force-sells known-delisted positions.  Missing universe
    # parquets → empty status → all -1 / all-True gates (no crash); strict
    # formal-mode failure for missing artifacts is enforced separately (§P0-7).
    nd_mask, mem_mask, delist_global, universe_status = _fold_universe_gates(
        global_dates, panel_stocks, args.universe, data_dir,
        formal=_formal_mode(args),
    )
    # §P0-6: content hashes of the exact universe records the gates consumed —
    # every fold tape embeds these so replay can prove it used the same
    # delist / membership artifacts.
    universe_hashes = _universe_artifact_hashes(
        universe_status, data_dir, args.universe)
    # §T6/§十四: the universe-membership PROVENANCE (Baostock monthly
    # reconstruction, latest-reconstructed) for CSI universes that consume
    # membership.parquet, None otherwise — declared EXPLICITLY in the summary
    # and the trial signature, never implied-bypassed by feature-vintage
    # revision-safe.  Separated from the feature VintagePolicy on purpose (audit §十四).
    universe_membership = universe_membership_provenance(args.universe)
    # §八.3: record what gates each split consumes so a run is self-describing.
    # inner_train default is the broad historical-member union (ungated);
    # --strict-index-training additionally gates its loss by per-day index
    # membership.  Evaluation always gates 未退市, plus per-day membership for
    # universes that consume membership.parquet.
    eval_gate_desc, train_gate_desc = _gate_descriptions(
        mem_mask is not None, args.strict_index_training)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    # Static features are (N, T, D) PIT — feature dim is axis 2.
    static_dim = panel_data["static_features"].shape[2]
    dims = f"S={static_dim} " \
           f"PK={panel_data['past_known'].shape[2]} " \
           f"PO={panel_data['past_observed'].shape[2]}"
    logger.info("Panel data: %d stocks × %d timesteps  dims: %s  horizon=%d",
                n_stocks, n_timesteps, dims, args.horizon)
    # §七-P0: refuse/warn when this universe's panel cannot realistically fit in
    # RAM.  --universe all (5530 stocks ~= 225 GB) is refused by default unless
    # --allow-high-risk-universe; csi800 (historical member union) warns and is
    # refused above the hard ceiling or when it exceeds host available memory.
    n_features = (
        static_dim + panel_data["past_known"].shape[2]
        + panel_data["past_observed"].shape[2]
    )
    _enforce_universe_memory(
        args.universe, n_stocks, n_timesteps, n_features,
        allow_override=args.allow_high_risk_universe,
        available_gb=_host_available_gb(),
    )

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=static_dim,
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        hidden_dim=args.hidden_dim,
        xlstm_num_blocks=args.xlstm_blocks,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
        num_workers=0,
        horizon=args.horizon,
        rank_loss_weight=args.rank_weight,
        seed=args.seed,
        log_gradient_flow=args.log_gradient_flow,
    )
    # §十一.3: apply the architecture-ablation overrides AFTER the base config
    # is built, so a plain run is byte-for-byte the formal baseline and an
    # ablation run only flips the switches in _ABLATIONS.
    if args.ablation:
        config = dataclasses.replace(config, **_ABLATIONS[args.ablation])
    logger.info("VSN+xLSTM config: hidden=%d blocks=%d heads=%d batch=%d lr=%.1e "
                "rank_w=%.2f ablation=%s",
                config.hidden_dim, config.xlstm_num_blocks, config.xlstm_num_heads,
                config.batch_size, config.learning_rate, config.rank_loss_weight,
                args.ablation or "full")

    # Freeze the data/code/feature versions up front so the run
    # stays explainable even if every fold fails.  Written to version.json
    # unconditionally; also embedded in summary.json when folds complete.
    version_info = _experiment_version(
        data_dir, universe_used, args.prebuilt,
        static_dim,
        panel_data["past_known"].shape[2],
        panel_data["past_observed"].shape[2],
        config, args.start, args.end, args.seed,
    )
    logger.info(
        "Version freeze: commit=%s data=%s feat=%s uni=%s cal=%s eval=%s",
        version_info["git_commit"][:10], version_info["data_manifest_hash"],
        version_info["feature_schema_hash"], version_info["universe_hash"],
        version_info["calendar_version"], version_info["evaluator_version"],
    )

    # Purged walk-forward splits
    if args.minute:
        val_len = 250      # ~62 trading days
    else:
        val_len = 126      # ~6 months daily
    # OOS folds are NON-OVERLAPPING — step == val_len, so
    # adjacent folds evaluate disjoint SIGNAL windows (strictly non-overlapping
    # signal/entry days; a sleeve's exit may extend past a fold boundary, which
    # is why the fold_note says "signal windows", never "return windows").  The
    # old step < val_len made every fold share test days with its neighbours,
    # inflating fold count and letting mean±std masquerade as independent
    # dispersion.
    step = val_len
    purge = config.seq_len
    all_sharpes = []
    fold_histories = []

    # Reserve the last N months as an untouched lockbox.
    # No fold trains on or evaluates it; it is kept for a single final run
    # once the design freezes.  Daily ≈ 21 bars/month; minute mode scales by
    # bars per day so lockbox_months spans the same wall-clock time.
    bars_per_day = {"5": 48, "15": 16, "30": 8, "60": 4}[args.minute_frequency]
    lockbox_len = int(args.lockbox_months * 21 * (bars_per_day if args.minute else 1))
    lockbox_start = max(0, n_timesteps - lockbox_len)
    if lockbox_start <= 0:
        logger.error("Lockbox (%d steps) leaves no trainable panel "
                     "(n_timesteps=%d) — reduce --lockbox-months",
                     lockbox_len, n_timesteps)
        sys.exit(1)
    logger.info("Lockbox [%d:%d] %d steps (%.1f months) — %s .. %s",
                lockbox_start, n_timesteps, lockbox_len, args.lockbox_months,
                _fmt_date(global_dates, lockbox_start),
                _fmt_date(global_dates, n_timesteps - 1))

    # Resolve the outdir FIRST so the lockbox marker records the real output
    # directory (not null) when a default outdir is used (§二十).  The marker is
    # written here (as the lockbox is opened) so even an aborted first run
    # consumes the single use; a second formal run — into any outdir — is
    # refused instead of re-opening the untouched period.  The output directory
    # itself is NOT created until after the lockbox contract passes, so a
    # refused run leaves no empty experiment dir behind.
    outdir = args.outdir or os.path.join(
        "reports", "experiments", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    # §二十: the lockbox is a SINGLE-USE resource.  Exploratory runs
    # (--no-require-quality-gate / --no-formal) and --lockbox-months 0 are
    # never blocked.
    _require_single_use_lockbox(
        args.lockbox_months,
        formal=_gate_enforced(args) and _formal_mode(args),
        info={
            "lockbox_months": args.lockbox_months,
            "universe": universe_desc,
            "lockbox_start": _fmt_date(global_dates, lockbox_start),
            "lockbox_end": _fmt_date(global_dates, n_timesteps - 1),
            "outdir": outdir,
        },
    )

    oos_dir = os.path.join(outdir, "oos_preds")
    os.makedirs(oos_dir, exist_ok=True)
    # §八-2: the --allow-missing-universe escape proceeded despite requested
    # stocks missing from disk — record the gap in the experiment artifacts so
    # the run's universe is never "silently whatever is on disk".
    if args.allow_missing_universe and _gate_enforced(args):
        recon = (gate_report.get("universe_reconciliation") or {})
        missing = sorted(str(c) for c in (recon.get("missing_codes") or []))
        if missing:
            missing_path = os.path.join(outdir, "universe_missing.txt")
            with open(missing_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(missing) + "\n")
            logger.warning(
                "--allow-missing-universe: %d requested stocks missing from "
                "disk, recorded in %s (§八-2)",
                len(missing), missing_path,
            )
    # §十五-1 / §十二.6: this run is one more DISTINCT experiment in the
    # project-wide registry — the DSR deflation N counts distinct experiments
    # iterated so far, not runs and not the strategies inside one report.  A
    # prior row with this run's experiment_signature is the SAME experiment
    # re-run, so it is replaced and N does not grow.
    experiment_registry = _load_experiment_registry(_EXPERIMENT_REGISTRY_PATH)
    experiment_signature = _experiment_signature(
        version_info, config, augmentation=bool(args.augment),
        vintage_policy=args.vintage_policy, feature_profile=profile_name,
        universe_membership=universe_membership)
    n_trials = _distinct_trial_count(experiment_registry, experiment_signature)
    oos_dates_all: list[str] = []
    oos_stocks_all: list[str] = []
    oos_preds_all: list[np.ndarray] = []
    oos_pool_all: list[np.ndarray] = []
    oos_ledgers: list[pd.DataFrame] = []
    oos_fold_all: list[int] = []
    oos_weight_hash_all: list[str] = []

    rng = np.random.RandomState(args.seed)
    fold = 0
    # Walk BACKWARD from the lockbox boundary so the (max_folds) validation
    # windows cover the newest period instead of the earliest.  The training
    # window GROWS from position 0 out to (val_start - purge) each fold, so
    # the 2000-2015 history is genuinely in the training set — the old
    # fixed-width 756-day scheme left [0, n_timesteps-train_len-purge-val_len)
    # permanently unused and put short-history stocks' data entirely before
    # every fold window.
    # Reserve the `horizon` steps before the lockbox as a
    # settlement buffer — the last outer entry <= lockbox_start - horizon, so
    # its liquidation reads prices that end BEFORE the lockbox opens.  Without
    # this the final fold's sleeve would settle inside the untouched lockbox
    # and those gains would be counted as "evaluated" OOS performance.
    last_val_start = n_timesteps - config.horizon - val_len - lockbox_len
    val_start = last_val_start
    while val_start >= 0:
        if args.max_folds and fold >= args.max_folds:
            break
        train_end = val_start - purge
        if train_end < config.seq_len + 1:
            # The panel dataset needs at least seq_len+1 rows for one window
            # (n_windows = n_timesteps - seq_len must be >= 1).
            break
        fold += 1
        train_start = 0
        val_end = min(val_start + val_len, n_timesteps)

        # Carve the last ~15% of the trainable span as inner_val — used ONLY
        # for checkpoint selection inside train_panel.  The outer test (the old
        # val window) is fully held out of training and evaluated exactly once
        # on the deployed checkpoint.
        n_train_targets = train_end - config.seq_len
        inner_val_len = max(1, int(round(0.15 * n_train_targets)))
        inner_val_context_start = train_end - inner_val_len - config.seq_len
        if inner_val_context_start < config.seq_len + 1:
            # not enough rows left for one inner_train + one inner_val window
            break
        inner_train_end = inner_val_context_start
        val_context_start = val_start - config.seq_len

        inner_train_slice = slice(0, inner_train_end)
        inner_val_slice = slice(inner_val_context_start, train_end)
        outer_test_slice = slice(val_context_start, val_end)

        inner_train_data = _slice_panel(panel_data, inner_train_slice, price_pad=config.horizon)
        inner_val_data = _slice_panel(panel_data, inner_val_slice, price_pad=config.horizon)
        outer_test_data = _slice_panel(panel_data, outer_test_slice, price_pad=config.horizon)

        # Per-fold PIT stock-level eligibility: judge a stock
        # ONLY on data before train_end, never the full 2000→2099 history.  The
        # old global _filter_quality ejected a stock from EVERY fold because of
        # one bad 2025 row; now each fold judges its own past.  Row-level
        # badness remains masked (pipeline), not a reason to eject.
        fold_eligible = _fold_eligible_stocks(panel_data, train_end)
        fold_stocks = [panel_stocks[i] for i in np.where(fold_eligible)[0]]
        if len(fold_stocks) < 20:
            logger.warning("Fold %d: only %d stocks eligible PIT (need >= 20) — "
                           "skipping fold", fold, len(fold_stocks))
            val_start -= step
            continue
        inner_train_data = _mask_stocks(inner_train_data, fold_eligible)
        inner_val_data = _mask_stocks(inner_val_data, fold_eligible)
        outer_test_data = _mask_stocks(outer_test_data, fold_eligible)

        # Per-fold dead-feature drop (§十一-4): a column constant across every
        # observed day of THIS fold's training window is dropped from all three
        # slices (they share the column layout).  Judged only on the training
        # period — never validation/test — so a future fold can't decide an
        # earlier fold's feature set.  The full-history sparsity report is NOT
        # used for selection.  Config dims shrink by the same count so the
        # model's VSN input widths match the sliced grids.
        pk_drop, po_drop = fold_dead_feature_columns(
            inner_train_data,
            panel_data["past_known_cols"],
            panel_data["past_observed_cols"],
        )
        if pk_drop or po_drop:
            for dd in (inner_train_data, inner_val_data, outer_test_data):
                if pk_drop:
                    dd["past_known"] = np.delete(dd["past_known"], pk_drop, axis=2)
                if po_drop:
                    dd["past_observed"] = np.delete(dd["past_observed"], po_drop, axis=2)
            fold_config = dataclasses.replace(
                config,
                past_known_dim=config.past_known_dim - len(pk_drop),
                past_observed_dim=config.past_observed_dim - len(po_drop),
            )
            logger.info("Fold %d: dropped %d dead past_known + %d past_observed "
                        "columns (train-window constancy)",
                        fold, len(pk_drop), len(po_drop))
        else:
            fold_config = config

        # Merge the §七-3 universe gates into the EVALUATION candidate pools:
        # 未退市 for every universe, plus 当日是该指数成员 (per-day index
        # membership) for csi300/csi500/csi800.  §八.3: inner_train is by
        # DEFAULT left ungated — the model learns from the broad
        # historical-member union; only what gets RANKED as a tradable
        # candidate (inner_val/outer_test) is restricted.  That asymmetry is
        # recorded in the summary (train_gate/eval_gate).  Applied in this
        # fold's row/column space (rows = surviving original stock rows, cols
        # = the slice's global columns) so _candidate_pool picks the gates up
        # automatically.
        rows = np.where(fold_eligible)[0]
        for name, tslice, dd in (
            ("inner_val", inner_val_slice, inner_val_data),
            ("outer_test", outer_test_slice, outer_test_data),
        ):
            _apply_candidate_gates(dd, tslice, rows, nd_mask, mem_mask)

        # §八.3 strict mode: also gate the inner-TRAIN loss by per-day index
        # membership (see _gate_inner_train_membership).  Default: off —
        # inner_train learns from the broad historical-member union.
        if args.strict_index_training and mem_mask is not None:
            _gate_inner_train_membership(
                inner_train_data, mem_mask, rows, np.arange(0, inner_train_end))

        # y_return: cross-sectional z-score per date — preserves relative
        # ordering across stocks so ranking loss and IC evaluation work on a
        # consistent scale.  Normalize using the RETURN-target mask (clean
        # open-to-open returns) so dirty/missing positions don't skew the
        # z-score.  y_volatility: kept as the raw positive future-vol target
        # (std of the next-horizon daily returns).  VolatilityHead outputs
        # softplus > 0, so z-scoring the target would reintroduce the
        # negative-target-vs-positive-output contradiction — it must stay in
        # original units.
        inner_train_data["y_return"] = _cross_sectional_normalize(
            inner_train_data["y_return"], inner_train_data["return_target_mask"],
        )
        inner_val_data["y_return"] = _cross_sectional_normalize(
            inner_val_data["y_return"], inner_val_data["return_target_mask"],
        )
        outer_test_data["y_return"] = _cross_sectional_normalize(
            outer_test_data["y_return"], outer_test_data["return_target_mask"],
        )
        # Clip normalized targets to [-5, 5] — same band for train and val so
        # the model is never asked to fit z-scores beyond the eval range.
        # (Only y_return is z-scored; y_volatility stays in original units,
        # well below 5, so the clip applies to y_return only.)
        for dd in (inner_train_data, inner_val_data, outer_test_data):
            np.clip(dd["y_return"], -5.0, 5.0, out=dd["y_return"])

        # §十一.1: OPTIONAL fixed corruption pass on the inner-training data.
        # OFF by default.  This is NOT online per-sample augmentation — the
        # Gaussian noise is per-element independent (gated by observation_mask
        # so zero-padded history of new listings stays exactly zero), but the
        # time mask zeroes the SAME global time segment and the feature dropout
        # the SAME feature set for every stock; the pass runs once per fold and
        # every epoch reuses the identical corrupted copy.  Use --augment for
        # ablation only.
        if args.augment:
            pk_aug, po_aug = _augment_sequence(
                inner_train_data["past_known"],
                inner_train_data["past_observed"],
                obs_mask=inner_train_data["observation_mask"],
                noise_std=0.005,
                mask_prob=0.03,
                feat_dropout=0.01,
                rng=rng,
            )
            inner_train_data["past_known"] = pk_aug
            inner_train_data["past_observed"] = po_aug

        logger.info("Fold %d/%d: inner_train [%d:%d] inner_val [%d:%d] "
                    "outer_test [%d:%d]",
                    fold, args.max_folds or "∞",
                    0, inner_train_end,
                    inner_val_context_start, train_end,
                    val_context_start, val_end)

        t0 = time.time()
        # Checkpoint selection runs on inner_val inside train_panel; the
        # returned model is the best-inner-val checkpoint.
        model, history = train_panel(
            fold_config, inner_train_data, inner_val_data, device,
            raw_val_returns=inner_val_data["realized_return"],
        )
        elapsed = time.time() - t0

        # §十二-1: persist the best-inner-val checkpoint per fold so a fold's
        # OOS tape is reproducible / deployable, not only an in-memory state.
        # version_info["model_hash"] fingerprints config + architecture source
        # (shared by every fold); weight_hash below fingerprints the actual
        # trained parameters so an OOS row maps to exactly one set of weights.
        weight_hash = _weight_hash(model)
        if pk_drop or po_drop:
            pk_cols = [c for j, c in enumerate(panel_data["past_known_cols"])
                       if j not in pk_drop]
            po_cols = [c for j, c in enumerate(panel_data["past_observed_cols"])
                       if j not in po_drop]
        else:
            pk_cols = list(panel_data["past_known_cols"])
            po_cols = list(panel_data["past_observed_cols"])
        model_path = os.path.join(oos_dir, f"fold_{fold:03d}_model.pt")
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": dataclasses.asdict(fold_config),
            "feature_schema": {
                "static_cols": list(_PIT_STATIC_COLS)[:fold_config.static_dim],
                "past_known_cols": pk_cols,
                "past_observed_cols": po_cols,
            },
            "fold_range": {
                "train_start": _fmt_date(global_dates, 0),
                "train_end": _fmt_date(global_dates, inner_train_end - 1),
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "context_start": _fmt_date(global_dates, val_context_start),
                "signal_start": _fmt_date(global_dates, val_start - 1),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "exit_end": _fmt_date(
                    global_dates,
                    min(val_start + val_len - 1 + config.horizon,
                        n_timesteps - 1)),
                "test_start": _fmt_date(global_dates, val_context_start),
                "test_end": _fmt_date(global_dates, val_end - 1),
            },
            "weight_hash": weight_hash,
            "model_source_hash": version_info["model_source_hash"],
            "model_config_hash": version_info["model_config_hash"],
            "best_epoch": history.get("best_epoch_idx", 0) + 1,
            "data_version": version_info["data_manifest_hash"],
            "feature_schema_hash": version_info["feature_schema_hash"],
            "git_commit": version_info["git_commit"],
            "evaluator_version": version_info["evaluator_version"],
            "seed": args.seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, model_path)
        logger.info("  Fold %d: best checkpoint (epoch %d, weights %s) -> %s",
                    fold, history.get("best_epoch_idx", 0) + 1,
                    weight_hash, model_path)

        # Evaluate the exact deployed checkpoint ONCE on the held-out outer
        # test — the honest out-of-sample number, never used for selection.
        # Delist-day index in the fold's simulation column space, via the
        # shared helper (§P0-5) so the baselines force-sell delisted positions
        # exactly as the deep model does.
        Wp = outer_test_data["close_price"].shape[1] - config.seq_len
        delist_day = _fold_delist_day(
            delist_global, fold_eligible, val_start, Wp)
        outer_m = evaluate_portfolio(
            model, outer_test_data, config, device,
            horizon=config.horizon,
            top_fraction=0.1,
            raw_returns=outer_test_data["realized_return"],
            # Formal training must use the chronological
            # sleeve account — a prebuilt panel without price paths is a data
            # bug, not a reason to silently downgrade to the legacy estimator.
            require_price_path=True,
            # Emit the per-position ledger so the OOS tape
            # records every fill the account actually made, offline-replayable.
            return_ledger=True,
            # Known-delisted stocks are force-sold at the delisting close
            # instead of dangling as UNRESOLVED (§七-1).
            delist_day=delist_day,
            # §十五-1: project-wide trial count for the DSR multiplicity.
            n_trials=n_trials,
        )
        best_epoch = history.get("best_epoch_idx", 0) + 1

        # Daily OOS predictions: one return forecast per
        # (stock, entry day).  A window's entry is global column val_start+d,
        # so entry dates run global_dates[val_start .. val_start+val_len-1].
        oos_preds = _predict_outer(model, outer_test_data, config, device)
        if oos_preds is not None:
            n_w = oos_preds.shape[1]
            p0 = config.seq_len
            entry_dates = [_fmt_date(global_dates, val_start + d) for d in range(n_w)]
            # Window-day grid arrays (column d ↔ panel column seq_len+d), all
            # aligned exactly as evaluate_portfolio slices them, so a tape
            # consumer can reconstruct the sleeve account offline: the
            # selection pool (decision & history), the entry/open-validity
            # fill gate, the clean open->open return target (saved before
            # z-score) and its mask.
            dec = outer_test_data["decision_eligible_mask"][:, p0:p0 + n_w]
            hist = outer_test_data["history_eligible_mask"][:, p0:p0 + n_w]
            pool = dec & hist
            elig = outer_test_data["entry_eligible_mask"][:, p0:p0 + n_w]
            rt_mask = outer_test_data["return_target_mask"][:, p0:p0 + n_w]
            rt = outer_test_data["y_return_raw"][:, p0:p0 + n_w]
            # Price paths on the same grid, with `horizon` EXTRA columns so the
            # sleeve entered on the last signal day W-1 can still liquidate at
            # open[W-1+horizon] — identical to the grid
            # evaluate_portfolio passes to the sleeve simulator.
            price_grid = outer_test_data["close_price"][:, p0:p0 + n_w + config.horizon]
            open_grid = outer_test_data["open_price"][:, p0:p0 + n_w + config.horizon]
            price_dates = [_fmt_date(global_dates, val_start + d)
                           for d in range(n_w + config.horizon)]
            np.savez(
                os.path.join(oos_dir, f"fold_{fold:03d}.npz"),
                preds=oos_preds,
                dates=np.array(entry_dates),
                stocks=np.array(fold_stocks),
                decision_eligible=dec,
                history_eligible=hist,
                pool=pool,
                entry_eligible=elig,
                return_target_mask=rt_mask,
                return_target=rt,
                close_price=price_grid,
                open_price=open_grid,
                price_dates=np.array(price_dates),
                horizon=config.horizon,
                seq_len=config.seq_len,
                top_fraction=0.1,
                cost=config.txn_cost,
                # §P0-6: the force-sell delist-day grid (in this fold's sim
                # column space) so an offline replay force-sells delisted
                # positions exactly as the live sleeve did, and the content
                # hashes of the universe records the gates consumed.
                delist_day=delist_day,
                universe_status_hash=universe_hashes["universe_status_hash"],
                membership_hash=universe_hashes["membership_hash"],
                # §十二.2: calendar content hash — the tape must not blend folds
                # trained under a different holiday set / verified_until.
                calendar_hash=version_info["calendar_artifact_hash"],
                # A tape row must identify the data + model
                # that produced it, so every return number is traceable.
                data_version=version_info["data_manifest_hash"],
                model_hash=version_info["model_hash"],
                # §十六: the split model-identity hashes the formal continuous
                # replay REQUIRES to be identical across folds (architecture /
                # config / feature-schema).  weight_hash below — the actual
                # trained parameters — is allowed to differ per fold.
                model_source_hash=version_info["model_source_hash"],
                model_config_hash=version_info["model_config_hash"],
                feature_schema_hash=version_info["feature_schema_hash"],
                # Trained-parameter hash — the fold's tape maps
                # to the exact weights in fold_XXX_model.pt (config+source
                # model_hash is shared by all folds, this one is not).
                weight_hash=weight_hash,
                # §十五-1: the strategy policy + evaluator identity that produced
                # this tape.  The continuous replay REJECTS a directory whose
                # folds disagree on any of these — otherwise a horizon=5 fold
                # mixed with a horizon=20 fold would be replayed with the first
                # tape's policy explaining the whole account.
                evaluator_version=version_info["evaluator_version"],
                price_convention="open_to_open",
                exit_policy="scheduled_horizon_delayed_delist_force_sell",
                strategy_mode="long_top_fraction",
            )
            # Per-position ledger: the exact fills the long
            # sleeve account made — entry/exit price, exit status, gross/net
            # PnL and attributed costs — mapped to dates and stock codes so the
            # tape is self-contained.  Sum over net_pnl (resolved) +
            # unrealized_pnl (unresolved) == final_nav - 1 holds per fold by
            # construction (enforced inside _run_sleeve_sim, §十三-3).
            ledger_rows = outer_m.get("long_ledger")
            if ledger_rows:
                ldf = pd.DataFrame(ledger_rows)
                si = ldf["stock"].to_numpy(dtype=int)
                di = ldf["entry_day"].to_numpy(dtype=int)
                ldf["entry_date"] = [entry_dates[c] for c in di]
                ldf["stock_code"] = [fold_stocks[i] for i in si]
                ldf["prediction"] = oos_preds[si, di]
                ldf["candidate_eligible"] = pool[si, di]
                ldf["entry_eligible"] = elig[si, di]
                ldf["fold"] = fold
                # Data/model provenance columns on every tape
                # row — realized return + executed weight make the P&L fully
                # recomputable from prices alone.
                ldf["data_version"] = version_info["data_manifest_hash"]
                ldf["model_hash"] = version_info["model_hash"]
                # §十四-2: `entry_value`/`executed_weight` (both the nominal) are
                # split into entry_notional + target_weight + executed_weight
                # (notional/entry_nav) + entry_nav, so an offline consumer can
                # distinguish the intended weight from the cash-cap-reduced one.
                ldf = ldf[["fold", "entry_day", "entry_date", "stock",
                           "stock_code", "mode", "prediction",
                           "candidate_eligible", "entry_eligible",
                           "entry_price", "entry_notional", "target_weight",
                           "executed_weight", "entry_nav",
                           "shares", "scheduled_exit_day", "actual_exit_day",
                           "exit_status", "exit_price", "realized_return",
                           "mark_day", "mark_price", "gross_pnl",
                           "entry_cost", "exit_cost", "net_pnl",
                           "unrealized_pnl"]]
                ledger_path = os.path.join(oos_dir, f"fold_{fold:03d}_ledger.parquet")
                ldf.to_parquet(ledger_path)
                oos_ledgers.append(ldf)
                logger.info("  Fold %d: ledger %d filled positions -> %s",
                            fold, len(ldf), ledger_path)
            for d in range(n_w):
                oos_dates_all.extend([entry_dates[d]] * len(fold_stocks))
                oos_stocks_all.extend(fold_stocks)
                oos_preds_all.append(oos_preds[:, d])
                oos_pool_all.append(pool[:, d])
                oos_fold_all.extend([fold] * len(fold_stocks))
                oos_weight_hash_all.extend([weight_hash] * len(fold_stocks))

        if outer_m["n_periods"] >= 2:
            best_ls = outer_m["ls_sharpe"]
            all_sharpes.append(best_ls)
            # Inner-val eval nearest the deployed checkpoint — what selection
            # actually saw, reported honestly alongside the held-out outer
            # metrics (never report a post-hoc max).
            inner_eval_m, inner_eval_epoch = _best_eval_metrics(history)
            # Input-context date bounds of each segment — column t of the panel
            # is global_dates[t], so a slice [a,b) covers dates [a, b-1].
            # Semantic dates: entry day e buys at open[e],
            # the signal is produced after close[e-1], and the input context is
            # the seq_len days [e-seq_len, e).
            fold_histories.append({
                "history": history,
                "outer_metrics": outer_m,
                "best_epoch": best_epoch,
                "inner_eval_epoch": inner_eval_epoch,
                "inner_eval_ls_sharpe": inner_eval_m.get("ls_sharpe"),
                "inner_eval_ic": inner_eval_m.get("ic_mean"),
                "weight_hash": weight_hash,
                # §P0-6: the universe records this fold's gates consumed, so the
                # fold result is provably tied to those delist/membership files.
                "universe_status_hash": universe_hashes["universe_status_hash"],
                "membership_hash": universe_hashes["membership_hash"],
                "model_path": f"oos_preds/fold_{fold:03d}_model.pt",
                "train_start": _fmt_date(global_dates, 0),
                "train_end": _fmt_date(global_dates, inner_train_end - 1),
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "context_start": _fmt_date(global_dates, val_context_start),
                "signal_start": _fmt_date(global_dates, val_start - 1),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "exit_end": _fmt_date(
                    global_dates,
                    min(val_start + val_len - 1 + config.horizon, n_timesteps - 1)),
                "test_start": _fmt_date(global_dates, val_context_start),
                "test_end": _fmt_date(global_dates, val_end - 1),
            })
            logger.info(
                "  Fold %d: best@epoch%d OUTER-TEST LS_Sharpe=%.2f IC=%.4f(IR=%.2f) "
                "Long_Sharpe=%.2f Q5-Q1=%.1fbp ElgEW_Sharpe=%.2f SelUniEW_Sharpe=%.2f (%.1fs)",
                fold, best_epoch, best_ls,
                outer_m.get("ic_mean", 0), outer_m.get("ic_ir", 0),
                outer_m.get("long_sharpe", 0),
                outer_m.get("q5mq1_ret", 0) * 10000,
                outer_m.get("eligible_ew_sharpe", 0),
                outer_m.get("selected_universe_ew_sharpe", 0),
                elapsed,
            )
        else:
            logger.warning(
                "  Fold %d: outer-test too short for metrics (%.1fs)", fold, elapsed,
            )

        val_start -= step

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


if __name__ == "__main__":
    main()
