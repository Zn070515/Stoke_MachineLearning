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
import dataclasses
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from stoke_ml.config import get_project_root, load_config
from stoke_ml.data.calendar import (
    TradingCalendar,  # noqa: F401  re-exported for import-compat
)
from stoke_ml.data.vintage_policy import universe_membership_provenance
from stoke_ml.features.pipeline import (
    _PIT_STATIC_COLS,  # noqa: F401  re-exported for import-compat
    fold_dead_feature_columns,  # noqa: F401  re-exported for import-compat
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.panel_store import (
    panel_store_complete,
)
from stoke_ml.models.panel.train import train_panel  # noqa: F401  re-exported for import-compat
from stoke_ml.models.panel.evaluate import (
    _run_sleeve_sim,  # noqa: F401  re-exported for import-compat
    compute_equity_curve,  # noqa: F401  re-exported for import-compat
    compute_max_drawdown,  # noqa: F401  re-exported for import-compat
    compute_sharpe,  # noqa: F401  re-exported for import-compat
    evaluate_portfolio,  # noqa: F401  re-exported for import-compat
)
from scripts.production.data_quality_gate import (
    QUALITY_GATE_VERSION, contract_version, dataset_fingerprint,
)
from scripts.production.train_panel_oos import (
    _file_sha256,  # noqa: F401  re-exported for import-compat
    _state_dict_hash,
    _verify_tape_weight_hash,  # noqa: F401  re-exported for import-compat
    _replay_continuous_oos,  # noqa: F401  re-exported for import-compat
)
from scripts.production.train_panel_registry import (
    _calendar_freeze,
    _experiment_version,
    _EXPERIMENT_REGISTRY_PATH,
    _LOCKBOX_MARKER_PATH,  # noqa: F401  re-exported for import-compat
    _read_lockbox_marker,  # noqa: F401  re-exported for import-compat
    _mark_lockbox_used,  # noqa: F401  re-exported for import-compat
    _require_single_use_lockbox,
    _ablation_desc,  # noqa: F401  re-exported for import-compat
    _objective_desc,  # noqa: F401  re-exported for import-compat
    _experiment_signature,
    _asset_identity_digest,
    _distinct_trial_count,
    _registry_lock,  # noqa: F401  re-exported for import-compat
    _load_experiment_registry,
    _append_experiment_registry,  # noqa: F401  re-exported for import-compat
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
    _fold_eligible_stocks,  # noqa: F401  re-exported for import-compat
    _mask_stocks,  # noqa: F401  re-exported for import-compat
    _require_universe_artifacts,  # noqa: F401  re-exported for import-compat
    _fold_universe_gates,
    _apply_candidate_gates,  # noqa: F401  re-exported for import-compat
    _gate_inner_train_membership,  # noqa: F401  re-exported for import-compat
    _gate_descriptions,
    _fold_delist_day,  # noqa: F401  re-exported for import-compat
    _universe_artifact_hashes,
    _cross_sectional_normalize,  # noqa: F401  re-exported for import-compat
    _slice_panel,  # noqa: F401  re-exported for import-compat
    _fmt_date,
    _augment_sequence,  # noqa: F401  re-exported for import-compat
    _save_artifacts,  # noqa: F401  re-exported for import-compat
    _run_fold_loop,
    _FoldLoopResult,  # noqa: F401  re-exported for import-compat
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
    _gold_manifest_paths,  # noqa: F401  re-exported for import-compat
    _stock_era_coverage,  # noqa: F401  re-exported for import-compat
    _probe_era_coverage,  # noqa: F401  re-exported for import-compat
    _merge_era_coverage,  # noqa: F401  re-exported for import-compat
    _era_capable_channels,  # noqa: F401  re-exported for import-compat
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
    _weight_hash,  # noqa: F401  re-exported for import-compat
    _predict_outer,  # noqa: F401  re-exported for import-compat
    _best_eval_metrics,  # noqa: F401  re-exported for import-compat
)
from scripts.production.train_panel_cli import (
    _ABLATIONS,  # noqa: F401  re-exported for import-compat
    build_parser,
)
from scripts.production.train_panel_summary import (
    _SummaryInputs,
    _finalize_summary,  # noqa: F401  re-exported for import-compat
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    args = build_parser().parse_args()

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
    # §P2-15: bind the profile's coverage-contract CONTENT (the profile NAME is
    # bound via feature_profile above; the per-channel (metric, minimum) is
    # serialized here so a retuned threshold is a distinct trial) and the asset
    # identity (canonical digest of the DataAssetContract definitions the
    # consumed channels adopted).  An inactive profile carries {} → None → 'none'
    # so profile-inactive runs' signatures stay stable.
    _coverage_contracts = {
        ch: {"metric": c.metric, "threshold": c.threshold}
        for ch, c in coverage_contracts.items()
    }
    experiment_signature = _experiment_signature(
        version_info, config, augmentation=bool(args.augment),
        vintage_policy=args.vintage_policy, feature_profile=profile_name,
        universe_membership=universe_membership,
        coverage_contracts=_coverage_contracts or None,
        asset_identity=_asset_identity_digest(
            ch for ch in channel_manifest if not ch.startswith("_")))
    n_trials = _distinct_trial_count(experiment_registry, experiment_signature)
    fold_result = _run_fold_loop(
        panel_data, panel_stocks, config, device, args,
        global_dates, n_timesteps, val_len, step, purge, lockbox_len,
        nd_mask, mem_mask, delist_global, universe_hashes, version_info,
        oos_dir, n_trials,
    )

    _finalize_summary(_SummaryInputs(
        fold_result=fold_result,
        version_info=version_info,
        outdir=outdir,
        oos_dir=oos_dir,
        n_trials=n_trials,
        experiment_registry=experiment_registry,
        experiment_signature=experiment_signature,
        universe_desc=universe_desc,
        args=args,
        profile_name=profile_name,
        train_gate_desc=train_gate_desc,
        eval_gate_desc=eval_gate_desc,
        universe_hashes=universe_hashes,
        universe_membership=universe_membership,
        global_dates=global_dates,
        lockbox_start=lockbox_start,
        lockbox_len=lockbox_len,
        n_timesteps=n_timesteps,
        channel_manifest=channel_manifest,
        universe_resolved=universe_resolved,
        universe_used=universe_used,
        config=config,
    ))


if __name__ == "__main__":
    main()
