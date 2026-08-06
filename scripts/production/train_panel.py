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
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stoke_ml.config import get_project_root, load_config
from stoke_ml.data.calendar import (
    TradingCalendar,
    get_research_calendar,
)
from stoke_ml.data.universe import (
    delist_global_index,
    index_membership_mask,
    load_index_membership,
    load_universe_status,
    not_delisted_mask,
)
from stoke_ml.data.vintage_policy import VintagePolicy, channel_allowed
from stoke_ml.features.cache_manifest import current_config_hash, git_head
from stoke_ml.features.pipeline import (
    FeaturePipeline, _PIT_STATIC_COLS, fold_dead_feature_columns,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.panel_store import (
    load_panel_memmap,
    panel_store_complete,
    save_panel_memmap,
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
    _calendar_freeze,  # noqa: F401  re-exported for import-compat
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _discover_stocks(data_dir: str, limit: int | None = None) -> list[str]:
    from stoke_ml.data.storage import DataStorage
    stocks = DataStorage(data_dir).list_stocks()
    return stocks[:limit] if limit else stocks


def _exchange_group(stock_code: str) -> str:
    """Exchange bucket via the single market authority (§六): SH/SZ/BJ.

    The old ``6→SH; 0/3→SZ; else BJ`` heuristic already bucketed the real BSE
    ranges (43/83/87/88/920) into BJ, but it re-derived the market locally.
    Route through market_of_code so every caller shares one authority.
    """
    from stoke_ml.data.codes import market_of_code

    market = market_of_code(stock_code)
    # The panel daily store only holds A-share common equity, so market is
    # always SH/SZ/BJ.  market_of_code returns None for anything that is not a
    # known equity prefix (incl. legacy 老三板 4/8 codes and garbage input); the
    # fallback keeps that in the BJ bucket rather than KeyError-ing into
    # by_group or silently mis-bucketing it as SH.
    if market is not None:
        return market
    return "BJ"


def _load_index_universe(data_dir: str, index_codes: set[str]) -> list[str]:
    """Stocks ever in the given indices (PIT in_date/out_date union).

    Any stock that was a member at any point in history is included — the
    training span runs 2000-2026, so a fixed-date snapshot would silently
    drop stocks that left the index mid-period.
    """
    path = os.path.join(
        data_dir, "a_shares", "index_constituents_hist", "membership.parquet",
    )
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    members = df[df["index_code"].isin(index_codes)]["stock_code"].unique()
    return sorted(members)


def _assess_universe_reconciliation(report: dict, allow_missing: bool) -> list[str]:
    """§八-2: validate the gate report's universe_reconciliation.

    EVERY enforced run consumes a DEFINED universe, so the report must carry a
    reconciliation of the download Run Manifest's requested universe — training
    is never "whatever is on disk".  Requested stocks missing from disk are
    refused unless ``allow_missing`` (the explicit --allow-missing-universe
    escape); present-but-degraded stocks are NEVER escapable.  Returns the
    problems (empty list = acceptable).
    """
    recon = report.get("universe_reconciliation")
    if not isinstance(recon, dict):
        return [
            "universe reconciliation missing from gate report — run the gate "
            "with the download run manifest (build_features.py --quality-gate "
            "forwards it by default) so training can verify every requested "
            "stock is present (§八-2)"
        ]
    missing_codes = recon.get("missing_codes") or []
    missing_count = int(recon.get("missing_count") or len(missing_codes))
    degraded_codes = recon.get("degraded_codes") or []
    degraded_count = int(recon.get("degraded_count") or len(degraded_codes))
    problems: list[str] = []
    if missing_count and not allow_missing:
        problems.append(
            "gate universe reconciliation reports missing stocks: "
            + ", ".join(sorted(str(c) for c in missing_codes))
            + f" (requested={recon.get('requested_count')}, "
            f"present={recon.get('present_count')}).  Re-download the missing "
            f"members or pass --allow-missing-universe to proceed explicitly "
            f"and record the gap (§八-2)."
        )
    if degraded_count:
        problems.append(
            "gate universe reconciliation reports degraded stocks: "
            + ", ".join(
                sorted(
                    str(d.get("code"))
                    for d in degraded_codes
                    if isinstance(d, dict) and d.get("code")
                )
            )
            + f" (requested={recon.get('requested_count')}, "
            f"degraded={degraded_count}).  Present-but-degraded stocks are "
            f"not covered by --allow-missing-universe (§八-2)."
        )
    return problems


def _gate_enforced(args) -> bool:
    """Quality-gate report required and verified for this run (§六-2 / §八-2).

    ``--no-require-quality-gate`` (dev smoke) disables the whole gate.  A
    gate-enforced run's report must carry the requested-universe reconciliation
    (§八-2), so this predicate is also "universe reconciliation is enforced" —
    the csi* member-drop refusal in :func:`_resolve_universe` keys on it, and
    ``--allow-missing-universe`` only makes sense while the gate is enforced.
    """
    return not args.no_require_quality_gate


def _formal_mode(args) -> bool:
    """Formal (non-exploratory) methodology mode (§P0-7).

    ``--no-formal`` marks an exploratory run that may degrade universe gates
    when a required PIT artifact is missing instead of refusing to start.  The
    single-use lockbox additionally requires the quality gate to be enforced,
    so it combines ``_gate_enforced`` AND ``_formal_mode``.
    """
    return not args.no_formal


def _require_quality_gate(
    data_dir: str,
    prebuilt_dir: str | None,
    report_path: str,
    allow_missing: bool = False,
    **deprecated,
) -> dict:
    """Verify a matching quality-gate report covers the data this run consumes.

    §六-2: training must not read data the gate has not validated, or that
    changed since the gate PASS.  A missing / stale / mismatched report exits
    the run; --no-require-quality-gate disables the whole gate (dev smoke).

    ``universe_name`` / ``requested`` were historically accepted but never read
    by the body (the reconciliation is validated from the report's own
    ``universe_reconciliation`` section, not re-scoped by the caller), so they
    are dropped.  A legacy caller still passing them is absorbed via
    ``**deprecated`` so the shared baselines entry point keeps working.

    §八-2: EVERY enforced run consumes a DEFINED universe, so the gate report
    must carry a ``universe_reconciliation`` of the download Run Manifest's
    requested universe — training is never "whatever is on disk".  Requested
    stocks missing from disk are refused outright unless ``allow_missing`` (the
    explicit --allow-missing-universe escape) is passed; the escape still
    surfaces the missing list so the caller records it.  Present-but-degraded
    stocks are never escapable.  A ``scope=="full"`` report additionally must
    have actually audited the ``manifest`` AND ``contract_schema`` checks (both
    present and passed) — the full-scan floor §八-2 wants, verified from the
    checks array rather than the boolean flag alone.
    """
    if deprecated:
        bad = sorted(k for k in deprecated if k not in ("universe_name", "requested"))
        if bad:
            raise TypeError(
                f"_require_quality_gate got unexpected keyword arguments: {bad}"
            )

    if not os.path.isfile(report_path):
        raise SystemExit(
            f"quality gate report not found at {report_path} — run "
            f"scripts/production/build_features.py --quality-gate (or the gate directly) "
            f"before training, or pass --no-require-quality-gate to bypass."
        )
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    problems: list[str] = []
    if report.get("quality_gate_version") != QUALITY_GATE_VERSION:
        problems.append(
            f"gate version {report.get('quality_gate_version')!r} "
            f"!= {QUALITY_GATE_VERSION!r}"
        )
    if os.path.realpath(str(report.get("data_root") or "")) != os.path.realpath(data_dir):
        problems.append(f"data_root {report.get('data_root')!r} != {data_dir!r}")
    if report.get("calendar_version") != TradingCalendar.CALENDAR_VERSION:
        problems.append(
            f"calendar {report.get('calendar_version')!r} "
            f"!= {TradingCalendar.CALENDAR_VERSION!r}"
        )
    if report.get("contract_version") != contract_version():
        problems.append("daily contract changed since the gate ran")
    # §七.2: a sampled-scope report is only acceptable for a formal run when
    # the two cheap-to-scan-audit layers (manifest + contract) really covered
    # every file — the reviewer's "at least full manifest/contract + sampled
    # deep feature audit" floor.  A sampled report that skipped them means the
    # gate never saw whole files it could have refused on.
    if (report.get("scope") == "sample"
            and not report.get("manifest_contract_full_scan")):
        problems.append(
            "sample-scope gate without a full manifest/contract scan — §七.2 "
            "requires formal training to consume at least a full "
            "manifest+contract audit"
        )
    consumed = {"daily"}
    if prebuilt_dir:
        name = os.path.basename(os.path.normpath(prebuilt_dir))
        consumed.add(name)
        # §九.1: the gate must have validated the SAME absolute directory this
        # run reads, not just a matching basename.  The gate records where it
        # validated each dataset; a prebuilt at a different real path than the
        # gate reviewed is refused instead of trusted.
        bound = (report.get("dataset_paths") or {}).get(name)
        if bound and os.path.realpath(bound) != os.path.realpath(prebuilt_dir):
            problems.append(
                f"quality gate validated {name} at {bound}, but training reads "
                f"prebuilt from {os.path.realpath(prebuilt_dir)} — §九.1 path "
                f"mismatch"
            )
    gate_required = set(report.get("required_datasets") or [])
    if not consumed <= gate_required:
        problems.append(
            f"gate required datasets {sorted(gate_required)} do not cover "
            f"consumed {sorted(consumed)}"
        )
    live_hash = dataset_fingerprint(data_dir, sorted(gate_required))
    if report.get("data_manifest_hash") != live_hash:
        problems.append("dataset fingerprint changed since the gate PASS")
    # §八-2: EVERY enforced run consumes a DEFINED universe — the gate report
    # must reconcile the download Run Manifest's requested universe (missing
    # stocks are refused outright unless --allow-missing-universe; degraded
    # stocks are never escapable).
    problems.extend(_assess_universe_reconciliation(report, allow_missing))
    # §八-2: a FULL-scope report must have really audited the manifest +
    # contract_schema checks — both present in the checks array and both passed.
    # A full-scan report whose checks array lacks either (or ran neither) is a
    # report that never saw whole files it could have refused on.
    if report.get("scope") == "full":
        checks = report.get("checks")
        if not isinstance(checks, list):
            problems.append(
                "full-scope gate report lacks a checks array — cannot verify "
                "the manifest/contract_schema audit floor (§八-2)"
            )
        else:
            by_name = {c.get("name"): c for c in checks if isinstance(c, dict)}
            for need in ("manifest", "contract_schema"):
                entry = by_name.get(need)
                if entry is None or not entry.get("passed"):
                    problems.append(
                        f"full-scope gate check '{need}' missing or not passed "
                        f"— the manifest/contract_schema audit floor is not met "
                        f"(§八-2)"
                    )
    # §八-2: --allow-missing-universe explicitly tolerates a gate report that
    # FAILED purely on the universe reconciliation (requested stocks missing
    # beyond tolerance).  Every OTHER check must still pass and nothing may be
    # degraded — otherwise the report's failure is not the escapable gap, and
    # the missing list must still be surfaced for the caller to record.
    checks_arr = report.get("checks")
    other_failed = (
        [
            c.get("name")
            for c in checks_arr
            if isinstance(c, dict) and c.get("name") != "universe" and not c.get("passed")
        ]
        if isinstance(checks_arr, list)
        else ["<no checks array>"]
    )
    recon = report.get("universe_reconciliation")
    universe_escape = (
        allow_missing
        and isinstance(recon, dict)
        and int(recon.get("missing_count") or 0) > 0
        and int(recon.get("degraded_count") or 0) == 0
        and not other_failed
    )
    if not report.get("passed") and not universe_escape:
        problems.append("gate did not PASS")
    if problems:
        raise SystemExit(
            "quality-gate report mismatch — refusing to train on unvalidated "
            "data: " + "; ".join(problems)
        )
    return report


def _check_verified_until_scope(global_dates, enforce: bool) -> str | None:
    """§九-3: refuse a formal run whose panel axis reaches past verified_until.

    Forward-estimate trading days (2027+ A-share closures) are not verified
    exchange fact, so a formal run (quality gate enforced) that spans them is
    refused instead of silently mixing guessed holidays into the panel axis.
    Uses the strict calendar's own bound so the refusal is exactly the
    strict-mode contract.  ``enforce`` is False for exploratory runs that opt
    out via --no-require-quality-gate.  Returns the refusal message (caller
    raises SystemExit) or None when in scope.
    """
    if not enforce or global_dates is None or not len(global_dates):
        return None
    strict_cal = get_research_calendar(strict=True)
    lo = pd.Timestamp(global_dates[0]).date()
    hi = pd.Timestamp(global_dates[-1]).date()
    try:
        strict_cal.get_trading_days(lo, hi)
    except ValueError as exc:
        return f"formal run refused: {exc}"
    return None


def _resolve_universe(
    all_stocks: list[str],
    universe: str,
    limit: int | None,
    seed: int,
    data_dir: str,
    formal: bool = False,
) -> tuple[list[str], str]:
    """Select the training universe by name.

    Returns (resolved_stocks, description).  The description records exactly
    what was selected (mode / seed / count) for the experiment artifacts.

    Modes:
      first      — first N by code (legacy behaviour, alphabetical bias)
      random     — seeded sample of N from all stocks (no code-order bias)
      stratified — seeded sample targeting N, balanced across SH/SZ/BJ
      all        — every stock (ignores --stocks)
      csi300/500/800 — index constituents (PIT union), --stocks caps count

    §八-2: index membership is never silently truncated by missing data.
    csi* members with no daily K-line on disk are counted and surfaced.  When
    universe reconciliation is ENFORCED (the quality gate is required — a run
    whose gate report must reconcile the requested universe, §八-2) the run
    refuses to start: the dropped members are listed, so the gap is visible
    instead of silently shrinking the index to what happened to be downloaded.
    When reconciliation is not enforced (exploratory / --no-formal / dev smoke
    with --no-require-quality-gate), the run degrades to a prominent warning
    and records the drop count in the description so the artifact still
    captures what was dropped.

    ``formal`` is the gate-enforcement predicate (:func:`_gate_enforced`), not
    merely ``--no-formal`` — the refusal must key on whether the run is
    actually required to reconcile the requested universe.
    """
    if limit is None:
        limit = len(all_stocks)
    all_sorted = sorted(all_stocks)
    if universe == "all":
        return all_sorted, f"all ({len(all_sorted)} stocks)"
    if universe == "first":
        chosen = all_sorted[:limit]
        return chosen, f"first {len(chosen)} (sorted by code)"
    if universe == "random":
        rng = np.random.RandomState(seed)
        k = min(limit, len(all_stocks))
        chosen = rng.choice(all_sorted, size=k, replace=False).tolist()
        return chosen, f"random {len(chosen)} (seed={seed})"
    if universe == "stratified":
        rng = np.random.RandomState(seed)
        by_group: dict[str, list[str]] = {"SH": [], "SZ": [], "BJ": []}
        for code in all_sorted:
            by_group[_exchange_group(code)].append(code)
        chosen: list[str] = []
        remaining = limit
        for group, codes in by_group.items():
            share = min(remaining // 3, len(codes))
            chosen.extend(rng.choice(codes, size=share, replace=False).tolist())
            remaining -= share
        if remaining > 0:
            used = set(chosen)
            leftover = [c for codes in by_group.values() for c in codes if c not in used]
            chosen.extend(
                rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist()
            )
        rng.shuffle(chosen)
        return chosen, f"stratified {len(chosen)} (seed={seed}, SH/SZ/BJ balanced)"
    if universe in ("csi300", "csi500", "csi800"):
        idx_map = {"csi300": {"000300"}, "csi500": {"000905"}, "csi800": {"000300", "000905"}}
        members = _load_index_universe(data_dir, idx_map[universe])
        if not members:
            return [], f"{universe}: no membership data"
        # §八-2: keep only members we have daily K-line for, but NEVER silently.
        # The membership file includes codes with no data (delisted / not
        # downloaded); a formal run refuses so the index is not silently shrunk
        # to whatever was downloaded, and exploratory runs record the drop so
        # the artifact still exposes it.
        have = set(all_stocks)
        dropped = sorted(c for c in members if c not in have)
        members = [c for c in members if c in have]
        if dropped:
            listing = ", ".join(dropped[:20]) + (" ..." if len(dropped) > 20 else "")
            if formal:
                raise SystemExit(
                    f"universe={universe}: {len(dropped)} index members have no "
                    f"daily K-line on disk and would be silently dropped — "
                    f"refusing to run because universe reconciliation is "
                    f"enforced: the run must NOT silently shrink the requested "
                    f"index (§八-2).  Missing: {listing}.  Download the missing "
                    f"members or re-run with --no-require-quality-gate to "
                    f"degrade explicitly and record the drop."
                )
            logger.warning(
                "universe=%s: dropped %d/%d index members with no daily K-line "
                "on disk (recorded in description): %s",
                universe, len(dropped), len(dropped) + len(members), listing,
            )
        chosen = members[:limit] if limit else members
        note = f" (PIT union, cap={limit})"
        if dropped and not formal:
            note += f", dropped {len(dropped)} no-data members"
        return chosen, f"{universe} {len(chosen)}{note}"
    raise ValueError(f"unknown --universe: {universe}")


def _require_all_universe_prebuilt(
    universe: str, prebuilt: str | None, store_complete: bool = False,
) -> None:
    """§七-P0: ``--universe all`` without ``--prebuilt`` is refused outright.

    The full market cannot be feature-engineered in RAM (~225GB of feature
    arrays on a ~96GB host); it must read prebuilt panel features
    (build_features.py --panel-mode).  A complete ``--panel-store`` is an
    equivalent source (the full panel already persisted as mmap'd arrays), so
    it lifts the requirement.  Extracted from main() so the refusal is
    unit-testable.
    """
    if universe == "all" and not prebuilt and not store_complete:
        raise SystemExit(
            "--universe all requires --prebuilt (or a complete --panel-store): "
            "the full market cannot be feature-engineered in memory (5530 "
            "stocks x the full feature grid ~= 225GB on a ~96GB host, §七-P0).  "
            "Run scripts/production/build_features.py --panel-mode first, then "
            "re-run with --prebuilt data/features_panel, or point --panel-store "
            "at a previously-built store."
        )


# §T2: base preference for each documented use_* dimension — what the feature
# set would include with an unrestricted vintage policy.  Exactly matches the
# switches the pipeline used before this change: FeaturePipeline defaults every
# use_* to True; board/sector/concept/limit_up/topic are OFF by default for
# non-vintage engineering reasons (deferred / ablation-only / low density).
_BASE_DIM_PREFERENCE = {
    "sentiment": True, "guba": True, "comment": True, "announcement": True,
    "margin": True, "northbound": True, "dragon_tiger": True,
    "fundamental": True, "earnings": True, "valuation": True,
    "etf_flow": True, "capital_flow": True, "block_trade": True,
    "shareholder": True, "lockup": True, "dividend": True,
    "industry": True, "macro": True, "pledge": True,
    "index_membership": True, "market_env": True, "market_env_refine": True,
    "board": False, "sector": False, "concept": False,
    "limit_up": False, "topic": False,
}
# dim → FeaturePipeline kwarg name; only "announcement" differs (use_announcements).
_SWITCH_KEY = {"announcement": "use_announcements"}


def _panel_pipeline_kwargs(args, seq_len: int) -> dict:
    """FeaturePipeline constructor kwargs for the panel build.

    Single source of truth for the ``use_*`` switch set — shared by the live
    pipeline construction AND the panel-store meta fingerprint, so a change to
    the switches OR the vintage policy is caught by the store staleness guard.
    ``--vintage-policy`` is applied as an AND-filter over each channel's base
    preference (the policy can only turn channels OFF, never force one ON).
    """
    policy = VintagePolicy(args.vintage_policy)
    kwargs = {
        _SWITCH_KEY.get(dim, f"use_{dim}"): pref and channel_allowed(dim, policy)
        for dim, pref in _BASE_DIM_PREFERENCE.items()
    }
    kwargs["seq_len"] = seq_len
    kwargs["minute_mode"] = args.minute
    return kwargs


def _panel_store_meta(args, seq_len: int, n_stocks: int | None = None) -> dict:
    """Build-time fingerprint persisted in a panel store's meta.json.

    Re-checked by load_panel_memmap on a store-backed re-run so a stale store
    (different horizon / universe / feature switches / date window) is refused
    instead of silently training on wrong targets — mirrors cache_manifest's
    config_hash + range staleness logic.
    """
    return {
        "horizon": args.horizon,
        "seq_len": seq_len,
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "n_stocks": n_stocks,
        "feature_switches": _panel_pipeline_kwargs(args, seq_len),
        "config_hash": current_config_hash(),
        "git_commit": git_head(),
    }


def _validate_panel_store_path(path: str) -> None:
    """Refuse a --panel-store that points at an existing FILE, not a dir.

    save_panel_memmap raises on the same condition; this surfaces it as a
    clear CLI error before any K-line work happens.
    """
    p = Path(path)
    if p.exists() and not p.is_dir():
        raise SystemExit(
            f"--panel-store {path} exists but is not a directory — a panel "
            "store is a directory of .npy/.json files.  Point at a new/empty "
            "directory or remove the conflicting file."
        )


def _resolve_panel(
    args, stock_list: list[str], seq_len: int, data_dir,
    required_set: set[str], _store_load: bool,
) -> tuple[dict, dict]:
    """Resolve ``(panel_data, channel_manifest)`` for a training run.

    §十六: when a COMPLETE ``--panel-store`` is present (``_store_load``), the
    K-line load AND the feature build are skipped entirely — the stored panel's
    arrays are mmap'd and read lazily downstream, so a re-run never reads 5530
    stocks' OHLCV only to discard it.  The store is loaded under its meta.json
    config guard (refuses stale horizon/universe/feature-switch targets).

    Otherwise the panel is engineered live from K-line (+ aux), and when
    ``--panel-store`` is set the result is persisted there with its meta.json
    fingerprint for a future fast re-run.

    Returns ``(panel_data, channel_manifest)``.  ``channel_manifest`` is the
    channel-coverage dict for the required-channel gate; the store path probes
    it from the stored panel's ``has_*`` flags (prebuilt semantics), the live
    path from ``load_aux_data`` (or the same flag probe under ``--prebuilt``).
    """
    if _store_load:
        logger.info("Loading panel memmap store from %s (skipping K-line load "
                    "+ feature build)", args.panel_store)
        panel_data = load_panel_memmap(
            args.panel_store,
            expected_meta=_panel_store_meta(args, seq_len, len(stock_list)))
        channel_manifest = _prebuilt_channel_coverage(panel_data)
        return panel_data, channel_manifest

    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)
    if args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        ms = MinuteStorage(data_dir)
        frames = []
        for code in stock_list:
            df = ms.load(code, args.start, args.end, args.minute_frequency)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date
                df["stock_code"] = code
                frames.append(df)
        if not frames:
            logger.error("No minute data loaded for any stock — run download_minute.py first")
            sys.exit(1)
        logger.info("Minute mode: %d stocks @ %s-min, %d available in storage",
                    len(frames), args.minute_frequency,
                    len(ms.list_stocks(args.minute_frequency)))
    else:
        from stoke_ml.data.storage import DataStorage
        ds = DataStorage(data_dir)
        frames = []
        for code in stock_list:
            df = ds.load_daily(code, args.start, args.end,
                               require_valid_manifest=True)
            if df is not None and not df.empty:
                df["stock_code"] = code
                frames.append(df)
        if not frames:
            logger.error("No data loaded for any stock")
            sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    logger.info("Panel shape: %s", panel.shape)

    # Load auxiliary data (unless --no-aux / --prebuilt — the store path
    # skips this entirely; the stored panel already has every aux channel
    # baked in).
    aux_data = None
    channel_manifest = {}
    if not args.no_aux and not args.prebuilt:
        logger.info("Loading auxiliary data...")
        t_aux = time.time()
        aux_data, channel_manifest = load_aux_data(
            stock_list, data_dir, args.start, args.end,
            required_channels=required_set,
        )
        logger.info("Aux data loaded in %.1fs", time.time() - t_aux)

    fp = FeaturePipeline(**_panel_pipeline_kwargs(args, seq_len))
    panel_data = fp.build_panel_features(
        panel, aux_data=aux_data, horizon=args.horizon, prebuilt_dir=args.prebuilt,
        require_feature_manifest=args.require_feature_manifest,
    )
    if args.panel_store:
        save_panel_memmap(
            panel_data, args.panel_store,
            meta=_panel_store_meta(args, seq_len, len(stock_list)))
        logger.info("Saved panel memmap store to %s", args.panel_store)
    if args.prebuilt:
        # Live per-channel loading is skipped in prebuilt mode; probe the
        # panel's has_* flags instead so the experiment still records what
        # actually got in.
        channel_manifest = _prebuilt_channel_coverage(panel_data)
    return panel_data, channel_manifest


# §七-P0 universe memory guard thresholds (GB).  --universe all is refused
# above the safety line by default; csi800 warns at the same line and is
# refused only above the hard ceiling (or when the estimate exceeds the host's
# actual available memory, when that is introspectable).
_UNIVERSE_MEMORY_WARN_GB = 48.0
_UNIVERSE_MEMORY_REFUSE_GB = 48.0
_UNIVERSE_MEMORY_HARD_GB = 96.0


def _panel_memory_gb(n_stocks: int, n_timesteps: int, n_features: int) -> float:
    """§七-P0: dominant resident panel memory estimate in GB (float32 arrays)."""
    return int(n_stocks) * int(n_timesteps) * int(n_features) * 4 / (1024 ** 3)


def _enforce_universe_memory(
    universe: str,
    n_stocks: int,
    n_timesteps: int,
    n_features: int,
    *,
    allow_override: bool = False,
    available_gb: float | None = None,
) -> tuple[float, str]:
    """§七-P0: refuse / warn when a universe's panel cannot realistically fit in RAM.

    Returns ``(est_gb, action)`` where ``action`` is the UN-overridden verdict:
    "refuse" / "warn" / "ok".

    Verdict rules:
      * universe == "all": est > _UNIVERSE_MEMORY_REFUSE_GB → "refuse".
      * universe == "csi800": est > _UNIVERSE_MEMORY_HARD_GB → "refuse";
        est > _UNIVERSE_MEMORY_WARN_GB → "warn".
      * any other universe: "ok".
      * universe in ("all", "csi800") and ``available_gb`` is known and
        est > available_gb → "refuse" (the estimate alone already guarantees
        the panel will not fit on THIS host).

    Side effect: when the verdict is "refuse" and ``allow_override`` is False,
    raises SystemExit with a message that names the estimate, the available
    memory (when known), the threshold, and the ``--allow-high-risk-universe``
    escape hatch.  When the verdict is "warn", OR "refuse" with
    ``allow_override=True``, logs a prominent WARNING instead.
    """
    est_gb = _panel_memory_gb(n_stocks, n_timesteps, n_features)
    action = "ok"
    if universe == "all" and est_gb > _UNIVERSE_MEMORY_REFUSE_GB:
        action = "refuse"
    elif universe == "csi800":
        if est_gb > _UNIVERSE_MEMORY_HARD_GB:
            action = "refuse"
        elif est_gb > _UNIVERSE_MEMORY_WARN_GB:
            action = "warn"
    # §七-P0 precheck (the plan's "预检 by available memory" for csi800): a
    # risky-universe panel that cannot fit the host's ACTUAL available memory
    # is refused even below the static lines.  Other universes have no static
    # guard and must never be refused here (a transiently low `available`
    # snapshot must not block a documented default run).
    if (
        universe in ("all", "csi800")
        and available_gb is not None
        and est_gb > available_gb
        and action != "refuse"
    ):
        action = "refuse"
    if action == "refuse" and not allow_override:
        avail = f"vs available {available_gb:.1f} GB" if available_gb is not None else \
            "vs host available memory (unknown — psutil not installed)"
        raise SystemExit(
            f"universe={universe}: panel memory estimate {est_gb:.1f} GB "
            f"(n_stocks={n_stocks} x n_timesteps={n_timesteps} x "
            f"n_features={n_features} x 4B) {avail} — this panel will very "
            f"likely OOM the host (§七-P0).  Re-scope with a smaller "
            f"--stocks cap (default 500 is safe) or pass "
            f"--allow-high-risk-universe to run it anyway."
        )
    if action in ("warn", "refuse"):
        logger.warning(
            "universe=%s: panel memory estimate %.1f GB (n_stocks=%d x "
            "n_timesteps=%d x n_features=%d x 4B) — §七-P0 risk, the feature "
            "panel may not fit in RAM.  Re-scope or pass "
            "--allow-high-risk-universe.",
            universe, est_gb, n_stocks, n_timesteps, n_features,
        )
    return est_gb, action


# Channel coverage manifest: every aux channel is loaded
# per-stock with per-stock error counting, so an experiment that silently lost a
# whole channel (storage schema update, missing dir) is caught instead of
# finishing quietly.  `status` distinguishes three empty states:
#   MISSING — channel absent from disk (loaded 0, errors 0)
#   FAILED  — storage construction/read broke (errors == n_stocks)
#   PARTIAL — some stocks loaded, some errored
_HAS_FLAG_CHANNELS = {
    "has_news": "sentiment",
    "has_guba_post": "guba",
    "has_comment": "comment",
    "has_announce": "announcement",
    "has_forecast": "earnings",
    "has_pledge": "pledge",
    "has_hot_board": "concept",
}


def _new_channel_entry(requested: bool, required: bool) -> dict:
    return {
        "requested": requested,
        "required": required,
        "loaded_stocks": 0,
        "coverage": 0.0,
        "errors": 0,
        "status": "MISSING",
    }


def _finalize_channel(entry: dict, name: str, loaded: int, errors: int, n: int) -> None:
    entry["loaded_stocks"] = loaded
    entry["errors"] = errors
    entry["coverage"] = round(loaded / n, 4) if n else 0.0
    entry["status"] = (
        "FAILED" if loaded == 0 and errors > 0 else
        "MISSING" if loaded == 0 else
        "PARTIAL" if loaded < n else "OK"
    )
    logger.info("[%s] loaded %d/%d stocks (errors=%d) %s",
                name, loaded, n, errors, entry["status"])


def _load_channel_aux(
    name: str,
    stock_list: list[str],
    result: dict[str, dict[str, pd.DataFrame]],
    manifest: dict[str, dict],
    make_storage,      # Callable[[], object] — storage construction (raises → channel FAILED)
    load_one,          # Callable[[object, str], pd.DataFrame | None]
    required: bool = False,
) -> None:
    """Per-stock aux load with per-stock error counting."""
    entry = _new_channel_entry(True, required)
    manifest[name] = entry
    n = len(stock_list)
    try:
        storage = make_storage()
    except Exception as exc:
        entry["errors"] = n
        entry["status"] = "FAILED"
        entry["note"] = f"storage construction failed: {exc}"
        logger.warning("[%s] storage unavailable — %s", name, exc)
        return
    loaded = 0
    errors = 0
    for code in stock_list:
        try:
            df = load_one(storage, code)
            if df is not None and not df.empty:
                result[code][name] = df
                loaded += 1
        except Exception:
            errors += 1
    _finalize_channel(entry, name, loaded, errors, n)


def load_aux_data(
    stock_list: list[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    required_channels: set[str] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    """Load auxiliary data (sentiment, guba, margin, etc.) per stock.

    Returns (result, manifest):
      result   — {stock_code: {"sentiment": df, "guba": df, ...}}
      manifest — per-channel coverage (requested/required/loaded_stocks/
                 coverage/errors/status).
    """
    from stoke_ml.data.news_storage import NewsStorage
    from stoke_ml.data.guba_storage import GubaStorage
    from stoke_ml.data.market_wide_storage import MarketWideStorage
    from stoke_ml.data.fundamental_storage import FundamentalStorage
    from stoke_ml.data.comment_storage import CommentStorage
    from stoke_ml.data.announcement_storage import AnnouncementStorage

    result: dict[str, dict[str, pd.DataFrame]] = {c: {} for c in stock_list}
    manifest: dict[str, dict] = {}
    required_set = set(required_channels or ())

    # Sentiment (news)
    _load_channel_aux(
        "sentiment", stock_list, result, manifest,
        make_storage=lambda: NewsStorage(data_dir),
        load_one=lambda ns, code: ns.load_daily_sentiment(code, start_date, end_date),
        required=("sentiment" in required_set),
    )

    # Announcements (CNINFO PDF body sentiment preferred, EastMoney fallback)
    def _make_ann():
        cninfo_dir = os.path.join(data_dir, "a_shares", "cninfo_announcements", "sentiment")
        return (cninfo_dir, AnnouncementStorage(data_dir))

    def _load_ann(storage_tuple, code):
        cninfo_dir, a_store = storage_tuple
        path = os.path.join(cninfo_dir, f"{code}.parquet")
        if os.path.isfile(path):
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            if not df.empty:
                return df.sort_values("date").reset_index(drop=True)
            return None
        return a_store.load_daily_sentiment(code, start_date, end_date)

    _load_channel_aux(
        "announcement", stock_list, result, manifest,
        make_storage=_make_ann,
        load_one=_load_ann,
        required=("announcement" in required_set),
    )

    # Guba
    _load_channel_aux(
        "guba", stock_list, result, manifest,
        make_storage=lambda: GubaStorage(data_dir),
        load_one=lambda gs, code: gs.load_daily_sentiment(code, start_date, end_date),
        required=("guba" in required_set),
    )

    # Comment
    _load_channel_aux(
        "comment", stock_list, result, manifest,
        make_storage=lambda: CommentStorage(data_dir),
        load_one=lambda cs, code: cs.build_features(code, start_date, end_date),
        required=("comment" in required_set),
    )

    # Fundamental (quarterly, backfilled from 2010 so the forward-fill spans)
    _load_channel_aux(
        "fundamental", stock_list, result, manifest,
        make_storage=lambda: FundamentalStorage(data_dir),
        load_one=lambda fs, code: fs.load(code, "2010-01-01", end_date),
        required=("fundamental" in required_set),
    )

    # MarketWideStorage channels (margin/northbound/dragon_tiger/capital_flow/
    # block_trade/shareholder/lockup/dividend/valuation) — identical pattern.
    for ch in (
        "margin", "northbound", "dragon_tiger", "capital_flow", "block_trade",
        "shareholder", "lockup", "dividend", "valuation",
    ):
        _load_channel_aux(
            ch, stock_list, result, manifest,
            make_storage=lambda ch=ch: MarketWideStorage(data_dir, ch),
            load_one=lambda st, code, ch=ch: st.load(code, start_date, end_date),
            required=(ch in required_set),
        )

    # ETF Flow (sector-level, aggregated to market-wide per date, broadcast to
    # every stock — not a per-stock channel).
    entry = _new_channel_entry(True, "etf_flow" in required_set)
    manifest["etf_flow"] = entry
    try:
        etf_base = os.path.join(data_dir, "a_shares", "etf_flow")
        etf_frames = []
        if os.path.isdir(etf_base):
            for f in os.listdir(etf_base):
                if f.startswith("sector_") and f.endswith(".parquet"):
                    etf_frames.append(pd.read_parquet(os.path.join(etf_base, f)))
        if etf_frames:
            etf_all = pd.concat(etf_frames, ignore_index=True)
            etf_all["date"] = pd.to_datetime(etf_all["date"])
            etf_agg = etf_all.groupby("date").agg(
                etf_flow_sum=("etf_flow_sum", "sum"),
                etf_amount_sum=("etf_amount_sum", "sum"),
            ).reset_index()
            for code in stock_list:
                result[code]["etf_flow"] = etf_agg
            entry["loaded_stocks"] = len(stock_list)
            entry["coverage"] = 1.0 if stock_list else 0.0
            entry["status"] = "OK"
            logger.info("[etf_flow] aggregated from %d sector files "
                        "(broadcast to %d stocks)", len(etf_frames), len(stock_list))
        else:
            logger.info("[etf_flow] no sector files found — MISSING")
    except Exception as exc:
        entry["errors"] = len(stock_list)
        entry["status"] = "FAILED"
        entry["note"] = str(exc)
        logger.warning("[etf_flow] aggregation failed — %s", exc)

    loaded = sum(1 for v in result.values() if v)
    logger.info("Aux data loaded for %d/%d stocks", loaded, len(stock_list))
    return result, manifest


def _prebuilt_channel_coverage(panel_data: dict) -> dict:
    """Channel coverage probed from a prebuilt panel's has_* flags.

    past_observed is (N, T, D); each has_* flag is True on exactly the
    (stock, day) cells where that aux channel delivered data (the pipeline
    ZI-fills absent cells, so False == no data).  Coverage = fraction of grid
    cells with the flag set, across the whole loaded panel.  Channels without
    a has_* flag carry no presence marker in the arrays, so their coverage is
    left null (not decodable).
    """
    po = panel_data.get("past_observed")
    col_names = panel_data.get("past_observed_cols") or []
    index = {name: i for i, name in enumerate(col_names)}
    channels: dict[str, dict] = {}
    if po is None or po.ndim != 3:
        channels["_note"] = {
            "status": "UNKNOWN",
            "message": "panel lacks a past_observed grid",
        }
        return channels
    for flag, channel in _HAS_FLAG_CHANNELS.items():
        if flag not in index:
            continue
        mask = po[:, :, index[flag]] > 0
        present = int(np.count_nonzero(mask))
        channels[channel] = {
            "requested": True,
            "required": False,
            "loaded_stocks": None,  # cell-level probe, not per-stock
            "coverage": round(float(mask.mean()), 4),
            "errors": 0,
            "status": "OK" if present else "MISSING",
            "cells": int(mask.size),
            "flag": flag,
        }
    return channels


def _quality_fail_reason(close: np.ndarray) -> str | None:
    """Structural quality check on a stock's CLOSE prefix.

    `close` must be ONLY the rows up to the fold's train_end (caller slices the
    panel) — the check is inherently PIT.  Returns a reason string if the stock
    is unusable on that prefix, else None.  Row-level badness (a non-positive
    close, a dead row) is NOT a reason to eject the stock — the pipeline masks
    those rows; only structural corruption (all-NaN prefix,
    >50 % daily vol, >1000 % forward move) excludes the stock from THAT fold.
    """
    if close.size == 0 or np.isnan(close).all():
        return "all_nan"
    ret = np.diff(close) / (close[:-1] + 1e-8)
    if np.nanstd(ret) > 0.50:  # >50 % daily vol = data error
        return "hi_vol"
    if len(close) > 5:
        fwd_ret = (close[5:] - close[:-5]) / (close[:-5] + 1e-8)
        if np.nanmax(np.abs(fwd_ret)) > 10.0:
            return "extreme_fwd"
    return None


def _fold_eligible_stocks(panel_data: dict, train_end: int) -> np.ndarray:
    """Per-fold PIT stock-level eligibility.

    A stock is eligible for a fold iff its close path is structurally clean on
    columns [0, train_end) — ONLY data before the fold's train boundary.  The
    old global `_filter_quality` loaded 2015→2099 once and ejected a stock from
    EVERY fold if a 2025 row was bad; this judges each fold on its own past, so
    real-market volatility can no longer masquerade as a "bad stock".

    Uses the panel's close_price grid (N, T), aligned to panel_stocks order, so
    no per-fold DataFrame regroup is needed.
    """
    close_price = panel_data["close_price"]  # (N, T) float32, NaN = no data
    n_stocks = close_price.shape[0]
    keep = np.ones(n_stocks, dtype=bool)
    for i in range(n_stocks):
        if _quality_fail_reason(close_price[i, :train_end]) is not None:
            keep[i] = False
    return keep


def _mask_stocks(data: dict, keep: np.ndarray) -> dict:
    """Drop ineligible stocks (axis 0) from every panel slice array."""
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[keep]
        else:
            out[k] = v
    return out


def _require_universe_artifacts(
    data_dir: str, universe_name: str, formal: bool,
) -> None:
    """§P0-7: a FORMAL experiment must never silently no-op its universe gates.

    ``exploratory`` runs (``--no-formal``) may degrade with a prominent marker;
    a formal run REFUSES to start when a gate's required artifact is missing:

      - csi300/csi500/csi800 require PIT ``membership.parquet`` intervals
        (without them the "per-day member" gate collapses to the historical
        union — the exact silent no-op §P0-7 calls out);
      - every universe requires delisting records (``delisted.parquet``) — the
        sleeve's force-sell policy (§七-1) is part of the executed task;
      - ``all`` additionally requires IPO records so the delisted merge is real.

    Returns normally when everything present; exits 1 with a precise list when
    a required artifact is missing in formal mode.
    """
    missing: list[str] = []

    def _present(*relparts: str) -> bool:
        path = os.path.join(data_dir, *relparts)
        if not os.path.isfile(path):
            return False
        try:
            return not pd.read_parquet(path).empty
        except Exception as exc:  # noqa: BLE001 — a corrupt artifact is as bad as absent
            logger.warning("universe artifact unreadable: %s (%s)", path, exc)
            return False

    if universe_name in ("csi300", "csi500", "csi800"):
        if not _present("a_shares", "index_constituents_hist", "membership.parquet"):
            missing.append("membership.parquet (PIT index-membership gate)")
    if not _present("a_shares", "universe", "delisted.parquet"):
        missing.append("delisted.parquet (delisting force-sell policy)")
    if universe_name == "all":
        if not _present("a_shares", "universe", "ipo.parquet"):
            missing.append("ipo.parquet (delisted-stock universe merge)")

    if not missing:
        return
    if formal:
        logger.error(
            "universe=%s: required PIT artifacts missing — %s.  A formal "
            "experiment must NOT silently no-op its universe gates (that would "
            "measure a different task than intended); rerun with --no-formal "
            "for an explicitly-degraded exploratory run (§P0-7).",
            universe_name, "; ".join(missing),
        )
        sys.exit(1)
    logger.warning(
        "[exploratory] universe=%s: %s missing — universe gate DEGRADED "
        "(silent no-op); formal runs refuse to start in this state (§P0-7).",
        universe_name, "; ".join(missing),
    )


def _fold_universe_gates(
    global_dates: np.ndarray,
    panel_stocks: list[str],
    universe_name: str,
    data_dir: str,
    formal: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, pd.DataFrame]:
    """§七-1/§七-3: compute the whole-run universe gates ONCE, before the fold loop.

    Both the deep model (train_panel.py) and the baselines
    (train_baselines_panel.py) must consume the SAME candidate-pool gates
    (§P0-5), so this is the single shared construction.  Returns, all in the
    panel's (N_stocks, T) grid space:

      nd_mask       — 未退市: blocks ENTRY from a known delisting column on.
      mem_mask      — per-day index membership (in_date <= date < out_date) for
                      csi300/csi500/csi800; None for other universes so fold
                      gates become a no-op for the "all A" / stratified studies.
      delist_global — per-stock delisting day in global panel-column space
                      (each fold's delist_day for the sleeve simulator).
      universe_status — the raw universe frame; returned so callers can hash
                      it into their artifacts (§P0-6 universe_status_hash).

    Missing universe parquets → empty status → delist_global all -1 and
    nd_mask all True (no force-sell, no entry gate), so a data-dir without
    records never crashes — the strict formal-mode failure for missing
    artifacts is enforced here up front (§P0-7).
    """
    _require_universe_artifacts(data_dir, universe_name, formal)
    universe_status = load_universe_status(data_dir)
    delist_global = delist_global_index(
        global_dates, universe_status, panel_stocks,
    )
    nd_mask = not_delisted_mask(global_dates, panel_stocks, universe_status)
    universe_index_codes = {
        "csi300": {"000300"},
        "csi500": {"000905"},
        "csi800": {"000300", "000905"},
    }.get(universe_name, set())
    mem_mask = None
    if universe_index_codes:
        membership_df = load_index_membership(data_dir, sorted(universe_index_codes))
        if membership_df.empty:
            logger.warning(
                "universe=%s: no membership.parquet intervals — per-day "
                "index-membership gate is a no-op (candidate pool keeps the "
                "full historical-member union)", universe_name,
            )
        else:
            mem_mask = index_membership_mask(global_dates, panel_stocks, membership_df)
    return nd_mask, mem_mask, delist_global, universe_status


def _apply_candidate_gates(
    dd: dict,
    tslice: slice,
    rows: np.ndarray,
    nd_mask: np.ndarray,
    mem_mask: np.ndarray | None,
) -> None:
    """§七-3: merge the universe gates into ONE evaluation candidate pool.

    ``dd`` is a fold slice already masked to eligible stocks; ``rows`` maps its
    row axis back to original panel-stock rows and ``tslice`` maps its columns
    to global panel columns, so the gates AND into
    ``dd["decision_eligible_mask"]`` in this fold's row/column space and
    ``_candidate_pool`` picks them up automatically.  §八.3: inner_train is by
    default left ungated — the model still learns from the broad
    historical-member union; only what gets RANKED as a tradable candidate is
    restricted.  ``--strict-index-training`` instead ANDs the per-day
    membership gate into inner_train's ``entry_eligible_mask`` (the dataset
    valid_mask) so the training loss matches the evaluation candidate pool.
    """
    cols = np.arange(tslice.start, tslice.stop)
    gate = nd_mask[np.ix_(rows, cols)]
    if mem_mask is not None:
        gate = gate & mem_mask[np.ix_(rows, cols)]
    dd["decision_eligible_mask"] &= gate


def _gate_inner_train_membership(
    inner_train: dict,
    mem_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    """§八.3 strict mode: AND per-day index membership into the inner-TRAIN
    ``entry_eligible_mask`` (which feeds the dataset valid_mask, the per-sample
    training-loss mask) so the model learns only from index-member days —
    matching the evaluation candidate pool.  ``rows`` maps the fold's row axis
    back to original panel-stock rows and ``cols`` are the inner_train grid
    columns (both aligned by the fold's slicing order).
    """
    inner_train["entry_eligible_mask"] &= mem_mask[np.ix_(rows, cols)]


def _gate_descriptions(
    consumes_membership: bool, strict_index_training: bool,
) -> tuple[str, str]:
    """§八.3: human-readable ``(eval_gate, train_gate)`` descriptions for the
    summary.  eval_gate always gates 未退市, plus per-day membership for
    universes that consume membership.parquet.  train_gate is the broad
    historical-member union (ungated) unless strict_index_training gates it by
    per-day membership.
    """
    eval_gate = "not_delisted" + (
        " + per-day-membership" if consumes_membership else "")
    train_gate = (
        "per-day-membership"
        if strict_index_training and consumes_membership
        else "union (ungated)"
    )
    return eval_gate, train_gate


def _fold_delist_day(
    delist_global: np.ndarray,
    fold_eligible: np.ndarray,
    val_start: int,
    Wp: int,
) -> np.ndarray:
    """Delist-day index in the fold's simulation column space.

    Sim column d ↔ global column val_start+d (evaluate_portfolio slices prices
    at seq_len within the outer_test window starting at val_context_start =
    val_start - seq_len), so subtract val_start.  Values outside [0, Wp) clamp
    to -1 — the force-sell never fires within this window.
    """
    dd_global = delist_global[fold_eligible]
    return np.where(
        (dd_global >= val_start) & (dd_global < val_start + Wp),
        dd_global - val_start, -1,
    )


def _universe_artifact_hashes(
    universe_status: pd.DataFrame,
    data_dir: str,
    universe_name: str,
) -> dict:
    """§P0-6: content hashes of the universe artifacts a run's gates consumed.

    ``universe_status_hash`` covers the delist/list records that drive
    ``delist_global`` and ``nd_mask``; ``membership_hash`` covers the
    index-membership intervals that drive ``mem_mask`` for csi300/csi500/csi800
    (None for universes where membership is not consumed).  Every fold tape and
    summary embeds these so a later replay can prove it used the SAME universe
    records — a delist-file or membership edit between runs invalidates the
    OOS tape instead of passing silently.
    """
    status_hash = hashlib.sha1()
    if universe_status is not None and not universe_status.empty:
        status_hash.update(universe_status.to_csv(index=False).encode("utf-8"))
    membership_hash: str | None = None
    universe_index_codes = {
        "csi300": {"000300"},
        "csi500": {"000905"},
        "csi800": {"000300", "000905"},
    }.get(universe_name, set())
    if universe_index_codes:
        path = os.path.join(
            data_dir, "a_shares", "index_constituents_hist", "membership.parquet")
        h = hashlib.sha1()
        if os.path.isfile(path):
            h.update(pd.read_parquet(path).to_csv(index=False).encode("utf-8"))
        membership_hash = h.hexdigest()[:16]
    return {
        "universe_status_hash": status_hash.hexdigest()[:16],
        "membership_hash": membership_hash,
    }


def _cross_sectional_normalize(
    y_arr: np.ndarray,
    mask_arr: np.ndarray,
    min_stocks: int = 5,
) -> np.ndarray:
    """Z-score normalize returns across stocks within each date.

    Preserves cross-sectional ordering while giving each date's return
    distribution zero mean and unit variance.  Dates with too few valid
    stocks are left unchanged.

    Returns a new array (does not mutate input).
    """
    y_out = y_arr.copy()
    n_stocks, n_dates = y_arr.shape
    for t in range(n_dates):
        valid = mask_arr[:, t] if mask_arr is not None else np.ones(n_stocks, dtype=bool)
        if valid.sum() < min_stocks:
            continue
        vals = y_arr[valid, t]
        mean_t = float(np.nanmean(vals))
        std_t = max(float(np.nanstd(vals)), 1e-8)
        y_out[valid, t] = (y_arr[valid, t] - mean_t) / std_t
    return y_out


def _slice_panel(panel_data: dict, tslice: slice, price_pad: int = 0) -> dict:
    """Slice every time-axis array of the panel by `tslice`.

    Static features are (N, T, D) PIT — sliced on the time
    axis like every other panel array.  Arrays that downstream code mutates in
    place (y_return z-score + clip, and their neighbours) are copied so one
    fold's normalization never corrupts the shared panel for later folds.

    `price_pad`: extend the close/open price columns by this many beyond
    `tslice.stop` (capped at the panel end).  The sleeve-account evaluation
    needs open[t+h] to liquidate a position entered at open[t],
    so the last `price_pad` sleeves get a real exit instead of a forced carry.

    `y_return_raw` is a copy of the RAW open-to-open return saved BEFORE the
    caller z-scores/clips `y_return`: clean IC and quintile
    spreads must be computed on raw returns, not on the normalized model target.
    """
    stop = tslice.stop
    out = {
        "static_features": panel_data["static_features"][:, tslice, :],
        "past_known": panel_data["past_known"][:, tslice],
        "past_observed": panel_data["past_observed"][:, tslice],
        "y_direction": panel_data["y_direction"][:, tslice],
        "y_return_raw": panel_data["y_return"][:, tslice].copy(),
        "y_return": panel_data["y_return"][:, tslice].copy(),
        "y_volatility": panel_data["y_volatility"][:, tslice].copy(),
        "observation_mask": panel_data["observation_mask"][:, tslice],
        "entry_eligible_mask": panel_data["entry_eligible_mask"][:, tslice],
        "return_target_mask": panel_data["return_target_mask"][:, tslice],
        "vol_target_mask": panel_data["vol_target_mask"][:, tslice],
        "realized_return": panel_data["realized_return"][:, tslice].copy(),
        # REBASE date_indices to LOCAL column space.  panel_builder emits the
        # GLOBAL calendar position (0..max_T-1); a fold slice with start > 0
        # must restart at 0 so date-centric consumers' window placement
        # ``window_idx = date_idx - seq_len`` stays inside the (N, n_windows)
        # grid.  Same-date stocks keep equal local indices within a dataset,
        # so PairwiseRankingLoss grouping is unchanged.
        "date_indices": (
            panel_data["date_indices"][:, tslice].copy() - (tslice.start or 0)
        ),
        "decision_eligible_mask": panel_data["decision_eligible_mask"][:, tslice],
        "history_eligible_mask": panel_data["history_eligible_mask"][:, tslice],
    }
    # Price paths feed the sleeve-account evaluation; a stale prebuilt panel
    # without them just falls back to the legacy path in evaluate_portfolio.
    if "close_price" in panel_data and "open_price" in panel_data:
        max_T = panel_data["close_price"].shape[1]
        pstop = min(stop + price_pad, max_T) if price_pad > 0 else stop
        out["close_price"] = panel_data["close_price"][:, tslice.start:pstop]
        out["open_price"] = panel_data["open_price"][:, tslice.start:pstop]
    return out


def _fmt_date(global_dates, idx):
    """Global-calendar position → 'YYYY-MM-DD'.  Out of range → None."""
    if global_dates is None or idx < 0 or idx >= len(global_dates):
        return None
    return str(np.datetime_as_string(global_dates[idx], unit="D"))


def _weight_hash(model) -> str:
    """Content hash of a model's TRAINED parameters (float32, CPU).

    The version dict's `model_hash` only fingerprints config + architecture
    source — every fold shares it.  This one hashes the actual state_dict so
    an OOS tape row / checkpoint can be tied to the exact weights that
    produced it, and two differently-trained folds get different digests.
    """
    return _state_dict_hash(model.state_dict())


def _augment_sequence(
    pk: np.ndarray,
    po: np.ndarray,
    obs_mask: np.ndarray | None = None,
    noise_std: float = 0.01,
    mask_prob: float = 0.05,
    feat_dropout: float = 0.02,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FIXED per-fold corruption pass on a training panel (§十一.1).

    This is NOT online, per-sample augmentation.  It corrupts a whole
    (n_stocks, T, F) block once and returns a fixed copy that every epoch
    reuses verbatim, so the model never sees re-sampled noise:

    1. Gaussian noise ~ N(0, noise_std) — per-element independent, gated by
       `obs_mask` (True = real observation) so zero-padded history of new
       listings stays exactly zero instead of gaining fake noise the model
       would read as real data.
    2. Time masking — ONE global contiguous segment is zeroed for EVERY
       stock in the block (a single ``start``/``mask_len`` shared across the
       stock axis), not an independent segment per stock.
    3. Feature dropout — ONE global feature subset is zeroed for EVERY
       stock (a single boolean mask over the feature axis), not an
       independent subset per stock.

    Conservative magnitudes are the point: this exists to probe robustness,
    not to expand the effective sample size.  Because the corruption is
    global and static across epochs, it is opt-in ablation only and OFF by
    default in the formal baseline.
    """
    if rng is None:
        rng = np.random.RandomState()

    pk_aug = pk.copy()
    po_aug = po.copy()

    # 1. Gaussian noise (per-element, independent), only on real-observation days
    if noise_std > 0:
        noise_pk = rng.randn(*pk.shape).astype(np.float32) * noise_std
        noise_po = rng.randn(*po.shape).astype(np.float32) * noise_std
        if obs_mask is not None:
            obs_b = obs_mask[..., None].astype(np.float32)
            noise_pk *= obs_b
            noise_po *= obs_b
        pk_aug += noise_pk
        po_aug += noise_po

    # 2. Time masking: zero out a random contiguous block of length 1-5
    if mask_prob > 0 and pk.shape[1] >= 3:
        T = pk.shape[1]
        mask_len = rng.randint(1, min(6, T // 2 + 1))
        if rng.random() < mask_prob:
            start = rng.randint(0, T - mask_len)
            pk_aug[:, start:start + mask_len, :] = 0.0
            po_aug[:, start:start + mask_len, :] = 0.0

    # 3. Feature dropout: zero out random feature dimensions
    if feat_dropout > 0:
        for arr in [pk_aug, po_aug]:
            if arr.shape[2] > 0:
                mask = rng.random(arr.shape[2]) < feat_dropout
                arr[:, :, mask] = 0.0

    return pk_aug, po_aug


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


def _save_artifacts(
    outdir: str,
    args: argparse.Namespace,
    resolved: list[str],
    used: list[str],
    universe_desc: str,
    summary: dict | None,
    channel_manifest: dict | None = None,
    version: dict | None = None,
) -> str:
    """Persist the experiment: args, resolved/used universes, fold summary,
    the channel-coverage manifest, and the frozen data/code
    versions."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if version is not None:
        with open(os.path.join(outdir, "version.json"), "w", encoding="utf-8") as f:
            json.dump(version, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "universe_resolved.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n# n={len(resolved)}\n")
        f.write("\n".join(resolved))
        f.write("\n")
    with open(os.path.join(outdir, "universe_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n"
                f"# n={len(used)} (per-fold PIT eligibility applied inside the "
                f"fold loop)\n")
        f.write("\n".join(used))
        f.write("\n")
    if channel_manifest is not None:
        with open(os.path.join(outdir, "channel_coverage.json"),
                  "w", encoding="utf-8") as f:
            json.dump(channel_manifest, f, indent=2, ensure_ascii=False)
    if summary is not None:
        if channel_manifest:
            summary["channel_coverage"] = {
                k: {"status": v.get("status"),
                    "coverage": v.get("coverage")}
                for k, v in sorted(channel_manifest.items())
                if not k.startswith("_")
            }
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Experiment artifacts saved to %s", outdir)
    return outdir


def _predict_outer(model, outer_data, config, device) -> np.ndarray | None:
    """Run the deployed checkpoint over the outer-test panel.

    Date-centric (§七/§十六): uses the same eval-mode no-sampler DataLoader as
    evaluate_portfolio (training=False → eval_mask / full candidate pool,
    max_stocks_per_date=None, batch_size=1).  Each ``__getitem__`` returns one
    date's (M, ...) tensors; the return prediction is placed directly at
    ``preds[stock_indices, window_idx]`` so the sparse grid is reconstructed
    without a flat cat+reshape (which would mismatch when sum(M_i) !=
    n_stocks*n_windows).  Cells for ineligible stocks/windows stay NaN.  Window
    d enters at panel column seq_len + d — i.e. global column val_start + d of
    the full panel.  Returns None only when the outer panel has no windows
    (all-NaN grid).
    """
    n_stocks = outer_data["static_features"].shape[0]
    val_ds = PanelDataset(outer_data, seq_len=config.seq_len,
                          min_history=config.min_history,
                          max_stocks_per_date=None, training=False)
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )
    n_windows = val_ds.n_windows
    seq_len = val_ds.seq_len
    model.eval()
    preds = torch.full((n_stocks, n_windows), float("nan"))
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_y, date_idx_t, _dm, _rm, _vm, stock_indices = batch
            if stock_indices.numel() == 0:
                continue
            # Per-stock window indices (supports mixed-date batches).
            window_idx = date_idx_t - seq_len
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            preds[stock_indices, window_idx] = pred_ret.cpu().squeeze(-1)
    if torch.isnan(preds).all():
        return None
    return preds.numpy()


def _best_eval_metrics(history: dict) -> tuple[dict, int]:
    """Metrics of the inner-val eval nearest the deployed checkpoint.

    Returns (metrics_dict, eval_epoch) for the evaluation whose 1-based epoch
    sits closest to best_epoch_idx+1 — NOT the post-hoc max, which would
    double-count hindsight.  Histories without val_eval_epochs (legacy) are
    assumed to have evaluated on the 5,10,15,... grid.  Empty histories yield
    ({}, 0).
    """
    metrics = history.get("val_metrics") or []
    if not metrics:
        return {}, 0
    best = history.get("best_epoch_idx", 0) + 1  # 1-based deployed epoch
    eval_epochs = history.get("val_eval_epochs")
    if not eval_epochs:
        eval_epochs = [5 + 5 * i for i in range(len(metrics))]
    nearest = min(range(len(metrics)), key=lambda i: abs(eval_epochs[i] - best))
    return metrics[nearest], eval_epochs[nearest]


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
                             "and read lazily by PanelDataset / _slice_panel).  A "
                             "store's meta.json config fingerprint is checked on "
                             "load — a mismatch (horizon/seq_len/start/end/"
                             "universe/feature switches) REFUSES the run so a stale "
                             "store can't silently train on wrong targets.  "
                             "Otherwise the panel built this run is persisted there "
                             "for future runs.  Default: off — build in memory as "
                             "always.")
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
                        action=argparse.BooleanOptionalAction, default=False,
                        help="§八.3: gate the inner-TRAIN loss by per-day index "
                             "membership for csi300/csi500/csi800.  Default: "
                             "off — inner_train learns from the broad "
                             "historical-member union and only the RANKED "
                             "candidate pools (inner_val/outer_test) are "
                             "membership-gated.  Only meaningful when the "
                             "universe consumes membership.parquet")
    parser.add_argument("--vintage-policy", type=str, default="safe-only",
                        choices=["safe-only", "allow-revised"],
                        help="§T2: vintage-admission policy for the feature set.  "
                             "safe-only (default) admits raw_vintage_safe + "
                             "derived_versioned channels and DENIES "
                             "latest_revised_aligned ones (fundamental/macro/"
                             "earnings/valuation/pledge/shareholder/"
                             "index_membership/market_env_refine); allow-revised "
                             "additionally admits latest_revised_aligned channels "
                             "(legacy / ablation use).")
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
    required_set = {c.strip() for c in (args.require_aux_channels or "").split(",") if c.strip()}
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

    # Required-channel gate: a required channel with ZERO
    # coverage aborts the experiment instead of silently training on air.
    missing_required = sorted(
        ch for ch in required_set
        if channel_manifest.get(ch, {}).get("loaded_stocks", 0) == 0
        and channel_manifest.get(ch, {}).get("coverage", 0.0) == 0
    )
    if missing_required:
        logger.error("Required aux channels have ZERO coverage: %s — aborting",
                     ", ".join(missing_required))
        sys.exit(1)
    for ch in sorted(required_set):
        if ch not in channel_manifest:
            logger.warning("Required aux channel '%s' has no coverage probe in "
                           "this mode (prebuilt panel without has_* flag) — "
                           "coverage cannot be verified", ch)
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
        global_dates, enforce=_gate_enforced(args))
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
    available_gb = None
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        available_gb = None  # psutil optional — the static thresholds still guard
    _enforce_universe_memory(
        args.universe, n_stocks, n_timesteps, n_features,
        allow_override=args.allow_high_risk_universe,
        available_gb=available_gb,
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
        version_info, config, augmentation=bool(args.augment))
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
            # PanelDataset needs at least seq_len+1 rows for one window
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
            "train_gate": train_gate_desc,
            "eval_gate": eval_gate_desc,
            # §P0-6: content hashes of the universe records the whole-run gates
            # consumed (delist status + index membership), so the summary is
            # provably tied to those artifact files.
            "universe_status_hash": universe_hashes["universe_status_hash"],
            "membership_hash": universe_hashes["membership_hash"],
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
