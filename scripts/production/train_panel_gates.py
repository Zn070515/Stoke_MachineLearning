"""Quality-gate and universe-reconciliation gates for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — the quality-gate report
verification, the universe-reconciliation assessment, the formal/exploratory
mode predicates, the frozen feature-profile required-channel set, the
channel-coverage gate, and the verified-calendar scope check.  ``train_panel``
re-exports these names for backward compatibility.
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from stoke_ml.config.feature_profile import FEATURE_PROFILES, profile_for
from stoke_ml.data.calendar import TradingCalendar, get_research_calendar
from scripts.production.data_quality_gate import (
    QUALITY_GATE_VERSION,
    contract_version,
    dataset_fingerprint,
)

logger = logging.getLogger(__name__)


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

def _resolve_required_set(args) -> tuple[set[str], dict, str]:
    """Resolve the required-channel set + minimum-coverage map (§十四).

    ``--require-aux-channels`` always feeds the set (existing behavior); the
    frozen feature profile (feature_profile.py) ADDS its ``required_channels``
    when the gate is active — a FORMAL, gate-enforced run with a named profile
    (default ``headline_v1``).  ``--feature-profile none``, ``--no-formal`` and
    ``--no-require-quality-gate`` each make the profile contribution empty, so
    ``required_set`` is just the explicit channels and ``min_cov`` is empty.

    Returns ``(required_set, min_cov, profile_name)``; ``profile_name`` is
    ``"none"`` when no profile is active.  An UNKNOWN named profile on an
    active gate aborts loudly (a typo must not silently skip the coverage
    gate).
    """
    extra = {
        c.strip() for c in (args.require_aux_channels or "").split(",")
        if c.strip()
    }
    profile_name = getattr(args, "feature_profile", None)
    active = (
        _gate_enforced(args) and _formal_mode(args)
        and profile_name not in (None, "", "none")
    )
    if not active:
        return extra, {}, "none"
    profile = profile_for(profile_name)
    if profile is None:
        raise SystemExit(
            f"unknown feature profile {profile_name!r} — --feature-profile "
            f"must be one of {sorted(FEATURE_PROFILES)} or 'none'")
    required_set = set(profile.required_channels) | extra
    return required_set, dict(profile.minimum_coverage), profile_name

def _enforce_channel_coverage(
    required_set: set[str],
    channel_manifest: dict,
    min_cov: dict[str, float] | None = None,
) -> None:
    """§十四 required-channel + minimum-coverage gate.

    (1) A REQUIRED channel that IS probed with ZERO coverage (manifest entry
    present, loaded 0, coverage 0) aborts the experiment instead of silently
    training on air.  (2) A required channel with NO coverage probe in this
    mode (prebuilt panel without a has_* flag, e.g. margin/northbound/
    capital_flow) warns — coverage cannot be verified.  (3) A channel with a
    ``min_cov`` threshold that IS probeable (manifest entry present with a
    finite float ``coverage``) must meet its threshold or the experiment
    aborts; a ``min_cov`` channel that is NOT probeable falls through to the
    required warn loop above (a ``min_cov`` channel is always also in
    ``required_set`` for the shipped profiles).

    Note the "IS probed" scope on (1): an ABSENT channel is not "zero
    coverage", it is not decodable in this mode — requiring e.g. margin on the
    prebuilt path (no has_* flag) must warn, not abort, or the default
    headline_v1 profile would refuse every prebuilt run.
    """
    missing_required = sorted(
        ch for ch in required_set
        if ch in channel_manifest
        and channel_manifest[ch].get("loaded_stocks", 0) == 0
        and channel_manifest[ch].get("coverage", 0.0) == 0
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
    for ch in sorted(min_cov or {}):
        entry = channel_manifest.get(ch)
        if entry is None:
            continue  # not probeable → covered by the required warn loop above
        cov = entry.get("coverage")
        if not isinstance(cov, (int, float)) or isinstance(cov, bool):
            continue  # null / non-numeric coverage = not probeable
        if not np.isfinite(float(cov)):
            continue
        if float(cov) < min_cov[ch]:
            logger.error(
                "Required aux channel '%s' coverage %.4f < minimum %.4f "
                "(--feature-profile) — aborting",
                ch, float(cov), min_cov[ch])
            sys.exit(1)

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
