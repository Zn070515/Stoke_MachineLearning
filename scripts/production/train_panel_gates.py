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

from stoke_ml.config.feature_profile import (
    CoverageContract, FEATURE_PROFILES, profile_for,
)
from stoke_ml.data.calendar import (
    TradingCalendar, calendar_artifact_hash, get_research_calendar,
)
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
    """Resolve the required-channel set + per-channel coverage contracts (§十四).

    ``--require-aux-channels`` always feeds the set (existing behavior); the
    frozen feature profile (feature_profile.py) ADDS its ``required_channels``
    when the gate is active — a FORMAL, gate-enforced run with a named profile
    (default ``headline_v1``).  ``--feature-profile none``, ``--no-formal`` and
    ``--no-require-quality-gate`` each make the profile contribution empty, so
    ``required_set`` is just the explicit channels and ``coverage_contracts`` is
    empty.

    Two §二十-1 safety properties hold when a named profile is ACTIVE:

    * A named profile is ONE indivisible research recipe — the run's
      ``--vintage-policy`` MUST equal the profile's declared ``vintage_policy``
      (headline_v1 declares ``revision-safe``), or the required set resolved
      under the profile would not match the channels the pipeline actually
      opens.  A mismatch ABORTS loudly (SystemExit) instead of silently gating
      a different recipe than the one the model consumes.
    * ``--allow-fundamental-ablation`` forces the fundamental channel ON in the
      pipeline, so under an active profile the flag ADDS ``"fundamental"`` to
      the required set — a channel the model actually consumes must be gated,
      not ride ungated.

    Returns ``(required_set, coverage_contracts, profile_name)``;
    ``coverage_contracts`` maps each profile channel to its CoverageContract
    (the metric measured + the minimum), and ``profile_name`` is ``"none"`` when
    no profile is active.  An UNKNOWN named profile on an active gate aborts
    loudly (a typo must not silently skip the coverage gate).
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
    # §二十-1: a named profile is ONE indivisible research recipe — the run's
    # --vintage-policy MUST equal the profile's declared vintage_policy, or the
    # required-channel set (resolved under the profile) would not match the
    # channels the pipeline actually opens (headline-strict closes industry;
    # allow-revised opens 8 more latest_revised channels).  Refuse loudly — a
    # mismatched vintage silently gates a different recipe than the one the
    # model consumes.
    args_vintage = getattr(args, "vintage_policy", None)
    if args_vintage != profile.vintage_policy:
        raise SystemExit(
            f"--feature-profile {profile_name!r} declares vintage_policy "
            f"{profile.vintage_policy!r} but --vintage-policy is "
            f"{args_vintage!r} — a named profile and its vintage policy are "
            f"one indivisible research recipe (§二十-1).  Use --feature-profile "
            f"none for a custom vintage policy, or align --vintage-policy to "
            f"the profile.")
    required_set = set(profile.required_channels) | extra
    # §二十-1: --allow-fundamental-ablation forces the fundamental channel ON
    # regardless of policy — the same class of hole as a mismatched vintage.  A
    # channel the model actually consumes must be REQUIRED (gated), so under an
    # active profile the ablation flag ADDS fundamental to the required set
    # rather than letting it ride ungated.
    if getattr(args, "allow_fundamental_ablation", False):
        required_set.add("fundamental")
    return required_set, dict(profile.coverage_contracts), profile_name

def _enforce_channel_coverage(
    required_set: set[str],
    channel_manifest: dict,
    coverage_contracts: dict[str, CoverageContract] | None = None,
    formal: bool = False,
) -> None:
    """§十四 required-channel + per-channel coverage-contract gate (§T4).

    For each required channel, in sorted order:

    * UNPROBEABLE — no manifest entry at all, OR a manifest entry whose DECLARED
      metric is missing / non-finite (the strict-None-refusal: a channel cannot
      be verified in this mode, e.g. a prebuilt panel without a has_* flag and
      no persisted store manifest).  FORMAL mode ABORTS (coverage cannot be
      verified, so a formal run must not proceed); EXPLORE mode warns.  This is
      the semantic-strictness fix: the old gate only warned here, silently
      accepting an unverifiable required channel.
    * ZERO coverage — the declared metric is a finite number <= 0 (a required
      channel that IS probed but delivered nothing) aborts unconditionally,
      exactly the legacy ``loaded 0 / coverage 0`` abort.
    * BELOW-MINIMUM — a channel with a ``CoverageContract`` whose declared-metric
      value is below the contract's threshold aborts.

    The DECLARED metric is read from the channel's ``CoverageContract`` (e.g.
    ``date_coverage`` for the market-wide broadcast channels etf_flow /
    industry / market_env, ``era_coverage`` for the §T8 sparse text channels
    sentiment / guba).  ``era_coverage`` is the provider-era retrieval coverage
    probed by ``_resolve_panel`` (``_merge_era_coverage``): the mean over the
    era-observable stocks of each stock's calendar-day fraction of its provider
    window that was actually retrieved — distinguishing a stock with genuinely
    no events (no_event, covered) from an era we never observed (not_observed,
    excluded from the numerator and reported via ``era_not_observed_stocks``).
    A channel with ZERO era-observable stocks leaves ``era_coverage`` absent, so
    it is UNPROBEABLE here and aborts a formal run (correct: nothing was
    observed).  A presence-only channel (required but no contract, e.g.
    dragon_tiger) defaults to ``stock_coverage``.  There is NO legacy
    ``entry["coverage"]`` fallback in the gate — the declared-metric key must
    be present and finite, else the channel is unprobeable.  A channel in the
    manifest but NOT in ``required_set`` is not gated at all.
    """
    coverage_contracts = coverage_contracts or {}

    def _unprobeable(ch: str) -> None:
        if formal:
            logger.error(
                "formal mode: required aux channel '%s' has no coverage probe "
                "in this mode (prebuilt/store without a persisted manifest or "
                "has_* flag) — cannot verify coverage; build with "
                "--panel-store (live) or pass --no-formal", ch)
            sys.exit(1)
        logger.warning("Required aux channel '%s' has no coverage probe in "
                       "this mode (prebuilt panel without has_* flag) — "
                       "coverage cannot be verified", ch)

    for ch in sorted(required_set):
        entry = channel_manifest.get(ch)
        if entry is None:
            _unprobeable(ch)
            continue
        contract = coverage_contracts.get(ch)
        metric = contract.metric if contract is not None else "stock_coverage"
        cov = entry.get(metric)
        if not (
            isinstance(cov, (int, float))
            and not isinstance(cov, bool)
            and np.isfinite(float(cov))
        ):
            _unprobeable(ch)
            continue
        if float(cov) <= 0:
            logger.error("Required aux channels have ZERO coverage: %s — "
                         "aborting", ch)
            sys.exit(1)
        if contract is not None and float(cov) < contract.threshold:
            logger.error(
                "Required aux channel '%s' coverage %.4f < minimum %.4f "
                "(--feature-profile) — aborting",
                ch, float(cov), contract.threshold)
            sys.exit(1)

def _parse_quality_gate_report(report_path: str) -> dict:
    """Load and return the quality-gate report at ``report_path`` (§六-2).

    The file-existence check + ``json.load`` — the read half of
    :func:`_require_quality_gate`.  A missing report exits the run (the gate
    cannot verify data it never saw); a corrupt JSON payload propagates as a
    ``json.JSONDecodeError`` so the caller sees exactly what is wrong with the
    file.  Returns the parsed report dict.
    """
    if not os.path.isfile(report_path):
        raise SystemExit(
            f"quality gate report not found at {report_path} — run "
            f"scripts/production/build_features.py --quality-gate (or the gate directly) "
            f"before training, or pass --no-require-quality-gate to bypass."
        )
    with open(report_path, encoding="utf-8") as fh:
        return json.load(fh)

def _validate_quality_gate_report(
    report: dict,
    data_dir: str,
    prebuilt_dir: str | None,
    allow_missing: bool = False,
) -> list[str]:
    """Verify a loaded quality-gate report against the data this run consumes.

    The validation half of :func:`_require_quality_gate` — accumulates every
    mismatch into a ``problems`` list and returns it (empty list = the report
    passes).  Version / data-root / calendar / contract / scope / dataset-
    coverage / fingerprint / universe-reconciliation / full-scan / passed-flag
    checks are all here; the caller decides whether to raise on the result.
    ``prebuilt_dir`` ``None`` → the gate only needs to cover the ``daily``
    dataset; otherwise the report must have validated the SAME absolute
    prebuilt directory this run reads (§九.1).  ``allow_missing`` tolerates a
    report that failed purely on the universe reconciliation (§八-2 escape).
    """
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
    # §八: bind the calendar artifact's CONTENT hash, not just the version
    # string.  A content edit (holiday rows flipped) that keeps
    # CALENDAR_VERSION would otherwise pass the version check while training
    # reads a different calendar.  calendar_artifact_hash NEVER returns None
    # (it falls back to the code-derived frame when the artifact is absent),
    # so a report with a None/missing hash — an old gate report, or a gate that
    # ran with NO calendar artifact present — is REFUSED: the gate cannot vouch
    # for calendar content it never bound.  A None on one side is a refusal,
    # never a skip-the-comparison escape.
    if report.get("calendar_artifact_hash") != calendar_artifact_hash(data_dir, "a_shares"):
        problems.append("calendar artifact changed since the gate PASS")
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
    return problems

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

    Composition of :func:`_parse_quality_gate_report` (read the report,
    exiting when the file is missing) and :func:`_validate_quality_gate_report`
    (accumulate the mismatch list), raising ``SystemExit`` on any problem.
    """
    if deprecated:
        bad = sorted(k for k in deprecated if k not in ("universe_name", "requested"))
        if bad:
            raise TypeError(
                f"_require_quality_gate got unexpected keyword arguments: {bad}"
            )
    report = _parse_quality_gate_report(report_path)
    problems = _validate_quality_gate_report(
        report, data_dir, prebuilt_dir, allow_missing)
    if problems:
        raise SystemExit(
            "quality-gate report mismatch — refusing to train on unvalidated "
            "data: " + "; ".join(problems)
        )
    return report

def _check_verified_until_scope(global_dates, enforce: bool, data_dir) -> str | None:
    """§九-3: refuse a formal run whose panel axis reaches past verified_until.

    Forward-estimate trading days (2027+ A-share closures) are not verified
    exchange fact, so a formal run (quality gate enforced) that spans them is
    refused instead of silently mixing guessed holidays into the panel axis.
    Uses the strict calendar's own bound so the refusal is exactly the
    strict-mode contract.  ``enforce`` is False for exploratory runs that opt
    out via --no-require-quality-gate.  ``data_dir`` is REQUIRED (no default) so
    every caller is forced to thread the data root it actually reads — the
    strict calendar must follow the frozen ``exchange_calendar`` artifact at
    that root, never the process config default.  Returns the refusal message
    (caller raises SystemExit) or None when in scope.
    """
    if not enforce or global_dates is None or not len(global_dates):
        return None
    strict_cal = get_research_calendar(strict=True, data_dir=data_dir)
    lo = pd.Timestamp(global_dates[0]).date()
    hi = pd.Timestamp(global_dates[-1]).date()
    try:
        strict_cal.get_trading_days(lo, hi)
    except ValueError as exc:
        return f"formal run refused: {exc}"
    return None
