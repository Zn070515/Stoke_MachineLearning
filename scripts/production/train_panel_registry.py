"""Experiment bookkeeping for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — the frozen experiment
version (``_experiment_version`` / ``_calendar_freeze``), the DSR multiplicity
trial signature, the durable cross-process experiment registry, and the
single-use lockbox gate.  ``train_panel`` re-exports these names for backward
compatibility.
"""
import contextlib
import hashlib
import json
import logging
import os
import time
from datetime import datetime

import pandas as pd

from stoke_ml.config import get_project_root
from stoke_ml.data.calendar import (
    TradingCalendar,
    build_calendar_frame,
    load_calendar,
)
from stoke_ml.features import cache_manifest
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import EVALUATOR_VERSION

logger = logging.getLogger(__name__)


def _calendar_freeze(data_dir: str) -> dict:
    """§九-4: freeze the calendar CONTENT an experiment consumed.

    A bare manual version string (``CALENDAR_VERSION``) does not pin the
    actual trading days — a holiday-set edit with a forgotten version bump
    would go unnoticed.  Hash the materialized calendar (dates + is_open +
    source + verified_until + status), which is invariant to the
    non-deterministic ``generated_at`` stamp, and record ``verified_until`` +
    ``source``.  The on-disk artifact is the authoritative frame when present
    (data-driven corrections win over code); a malformed/absent artifact falls
    back to the code-derived frame the panel pipeline actually uses.
    """
    try:
        frame = load_calendar(data_dir, "a_shares")
    except Exception:
        frame = None
    if frame is None:
        frame = build_calendar_frame("a_shares")
    verified_until = str(pd.Timestamp(frame["verified_until"].iloc[0]).date())
    source = str(frame["source"].iloc[0])
    canonical = (frame.drop(columns=["generated_at"])
                 if "generated_at" in frame.columns else frame)
    # Deterministic text encoding (not parquet bytes): pandas str() normalizes
    # datetime values regardless of dtype unit (datetime64[s] vs [ms] after a
    # parquet round-trip), so a disk artifact and the code formula hash equal.
    cols = sorted(canonical.columns)
    key = "date"
    canon_sorted = canonical.sort_values(key).reset_index(drop=True)
    lines = []
    for _, row in canon_sorted.iterrows():
        lines.append("|".join(str(row[c]) for c in cols))
    digest = "\n".join(lines).encode("utf-8")
    checksum = hashlib.sha1(digest).hexdigest()[:16]
    return {
        "calendar_artifact_hash": checksum,
        "verified_until": verified_until,
        "calendar_source": source,
    }


def _experiment_version(
    data_dir: str,
    universe_used: list[str],
    prebuilt_dir: str | None,
    static_dim: int,
    past_known_dim: int,
    past_observed_dim: int,
    config: PanelConfig,
    start: str,
    end: str,
    seed: int,
) -> dict:
    """Freeze the data/code/feature versions an experiment consumed.

    Every hash is content-addressed and deterministic — the same commit +
    source files + feature schemas + universe must yield the same digest, so a
    days-old run stays explainable.  `data_manifest_hash` covers the raw source
    files actually fed to feature engineering; `feature_schema_hash` covers the
    feature column set (prebuilt sidecar manifests when available, else the
    panel dims).
    """
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    src = hashlib.sha1()
    feat = hashlib.sha1()
    feat.update(f"S={static_dim}|PK={past_known_dim}|PO={past_observed_dim}|".encode())

    manifest_dir = os.path.join(prebuilt_dir, ".manifests") if prebuilt_dir else None
    for code in sorted(universe_used):
        src.update(code.encode())
        src.update(b":")
        if manifest_dir is not None:
            mp = os.path.join(manifest_dir, f"{code}.json")
            m = None
            if os.path.isfile(mp):
                try:
                    with open(mp, encoding="utf-8") as f:
                        m = json.load(f)
                except Exception:
                    m = None
            if m:
                for name in sorted(m.get("source_files") or {}):
                    src.update(name.encode())
                    src.update(b"=")
                    src.update(str(m["source_files"][name].get("hash")).encode())
                    src.update(b";")
                feat.update(str(m.get("feature_schema_hash", "")).encode())
                feat.update(b";")
                continue
            # No readable manifest → fingerprint the prebuilt feature file.
            p = os.path.join(prebuilt_dir, f"{code}.parquet")
            src.update(b"prebuilt=")
            src.update(str(cache_manifest.file_fingerprint(p)).encode())
            src.update(b";")
            feat.update(str(cache_manifest.schema_hash(p)).encode())
            feat.update(b";")
        else:
            _, source_files = cache_manifest.channels_and_source_files(
                data_dir, code, start, end,
            )
            for name in sorted(source_files):
                src.update(name.encode())
                src.update(b"=")
                src.update(str(source_files[name].get("hash")).encode())
                src.update(b";")

    # §十六: model identity is split into THREE content hashes so a continuous-OOS
    # replay can tell a config change from a source change from a schema change,
    # and reject a "first two folds VSN+xLSTM, last three plain LSTM" switch
    # instead of silently blending it.
    #   * model_source_hash — architecture + training code source files only
    #     (hyper-parameter VALUES do not enter it);
    #   * model_config_hash — the PanelConfig hyper-parameters only;
    #   * model_hash        — LEGACY combined digest (config + architecture
    #     source), kept byte-for-byte stable so old readers / registry
    #     signatures that consume `model_hash` keep working.
    _root = str(get_project_root())
    _model_source_files = (
        "stoke_ml/models/panel/model.py",
        "stoke_ml/models/panel/vsn.py",
        "stoke_ml/models/panel/xlstm.py",
        "stoke_ml/models/panel/heads.py",
        "stoke_ml/models/panel/config.py",
        "stoke_ml/models/panel/loss.py",
    )
    model_source_h = hashlib.sha1()
    for rel in (*_model_source_files,
                "stoke_ml/models/panel/train.py",
                "scripts/production/train_panel.py"):
        fp = cache_manifest.file_fingerprint(os.path.join(_root, rel)) or "absent"
        model_source_h.update(rel.encode("utf-8"))
        model_source_h.update(b"=")
        model_source_h.update(fp.encode("utf-8"))
        model_source_h.update(b";")
    model_config_h = hashlib.sha1()
    model_config_h.update(repr(config).encode("utf-8"))
    model_h = hashlib.sha1()
    model_h.update(repr(config).encode("utf-8"))
    for rel in _model_source_files:
        fp = cache_manifest.file_fingerprint(os.path.join(_root, rel)) or "absent"
        model_h.update(rel.encode("utf-8"))
        model_h.update(b"=")
        model_h.update(fp.encode("utf-8"))
        model_h.update(b";")

    # §九-4: calendar content freeze — artifact hash + verified_until + source
    # (see _calendar_freeze), alongside the manual version string.
    calendar_freeze = _calendar_freeze(data_dir)
    return {
        "git_commit": cache_manifest.git_head(),
        "data_manifest_hash": src.hexdigest()[:16],
        "calendar_version": TradingCalendar.CALENDAR_VERSION,
        **calendar_freeze,
        "feature_schema_hash": feat.hexdigest()[:16],
        "universe_hash": _sha1("\n".join(sorted(universe_used))),
        "model_hash": model_h.hexdigest()[:16],
        "model_source_hash": model_source_h.hexdigest()[:16],
        "model_config_hash": model_config_h.hexdigest()[:16],
        "evaluator_version": EVALUATOR_VERSION,
        "cost_model": f"sleeve per-side txn_cost={config.txn_cost}, top_fraction=0.1",
        "random_seed": seed,
    }


# §十五-1 experiment registry — a durable count of every research trial, so the
# DSR deflation has a defensible N (the number of trials iterated across the
# project, not just the strategies inside one report).  Lives at a STABLE path
# (independent of the timestamped outdir) so it accumulates across runs.
#
# §十二.6: hardened as a real experiment ledger —
#   * path is anchored to the project root (a relative path silently breaks
#     when the script runs from another cwd);
#   * the append is a read → dedup → atomic-replace guarded by a cross-process
#     file lock, so two concurrent experiments cannot overwrite each other;
#   * entries are deduplicated by `experiment_signature` (data + features +
#     model + universe + horizon + objective): re-running the SAME experiment
#     into a NEW outdir replaces the old row instead of double-counting;
#   * a corrupt registry FAILS LOUDLY (never silently returns [] and resets
#     the DSR trial count).
_EXPERIMENT_REGISTRY_PATH = str(
    get_project_root() / "reports" / "experiments" / "experiment_registry.json")


# §二十: the lockbox is a SINGLE-USE resource — reserved for ONE final run once
# the design freezes.  Opening it is recorded at a stable project-level marker,
# and any subsequent FORMAL run that tries to open the lockbox again is refused
# up front (a dev/exploratory run may pass --no-require-quality-gate or
# --no-formal, or --lockbox-months 0, to proceed explicitly).  The marker lives
# outside any single outdir so a second run into a NEW outdir is still caught.
_LOCKBOX_MARKER_PATH = str(
    get_project_root() / "reports" / "lockbox_used.json")


def _read_lockbox_marker(path: str | None = None) -> dict | None:
    """Prior lockbox-open record; None when never opened (or unreadable).

    §二十: a corrupt/unreadable marker is treated as ABSENT (the run may
    proceed and re-record its use) — refusing on a broken marker would brick
    every future run with no recovery.
    """
    p = path or _LOCKBOX_MARKER_PATH
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 — corrupt marker must not wedge the run
        logger.warning("lockbox marker %s unreadable — treating as absent", p)
        return None
    return data if isinstance(data, dict) else None


def _mark_lockbox_used(path: str | None, info: dict) -> None:
    """Persist that a formal run opened the lockbox (single-use gate §二十)."""
    p = path or _LOCKBOX_MARKER_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)


def _require_single_use_lockbox(
    lockbox_months: int,
    *,
    formal: bool,
    marker_path: str | None = None,
    info: dict | None = None,
) -> None:
    """§二十: enforce the single-open lockbox contract.

    A FORMAL run that opens the lockbox (``lockbox_months > 0``) must be the
    first — if a prior formal run already recorded its use, this run is refused
    with the prior use's provenance.  The marker is written as the lockbox is
    opened (before training), so even an aborted first run still consumes the
    single use rather than allowing a silent re-open.  Non-formal runs and
    ``--lockbox-months 0`` never touch the marker.
    """
    if lockbox_months <= 0 or not formal:
        return
    p = marker_path or _LOCKBOX_MARKER_PATH
    prior = _read_lockbox_marker(p)
    if prior is not None:
        raise SystemExit(
            "lockbox 只允许单次开启 — a previous formal run already opened the "
            f"lockbox (recorded in {p}: opened_at={prior.get('opened_at', '?')}, "
            f"universe={prior.get('universe', '?')}, lockbox="
            f"{prior.get('lockbox_months')}).  The lockbox is reserved for a "
            "single final run once the design freezes; to proceed explicitly "
            "re-run with --lockbox-months 0 or delete the marker."
        )
    _mark_lockbox_used(p, dict(info or {}, opened_at=datetime.now().isoformat(
        timespec="seconds")))


def _ablation_desc(config: PanelConfig) -> str:
    """Human-readable description of the active architecture-ablation switches."""
    parts = []
    if config.backbone != "xlstm":
        parts.append(f"backbone={config.backbone}")
    if not config.use_vsn:
        parts.append("no-vsn")
    if not config.use_dir_head:
        parts.append("no-dir-head")
    if not config.use_vol_head:
        parts.append("no-vol-head")
    if not config.use_ranking_loss:
        parts.append("no-ranking")
    if config.fixed_task_weights:
        parts.append("fixed-task-weights")
    if not config.use_pit_static:
        parts.append("no-pit-static")
    return ";".join(parts) if parts else "full"


def _objective_desc(config: PanelConfig) -> str:
    """Stable description of the training objective for the trial signature."""
    return (f"multi-task(direction/return/vol);rank_loss_weight={config.rank_loss_weight}"
            f";ablation={_ablation_desc(config)}")


def _experiment_signature(version: dict, config: PanelConfig,
                          model_key: str | None = None,
                          *,
                          augmentation: bool | None = None,
                          seq_features: bool | None = None,
                          vintage_policy: str | None = None,
                          feature_profile: str | None = None,
                          universe_membership: dict | None = None,
                          baseline_hyperparameter_hash: str | None = None,
                          baseline_input_recipe_hash: str | None = None,
                          training_sample_policy_hash: str | None = None,
                          scaler_hash: str | None = None) -> str:
    """Content signature of a research trial: the keys the DSR multiplicity must
    NOT conflate.  Two runs sharing a signature ARE the same experiment (a
    re-run); differing on any key is a NEW trial (§十二.6).  `model_key` lets a
    caller override the model identity (e.g. baselines, whose version's
    model_hash is computed for the deep architecture).

    §十八-2 best-effort: the signature covers at least — random seed, model
    architecture (model_hash) + loss weights (objective), features
    (feature_schema_hash), horizon, top fraction, universe, cost (txn_cost),
    data version, augmentation, baseline sequence features, and a code-tree
    proxy (git commit; the model source-file fingerprints already enter
    model_hash).  `augmentation`/`seq_features` are the caller's flags (the
    deep run's --augment, the baselines' --with-seq-features); each defaults to
    'none' so the signature stays meaningful for callers that lack the switch.

    §十七: the baseline-run signature additionally binds the input-recipe, the
    model hyperparameters, the training-sample policy and the feature-scaling
    recipe — a baseline re-run that flips --with-seq-features, changes a
    hyperparameter, caps training rows differently, or changes the scaler is a
    NEW trial.  Each defaults to 'none' so non-baseline callers are unaffected.

    §T19: the deep-run signature additionally binds the vintage-admission
    policy (--vintage-policy: safe-only vs allow-revised) and the frozen
    feature-profile identity — runs differing ONLY in either lever train on
    materially different channels and MUST be distinct trials.  Both default to
    'none' so the baseline run (which has neither switch) is unaffected.

    §十四: the signature ALSO binds the universe-membership provenance — a CSI
    run's universe gate consumes membership.parquet (Baostock monthly
    reconstruction, latest-reconstructed), and two runs whose membership
    provenance differs (monthly-reconstructed vs a future official
    effective-date artifact) are distinct trials.  Defaults to 'none' so
    non-CSI / non-deep callers are unaffected.
    """
    h = hashlib.sha1()
    for key in ("data_manifest_hash", "feature_schema_hash", "universe_hash"):
        h.update(f"{key}={version.get(key) or 'unknown'};".encode("utf-8"))
    model_hash = model_key if model_key else version.get("model_hash")
    h.update(f"model_hash={model_hash or 'unknown'};".encode("utf-8"))
    # §十二.5/§P1-8: the DSR multiplicity must treat different RESEARCH CHOICES
    # as distinct trials — the review's canonical example is trying five random
    # seeds and keeping the best, which the old signature (no seed) conflated
    # into one trial.  Seed, horizon, sequence length, transaction cost, the
    # evaluator identity and the frozen calendar all enter the signature, so a
    # re-run that changes any research choice counts as a NEW experiment.
    h.update(f"seed={version.get('random_seed', getattr(config, 'seed', None))};"
             .encode("utf-8"))
    h.update(f"horizon={config.horizon};".encode("utf-8"))
    h.update(f"seq_len={config.seq_len};".encode("utf-8"))
    h.update(f"txn_cost={getattr(config, 'txn_cost', None)};".encode("utf-8"))
    h.update(f"objective={_objective_desc(config)};".encode("utf-8"))
    h.update(f"evaluator_version={version.get('evaluator_version') or 'unknown'};"
             .encode("utf-8"))
    h.update(f"calendar_version={version.get('calendar_version') or 'unknown'};"
             .encode("utf-8"))
    h.update(f"calendar_artifact_hash={version.get('calendar_artifact_hash') or 'unknown'};"
             .encode("utf-8"))
    # §十八-2: the remaining research levers the review wants pinned — top
    # fraction (constant 0.1 across deep + baseline runs), the augmentation
    # flag, the baseline sequence-feature flag, and a code-tree hash proxy
    # (git commit; model source-file fingerprints already enter model_hash).
    h.update("top_fraction=0.1;".encode("utf-8"))
    h.update(f"augmentation={augmentation if augmentation is not None else 'none'};"
             .encode("utf-8"))
    h.update(f"seq_features={seq_features if seq_features is not None else 'none'};"
             .encode("utf-8"))
    # §T19 (§T2/§T7): the vintage-admission policy and the frozen feature-profile
    # identity are part of what a deep run IS.  A safe-only run DENIES
    # latest_revised-sourced channels (fundamental/macro/earnings/valuation/
    # pledge/shareholder/index_membership/market_env_refine/sector/concept)
    # that an allow-revised run admits, and a different frozen feature profile is a
    # different feature recipe (required_channels + per-channel coverage
    # minimums).  Two runs differing ONLY in either lever train on materially
    # different channels, so they MUST be distinct trials — never conflated into
    # one experiment.  Each defaults to 'none' so callers without the switch
    # (the baseline script) hash a stable signature.
    h.update(f"vintage_policy={vintage_policy or 'none'};".encode("utf-8"))
    # §十四: the universe-membership provenance is part of what a deep run IS —
    # a CSI universe gate consumes membership.parquet (Baostock monthly
    # reconstruction, latest-reconstructed), so feature-vintage safe-only does
    # NOT free the run of latest-reconstructed data in its universe gate, and
    # a different membership provenance (e.g. a future official effective-date
    # artifact) is a NEW trial.  None → 'none' so callers without the lever
    # (non-CSI / baseline) hash a stable signature.
    h.update(f"universe_membership="
             f"{json.dumps(universe_membership, sort_keys=True) if universe_membership else 'none'};"
             .encode("utf-8"))
    h.update(f"feature_profile={feature_profile or 'none'};".encode("utf-8"))
    # §十七: baseline identity levers — input recipe (with_seq + seq_len +
    # construction version), model hyperparameters, the training-sample policy
    # (max_train_rows / sampling strategy), and the feature-scaling recipe are
    # all part of what a baseline trial IS; changing any of them is a new
    # experiment.
    h.update(f"baseline_hyperparameter_hash="
             f"{baseline_hyperparameter_hash or 'none'};".encode("utf-8"))
    h.update(f"baseline_input_recipe_hash="
             f"{baseline_input_recipe_hash or 'none'};".encode("utf-8"))
    h.update(f"training_sample_policy_hash="
             f"{training_sample_policy_hash or 'none'};".encode("utf-8"))
    h.update(f"scaler_hash={scaler_hash or 'none'};".encode("utf-8"))
    h.update(f"code_tree={version.get('git_commit') or 'unknown'};".encode("utf-8"))
    return h.hexdigest()[:16]


def _distinct_trial_count(entries: list[dict],
                          current_signature: str | None = None) -> int:
    """DSR multiplicity N: distinct experiments among `entries`, where a prior
    row whose signature equals `current_signature` is the SAME experiment being
    re-run (replaced, not counted twice).  Legacy rows without a signature each
    count as one distinct trial.  `current_signature` None → N counts only the
    given entries (caller adds its own trials separately)."""
    sigs = {e.get("experiment_signature")
            for e in entries if e.get("experiment_signature")}
    if current_signature is not None:
        sigs.discard(current_signature)
    legacy = sum(1 for e in entries if not e.get("experiment_signature"))
    return len(sigs) + legacy + (1 if current_signature is not None else 0)


@contextlib.contextmanager
def _registry_lock(lock_path: str, timeout_s: float = 60.0,
                   stale_s: float = 120.0):
    """Cross-process exclusive lock via an O_CREAT|O_EXCL lockfile.

    The registry critical section (read → dedup → atomic replace) is short, so
    a lock older than `stale_s` — its writer presumably crashed mid-append — is
    reclaimed rather than wedging every later run.  Raises TimeoutError if the
    lock cannot be acquired in time.
    """
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({
                "pid": os.getpid(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                stale = time.time() - os.path.getmtime(lock_path) > stale_s
            except OSError:
                stale = True
            if stale:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                raise TimeoutError(
                    f"could not acquire experiment-registry lock {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _load_experiment_registry(path: str) -> list[dict]:
    """Prior experiment entries; [] only when the registry does not exist yet.

    §十二.6: a corrupt or non-list registry FAILS LOUDLY — silently returning []
    would reset the DSR trial count and deflate every future run's multiplicity.
    A missing file (first run) is the one legitimate empty case.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"experiment registry {path} is corrupt (could not parse JSON): "
            f"{exc}.  Inspect/repair it before re-running — a fresh registry "
            "would silently reset the DSR trial count.") from exc
    if not isinstance(data, list):
        raise RuntimeError(
            f"experiment registry {path} has an unexpected shape "
            f"({type(data).__name__}, expected a list of entries).")
    return data


def _append_experiment_registry(path: str, entry: dict) -> None:
    """Atomically append/replace an entry under a cross-process lock.

    An existing row with the same experiment_signature is REPLACED — re-running
    the same experiment into a NEW outdir is one trial, not two (§十二.6); an
    existing row for the same outdir is likewise replaced.  The whole
    read → dedup → atomic-replace happens under `_registry_lock` so concurrent
    experiments cannot overwrite each other.
    """
    sig = entry.get("experiment_signature")
    outdir = entry.get("outdir")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with _registry_lock(lock_path):
        entries = _load_experiment_registry(path)
        kept = [
            e for e in entries
            if not (e.get("outdir") == outdir
                    or (sig and e.get("experiment_signature") == sig))
        ]
        kept.append(entry)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
