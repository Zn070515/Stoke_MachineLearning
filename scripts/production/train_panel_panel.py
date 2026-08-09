"""Panel construction, memmap store, and aux-channel loading (§二十一).

Extracted from ``scripts.production.train_panel`` — the vintage-policy feature
switch set (``_panel_pipeline_kwargs``), the panel-store meta fingerprint, the
panel builder (``_resolve_panel``), the universe memory guards, the aux-channel
coverage manifest loading, and the has_* flag probe.  ``train_panel``
re-exports these names for backward compatibility.
"""
import glob
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.config import feature_profile as _fp
from stoke_ml.data.asset_contract import (
    manifest_body_digest as _manifest_body_digest,
    parse_era_coverage,
)
from stoke_ml.data.calendar import get_research_calendar
from stoke_ml.data.channel_sources import CHANNEL_SOURCE, live_data_type, source_dir
from stoke_ml.data.universe import (
    load_index_membership,
    load_universe_status,
)
from stoke_ml.data.vintage_policy import (
    VintagePolicy,
    channel_allowed,
    universe_membership_provenance,
)
from stoke_ml.features.aux_cols import FUNDAMENTAL_COLS
from stoke_ml.features.cache_manifest import (
    _dir_content_hash,
    current_config_hash,
    git_head,
    shared_inputs_hash,
)
from stoke_ml.models.panel.code_tree_hash import (
    feature_code_tree_hash,
    hash_json,
)
from stoke_ml.features.panel_builders._arrays import close_memmap_grids
from stoke_ml.features.panel_builder import (
    _DEFAULT_SCRATCH_SAFETY_MARGIN_GB,
    _DEFAULT_SCRATCH_STALE_DAYS,
    _scratch_run_id,
)
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.panel.panel_store import (
    load_panel_memmap,
    save_panel_memmap,
)
from scripts.production.data_quality_gate import dataset_fingerprint
from scripts.production.train_panel_folds import _universe_artifact_hashes
from scripts.production.train_panel_gates import _formal_mode
from scripts.production.train_panel_registry import _calendar_freeze
from scripts.production.train_panel_universe import _is_csi_universe

logger = logging.getLogger(__name__)


# §T2: MarketWideStorage feature channels load_aux_data iterates.  The live
# data type each storage reads is derived from the CHANNEL_SOURCE registry
# (``live_data_type``), so the a_shares subdir names live in ONE place.
_MARKET_WIDE_CHANNELS = (
    "margin", "northbound", "dragon_tiger", "capital_flow", "block_trade",
    "shareholder", "lockup", "dividend", "valuation",
)


# §v18-2: the channels the LIVE pipeline can actually read data for —
# load_aux_data provides these (per-stock loaders + MarketWideStorage + the
# etf_flow aggregation), or the aux_aligner reads the broadcast files directly
# (industry / market_env).  A use_*-OPEN channel NOT in this set has no live
# data path — earnings / macro / pledge / index_membership are prebuilt-only
# (never loaded live), and market_env_refine is a derived PROCESSING switch,
# not a data channel — so there is no manifest to gate and it is NOT consumed
# live.  Prebuilt runs consume these via the prebuilt feature manifest, never
# the per-channel live gate.
_LIVE_AUX_CHANNELS = frozenset({
    "sentiment", "guba", "comment", "fundamental", "announcement",
    "etf_flow", "industry", "market_env",
}) | set(_MARKET_WIDE_CHANNELS)


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
    # §T5: the market_env ACCOUNT sub-part is PROXY-PIT (ablation-only,
    # mirroring topic) — OFF by default so a revision-safe formal run consumes
    # only the verified PRICE part.  The PRICE part rides on use_market_env.
    "market_env_account": False,
}
# dim → FeaturePipeline kwarg name; only "announcement" differs (use_announcements).
_SWITCH_KEY = {"announcement": "use_announcements"}

def _panel_pipeline_kwargs(args, seq_len: int) -> dict:
    """FeaturePipeline constructor kwargs for the panel build.

    Single source of truth for the ``use_*`` switch set — shared by the live
    pipeline construction AND the panel-store meta fingerprint, so a change to
    the switches OR the vintage policy is caught by the store staleness guard.
    ``--vintage-policy`` is applied as an AND-filter over each channel's base
    preference (the policy can only turn channels OFF, never force one ON);
    ``--allow-fundamental-ablation`` is the one exception, forcing the
    fundamental channel ON (only that channel) regardless of policy.
    """
    policy = VintagePolicy(args.vintage_policy)
    kwargs = {
        _SWITCH_KEY.get(dim, f"use_{dim}"): pref and channel_allowed(dim, policy)
        for dim, pref in _BASE_DIM_PREFERENCE.items()
    }
    # T3 research decision #1: fundamental is denied under revision-safe (its
    # source_vintage is latest_revised) and may enter ONLY via an explicit
    # ablation.
    # --allow-fundamental-ablation forces use_fundamental=True REGARDLESS of
    # policy — but ONLY that channel; the other policy-denied channels stay
    # off.  Defensive getattr: callers passing a legacy args stub (no attr)
    # keep the policy-derived default.  The store-meta fingerprint below
    # consumes this dict, so an ablation store auto-differs from a
    # non-ablation run.
    if getattr(args, "allow_fundamental_ablation", False):
        kwargs["use_fundamental"] = True
    kwargs["seq_len"] = seq_len
    kwargs["minute_mode"] = args.minute
    return kwargs


def _consumed_channels(args, seq_len: int) -> set[str]:
    """The data channels the LIVE pipeline actually reads for this run (§v18-2).

    A channel is consumed exactly when its ``use_*`` switch resolves True after
    the base preference AND the vintage policy AND the explicit ablation
    opt-ins (mirrors ``_panel_pipeline_kwargs``) AND it has a LIVE data path
    (``_LIVE_AUX_CHANNELS``).  A switch-OPEN channel with no live loader —
    earnings / macro / pledge / index_membership (prebuilt-only, never loaded
    by ``load_aux_data``) and market_env_refine (a derived PROCESSING switch,
    not a data channel) — is NOT consumed live, so there is no manifest to gate
    and it must not be manifest-gated (prebuilt runs bind it via the prebuilt
    feature manifest instead).  Consumed ≠ Required: Required ⊂ Consumed, with
    the required subset additionally carrying coverage contracts.  Every
    CONSUMED channel must have a valid asset manifest under a formal gate
    (``_enforce_formal_manifests``); the required subset is the extra
    coverage-threshold layer (``_enforce_channel_coverage``).
    """
    kwargs = _panel_pipeline_kwargs(args, seq_len)
    return {
        dim for dim in _BASE_DIM_PREFERENCE
        if kwargs.get(_SWITCH_KEY.get(dim, f"use_{dim}"), False)
        and dim in _LIVE_AUX_CHANNELS
    }


def _entry_fill_prob_mean(panel_data: dict) -> float | None:
    """Period mean of the per-date ENTRY-side fill probability (§十八 T10a).

    ``entry_fill_prob[t]`` is the fraction of decision-eligible stocks at t
    that actually get a fillable entry open — the execution-risk summary the
    audit asked for.  NaN-ignoring mean over the panel dates; None when the
    panel lacks the array (a pre-T10a build) or every value is NaN (no
    decision-eligible date).  Recorded in the panel store's meta.json at
    build time; it is an informational diagnostic, not a config binding.
    """
    efp = panel_data.get("entry_fill_prob")
    if efp is None:
        return None
    efp = np.asarray(efp)
    if not np.isfinite(efp).any():
        return None
    return float(np.nanmean(efp))


def _asset_manifest_entries(root: str) -> dict[str, str]:
    """Sorted relpath → digest map of every ``*.manifest.json`` under ``root``.

    The value is the content digest of the sidecar with the per-write
    bookkeeping keys (``written_at`` / ``updated`` / ``run_id``) excluded
    (``_manifest_body_digest``).  The key is the file's name relative to the
    channel's LIVE asset root — so adding / removing / renaming a stock's
    sidecar changes the map even when every remaining sidecar's bytes are
    unchanged.  An absent / empty root yields an empty map (the channel has no
    asset manifests to bind — the §七 guard then treats "no aux assets bound"
    as the channel's honest state).
    """
    if not os.path.isdir(root):
        return {}
    entries: dict[str, str] = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".manifest.json"):
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            entries[name] = _manifest_body_digest(path)
    return entries


def _aux_asset_root_hash(
    data_dir: str, consumed_set: set[str], *, live_aux: bool,
) -> str:
    """Content hash of the CONSUMED channels' live asset-manifest roots (§T6).

    The §七 guard's "changed aux tomorrow" binding: for every CONSUMED channel
    (every channel the run's pipeline actually opens — §v18-2, not just the
    required subset), hash the sorted relpath → digest map of the
    ``*.manifest.json`` sidecars under that channel's CHANNEL_SOURCE
    ``live_dir`` (per-write bookkeeping keys ``written_at`` / ``updated`` /
    ``run_id`` excluded).  ``live_aux`` is itself part of the
    hash — a LIVE build (aux bound by these roots) is never interchangeable with
    a PREBUILT / ``--no-aux`` build, whose aux is bound differently (prebuilt
    feature manifest) or not at all.

    Fail-closed: a consumed channel with no CHANNEL_SOURCE entry is recorded as
    an explicit marker (``{"__missing_registry_entry__": True}``) — never
    silently dropped, so an unknown channel still differentiates the binding.
    An empty consumed set yields a stable hash over just the ``live_aux`` flag.
    """
    channels: dict[str, dict] = {}
    for ch in sorted(consumed_set or ()):
        spec = CHANNEL_SOURCE.get(ch)
        if spec is None:
            channels[ch] = {"__missing_registry_entry__": True}
            continue
        root = os.path.join(data_dir, *spec.live_dir.split("/"))
        channels[ch] = _asset_manifest_entries(root)
    return hash_json({"live_aux": bool(live_aux), "channels": channels})


def _panel_store_meta(
    args, seq_len: int, stock_list: list[str] | None = None,
    data_dir: str | None = None, prebuilt_dir: str | None = None,
    entry_fill_prob_mean: float | None = None,
    *, required_set: set[str],
) -> dict:
    """Build-time fingerprint persisted in a panel store's meta.json.

    ``required_set`` is a REQUIRED keyword-only parameter — the resolved set of
    required aux channels (from ``train_panel_gates._resolve_required_set``).
    §v18-2: the ``aux_asset_root_hash`` binding is computed over the CONSUMED
    channel set (``_consumed_channels(args, seq_len)``), and ``required_set``
    feeds the required ⊆ consumed invariant — a required channel the pipeline
    does not open is a contradiction (SystemExit), so the caller must state the
    channel contract the store is bound under explicitly (``set()`` is the
    honest value for a no-profile / no-required-channel build).  Making it
    required forces every caller to declare the contract rather than leave the
    invariant unchecked (a forgotten/omitted set would silently skip the
    §v18-2 contradiction check).

    Re-checked by load_panel_memmap on a store-backed re-run so a stale store
    (different horizon / universe / feature switches / date window) is refused
    instead of silently training on wrong targets — mirrors cache_manifest's
    config_hash + range staleness logic.  Carries the T4 §八 binding keys:

    * ``n_stocks`` — derived from ``len(stock_list)``, the REQUESTED candidate
      pool.  The store itself additionally records its own surviving-codes
      ``stock_order_hash`` / ``feature_schema_hash`` via ``save_panel_memmap``'s
      self-fingerprint merge (the authoritative row-identity binding recomputed
      against the store's own arrays/lists at load — panel_store).

    * ``_WARN_META_KEYS`` external-artifact hashes — data manifest, calendar,
      universe status/delist, index membership, prebuilt feature manifest.
      Each is computed only when the corresponding artifact is readable (None
      otherwise, skipped at load — mirrors the config_hash None-skip), and a
      mismatch warns-and-proceeds (each is re-derivable by rebuilding the
      store).

    §T6/§七 provenance — ``_WARN_META_KEYS`` content bindings that answer "was
    this store built from what is on disk RIGHT NOW":

    * ``feature_code_tree_hash`` — content hash of the ``stoke_ml/`` +
      ``scripts/production/`` source trees (every ``*.py`` file's bytes, NOT
      ``git_commit``).  An uncommitted code edit changes the hash, so a store
      built from edited feature code is refused instead of silently reused
      (code_tree_hash).  Recorded on EVERY build.
    * ``aux_asset_root_hash`` — content hash of the CONSUMED channels' live
      asset-manifest roots (``*.manifest.json`` sidecars, ``written_at``
      excluded; §v18-2: every channel the run's pipeline opens, not just the
      required subset).  Recorded ONLY for a LIVE-aux build (``data_dir`` set
      and no ``--prebuilt`` / ``--no-aux``): the store binds TODAY's aux roots,
      so changed aux TOMORROW makes a formal load refuse.  A prebuilt / no-aux
      store records no aux binding (its aux is bound via
      ``prebuilt_feature_manifest_hash`` or not at all).
    * ``panel_input_hash`` — SHA-256 aggregate (canonical JSON) of EVERY input
      provenance component: code tree, aux root, config hash, feature switches,
      label policy, vintage policy, feature profile, horizon / seq_len / window,
      universe + n_stocks, data manifest, calendar, universe status/membership,
      prebuilt feature manifest, and the shared-inputs hash.  One aggregate key
      so a change ANYWHERE is a single mismatch.  Recorded on every build.
    """
    # §T6 decision 2: CSI universes bake the daily-member cross-section
    # normalization into the panel arrays, so the store fingerprint must record
    # it (as a pseudo-switch) — otherwise a stale store built for the all-stock
    # z-norm would silently pass the staleness guard.  Copy the dict; never
    # mutate the kwargs cache.  `universe` is a separate meta key, but the
    # pseudo-switch makes the *normalization semantics* explicit and survives a
    # universe re-map.
    _switches = _panel_pipeline_kwargs(args, seq_len)
    if _is_csi_universe(args.universe):
        _switches = {**_switches, "daily_membership_norm": True}
    meta = {
        "horizon": args.horizon,
        "seq_len": seq_len,
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "n_stocks": len(stock_list) if stock_list is not None else None,
        "feature_switches": _switches,
        # §T13 decision 3: the return label carries non-fillable exits to the
        # last real close in (t, t+h] (aligned with evaluation realized).  A
        # critical key so a pre-T13 store (clean-open-only labels) is REFUSED
        # by the staleness guard instead of silently reused with different
        # y_return semantics — the store must be rebuilt for current labels.
        "label_policy": "carry_to_last_close_v1",
        "config_hash": current_config_hash(),
        "git_commit": git_head(),
    }
    if data_dir is not None:
        meta["data_manifest_hash"] = dataset_fingerprint(data_dir, ["daily"])
        try:
            meta["calendar_hash"] = _calendar_freeze(
                data_dir)["calendar_artifact_hash"]
        except Exception:
            # calendar could not be materialized (neither artifact nor code
            # frame) — record an explicit None so strict-mode loads REFUSE
            # (cannot vouch for calendar identity) instead of reusing a store
            # whose calendar the run cannot verify (T1).
            meta["calendar_hash"] = None
        universe_status = load_universe_status(data_dir)
        universe_hashes = _universe_artifact_hashes(
            universe_status, data_dir, args.universe)
        meta["universe_status_hash"] = universe_hashes["universe_status_hash"]
        # membership_hash is a "membership not consumed" sentinel (None) for
        # non-csi universes, NOT a compute failure — omit it so strict-mode
        # loads see a both-absent key (skip) instead of a both-explicit-None
        # (refuse): a --universe all store must stay reusable, not bricked.
        if universe_hashes["membership_hash"] is not None:
            meta["membership_hash"] = universe_hashes["membership_hash"]
            # §T6/§十四: when membership is consumed (a CSI universe) the store
            # ALSO self-describes the membership PROVENANCE — Baostock monthly
            # reconstruction (latest-reconstructed), SEPARATE from the feature
            # vintage policy, so a feature-vintage revision-safe store never
            # silently implies its universe gate avoided latest-reconstructed
            # data.  Non-CSI stores stay untouched (symmetric with the
            # membership_hash conditional).
            meta["universe_membership"] = universe_membership_provenance(
                args.universe)
    if prebuilt_dir:
        meta["prebuilt_feature_manifest_hash"] = _dir_content_hash(
            os.path.join(prebuilt_dir, ".manifests"))
    # §T6/§七 provenance block — computed AFTER the data_dir/prebuilt blocks so
    # the aggregate can fold every component above; placed BEFORE the
    # informational entry-fill key.  `live_aux` mirrors _resolve_panel's aux
    # branch (`if not args.no_aux and not args.prebuilt`) so the aux ROOT binding
    # is recorded exactly when the live aux path actually consumed those roots.
    live_aux = bool(
        not getattr(args, "no_aux", False)
        and not (prebuilt_dir or getattr(args, "prebuilt", None)))
    tree_hash = feature_code_tree_hash()
    meta["feature_code_tree_hash"] = tree_hash
    if data_dir is not None and live_aux:
        consumed = _consumed_channels(args, seq_len)
        # §v18-2 invariant: required ⊆ consumed — a required channel the
        # pipeline does not open is a contradiction (you demand coverage for a
        # channel the model never reads).  The §二十-1 vintage binding makes
        # this hold for every named-profile run.
        not_consumed = set(required_set) - consumed
        if not_consumed:
            raise SystemExit(
                "required channels not consumed by this run's pipeline "
                f"({sorted(not_consumed)}) — required ⊂ consumed must hold; "
                "align --require-aux-channels / --feature-profile with the "
                "--vintage-policy (§v18-2)")
        meta["aux_asset_root_hash"] = _aux_asset_root_hash(
            data_dir, consumed, live_aux=True)
    meta["panel_input_hash"] = hash_json({
        "schema_version": 1,
        "feature_code_tree_hash": tree_hash,
        "aux_asset_root_hash": meta.get("aux_asset_root_hash"),
        "config_hash": meta["config_hash"],
        "feature_switches": meta["feature_switches"],
        "label_policy": meta["label_policy"],
        "vintage_policy": getattr(args, "vintage_policy", None),
        "feature_profile": getattr(args, "feature_profile", None),
        "horizon": args.horizon,
        "seq_len": seq_len,
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "n_stocks": meta["n_stocks"],
        "data_manifest_hash": meta.get("data_manifest_hash"),
        "calendar_hash": meta.get("calendar_hash"),
        "universe_status_hash": meta.get("universe_status_hash"),
        "membership_hash": meta.get("membership_hash"),
        "universe_membership": meta.get("universe_membership"),
        "prebuilt_feature_manifest_hash": meta.get(
            "prebuilt_feature_manifest_hash"),
        "shared_inputs_hash": (
            shared_inputs_hash(data_dir) if data_dir is not None else None),
    })
    # §十八 (T10a): INFORMATIONAL execution-risk summary — the NaN-ignoring
    # mean of the per-date ENTRY-side fill probability.  Explicitly NOT added
    # to _CRITICAL_META_KEYS / _WARN_META_KEYS: it is a build-time diagnostic,
    # not a config binding, so the load-side exact-key guard never compares it
    # (a store built without it loads with the key simply absent).
    if entry_fill_prob_mean is not None:
        meta["entry_fill_prob_mean"] = entry_fill_prob_mean
    return meta

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

# §T8: the era-capable text channels — the channels whose gold (daily) asset
# manifests carry the provider-era fields (provider_available_start/end,
# retrieved_ranges, known_gaps) recorded by the storage write-end from the
# downloader per-stock manifest.  Only these can separate no_event from
# not_observed, so only these get an era_coverage probe.  The SET is DERIVED
# from the feature profiles' ``era_coverage`` contracts (single source of
# truth) rather than hard-coded, so a channel that gains an era_coverage
# contract is automatically probed (and one that drops it stops being probed).


def _era_capable_channels() -> frozenset[str]:
    """Channels with an ``era_coverage`` contract in ANY feature profile (§T8).

    Derived from ``feature_profile.FEATURE_PROFILES`` — the single source of
    truth for which channels are gated on provider-era retrieval coverage —
    instead of a hard-coded list that must be kept in sync by hand.  The UNION
    is taken over ALL profiles, not just the run's active profile: the probe is
    a data-capability read of the gold manifests, and it must not silently
    collapse to an empty set when ``profile_name`` is None (the live default,
    where no profile is active but sentiment/guba era contracts still gate
    headline_v1 formal runs).  Read via the module attribute (not a captured
    binding) so tests that mutate the profile registry are observed at call
    time.
    """
    channels = {
        ch
        for prof in _fp.FEATURE_PROFILES.values()
        for ch, contract in prof.coverage_contracts.items()
        if contract.metric == "era_coverage"
    }
    return frozenset(channels)


def _gold_manifest_paths(data_dir: str, channel: str, code: str) -> list[str]:
    """Candidate gold asset-manifest paths for a channel's per-stock file(s).

    sentiment is partitioned (``a_shares/sentiment/{year}/{month}/{code}.parquet``)
    with a flat ``sentiment/{code}.parquet`` fallback; the other three are flat
    (``guba_sentiment|comment_sentiment|announcements/sentiment/{code}.parquet``).
    Every gold file of a stock carries the SAME provider-era fields (all stamped
    from the one downloader manifest), so parsing ANY existing candidate yields
    the same era coverage — the first existing path wins.
    """
    base = os.path.join(data_dir, "a_shares")
    if channel == "sentiment":
        flat = os.path.join(base, "sentiment", f"{code}.parquet")
        if os.path.isfile(flat + ".manifest.json"):
            return [flat + ".manifest.json"]
        return sorted(glob.glob(
            os.path.join(base, "sentiment", "*", "*", f"{code}.parquet.manifest.json")))
    if channel == "guba":
        d = os.path.join(base, "guba_sentiment")
    elif channel == "comment":
        d = os.path.join(base, "comment_sentiment")
    else:  # announcement — the daily SENTIMENT asset, not the raw file
        d = os.path.join(base, "announcements", "sentiment")
    p = os.path.join(d, f"{code}.parquet.manifest.json")
    return [p] if os.path.isfile(p) else []


def _stock_era_coverage(data_dir: str, channel: str, code: str) -> tuple:
    """(era_coverage, not_observed) for one stock from its gold asset manifest.

    ``era_coverage`` is None and ``not_observed`` True when there is no gold
    manifest at all (nothing was ever built for this stock in the gold layer).
    """
    for mp in _gold_manifest_paths(data_dir, channel, code):
        try:
            with open(mp, "r", encoding="utf-8") as f:
                asset_manifest = json.load(f)
        except (OSError, ValueError):
            # A PRESENT but unreadable manifest is diagnostically different from
            # a MISSING one — a broken manifest is a tamper/format signal, not
            # "this stock was never observed".
            logger.debug("_stock_era_coverage: unreadable gold manifest %s",
                         mp)
            continue
        report = parse_era_coverage(asset_manifest)
        return report["era_covered"], report["not_observed"]
    return None, True


def _probe_era_coverage(data_dir: str, channel: str, stock_list: list[str]) -> tuple:
    """Aggregate per-stock era coverage over a universe (§T8).

    Returns ``(mean_coverage, era_observable_stocks, era_not_observed_stocks)``:
    mean over the stocks whose gold manifest records a provider era.  A
    ``not_observed`` stock (no provider era — we never looked) is excluded from
    the numerator and counted separately: it is NOT evidence of "no events".
    ``mean_coverage`` is ``None`` when no stock is era-observable (the channel
    is UNPROBEABLE for the era metric, which aborts a formal gate).
    """
    coverages: list[float] = []
    not_observed = 0
    for code in stock_list:
        cov, no = _stock_era_coverage(data_dir, channel, code)
        if no:
            not_observed += 1
        elif cov is not None:
            coverages.append(float(cov))
    if not coverages:
        return None, 0, not_observed
    return sum(coverages) / len(coverages), len(coverages), not_observed


def _merge_era_coverage(
    channel_manifest: dict, data_dir: str, stock_list: list[str], *,
    force: bool = True,
) -> dict:
    """Add the era-coverage probe to a channel manifest (§T8).

    Merges ONLY the era-capable channels that are ALREADY present — a
    ``--no-aux`` build's empty manifest stays empty (a channel not actually in
    the panel is not probed).  For each present channel, sets
    ``era_observable_stocks`` / ``era_not_observed_stocks``,
    ``era_observable_stock_fraction`` (n_obs / len(stock_list), the §v18-3
    composite half) and, when at least one stock is era-observable,
    ``era_coverage`` (the mean).  With ZERO era-observable stocks
    ``era_coverage`` is left ABSENT so the coverage gate treats the channel as
    unprobeable — a formal run must not proceed on a channel nothing was
    observed for.

    ``force=False`` (the store-LOAD path) skips a channel that already carries a
    persisted build-time ``era_coverage`` AND ``era_observable_stock_fraction``
    — a freshly-built store replays its frozen era coverage without re-probing
    the gold manifests on the fast path; only a LEGACY store (built before §T8
    or before §v18-3, missing the fraction) is probed.
    """
    for ch in _era_capable_channels():
        if ch not in channel_manifest:
            continue
        if (
            not force
            and "era_coverage" in channel_manifest[ch]
            and "era_observable_stock_fraction" in channel_manifest[ch]
        ):
            continue  # persisted build-time era coverage already present
        mean_cov, n_obs, n_not = _probe_era_coverage(data_dir, ch, stock_list)
        entry = channel_manifest[ch]
        entry["era_observable_stocks"] = n_obs
        entry["era_not_observed_stocks"] = n_not
        entry["era_observable_stock_fraction"] = (
            (n_obs / len(stock_list)) if stock_list else 0.0)
        if mean_cov is not None:
            entry["era_coverage"] = mean_cov
    return channel_manifest


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

    T1: the store's external-artifact hashes are a HARD-FAIL in formal mode —
    a store built from upstream data (manifest / calendar / membership /
    prebuilt features) that no longer matches this run is refused, not reused.
    ``strict_external_meta`` is threaded from ``_formal_mode(args)``, so
    ``--no-formal`` (exploratory) keeps the legacy warn-and-proceed.
    """
    strict = _formal_mode(args)
    if _store_load:
        logger.info("Loading panel memmap store from %s (skipping K-line load "
                    "+ feature build)", args.panel_store)
        panel_data = load_panel_memmap(
            args.panel_store,
            expected_meta=_panel_store_meta(
                args, seq_len, stock_list, data_dir, args.prebuilt,
                required_set=required_set),
            strict_external_meta=strict)
        # §T4: a store built with the manifest persisted reads the ACCURATE
        # build-time coverage directly; a legacy store without it falls back to
        # the has_* flag probe (which cannot cover flag-less channels).
        stored = panel_data.get("channel_coverage_manifest")
        channel_manifest = (stored if stored is not None
                            else _prebuilt_channel_coverage(panel_data))
        # §T8: merge the provider-era coverage probe for the text channels.
        # force=False: a store built under §T8 already persisted build-time
        # era_coverage — replay it without re-probing the gold manifests on the
        # fast path; only a LEGACY store (no persisted era) is probed.
        return panel_data, _merge_era_coverage(
            channel_manifest, data_dir, stock_list, force=False)

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
        # §T4: formal mode threads the formal flag into load_aux_data so the
        # required channels' Asset Manifests are ENFORCED (fail-hard), not just
        # warned on.  --no-formal (explore) keeps the legacy warn-and-proceed.
        aux_data, channel_manifest = load_aux_data(
            stock_list, data_dir, args.start, args.end,
            required_channels=required_set,
            consumed_channels=_consumed_channels(args, seq_len),
            formal=_formal_mode(args),
        )
        logger.info("Aux data loaded in %.1fs", time.time() - t_aux)

    # §T6 decision 2 (strict CSI): CSI universes restrict the per-date
    # cross-section STATISTICAL SET to that day's index members (half-open
    # in_date <= date < out_date).  Non-members are still z-scored but do NOT
    # contribute to the mean/std.  Missing membership.parquet degrades to the
    # all-stock z-norm (daily_membership=None) with a WARNING — the stock pool
    # gate already refuses csi universes without the artifact, so this is a
    # belt-and-suspenders guard, not a silent fallback.
    daily_membership = None
    if _is_csi_universe(args.universe):
        _idx = {"csi300": {"000300"}, "csi500": {"000905"},
                "csi800": {"000300", "000905"}}[args.universe]
        _mem = load_index_membership(data_dir, sorted(_idx))
        if _mem.empty:
            logger.warning(
                "CSI universe %s: index membership is empty/missing — falling "
                "back to the all-stock cross-section z-norm (no daily-member "
                "normalization; §T6 decision 2)", args.universe)
        else:
            daily_membership = _mem
    fp = FeaturePipeline(**_panel_pipeline_kwargs(args, seq_len))
    # T8 (§七-P0): when --panel-store is set, the three large (N,T,D) grids are
    # written directly to disk via open_memmap — the full dense grids never
    # reside in RAM.  After the build, the memmaps are flushed + closed so
    # save_panel_memmap can safely write the remaining small arrays + metadata
    # into the same directory (Windows locks open memmap files).  The store is
    # then re-loaded to get lazy read-only memmaps for downstream training.
    # §T7/§v18-6: resolve the streaming scratch dir (explicit --scratch-dir >
    # <panel-store>/scratch/<run_id>/ > system temp) and refuse up front when
    # the estimate-based disk footprint cannot fit — the panel_store and scratch
    # volumes are sized independently when they differ, summed when they share a
    # drive.  The scratch spec is threaded into the builder for crash-resume +
    # stale sweep.
    scratch_dir, scratch_run_id, cleanup_root, cleanup_prefix = _resolve_scratch_dir(args)
    _enforce_streaming_disk_space(args, stock_list, data_dir, scratch_dir)
    panel_data = fp.build_panel_features(
        panel, aux_data=aux_data, horizon=args.horizon, prebuilt_dir=args.prebuilt,
        require_feature_manifest=args.require_feature_manifest,
        daily_membership=daily_membership,
        memmap_dir=args.panel_store,
        scratch_dir=scratch_dir,
        run_id=scratch_run_id,
        scratch_stale_days=_scratch_stale_days(),
        scratch_cleanup_root=cleanup_root,
        scratch_cleanup_prefix=cleanup_prefix,
    )
    if args.prebuilt:
        # Live per-channel loading is skipped in prebuilt mode; probe the
        # panel's has_* flags instead so the experiment still records what
        # actually got in.  On a plain (no-store) build panel_data never carries
        # a manifest key yet — the probe is the only source here.  (On a store
        # build the manifest is set into panel_data AFTER this block and read
        # back from the reloaded store below.)
        channel_manifest = _prebuilt_channel_coverage(panel_data)
    # §T8: merge the provider-era coverage probe for the text channels BEFORE
    # any persist — the store BUILD persists build-time era_coverage (so a later
    # store-load replays it without re-probing the gold manifests), and the
    # live/prebuilt paths record it in the returned manifest.  The merge is
    # idempotent over the present era-capable channels, so the final return just
    # hands the already-merged manifest out.
    channel_manifest = _merge_era_coverage(channel_manifest, data_dir, stock_list)
    if args.panel_store:
        # §T4: persist the channel-coverage manifest into the store so a later
        # store-backed replay reads the accurate build-time coverage directly
        # instead of re-probing has_* flags (which cannot cover flag-less
        # channels).  Set BEFORE the save; the value is JSON, so
        # save_panel_memmap writes channel_coverage_manifest.json (an optional
        # JSON key — a legacy store without it still loads).
        panel_data["channel_coverage_manifest"] = channel_manifest
        # T8: flush + close the memmap grids so save_panel_memmap can write the
        # small arrays + metadata without file-lock collisions on Windows (open
        # memmaps keep their backing files locked).  close_memmap_grids (the
        # single source of truth for the close sequence) returns the set of
        # grids that were actually np.memmap — i.e. the arrays the sink wrote —
        # so a build that fell back to dense (e.g. a test stub) writes every
        # array normally.  The grids REMAIN in panel_data: save_panel_memmap's
        # self-consistency fingerprints (_feature_schema_hash) read their
        # .dtype (a closed memmap keeps header props) to record the T4 schema
        # binding — deleting them would silently drop feature_schema_hash.
        # skip_npy tells save_panel_memmap NOT to rewrite the files the sink
        # already wrote.
        sink_grids = close_memmap_grids(panel_data)
        save_panel_memmap(
            panel_data, args.panel_store,
            meta=_panel_store_meta(
                args, seq_len, stock_list, data_dir, args.prebuilt,
                entry_fill_prob_mean=_entry_fill_prob_mean(panel_data),
                required_set=required_set),
            skip_npy=sink_grids)
        logger.info("Saved panel memmap store to %s", args.panel_store)
        # Re-load the full store for downstream training — fresh lazy memmaps
        # for all arrays (the big grids page-fault only the rows/cols touched).
        panel_data = load_panel_memmap(
            args.panel_store,
            expected_meta=_panel_store_meta(
                args, seq_len, stock_list, data_dir, args.prebuilt,
                required_set=required_set),
            strict_external_meta=strict)
        # Replay the persisted manifest (preferred) or fall back to the probe.
        # `is not None`, NOT `or`: a persisted-but-EMPTY manifest ({} — e.g. a
        # --no-aux build) is a legitimately-empty build-time record that must be
        # replayed as-is, not silently replaced by the probe.
        stored = panel_data.get("channel_coverage_manifest")
        channel_manifest = (stored if stored is not None
                            else _prebuilt_channel_coverage(panel_data))
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

# §T10c: conservative RESIDENT-PEAK bound for the STREAMING panel build (T5),
# which memmap-sinks the three (N,T,D) grids to disk (open_memmap) so the dense
# grids never reside in RAM.  The resident peak is bounded and roughly
# INDEPENDENT of n_stocks.  Components:
#   * ONE stock's ZI-aligned feature frame at a time — n_timesteps ×
#     n_features float64 × pandas overhead (~2.5×) — the per-stock transient;
#   * the cross-sectional-fundamental panel cs_panel_df (~2-3 GB at full-market
#     scale) — ONLY when use_fundamental_refine is enabled (the ONE bounded-in-
#     size resident structure, kept through Pass 3 for the left-merge);
#   * the per-date normalizer-stats accumulator + scratch pickle I/O buffers
#     (~0.5 GB headroom).
_STREAMING_PER_STOCK_PANDAS_OVERHEAD = 2.5  # observed ~2.2x on a (6500,1700) float64 frame; 2.5x gives ~15% headroom
_STREAMING_CS_PANEL_GB = 3.0
_STREAMING_FIXED_BUFFERS_GB = 0.5

def _streaming_peak_memory_gb(
    n_timesteps: int, n_features: int, use_fundamental_refine: bool,
) -> float:
    """§T10c: bounded resident-peak estimate (GB) for the STREAMING panel build.

    See the module constants for the components.  The per-stock transient is
    ONE stock's frame (n_timesteps × n_features float64 × pandas overhead);
    the only other resident structure is the cross-sectional-fundamental panel
    when ``use_fundamental_refine`` is enabled.  Roughly independent of
    n_stocks — what scales with the universe is the ON-DISK memmap grids +
    scratch pickles, not the resident set.  The static 48/96 GB dense lines
    never trip on this bound; the streaming verdict relies on the host's
    actual available RAM as the only refusal floor.
    """
    per_stock = (
        int(n_timesteps) * int(n_features) * 8
        * _STREAMING_PER_STOCK_PANDAS_OVERHEAD / (1024 ** 3)
    )
    cs = _STREAMING_CS_PANEL_GB if use_fundamental_refine else 0.0
    return per_stock + cs + _STREAMING_FIXED_BUFFERS_GB

def _enforce_universe_memory(
    universe: str,
    n_stocks: int,
    n_timesteps: int,
    n_features: int,
    *,
    allow_override: bool = False,
    available_gb: float | None = None,
    streaming: bool = False,
    use_fundamental_refine: bool = False,
) -> tuple[float, str]:
    """§七-P0: refuse / warn when a universe's panel cannot realistically fit in RAM.

    Returns ``(est_gb, action)`` where ``action`` is the UN-overridden verdict:
    "refuse" / "warn" / "ok".

    Verdict rules (DENSE residency — the default, ``streaming=False``):
      * universe == "all": est > _UNIVERSE_MEMORY_REFUSE_GB → "refuse".
      * universe == "csi800": est > _UNIVERSE_MEMORY_HARD_GB → "refuse";
        est > _UNIVERSE_MEMORY_WARN_GB → "warn".
      * any other universe: "ok".
      * universe in ("all", "csi800") and ``available_gb`` is known and
        est > available_gb → "refuse" (the estimate alone already guarantees
        the panel will not fit on THIS host).

    Verdict rules (STREAMING build, ``streaming=True`` — §T10c): the build
    memmap-sinks the (N,T,D) grids to disk, so its RESIDENT peak is bounded and
    roughly independent of n_stocks (``_streaming_peak_memory_gb``).  The
    static 48/96 GB lines — written for the dense residency path — never trip
    on the ~few-GB bound.  Instead:
      * universe in ("all", "csi800"): "warn" — a heads-up about the build's
        size / disk-IO cost, not a RAM refusal;
      * any other universe: "ok";
      * ``available_gb`` known and est > available_gb → "refuse" — the ONLY
        refusal floor for a streaming build (the bounded peak must still fit
        the host's ACTUAL available RAM).
    ``use_fundamental_refine`` (only read when ``streaming=True``) gates the
    resident cs_panel_df term of the streaming peak.

    Side effect: when the verdict is "refuse" and ``allow_override`` is False,
    raises SystemExit with a message that names the estimate, the available
    memory (when known), the threshold, and the ``--allow-high-risk-universe``
    escape hatch.  When the verdict is "warn", OR "refuse" with
    ``allow_override=True``, logs a prominent WARNING instead.
    """
    if streaming:
        # §T10c: bounded streaming resident peak — the dense formula must NOT
        # gate a first build into --panel-store (a full-market streaming build
        # is ~3.7 GB resident, not the ~228 GB the dense estimate implies).
        est_gb = _streaming_peak_memory_gb(
            n_timesteps, n_features, use_fundamental_refine)
        action = "ok"
        if universe in ("all", "csi800"):
            action = "warn"
        if available_gb is not None and est_gb > available_gb:
            action = "refuse"
        if action == "refuse" and not allow_override:
            avail = f"vs available {available_gb:.1f} GB" if available_gb is not None else \
                "vs host available memory (unknown — psutil not installed)"
            raise SystemExit(
                f"universe={universe}: streaming panel build resident peak "
                f"{est_gb:.1f} GB (n_timesteps={n_timesteps} x "
                f"n_features={n_features}; the (N,T,D) grids are memmap-sunk "
                f"to disk) {avail} — the bounded streaming build cannot fit "
                f"the host's available memory (§七-P0).  Free up host RAM or "
                f"pass --allow-high-risk-universe to run it anyway."
            )
        if action in ("warn", "refuse"):
            logger.warning(
                "universe=%s: streaming panel build resident peak %.1f GB "
                "(memmap-sunk grids; bounded, roughly independent of "
                "n_stocks) — §七-P0 heads-up: a full-market streaming build "
                "is large on DISK/IO even though RAM stays bounded.  Pass "
                "--allow-high-risk-universe to run it anyway.",
                universe, est_gb,
            )
        return est_gb, action
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

def _host_available_gb() -> float | None:
    """Host currently-available memory in GB, or None when psutil is absent.

    Shared by the pre-build and post-build universe-memory guards (§七-P0) so
    both estimate against the same host snapshot.  psutil is optional — the
    static thresholds still guard when it is missing.
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None  # psutil optional — the static thresholds still guard

def _estimate_panel_memory(
    args, stock_list: list[str], data_dir,
) -> tuple[int, int, int] | None:
    """§七-P0 pre-build panel memory estimate: (n_stocks, n_timesteps, n_features).

    The post-build guard runs after build_panel_features has np.zeros-allocated
    the three dense (N, T, D) float32 grids, so a huge universe (--universe all,
    csi800) can OOM the host BEFORE that guard ever runs.  This estimate is
    computed from the resolved universe size plus a cheap schema read of the
    first prebuilt parquet (pyarrow reads only the schema, never the data), so
    an oversized universe is refused up front.

    Returns None (skip the early guard) when the feature dim cannot be known
    without building:
      * live builds (``not args.prebuilt``) — D is unknown without a build, and
        live builds are already bounded (default 500 stocks; --universe all
        without --prebuilt is refused outright);
      * a prebuilt dir with no readable ``*.parquet`` (first stock's parquet
        missing/unreadable, or no parquets at all) — never crash the estimate
        path.

    N is the RESOLVED universe size — an upper bound, since the build keeps only
    a surviving subset (conservative, as a safety guard should be).  T is the
    trading-day count in [args.start, args.end].  D is the surviving feature
    column count from the first parquet's schema after dropping exactly the
    columns the build drops (``*_lag{N}``, ``topic_*`` when use_topic is off,
    FUNDAMENTAL_COLS when use_fundamental is off, and the date/stock_code
    identifiers).  D is a small over-estimate of the true per-array dims (the
    parquet may carry a few label/price columns the arrays don't hold) — a
    safety upper bound; the exact post-build check is the backstop.
    """
    if not args.prebuilt:
        return None
    parquets = sorted(Path(args.prebuilt).glob("*.parquet"))
    if not parquets:
        return None
    try:
        import pyarrow.parquet as pq  # lazy — pyarrow is an optional dependency
        cols = list(pq.read_schema(str(parquets[0])).names)
    except Exception:
        return None  # unreadable schema / missing pyarrow — never crash the estimate path
    seq_len = args.seq_len or (64 if args.minute else 60)
    kwargs = _panel_pipeline_kwargs(args, seq_len)
    surviving = [
        c for c in cols
        if c not in ("date", "stock_code")
        and not re.search(r"_lag\d+$", c)
        and not (c.startswith("topic_") and not kwargs.get("use_topic", False))
        and not (c in FUNDAMENTAL_COLS and not kwargs.get("use_fundamental", False))
    ]
    n_stocks = len(stock_list)
    lo = pd.Timestamp(args.start).date()
    hi = pd.Timestamp(args.end).date()
    try:
        n_timesteps = len(get_research_calendar(
            strict=True, data_dir=data_dir).get_trading_days(lo, hi))
    except ValueError:
        # strict calendar extends only to verified_until — fall back to a
        # ~ trading-day fraction of the raw span for the estimate.
        n_timesteps = int((hi - lo).days * 0.7)
    return n_stocks, n_timesteps, len(surviving)

# ── §T7 streaming-build scratch-dir + disk pre-check ───────────────────────
# The STREAMING panel build (T5) memmap-sinks the (N,T,D) grids to disk AND
# writes one per-stock engineered-feature pickle to scratch per Pass-1 stock,
# so it needs a scratch dir with enough free space for the final footprint +
# the scratch pickles + a safety margin.  The pre-build estimate below is the
# egregious-case guard; the builder's exact post-Pass-1 backstop cannot
# underestimate (the authoritative net).
_SCRATCH_DISK_ASSUMED_N_FEATURES = 1700  # live-path D is unknown w/o a build; documented assumption (§十五-3)
_SCRATCH_PICKLE_OVERHEAD = 1.3  # float64 pickle ≈ 8B/feature × ~1.3 overhead
_SCRATCH_TEMP_PREFIX = "panel_stream_scratch_"


def _scratch_stale_days() -> int:
    """§T7 orphan-scratch stale window (days) — config.yaml ``panel_scratch.stale_days``."""
    try:
        from stoke_ml.config import load_config
        cfg = load_config().get("panel_scratch", {})
        return int(cfg.get("stale_days", _DEFAULT_SCRATCH_STALE_DAYS))
    except Exception:
        return _DEFAULT_SCRATCH_STALE_DAYS


def _scratch_safety_margin_gb() -> float:
    """§T7 disk safety margin (GB) — config.yaml ``panel_scratch.safety_margin_gb``."""
    try:
        from stoke_ml.config import load_config
        cfg = load_config().get("panel_scratch", {})
        return float(cfg.get("safety_margin_gb", _DEFAULT_SCRATCH_SAFETY_MARGIN_GB))
    except Exception:
        return _DEFAULT_SCRATCH_SAFETY_MARGIN_GB


def _resolve_scratch_dir(args) -> tuple[str, str, str | None, str | None]:
    """§T7 resolve the streaming build's scratch spec.

    Priority: explicit ``--scratch-dir`` > ``<panel-store>/scratch/<run_id>/``
    > system temp (``panel_stream_scratch_<run_id>``).  Returns
    ``(scratch_dir, run_id, cleanup_root, cleanup_prefix)``.

    An explicit ``--scratch-dir`` never participates in the startup stale sweep
    (``cleanup_root`` is None) — a user-chosen location is never swept.  The
    panel-store-derived and temp locations ARE swept (only dirs older than
    ``_scratch_stale_days()``), because they are owned by this tool.
    """
    run_id = _scratch_run_id()
    # Defensive getattr: legacy args stubs (unit tests / older callers) may not
    # carry the §T7 attr yet; the real CLI arg defaults to None.
    if getattr(args, "scratch_dir", None):
        return args.scratch_dir, run_id, None, None
    if args.panel_store:
        root = os.path.join(args.panel_store, "scratch")
        return os.path.join(root, run_id), run_id, root, None
    return (
        os.path.join(tempfile.gettempdir(), _SCRATCH_TEMP_PREFIX + run_id),
        run_id,
        tempfile.gettempdir(),
        _SCRATCH_TEMP_PREFIX,
    )


def _streaming_disk_required_gb(
    n_stocks: int, n_timesteps: int, n_features: int,
) -> tuple[float, float]:
    """§v18-6 disk required for a streaming build (GB), split by destination.

    Returns ``(final_panel_gb, scratch_gb)`` — the final grids land on the
    panel_store volume, the scratch pickles on the scratch volume.  A
    cross-filesystem build sizes each disk independently; a same-filesystem
    build sums them (the caller decides).  Never under-budgets.
    """
    final_panel = (
        int(n_stocks) * int(n_timesteps) * int(n_features) * 4 / (1024 ** 3)
    )
    scratch = (
        int(n_stocks) * int(n_timesteps) * int(n_features) * 8
        * _SCRATCH_PICKLE_OVERHEAD / (1024 ** 3)
    )
    return final_panel, scratch


def _streaming_disk_estimate(
    args, stock_list: list[str], data_dir,
) -> tuple[int, int, int] | None:
    """§T7 pre-build disk estimate: (n_stocks, n_timesteps, n_features).

    Mirrors ``_estimate_panel_memory`` for DISK: N is the resolved universe
    size (upper bound), T the trading-day count in [args.start, args.end].  D
    is read from the first prebuilt parquet's schema (surviving columns) when
    ``--prebuilt``; on a live build D is unknown without engineering, so the
    documented assumption ``_SCRATCH_DISK_ASSUMED_N_FEATURES`` is used.  Never
    raises — returns None when the estimate cannot be made (skip the early
    guard); the builder's exact backstop remains the authoritative net.
    """
    n_stocks = len(stock_list)
    lo = pd.Timestamp(args.start).date()
    hi = pd.Timestamp(args.end).date()
    if args.prebuilt:
        parquets = sorted(Path(args.prebuilt).glob("*.parquet"))
        if not parquets:
            return None
        try:
            import pyarrow.parquet as pq  # lazy — optional dependency
            cols = list(pq.read_schema(str(parquets[0])).names)
        except Exception:
            return None
        seq_len = args.seq_len or (64 if args.minute else 60)
        kwargs = _panel_pipeline_kwargs(args, seq_len)
        n_features = len([
            c for c in cols
            if c not in ("date", "stock_code")
            and not re.search(r"_lag\d+$", c)
            and not (c.startswith("topic_") and not kwargs.get("use_topic", False))
            and not (c in FUNDAMENTAL_COLS and not kwargs.get("use_fundamental", False))
        ])
    else:
        n_features = _SCRATCH_DISK_ASSUMED_N_FEATURES
    try:
        n_timesteps = len(get_research_calendar(
            strict=True, data_dir=data_dir).get_trading_days(lo, hi))
    except ValueError:
        n_timesteps = int((hi - lo).days * 0.7)
    return n_stocks, n_timesteps, n_features


def _same_filesystem(a: str, b: str) -> bool:
    """True when two paths resolve to the same volume (``os.stat`` device id).

    Windows and POSIX both expose the volume/device via ``st_dev``.  An
    un-stat-able path (not yet created) degrades to True (same) so the
    conservative COMBINED check applies.  On Windows, a junction/mount-point may
    report the HOST volume's ``st_dev``, misclassifying a cross-volume pair as
    same-fs → combined check → false-refuse, never false-pass (conservative).
    """
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return True


def _enforce_streaming_disk_space(
    args, stock_list: list[str], data_dir, scratch_dir: str,
) -> tuple[float, float] | None:
    """§v18-6 refuse to start a streaming build whose disk footprint cannot fit.

    Estimate-based pre-build check: when ``--panel-store`` is set (the STREAMING
    path), size the required disk PER FILESYSTEM — the final grids land on the
    panel_store volume, the scratch pickles on the scratch volume.  On a
    same-filesystem build the two are summed and compared against the shared
    drive; on a cross-filesystem build each volume is checked independently
    (a panel volume that fits only the grids, or a scratch volume that fits only
    the pickles, is not falsely refused nor falsely passed).  Insufficient →
    SystemExit BEFORE any engineering work.  No ``--panel-store`` (dense build)
    → no-op (None).  The builder's exact post-Pass-1 backstop is the second,
    un-underestimating net.

    Returns ``(required_gb, free_gb)`` when checked, else None.
    """
    if not args.panel_store:
        return None
    est = _streaming_disk_estimate(args, stock_list, data_dir)
    if est is None:
        return None
    # Create the scratch dir + panel_store first: shutil.disk_usage needs an
    # existing path on POSIX (os.statvfs raises on a missing one; Windows
    # tolerates it).  The builder also makedirs them idempotently, so this is
    # only an early-existence guarantee for the pre-check, never a semantic
    # difference.
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(args.panel_store, exist_ok=True)
    final_gb, scratch_gb = _streaming_disk_required_gb(*est)
    margin_gb = _scratch_safety_margin_gb()
    n_stocks, n_timesteps, n_features = est
    if _same_filesystem(scratch_dir, args.panel_store):
        need = final_gb + scratch_gb + margin_gb
        free = shutil.disk_usage(scratch_dir).free / (1024 ** 3)
        if need > free:
            raise SystemExit(
                f"streaming panel build into --panel-store requires "
                f"{need:.1f} GB which exceeds free space {free:.1f} GB on the "
                f"shared drive ({scratch_dir}) (n_stocks={n_stocks} x "
                f"n_timesteps={n_timesteps} x n_features={n_features} + "
                f"scratch pickles + margin) — the build cannot fit (§v18-6).  "
                f"Point --scratch-dir at a drive with more room, or shrink the "
                f"universe/stocks.")
        return need, free
    scratch_free = shutil.disk_usage(scratch_dir).free / (1024 ** 3)
    panel_free = shutil.disk_usage(args.panel_store).free / (1024 ** 3)
    # Margin is applied PER VOLUME deliberately: the scratch pickles persist on
    # the scratch volume while the final grids are written on the panel_store
    # volume, so reserving margin on both is conservative — do NOT "simplify"
    # this into a single shared margin.
    problems = []
    if scratch_gb + margin_gb > scratch_free:
        problems.append(
            f"scratch pickles {scratch_gb:.1f} GB + margin {margin_gb:.1f} GB "
            f"exceeds free space {scratch_free:.1f} GB on the scratch drive "
            f"({scratch_dir})")
    if final_gb + margin_gb > panel_free:
        problems.append(
            f"final panel {final_gb:.1f} GB + margin {margin_gb:.1f} GB "
            f"exceeds free space {panel_free:.1f} GB on the panel_store drive "
            f"({args.panel_store})")
    if problems:
        raise SystemExit(
            "streaming panel build disk pre-check FAILED (§v18-6): "
            + "; ".join(problems)
            + f" (n_stocks={n_stocks} x n_timesteps={n_timesteps} x "
            f"n_features={n_features}) — free disk space or re-point "
            f"--scratch-dir / --panel-store to a larger drive.")
    return final_gb + scratch_gb + margin_gb, min(scratch_free, panel_free)


def _early_panel_memory_guard(
    args, stock_list: list[str], data_dir, store_load: bool,
) -> tuple[float, str] | None:
    """§七-P0: enforce the pre-build universe memory estimate (main entry).

    Runs BEFORE _resolve_panel so an oversized universe is refused before the
    dense (N, T, D) grids are allocated.  Skipped when ``store_load`` is True
    (no build — the store is mmap'd lazily and its surviving subset may be far
    smaller than the requested universe) or when no estimate can be made.
    Returns the ``(est_gb, action)`` verdict from _enforce_universe_memory, or
    None when the early guard is skipped.  A refusal (no --allow-high-risk-
    universe) raises SystemExit.

    §T10c: the guard is BUILD-PATH-AWARE — a FIRST build into ``--panel-store``
    (``streaming=True``) is a memmap-sunk two-pass build whose resident peak is
    bounded and roughly independent of n_stocks, so it is judged against the
    bounded streaming peak (not the dense estimate), making a full-market
    streaming first build admissible.
    """
    if store_load:
        return None
    est = _estimate_panel_memory(args, stock_list, data_dir)
    if est is None:
        return None
    # §T10c: a FIRST build into --panel-store routes build_panel_features to the
    # STREAMING/two-pass path (memmap_dir set; store_load is already handled by
    # the early return above, so a set panel_store here means build-into-store,
    # not load).  The streaming resident peak is bounded and roughly independent
    # of n_stocks, so the DENSE estimate must not gate it.  use_fundamental_refine
    # gates the resident cs_panel_df term of the streaming peak; FeaturePipeline
    # defaults it True and train_panel.py has no CLI lever for it, so an absent
    # stub attr reads True (conservative).
    streaming = bool(getattr(args, "panel_store", None))
    use_fundamental_refine = bool(
        getattr(args, "use_fundamental_refine", True))
    return _enforce_universe_memory(
        args.universe, *est,
        allow_override=args.allow_high_risk_universe,
        available_gb=_host_available_gb(),
        streaming=streaming,
        use_fundamental_refine=use_fundamental_refine,
    )

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
    # §T4: stock-level coverage — the live path's per-stock metric.  The gate
    # reads the channel's declared metric (stock_coverage for per-stock
    # channels), so expose it under its canonical name.
    entry["stock_coverage"] = entry["coverage"]
    entry["status"] = (
        "FAILED" if loaded == 0 and errors > 0 else
        "MISSING" if loaded == 0 else
        "PARTIAL" if loaded < n else "OK"
    )
    logger.info("[%s] loaded %d/%d stocks (errors=%d) %s",
                name, loaded, n, errors, entry["status"])

def _trading_day_count(data_dir, start_date, end_date) -> int:
    """Trading-day denominator for broadcast date-coverage probes (§T4).

    Uses the strict research calendar (data_dir-threaded); a strict-calendar
    overflow past ``verified_until`` (forward-estimate days) falls back to a ~
    weekday fraction of the raw span so a coverage probe never crashes.  A
    falsy window yields 0 — the caller treats any present data as covered.
    """
    if not start_date or not end_date:
        return 0
    lo = pd.Timestamp(start_date).date()
    hi = pd.Timestamp(end_date).date()
    try:
        cal = get_research_calendar(strict=True, data_dir=data_dir)
        return len(cal.get_trading_days(lo, hi))
    except ValueError as exc:
        # Strict-calendar range overflow past verified_until (or a malformed
        # calendar artifact) — fall back to a ~weekday fraction of the raw span
        # so a coverage probe never crashes.  Logged at debug so a corrupted
        # artifact is surfaced instead of silently masked.
        logger.debug("trading-day count fell back to a weekday estimate for "
                     "[%s, %s]: %s", start_date, end_date, exc)
        span_days = (hi - lo).days
        return max(1, int(round(span_days * 5 / 7)))


def _date_coverage_fraction(n_dates_in_range, data_dir, start_date, end_date) -> float:
    """Date coverage = n_dates_in_range / trading days, for broadcast channels.

    A falsy window (no start/end) treats any present data as fully covered
    (1.0 when ``n_dates_in_range > 0``), 0.0 otherwise.  This is the MEANINGFUL
    metric for market-wide broadcast channels — their value is the same for
    every stock per date, so stock coverage is vacuous.  The fraction is CLIPPED
    at 1.0: a broadcast file written with ``resample("D")`` counts weekend dates
    in the numerator while the denominator is trading days only, so the raw
    ratio can exceed 1.0 — it only inflates (never false-aborts), and the clip
    keeps reported coverage honest (§T4 review).
    """
    if not start_date or not end_date:
        return 1.0 if n_dates_in_range > 0 else 0.0
    n_trading = _trading_day_count(data_dir, start_date, end_date)
    if not n_trading:
        return 0.0
    return round(min(1.0, n_dates_in_range / n_trading), 4)


def _filter_dates_in_range(dates, start_date, end_date):
    """Return ``dates`` normalized and restricted to [start_date, end_date].

    Shared by ``_probe_broadcast_dates`` AND the etf_flow aggregation so both
    count in-window dates the same way — a date-coverage numerator that counted
    every distinct date across full history would make the etf_flow contract
    vacuous (almost always >= 1.0, able to pass with zero in-window data; §T4
    review).  A falsy bound is treated as unbounded on that side.
    """
    dates = pd.to_datetime(dates).dt.normalize()
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    return dates


def _probe_broadcast_dates(path, start_date, end_date, data_dir):
    """Probe a broadcast channel's date coverage from its parquet file (§T4).

    Broadcast channels (etf_flow / industry / market_env) carry the same value
    for every stock per date, so the probe records DATE coverage — distinct
    dates in [start_date, end_date] over trading days in range (see
    ``_date_coverage_fraction``) — instead of the vacuous stock coverage.

    Normalizes a DatetimeIndex OR a date column like ``aux_aligner``'s
    ``_merge_industry`` / ``_merge_market_env``.  Returns ``(n_dates_in_range,
    status)`` where status is MISSING (file absent) / FAILED (unreadable) / OK.
    The WHOLE body (not just the parquet read) is inside the try — a malformed
    index/column must degrade to FAILED, never raise out of load_aux_data.
    """
    if not os.path.isfile(path):
        return 0, "MISSING"
    try:
        df = pd.read_parquet(path)
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            else:
                return 0, "FAILED"
        dates = _filter_dates_in_range(df["date"], start_date, end_date)
        return int(dates.nunique()), "OK"
    except Exception:
        return 0, "FAILED"


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

def _fmt_manifest_problem(path: str, report: dict) -> str:
    """Human-readable problem string from a ``validate_asset_manifest`` report."""
    if report.get("reason"):
        return f"{path}: {report['reason']}"
    return f"{path}: " + "; ".join(report.get("mismatches") or ["manifest mismatch"])


def _enforce_formal_manifests(
    stock_list: list[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    consumed_set: set[str],
) -> None:
    """§T4/§十九-9: formal-mode asset-manifest gate for the consumed aux channels.

    Runs once, BEFORE the lenient per-channel load, when ``formal=True``.  For
    every CONSUMED channel with data on disk (every channel the run's pipeline
    actually opens — §v18-2, not just the required subset), the channel's asset
    manifest(s) must be present AND match the parquet they guard (content
    ``schema_hash`` / ``rows`` / ``start``-``end`` extent / ``data_type`` /
    ``partition`` / vintage declaration).  A channel whose data file exists but
    whose manifest is missing or mismatched — or a consumed channel with NO
    manifest support at all (the cninfo announcement-sentiment path, and the
    unadopted MarketWideStorage types shareholder/valuation) — FAILS HARD
    (SystemExit) instead of the explore-mode warn-and-proceed.

    Channels with NO data on disk are NOT checked here: the coverage gate
    (``train_panel_gates._enforce_channel_coverage``) already aborts a required
    channel at zero coverage.  This gate is the MANIFEST side of §十九-9 only.

    WHY a FULL per-stock scan (not a single-stock sample): the per-stock
    channels (sentiment/guba/comment/fundamental/announcement) carry PER-STOCK
    asset manifests (``{code}.parquet`` + ``{code}.manifest.json``), so
    validating one representative stock would miss tampering on any other stock
    and silently weaken the formal guarantee — correctness over I/O.  A
    single-read refactor (threading ``require_valid_manifest=True`` into the
    lenient loads below) was considered and rejected: it would route manifest
    failures through ``_load_channel_aux``'s per-stock ``except Exception``
    error accounting (downgrading a hard abort to a per-channel FAILED/partial
    note) and would lose this gate's all-channels-at-once aggregate diagnostics
    (§T4 review #2).  The pre-pass's extra read cost is bounded by the ~500-stock
    formal live universe — the price of fail-closed verification.
    """
    from stoke_ml.data.news_storage import NewsStorage
    from stoke_ml.data.guba_storage import GubaStorage
    from stoke_ml.data.market_wide_storage import (
        MARKET_WIDE_ASSETS, MarketWideStorage,
    )
    from stoke_ml.data.fundamental_storage import FundamentalStorage
    from stoke_ml.data.comment_storage import CommentStorage
    from stoke_ml.data.announcement_storage import AnnouncementStorage
    from stoke_ml.data.broadcast_assets import INDUSTRY_ASSET, MARKET_ENV_ASSET
    from stoke_ml.data.etf_storage import ETF_FLOW_ASSET
    from stoke_ml.data.asset_contract import (
        validate_asset_manifest,
        validate_derived_asset,
    )

    problems: list[str] = []
    code_set = set(stock_list)

    def _per_stock(ch: str, storage, load_name: str) -> None:
        """Formal per-stock read of one channel.

        Validates EVERY stock's manifest for this channel: the storage's
        ``load_*(..., require_valid_manifest=True)`` raises ``ValueError`` (via
        ``asset_contract.check_asset_read``) the moment a present parquet's
        manifest is missing or mismatched, and ``OSError`` on a disk-level read
        failure; a stock with no file on disk returns an empty frame without
        raising (its absence is the coverage gate's concern).  A stock with a
        valid manifest loads normally and the loop advances.  On the FIRST
        failure the problem is recorded and the loop stops — the same
        channel-wide defect hits every stock, so one representative failure is
        enough to abort the channel.
        """
        load = getattr(storage, load_name)
        for code in stock_list:
            try:
                load(code, start_date, end_date, require_valid_manifest=True)
            except (ValueError, OSError) as exc:
                problems.append(f"{ch}[{code}]: {exc}")
                return

    for ch in sorted(consumed_set):
        if ch == "sentiment":
            _per_stock(ch, NewsStorage(data_dir), "load_daily_sentiment")
        elif ch == "guba":
            _per_stock(ch, GubaStorage(data_dir), "load_daily_sentiment")
        elif ch == "comment":
            _per_stock(ch, CommentStorage(data_dir), "load_daily")
        elif ch == "fundamental":
            _per_stock(ch, FundamentalStorage(data_dir), "load")
        elif ch == "announcement":
            # The loader PREFERS the cninfo announcement-sentiment path, which
            # has NO storage/manifest support — under formal it must fail loudly
            # (use --prebuilt or add a DataAssetContract writer; T5/T9).
            cninfo_dir = os.path.join(
                data_dir, "a_shares", "cninfo_announcements", "sentiment")
            cninfo_hits = []
            if os.path.isdir(cninfo_dir):
                cninfo_hits = [
                    f for f in os.listdir(cninfo_dir)
                    if f.endswith(".parquet")
                    and os.path.splitext(f)[0] in code_set
                ]
            if cninfo_hits:
                problems.append(
                    "announcement: cninfo announcement-sentiment "
                    f"({cninfo_dir}/) has no asset-manifest support — use "
                    "--prebuilt or add a DataAssetContract writer for it "
                    "(T5/T9)")
            else:
                _per_stock(ch, AnnouncementStorage(data_dir),
                           "load_daily_sentiment")
        elif ch in _MARKET_WIDE_CHANNELS:
            data_type = live_data_type(CHANNEL_SOURCE[ch])
            if data_type not in MARKET_WIDE_ASSETS:
                problems.append(
                    f"{ch}: MarketWideStorage data_type {data_type!r} has no "
                    "asset-manifest contract — use --prebuilt or adopt a "
                    "DataAssetContract for it (T9)")
            else:
                _per_stock(ch, MarketWideStorage(data_dir, data_type), "load")
        elif ch == "etf_flow":
            etf_base = os.path.join(data_dir, "a_shares", "etf_flow")
            if os.path.isdir(etf_base):
                for f in sorted(os.listdir(etf_base)):
                    if f.startswith("sector_") and f.endswith(".parquet"):
                        path = os.path.join(etf_base, f)
                        report = validate_asset_manifest(path, ETF_FLOW_ASSET)
                        if not report["ok"]:
                            problems.append(_fmt_manifest_problem(path, report))
                            break
            # Zero sector files (or no etf_flow dir) → NOT a manifest problem:
            # a required etf_flow channel at zero coverage is aborted by the
            # coverage gate (_enforce_channel_coverage), not here (§T4 review #5).
        elif ch in ("industry", "market_env"):
            fname = ("industry_returns.parquet" if ch == "industry"
                     else "market_env_daily.parquet")
            asset = INDUSTRY_ASSET if ch == "industry" else MARKET_ENV_ASSET
            rel = os.path.join(*source_dir(CHANNEL_SOURCE[ch]).split("/"), fname)
            path = os.path.join(data_dir, rel)
            if os.path.isfile(path):
                report = validate_asset_manifest(path, asset)
                if not report["ok"]:
                    problems.append(_fmt_manifest_problem(path, report))
                elif ch == "market_env":
                    # §v19 P0#2: market_env is a DERIVED asset — integrity
                    # (file matches manifest) passed above; NOW check freshness
                    # (was it built from the CURRENT upstreams / transform code
                    # / transform config?).  The lineage is recomputed from
                    # what is on disk right now; the recorded ``parts`` come
                    # from the manifest (the write-time config snapshot).  A
                    # stale lineage means the on-disk file predates a change to
                    # its inputs or builder → FAIL, rebuild required.
                    from scripts.production.build_market_env import compute_lineage
                    parts = (report["manifest"] or {}).get("parts", {})
                    lineage_now = compute_lineage(data_dir, parts)
                    lineage = validate_derived_asset(
                        report["manifest"] or {},
                        current_upstream_roots=lineage_now["upstream_roots"],
                        current_transform_code_hash=lineage_now["transform_code_hash"],
                        current_transform_config_hash=lineage_now["transform_config_hash"],
                    )
                    if lineage["stale"]:
                        problems.append(
                            f"{path}: DERIVED-ASSET STALE — "
                            + "; ".join(lineage["mismatches"])
                            + "; rebuild with build_market_env.py")
        else:
            problems.append(
                f"{ch}: consumed channel is not loaded by load_aux_data and "
                "has no manifest checker — cannot verify in formal mode; use "
                "--prebuilt")
    if problems:
        for p in problems:
            logger.error("formal manifest gate: %s", p)
        raise SystemExit(
            "formal mode: consumed aux-channel asset-manifest gate FAILED — "
            "refusing to train on unverified data (§T4/§十九-9): "
            + "; ".join(problems))


def load_aux_data(
    stock_list: list[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    required_channels: set[str] | None = None,
    consumed_channels: set[str] | None = None,
    formal: bool = False,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    """Load auxiliary data (sentiment, guba, margin, etc.) per stock.

    ``formal`` (§T4): a FORMAL run (``formal=True``, threaded from
    ``_formal_mode(args)`` in the live branch of ``_resolve_panel``) runs
    :func:`_enforce_formal_manifests` first — every CONSUMED channel's asset
    manifest must be present + matching, else the run aborts (SystemExit).
    ``consumed_channels`` (§v18-2) is the full channel set the run's pipeline
    opens (derived from ``args`` + ``seq_len`` via ``_consumed_channels``);
    ``required_channels`` stays the coverage-contract subset.  The gate covers
    ``consumed_channels`` when given, else falls back to ``required_channels``
    (backward compat).  The default ``formal=False`` (explore / legacy) keeps
    the warn-and-proceed behavior byte-for-byte: lenient reads, a
    present-but-mismatched manifest logs a warning, a manifest-less file reads
    as legacy data.

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

    # §T4/§十九-9: formal mode — every consumed channel's asset manifest must
    # be present + matching BEFORE any lenient load proceeds.  Explore mode
    # (formal=False) skips this entirely and keeps the legacy warn-and-proceed.
    # §v18-2: the gate binds the full CONSUMED set when threaded, not just the
    # required subset (which is the extra coverage-contract layer).
    if formal and (consumed_channels or required_set):
        _enforce_formal_manifests(
            stock_list, data_dir, start_date, end_date,
            consumed_channels or required_set)

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
    # The storage's data type is the channel's LIVE dir last segment (§T2).
    for ch in _MARKET_WIDE_CHANNELS:
        _load_channel_aux(
            ch, stock_list, result, manifest,
            make_storage=lambda ch=ch: MarketWideStorage(
                data_dir, live_data_type(CHANNEL_SOURCE[ch])),
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
            # §T4: broadcast channel — stock coverage is vacuous (1.0 whenever
            # the aggregated data exists); DATE coverage is the meaningful metric.
            entry["stock_coverage"] = entry["coverage"]
            # §T4 review: the numerator MUST be window-filtered — counting every
            # distinct date across all sector files (full history) would make
            # date_coverage ~1.0 with zero in-window data, vacating the 0.80
            # contract.  Reuse the same in-window filter as _probe_broadcast_dates.
            n_etf_dates = int(
                _filter_dates_in_range(etf_agg["date"], start_date, end_date).nunique())
            entry["date_coverage"] = _date_coverage_fraction(
                n_etf_dates, data_dir, start_date, end_date)
            entry["status"] = "OK"
            logger.info("[etf_flow] aggregated from %d sector files "
                        "(broadcast to %d stocks)", len(etf_frames), len(stock_list))
        else:
            entry["coverage"] = 0.0
            entry["stock_coverage"] = 0.0
            entry["date_coverage"] = 0.0
            logger.info("[etf_flow] no sector files found — MISSING")
    except Exception as exc:
        entry["errors"] = len(stock_list)
        entry["status"] = "FAILED"
        entry["note"] = str(exc)
        entry["stock_coverage"] = 0.0
        entry["date_coverage"] = 0.0
        logger.warning("[etf_flow] aggregation failed — %s", exc)

    # Broadcast channels (industry / market_env): market-wide per-date values,
    # the same for every stock per date — STOCK coverage is vacuous (1.0 when
    # the file exists); DATE coverage is the meaningful metric (§T4).  The
    # feature pipeline loads these itself (aux_aligner), so load_aux_data only
    # records the coverage probe in the manifest (broadcast, not per-stock).
    # §T2: the broadcast file's DIRECTORY comes from the CHANNEL_SOURCE
    # registry; the filename is the channel-specific artifact (not a path
    # literal) so the two paths can't drift.  The registry path is joined
    # segment-by-segment to stay byte-identical with the historical literal.
    for name, fname in (
        ("industry", "industry_returns.parquet"),
        ("market_env", "market_env_daily.parquet"),
    ):
        entry = _new_channel_entry(True, name in required_set)
        manifest[name] = entry
        rel_path = os.path.join(
            *source_dir(CHANNEL_SOURCE[name]).split("/"), fname)
        path = os.path.join(data_dir, rel_path)
        n_dates, status = _probe_broadcast_dates(
            path, start_date, end_date, data_dir)
        entry["loaded_stocks"] = None  # broadcast channel, not per-stock
        if status == "OK":
            entry["coverage"] = 1.0
            entry["stock_coverage"] = 1.0
            entry["date_coverage"] = _date_coverage_fraction(
                n_dates, data_dir, start_date, end_date)
            entry["status"] = "OK"
            logger.info("[%s] date coverage %.4f (%d dates in range) "
                        "(broadcast)", name, entry["date_coverage"], n_dates)
        elif status == "MISSING":
            entry["coverage"] = 0.0
            entry["stock_coverage"] = 0.0
            entry["date_coverage"] = 0.0
            entry["status"] = "MISSING"
            logger.info("[%s] %s not found — MISSING", name, rel_path)
        else:  # FAILED
            entry["coverage"] = 0.0
            entry["stock_coverage"] = 0.0
            entry["date_coverage"] = 0.0
            entry["status"] = "FAILED"
            entry["note"] = f"unreadable broadcast parquet: {path}"
            logger.warning("[%s] broadcast parquet unreadable — %s",
                           name, path)

    loaded = sum(1 for v in result.values() if v)
    logger.info("Aux data loaded for %d/%d stocks", loaded, len(stock_list))
    return result, manifest

def _prebuilt_channel_coverage(panel_data: dict) -> dict:
    """Channel coverage probed from a prebuilt panel's has_* flags.

    past_observed is (N, T, D); each has_* flag is True on exactly the
    (stock, day) cells where that aux channel delivered data (the pipeline
    ZI-fills absent cells, so False == no data).  §T4: coverage is reported per
    DECLARED METRIC — the coverage gate picks the channel's contract metric:

    * ``stock_coverage`` — ``mask.any(axis=1).mean()``, the fraction of STOCKS
      with >=1 present cell;
    * ``cell_coverage`` — ``mask.mean()``, the fraction of (stock, day) cells;
    * ``date_coverage`` — ``mask.any(axis=0).mean()``, the fraction of DATES
      with >=1 present cell.

    ``coverage`` remains as the legacy alias for ``cell_coverage`` so external
    readers of ``entry["coverage"]`` keep working.  Channels without a has_*
    flag carry no presence marker in the arrays, so they are left out entirely
    (unprobeable — the gate's formal mode aborts / explore warns on them).
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
        stock_coverage = round(float(mask.any(axis=1).mean()), 4)
        cell_coverage = round(float(mask.mean()), 4)
        date_coverage = round(float(mask.any(axis=0).mean()), 4)
        channels[channel] = {
            "requested": True,
            "required": False,
            "loaded_stocks": None,  # cell-level probe, not per-stock
            "coverage": cell_coverage,
            "stock_coverage": stock_coverage,
            "cell_coverage": cell_coverage,
            "date_coverage": date_coverage,
            "errors": 0,
            "status": "OK" if present else "MISSING",
            "cells": int(mask.size),
            "flag": flag,
        }
    return channels
