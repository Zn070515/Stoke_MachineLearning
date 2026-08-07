"""Panel construction, memmap store, and aux-channel loading (§二十一).

Extracted from ``scripts.production.train_panel`` — the vintage-policy feature
switch set (``_panel_pipeline_kwargs``), the panel-store meta fingerprint, the
panel builder (``_resolve_panel``), the universe memory guards, the aux-channel
coverage manifest loading, and the has_* flag probe.  ``train_panel``
re-exports these names for backward compatibility.
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stoke_ml.data.calendar import get_research_calendar
from stoke_ml.data.universe import (
    load_index_membership,
    load_universe_status,
)
from stoke_ml.data.vintage_policy import VintagePolicy, channel_allowed
from stoke_ml.features.aux_cols import FUNDAMENTAL_COLS
from stoke_ml.features.cache_manifest import (
    _dir_content_hash,
    current_config_hash,
    git_head,
)
from stoke_ml.features.panel_builders._arrays import close_memmap_grids
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
    preference (the policy can only turn channels OFF, never force one ON);
    ``--allow-fundamental-ablation`` is the one exception, forcing the
    fundamental channel ON (only that channel) regardless of policy.
    """
    policy = VintagePolicy(args.vintage_policy)
    kwargs = {
        _SWITCH_KEY.get(dim, f"use_{dim}"): pref and channel_allowed(dim, policy)
        for dim, pref in _BASE_DIM_PREFERENCE.items()
    }
    # T3 research decision #1: fundamental is denied under safe-only
    # (latest_revised_aligned) and may enter ONLY via an explicit ablation.
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

def _panel_store_meta(
    args, seq_len: int, stock_list: list[str] | None = None,
    data_dir: str | None = None, prebuilt_dir: str | None = None,
) -> dict:
    """Build-time fingerprint persisted in a panel store's meta.json.

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
    if prebuilt_dir:
        meta["prebuilt_feature_manifest_hash"] = _dir_content_hash(
            os.path.join(prebuilt_dir, ".manifests"))
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
                args, seq_len, stock_list, data_dir, args.prebuilt),
            strict_external_meta=strict)
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
    panel_data = fp.build_panel_features(
        panel, aux_data=aux_data, horizon=args.horizon, prebuilt_dir=args.prebuilt,
        require_feature_manifest=args.require_feature_manifest,
        daily_membership=daily_membership,
        memmap_dir=args.panel_store,
    )
    if args.panel_store:
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
                args, seq_len, stock_list, data_dir, args.prebuilt),
            skip_npy=sink_grids)
        logger.info("Saved panel memmap store to %s", args.panel_store)
        # Re-load the full store for downstream training — fresh lazy memmaps
        # for all arrays (the big grids page-fault only the rows/cols touched).
        panel_data = load_panel_memmap(
            args.panel_store,
            expected_meta=_panel_store_meta(
                args, seq_len, stock_list, data_dir, args.prebuilt),
            strict_external_meta=strict)
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
        cols = list(pq.read_schema(str(parquets[0])).names)
    except Exception:
        return None  # unreadable schema — never crash the estimate path
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
        n_timesteps = len(get_research_calendar(strict=True).get_trading_days(lo, hi))
    except ValueError:
        # strict calendar extends only to verified_until — fall back to a
        # ~ trading-day fraction of the raw span for the estimate.
        n_timesteps = int((hi - lo).days * 0.7)
    return n_stocks, n_timesteps, len(surviving)

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
    """
    if store_load:
        return None
    est = _estimate_panel_memory(args, stock_list, data_dir)
    if est is None:
        return None
    return _enforce_universe_memory(
        args.universe, *est,
        allow_override=args.allow_high_risk_universe,
        available_gb=_host_available_gb(),
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
