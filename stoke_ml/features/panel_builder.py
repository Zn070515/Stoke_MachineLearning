"""Panel-format feature construction for VSN+xLSTM training.

``build_panel_features`` builds the panel-format arrays (static / past-known /
past-observed grids plus the direction/return/volatility target masks) from a
multi-stock panel.  Extracted from ``FeaturePipeline.build_panel_features``
(§二十一); it operates on a ``FeaturePipeline`` instance passed as the first
argument so the public method keeps delegating through it.  Leaf-safe: imports
nothing from ``stoke_ml.features.pipeline`` (only the leaf ``panel_helpers``
plus data-layer lazy imports), so ``pipeline`` can import this module without
an import cycle.

The five builder concerns were extracted into ``stoke_ml.features.panel_builders/``
(§二十一 refactor):
  - ``_targets.py``        — TargetBuilder: per-stock labels / masks / PIT-static raw inputs
  - ``_eligibility.py``    — EligibilityBuilder: decision / history / universe-eligibility masks
  - ``_normalizer.py``     — DateWiseZScoreNormalizer: per-date z-score + _daily_member_flag
  - ``_arrays.py``         — PanelArrays: allocation + sanitization + dict assembly (T8 seam)
  - ``_static_context.py`` — StaticContextBuilder: static / pk / po grid population + quantile ranks
"""
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.config.feature_profile import (
    CHANNEL_COLUMNS,
    market_env_account_is_verified,
)
from stoke_ml.features import cache_manifest
from stoke_ml.features.fundamental import FundamentalRefiner
from stoke_ml.features.panel_builders._arrays import (
    PanelArrays,
    compute_entry_fill_prob,
)
from stoke_ml.features.panel_builders._eligibility import EligibilityBuilder
from stoke_ml.features.panel_builders._normalizer import (
    DateWiseZScoreNormalizer, _daily_member_flag,  # noqa: F401  re-exported for import-compat
)
from stoke_ml.features.panel_builders._static_context import StaticContextBuilder
from stoke_ml.features.panel_builders._targets import TargetBuilder
from stoke_ml.features.panel_helpers import (
    _get_panel_calendar,
    _manifest_check_config,
    _PIT_STATIC_COLS,
)
from stoke_ml.features.temporal import add_calendar_features

logger = logging.getLogger(__name__)

# §T7: streaming-build scratch-dir management.  The streaming / two-pass
# builder writes each stock's engineered frame to a per-stock scratch pickle
# instead of keeping the full panel resident.  The scratch dir is now
# directable (--scratch-dir / <panel-store>/scratch/<run_id>/), disk-pre-checked
# (estimate pre-build + exact post-Pass-1 backstop), carries a run_manifest.json,
# and orphan scratch dirs from a hard-killed build are swept at startup.
_RUN_MANIFEST_NAME = "run_manifest.json"
_DEFAULT_SCRATCH_STALE_DAYS = 7
_DEFAULT_SCRATCH_SAFETY_MARGIN_GB = 5.0


def _scratch_run_id() -> str:
    """Run id for the streaming-build scratch dir (§T7).

    Same ``YYYYMMDD-HHMMSS-<pid>`` convention as ``preprocess_new_data.py``'s
    write-manifest ``run_id``, so the two share one vocabulary.  Unique per
    process; a re-run is a NEW run unless the caller points ``--scratch-dir``
    at a previous run's scratch dir (see the resume logic in
    ``_build_panel_streaming``).
    """
    return f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"


def _feature_switches_hash(pipeline) -> str:
    """Stable fingerprint of the feature-engineering switch set (§T7).

    Snapshot of the pipeline's ``use_*`` flags plus ``seq_len`` / ``minute_mode``
    — the dimensions that change a stock's engineered frame schema.  Recorded in
    ``run_manifest.json`` so a same-scratch re-run can REFUSE to resume when the
    switches changed: resuming would skip old-schema pickles and engineer
    new-schema ones into a silent hybrid panel.

    The builder is a config-free leaf module, so the fingerprint is derived from
    the pipeline INSTANCE (the ``use_*`` attributes FeaturePipeline already
    stores), not from CLI/config state.
    """
    keys = {
        k: (bool(v) if isinstance(v, bool) else v)
        for k, v in vars(pipeline).items()
        if k.startswith("use_") or k in ("seq_len", "minute_mode")
    }
    return hashlib.sha256(
        json.dumps(keys, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _read_run_manifest(scratch: str) -> dict | None:
    """Read ``run_manifest.json`` from a scratch dir, or None when absent/corrupt.

    The manifest marks a scratch dir as belonging to a specific run (§T7): a
    dir that still holds its manifest + per-stock pickles is a crashed run a
    same-scratch re-run can resume (Pass 1 skips the pickles already on disk).
    """
    path = os.path.join(scratch, _RUN_MANIFEST_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_run_manifest(
    scratch: str, run_id: str, stage: str, *,
    pid: int | None = None, resumed: bool = False,
    feature_switches_hash: str | None = None,
) -> dict:
    """Write/overwrite ``run_manifest.json`` with the given stage (§T7).

    Fields: ``run_id`` / ``start_time`` / ``stage`` / ``pid`` (the §T7
    contract) plus ``resumed`` when this is a crash-recovery re-run, and
    ``feature_switches_hash`` (the §T7 review fingerprint) when the caller
    supplies it — the builder always does, so a resume target carries the
    switch set it was built under.  ``start_time`` is preserved from a prior
    manifest so a resumed run keeps its original birth timestamp.  A write
    failure (read-only / locked scratch) warns instead of crashing the build.
    """
    prev = _read_run_manifest(scratch)
    manifest = {
        "run_id": run_id,
        "start_time": (prev or {}).get(
            "start_time", datetime.now().isoformat(timespec="seconds")),
        "stage": stage,
        "pid": pid if pid is not None else os.getpid(),
    }
    if feature_switches_hash is not None:
        manifest["feature_switches_hash"] = feature_switches_hash
    if resumed:
        manifest["resumed"] = True
    try:
        with open(os.path.join(scratch, _RUN_MANIFEST_NAME), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
    except OSError as exc:  # never crash the build over a diagnostic file
        logger.warning("could not write %s in %s: %s",
                       _RUN_MANIFEST_NAME, scratch, exc)
    return manifest


def _cleanup_stale_scratch_dirs(
    root: str, n_days: int, *, prefix: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Remove scratch dirs under *root* whose mtime is older than *n_days* (§T7).

    Startup sweep for orphan scratch dirs a hard-killed build left behind (its
    ``finally`` never ran).  Only DIRECTORY entries are candidates; *prefix*
    (when given) restricts removal to entries with a matching name (the
    ``panel_stream_scratch_`` temp fallback), and *exclude* — the CURRENT run's
    dir — is never touched.  Every removal uses ``ignore_errors=True`` so a
    Windows file-lock on a leftover pickle logs instead of crashing the new
    build (mirrors the builder's own ``finally``).  Returns the removed paths.
    """
    removed: list[str] = []
    if not root or not os.path.isdir(root):
        return removed
    cutoff = time.time() - n_days * 86400
    exclude_abs = os.path.abspath(exclude) if exclude else None
    for name in os.listdir(root):
        if prefix is not None and not name.startswith(prefix):
            continue
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if exclude_abs is not None and os.path.abspath(p) == exclude_abs:
            continue
        try:
            if os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
        except OSError:
            continue
    return removed


def _engineer_stock(
    pipeline,
    code: str,
    prebuilt_dir: str | None,
    panel: pd.DataFrame,
    aux_data: dict[str, dict[str, pd.DataFrame]],
    data_dir: str | None,
    drop_reasons: Counter,
    drop_examples: dict[str, list[str]],
) -> pd.DataFrame | None:
    """Engineer features for a single stock (prebuilt or live path).

    Extracted from ``build_panel_features`` (§T5 streaming/two-pass) so both
    the dense and streaming paths share the same per-stock body.

    Returns the engineered feature DataFrame, or None if the stock is dropped.
    Side-effects: mutates *drop_reasons* and *drop_examples* for drop
    accounting.
    """
    if prebuilt_dir:
        path = os.path.join(prebuilt_dir, f"{code}.parquet")
        feats = pipeline.load_features(path)
        feats["date"] = pd.to_datetime(feats["date"])
        # Flat prebuilt (data/features/) carries temporal lag columns
        # (skip_temporal=False).  Panel training uses skip_temporal=True
        # (xLSTM learns the time structure itself), so drop *_lag{N}
        # columns — the remainder matches a --panel-mode build.
        lag_cols = [c for c in feats.columns if re.search(r"_lag\d+$", c)]
        if lag_cols:
            feats = feats.drop(columns=lag_cols)
        # §七: topic_* columns (global_frozen topic model) are OFF by
        # default — drop them on the PREBUILT path too, not just in
        # _engineer_features, so a prebuilt parquet that carried them
        # (built with use_topic=True, or a schema drift) cannot leak
        # the non-PIT representation into a default training run.
        if not pipeline.use_topic:
            feats = pipeline._drop_topic_columns(feats)
        # §T7/§十四: generic per-channel scrub.
        _channel_switch_attr = {"announcement": "use_announcements"}
        _off_cols: list[str] = []
        for _channel, _cols in CHANNEL_COLUMNS.items():
            # §T5: the market_env ACCOUNT part is dropped whenever
            # use_market_env_account is OFF (the proxy default) — EXCEPT once
            # the account part is declared verified, when it is part of the
            # required/verified set and must never be scrubbed by the ablation
            # flag being off (the verified state is global, so a verified
            # account part is consumed regardless of the run's flag).
            if _channel == "market_env_account" and market_env_account_is_verified():
                continue
            _switch = getattr(
                pipeline,
                _channel_switch_attr.get(_channel, f"use_{_channel}"),
                True,
            )
            if not _switch:
                _off_cols.extend(c for c in _cols if c in feats.columns)
        if _off_cols:
            feats = feats.drop(columns=_off_cols)
        # A stale/hand-built parquet may carry a
        # weekend/duplicate bar that would pollute the UNION date axis.
        feats = pipeline._clean_calendar_dates(feats, code, data_dir=data_dir)
        if feats is None:
            drop_reasons["calendar_clean_dropped"] += 1
            drop_examples["calendar_clean_dropped"].append(code)
            return None
        # Calendar features are idempotent (overwrite in place); safe
        # to re-apply even though save_features(panel_mode=True) already
        # added them — guards against hand-built parquets.
        feats = add_calendar_features(feats)
    else:
        mask = panel["stock_code"] == code
        df_stock = panel[mask].sort_values("date").reset_index(drop=True)
        # Drop phantom/duplicate/out-of-calendar rows before
        # feature engineering so a bad bar neither pollutes the UNION
        # date axis nor corrupts the rolling indicators around it.
        df_stock = pipeline._clean_calendar_dates(df_stock, code, data_dir=data_dir)
        if df_stock is None:
            drop_reasons["calendar_clean_dropped"] += 1
            drop_examples["calendar_clean_dropped"].append(code)
            return None
        stock_aux = aux_data.get(code, {})
        feats = pipeline._engineer_features(
            df_stock,
            sentiment_df=stock_aux.get("sentiment"),
            guba_df=stock_aux.get("guba"),
            comment_df=stock_aux.get("comment"),
            announcement_df=stock_aux.get("announcement"),
            margin_df=stock_aux.get("margin"),
            northbound_df=stock_aux.get("northbound"),
            dragon_tiger_df=stock_aux.get("dragon_tiger"),
            fundamental_df=stock_aux.get("fundamental"),
            valuation_df=stock_aux.get("valuation"),
            etf_flow_df=stock_aux.get("etf_flow"),
            capital_flow_df=stock_aux.get("capital_flow"),
            block_trade_df=stock_aux.get("block_trade"),
            shareholder_df=stock_aux.get("shareholder"),
            lockup_df=stock_aux.get("lockup"),
            dividend_df=stock_aux.get("dividend"),
            board_df=stock_aux.get("board"),
            sector_df=stock_aux.get("sector"),
            concept_df=stock_aux.get("concept"),
            skip_temporal=True,  # xLSTM learns temporal patterns natively
        )
        # Calendar features are normally added by the temporal path;
        # we still want them when skip_temporal=True (panel model benefits
        # from day-of-week/month/quarter signals for seasonality).
        feats = add_calendar_features(feats)
    # Defragment after many df["col"] = ... assignments in merge methods.
    # Without this, pandas emits PerformanceWarning and slows down
    # subsequent operations.
    feats = feats.copy()
    return feats


def _zi_align_df(
    df: pd.DataFrame, all_cols: set,
) -> pd.DataFrame:
    """ZI-align a single stock's feature frame to the union column set.

    Mirrors the dense-path ZI-alignment block (lines 423-444) — same
    column-fill rules: ``has_*`` → False, ``*_count``/``*_streak`` → int16 0,
    else float32 0.  Returns *df* with missing columns added (mutated in
    place for the original frame reference).
    """
    missing = all_cols - set(df.columns)
    if not missing:
        return df
    fill_data: dict[str, np.ndarray] = {}
    n = len(df)
    for col in missing:
        if col == "date":
            continue
        elif col.startswith("has_"):
            fill_data[col] = np.full(n, False)
        elif col.endswith("_count") or col.endswith("_streak"):
            fill_data[col] = np.zeros(n, dtype=np.int16)
        else:
            fill_data[col] = np.zeros(n, dtype=np.float32)
    if fill_data:
        fill_df = pd.DataFrame(fill_data, index=df.index)
        # NOTE: pd.concat returns a new DataFrame; the caller must rebind.
        return pd.concat([df, fill_df], axis=1)
    return df


def _build_panel_streaming(
    pipeline,
    target_col: str,
    horizon: int,
    prebuilt_dir: str | None,
    panel: pd.DataFrame,
    aux_data: dict,
    data_dir: str | None,
    daily_membership: pd.DataFrame | None,
    memmap_dir: str,
    min_history: int,
    codes: list[str],
    input_stocks: int,
    drop_reasons: Counter,
    drop_examples: dict[str, list[str]],
    scratch_dir: str | None = None,
    run_id: str | None = None,
    scratch_stale_days: int = _DEFAULT_SCRATCH_STALE_DAYS,
    scratch_cleanup_root: str | None = None,
    scratch_cleanup_prefix: str | None = None,
) -> dict:
    """Streaming / two-pass panel build (§T5).

    Eliminates the ``all_feat_dfs`` full-residence list: each stock's
    engineered feature frame is written to a scratch pickle in Pass 1,
    re-read per-pass, and the scratch directory is cleaned in ``finally``.
    The only bounded structure is the per-date normalizer-stats accumulator
    and the (optional) cross-sectional-fundamental panel (~9 cols x total
    rows).

    §T7 scratch management (the audit §四/§十五 remediation):
      * ``scratch_dir`` — where Pass-1 pickles land.  A given dir is reused
        (created if missing); ``None`` falls back to a fresh OS-temp dir
        (legacy behavior).
      * ``run_id`` — the run identity recorded in ``run_manifest.json``.
        ``None`` derives a fresh one (``_scratch_run_id``) UNLESS the dir
        already holds a manifest (a crashed run), in which case the crashed
        run's id is ADOPTED and Pass 1 RESUMES by skipping per-stock pickles
        that already exist (crash recovery).
      * ``scratch_stale_days`` + ``scratch_cleanup_root``/``prefix`` — a
        startup sweep removes orphan scratch dirs older than the stale window
        (never the current dir).  Callers pass ``None`` for the root to skip
        the sweep (an explicit ``--scratch-dir`` location is never swept).
      * An EXACT disk backstop runs after Pass 1 (once N/T/D and the actual
        scratch bytes are known) and refuses before the memmap grids are
        allocated if the scratch drive cannot hold the final footprint +
        margin — the estimate-based pre-build check (train_panel_panel) is the
        egregious-case guard; this backstop cannot underestimate.
    """
    if scratch_dir is None:
        scratch = tempfile.mkdtemp(prefix="panel_stream_scratch_")
    else:
        scratch = scratch_dir
        os.makedirs(scratch, exist_ok=True)

    # §T7 startup sweep of orphan scratch dirs (a hard-killed build's
    # leftovers).  Excludes the current dir; explicit --scratch-dir callers
    # pass no cleanup root, so a user-chosen location is never swept.
    if scratch_cleanup_root is not None:
        _cleanup_stale_scratch_dirs(
            scratch_cleanup_root, scratch_stale_days,
            prefix=scratch_cleanup_prefix, exclude=scratch,
        )

    # §T7 run manifest + crash-resume detection.  A scratch dir that already
    # carries run_manifest.json (+ surviving pickles) is a crashed run: the
    # re-run ADOPTS its run_id and Pass 1 skips stocks whose pickle already
    # exists.  A fresh dir generates a new run_id.
    #
    # §T7 review: resume is gated on TWO conditions.
    #   * The manifest's feature-switch fingerprint must MATCH the current run
    #     (else resuming would skip old-schema pickles and engineer new-schema
    #     ones into a silent hybrid panel) — REFUSE, don't guess.
    #   * The prior run must NOT have completed (stage != "done") — a preserved
    #     COMPLETED run is not a crash to recover: resuming it would be a silent
    #     no-op (every pickle already present).  Its run_id is not adopted; a
    #     fresh full rebuild runs with a warning.
    prev_manifest = _read_run_manifest(scratch)
    feature_hash = _feature_switches_hash(pipeline)
    resume = False
    if prev_manifest and prev_manifest.get("run_id"):
        prev_hash = prev_manifest.get("feature_switches_hash")
        if prev_hash and prev_hash != feature_hash:
            raise RuntimeError(
                "streaming panel build: scratch dir %s belongs to a previous "
                "run with DIFFERENT feature switches (recorded %s vs current "
                "%s) — resuming would mix two feature schemas into a hybrid "
                "panel (§T7).  Point --scratch-dir at a fresh dir, or delete "
                "the old pickles for a full rebuild." % (
                    scratch, prev_hash, feature_hash))
        if prev_manifest.get("stage") == "done":
            logger.warning(
                "scratch %s already holds a COMPLETED run (run_id=%s, "
                "stage=done) — not resuming; starting a fresh run (all "
                "pickles re-engineered).  Point --scratch-dir at a fresh dir "
                "to keep the completed run's artifacts.",
                scratch, prev_manifest.get("run_id"))
        else:
            resume = True
    if resume:
        run_id = prev_manifest["run_id"]
    elif run_id is None:
        run_id = _scratch_run_id()
    _write_run_manifest(scratch, run_id, stage="running", resumed=resume,
                        feature_switches_hash=feature_hash)

    try:
        # ── Pass 1: engineer → disk + metadata ──────────────────────────
        valid_codes: list[str] = []
        all_cols: set = set()
        all_dates: set = set()
        has_sector_code = False
        N_stocks = 0

        for code in codes:
            pkl_path = os.path.join(scratch, f"{code}.pkl")
            feats = None
            # §T7 crash-recovery: a same-scratch re-run skips engineering a
            # stock whose pickle already exists (it completed before the
            # crash) and re-reads it to collect the same Pass-1 metadata.  A
            # corrupt/partial pickle (a kill during the write) re-engineers.
            if resume and os.path.isfile(pkl_path):
                try:
                    feats = pd.read_pickle(pkl_path)
                except Exception:
                    logger.warning(
                        "resume: unreadable scratch pickle %s — re-engineering "
                        "stock %s", pkl_path, code)
                    feats = None
            if feats is None:
                feats = _engineer_stock(
                    pipeline, code, prebuilt_dir, panel, aux_data, data_dir,
                    drop_reasons, drop_examples,
                )
                if feats is None:
                    continue
                # Serialize the freshly-engineered frame to scratch pickle.
                feats.to_pickle(pkl_path)
            # Collect metadata from the loaded/engineered frame (a resumed stock
            # contributes it from its re-read pickle, without re-engineering).
            valid_codes.append(code)
            all_cols.update(feats.columns)
            sdates = {pd.Timestamp(d).date() for d in feats["date"]}
            all_dates.update(sdates)
            if not has_sector_code and "sector_code" in feats.columns:
                has_sector_code = True
            N_stocks += 1

        if not valid_codes:
            raise ValueError(
                f"build_panel_features: every input stock was dropped — "
                f"{input_stocks} input stock(s), {len(valid_codes)} survived "
                f"cleaning.  drop_reason_counts={dict(drop_reasons)}; drop "
                f"examples (first reason → codes): "
                f"{dict(list(drop_examples.items())[:6])}.  Check the "
                f"prebuilt dir / panel data / calendar alignment before "
                f"training."
            )

        # ── Global date axis (exact same code as dense path) ────────────
        all_dates_sorted = sorted(all_dates)
        if all_dates_sorted:
            _cal = _get_panel_calendar(data_dir)
            _official = set(_cal.get_trading_days(
                all_dates_sorted[0], all_dates_sorted[-1]))
            _off = [d.strftime("%Y-%m-%d") for d in all_dates_sorted
                    if d not in _official]
            if _off:
                raise ValueError(
                    "panel union axis contains dates that are not in the "
                    f"official a_shares trading calendar: "
                    f"{_off[:10]}{' ...' if len(_off) > 10 else ''}")
        max_T = len(all_dates_sorted)
        global_dates = np.array(
            [pd.Timestamp(d) for d in all_dates_sorted],
            dtype="datetime64[ns]",
        )
        date_to_pos = {str(d): i for i, d in enumerate(all_dates_sorted)}

        if max_T < pipeline.seq_len + 5:
            raise ValueError(
                f"Max timesteps ({max_T}) must be > seq_len+5 "
                f"({pipeline.seq_len + 5})"
            )

        # ── Arrays (memmap-backed grids) ────────────────────────────────
        arrays = PanelArrays(N_stocks, max_T, sink_dir=memmap_dir)

        # ── Pass 2cs: cross-sectional fundamental (if applicable) ───────
        cs_fund_cols = ["date", "stock_code", "sector_code",
                        "pe_ttm", "pb_mrq", "ps_ttm", "debt_ratio",
                        "pe_percentile_252d", "pb_percentile_252d"]
        new_cs_cols: list[str] = []
        # Cross-sectional-fundamental panel.  OPTIONAL and OFF by default: it
        # is built only when `pipeline._fundamental_refiner is not None` AND
        # the frames carry `sector_code` (train_panel_*.py enables it via
        # use_fundamental_refine).  It stays resident through Pass 3 because
        # every stock's frame is left-merged against it, so it is the ONE
        # bounded-in-size exception to the streaming residency rule.  Footprint
        # at full-market scale: ~5530 stocks x ~5000 dates = ~27.7M rows x
        # ~14 cols (9 source + ~5 add_cross_sectional), ~2-3 GB float64-
        # dominated — large but fixed, NOT per-pass and NOT a list of full
        # feature frames.
        cs_panel_df: pd.DataFrame | None = None
        if (pipeline._fundamental_refiner is not None
                and has_sector_code):
            panel_parts: list[pd.DataFrame] = []
            for i, code in enumerate(valid_codes):
                pkl_path = os.path.join(scratch, f"{code}.pkl")
                df = pd.read_pickle(pkl_path)
                if len(df) == 0:
                    continue
                avail = [c for c in cs_fund_cols if c in df.columns]
                if "sector_code" not in avail:
                    continue
                part = df[avail].copy()
                panel_parts.append(part)
            if panel_parts:
                cs_panel = pd.concat(panel_parts, ignore_index=True)
                cs_panel = FundamentalRefiner.add_cross_sectional(cs_panel)
                new_cs_cols = [c for c in cs_panel.columns
                               if c not in set(cs_fund_cols)]
                if new_cs_cols:
                    cs_panel_df = cs_panel  # keep for Pass 3 merges

        # ── Pass 2d: column discovery (first artifact) ──────────────────
        first_code = valid_codes[0]
        first_path = os.path.join(scratch, f"{first_code}.pkl")
        first_df = pd.read_pickle(first_path)
        # ZI-align the first frame.
        first_df = _zi_align_df(first_df, all_cols)
        # Merge cs cols into the first frame for discovery.
        if cs_panel_df is not None and new_cs_cols:
            stock_cs = cs_panel_df[
                cs_panel_df["stock_code"] == first_code
            ]
            if not stock_cs.empty:
                merge_df = stock_cs[["date"] + new_cs_cols].copy()
                first_df = first_df.merge(merge_df, on="date", how="left")
                for col in new_cs_cols:
                    if col not in first_df.columns:
                        first_df[col] = np.float32(0.0)
                    else:
                        first_df[col] = (
                            first_df[col].fillna(0.0).astype(np.float32)
                        )
            all_cols.update(new_cs_cols)

        # §T7 exact disk backstop: after Pass 1 + the cs discovery the
        # footprint is KNOWN (N_stocks x max_T x len(all_cols) float32 grids
        # + the actual scratch pickle bytes).  Refuse BEFORE the memmap grids
        # are allocated and the heavy passes run — the estimate-based
        # pre-check (train_panel_panel) guards the egregious cases up front;
        # this is the hard net that cannot underestimate, so a build never
        # dies mid-way on ENOSPC.
        if max_T and N_stocks:
            scratch_bytes = sum(
                os.path.getsize(os.path.join(scratch, f"{c}.pkl"))
                for c in valid_codes
            )
            grid_bytes = N_stocks * max_T * len(all_cols) * 4
            margin = _DEFAULT_SCRATCH_SAFETY_MARGIN_GB * (1024 ** 3)
            # §v18-6: per-filesystem — final grids on the memmap/panel_store
            # volume, scratch pickles on the scratch volume; combined when same.
            try:
                same_fs = os.stat(scratch).st_dev == os.stat(memmap_dir).st_dev
            except OSError:
                same_fs = True
            if same_fs:
                need = grid_bytes + scratch_bytes + margin
                free = shutil.disk_usage(scratch).free
                if need > free:
                    raise RuntimeError(
                        "streaming panel build: exact disk footprint "
                        f"{(grid_bytes + scratch_bytes) / (1024 ** 3):.1f} GB "
                        f"(final grids {N_stocks} x {max_T} x {len(all_cols)} x "
                        f"4B = {grid_bytes / (1024 ** 3):.1f} GB + scratch "
                        f"pickles {scratch_bytes / (1024 ** 3):.1f} GB) + safety "
                        f"margin {_DEFAULT_SCRATCH_SAFETY_MARGIN_GB:.0f} GB "
                        f"exceeds the scratch drive's free space "
                        f"{free / (1024 ** 3):.1f} GB at {scratch} (§v18-6).  Free "
                        f"disk space or point --scratch-dir at a larger drive."
                    )
            else:
                scratch_free = shutil.disk_usage(scratch).free
                panel_free = shutil.disk_usage(memmap_dir).free
                # Margin is applied PER VOLUME deliberately: the scratch pickles
                # persist on the scratch volume while the grids are written on
                # the panel volume, so reserving margin on both is conservative
                # — do NOT "simplify" into a single shared margin.
                problems = []
                if scratch_bytes + margin > scratch_free:
                    problems.append(
                        f"scratch pickles {scratch_bytes/(1024**3):.1f} GB + "
                        f"margin {margin/(1024**3):.1f} GB exceeds free "
                        f"{scratch_free/(1024**3):.1f} GB on the scratch volume "
                        f"({scratch})")
                if grid_bytes + margin > panel_free:
                    problems.append(
                        f"final grids {grid_bytes/(1024**3):.1f} GB + margin "
                        f"{margin/(1024**3):.1f} GB exceeds free "
                        f"{panel_free/(1024**3):.1f} GB on the panel volume "
                        f"({memmap_dir})")
                if problems:
                    raise RuntimeError(
                        "streaming panel build: exact disk footprint does not "
                        "fit (§v18-6): " + "; ".join(problems)
                        + f" (N_stocks={N_stocks} x max_T={max_T} x "
                        f"len(cols)={len(all_cols)})")

        # Discover PK / PO / static columns.
        static_cols_available = list(_PIT_STATIC_COLS)
        pk_cols_available = pipeline._discover_pk_columns(first_df)
        pk_set = set(pk_cols_available)
        po_cols_available = pipeline._discover_po_columns(first_df, pk_set)

        # norm_cols — same as the dense path.
        from stoke_ml.features.panel_helpers import _CS_NORM_SKIP_COLS
        norm_cols = [c for c in pk_cols_available + po_cols_available
                     if c not in _CS_NORM_SKIP_COLS]

        # ── Pass 2stats: streaming per-date cross-section stats (§T5) ────
        # Accumulate per-date (count/sum/sumsq) aggregates one stock frame
        # at a time instead of concat-ing all light frames — the old
        # _stats_frames list (~23-46GB at full-market scale) is eliminated,
        # keeping peak memory bounded (tracemalloc-verified sublinear growth).
        # Each chunk is ZI-aligned to all_cols FIRST so a norm column absent
        # from a stock contributes 0 exactly like the dense path's ZI-fill
        # (accumulate_stats_chunk skips columns that are missing from the
        # chunk, so the frame must already carry every norm_col).  Streaming
        # float64 accumulation vs the dense pandas groupby over the concat
        # frame shifts float summation order, so the z-scored grids differ at
        # ULP level — a CONTROLLED diff (§T5) asserted with
        # rtol=1e-5/atol=1e-6 in tests/features/test_panel_builders.py for
        # past_known/past_observed ONLY.
        normalizer = DateWiseZScoreNormalizer(daily_membership)
        normalizer.init_stats_accumulator()
        for code in valid_codes:
            pkl_path = os.path.join(scratch, f"{code}.pkl")
            df = pd.read_pickle(pkl_path)
            df = _zi_align_df(df, all_cols)
            normalizer.accumulate_stats_chunk(df, norm_cols)
            del df
        date_stats = normalizer.finalize_date_stats(norm_cols, all_dates)

        # ── Pass 3: targets + ZI-align + cs merge + z-score + scatter ──
        target_builder = TargetBuilder(horizon, target_col)
        static_builder = StaticContextBuilder()
        # Pre-size stock_pos so compute_stock can assign by index.
        arrays.stock_pos = [
            np.empty(0, dtype=np.int32) for _ in range(N_stocks)
        ]

        for i, code in enumerate(valid_codes):
            pkl_path = os.path.join(scratch, f"{code}.pkl")
            df = pd.read_pickle(pkl_path)

            # 3a. Targets from RAW close (before any mutation).
            target_builder.compute_stock(
                df, i, code, max_T, date_to_pos, arrays,
            )

            # 3b. ZI-align columns.
            df = _zi_align_df(df, all_cols)

            # 3c. Merge cross-sectional fundamental cols.
            if cs_panel_df is not None and new_cs_cols:
                stock_cs = cs_panel_df[
                    cs_panel_df["stock_code"] == code
                ]
                if not stock_cs.empty:
                    merge_df = stock_cs[["date"] + new_cs_cols].copy()
                    df = df.merge(merge_df, on="date", how="left")
                    for col in new_cs_cols:
                        if col not in df.columns:
                            df[col] = np.float32(0.0)
                        else:
                            df[col] = (
                                df[col].fillna(0.0).astype(np.float32)
                            )

            # 3d. Apply z-score (in-place mutation of norm_cols).
            DateWiseZScoreNormalizer.apply_zscore(
                df, norm_cols, date_stats,
            )

            # 3e. Scatter into feature grids.
            # First stock: allocate grids after column discovery.
            if i == 0:
                arrays.alloc_features(
                    len(static_cols_available),
                    len(pk_cols_available),
                    len(po_cols_available),
                )
            static_builder.build_stock(
                df, i, code,
                static_cols_available, pk_cols_available,
                po_cols_available, arrays,
            )

        # ── Post: finalize ──────────────────────────────────────────────
        # Quantile ranks over the full static grid.
        static_builder.compute_quantile_ranks(
            arrays, static_cols_available,
        )

        # Fill-probability array (same as dense path).
        fill_prob_arr = np.full(max_T, np.nan, dtype=np.float64)
        if max_T > horizon:
            denom = arrays.entry_counts[:-horizon]
            numer = arrays.filled_counts[:-horizon]
            fill_prob_arr[:max_T - horizon] = np.divide(
                numer, denom,
                out=np.full(max_T - horizon, np.nan),
                where=denom > 0,
            )

        # Eligibility masks.
        elig_builder = EligibilityBuilder(pipeline.seq_len, min_history)
        decision_arr, history_arr, universe_eligible_arr = (
            elig_builder.compute(
                arrays.obs, arrays.first_col,
                arrays.amt60_raw, arrays.has_amount,
            )
        )

        # §十八: ENTRY-side fill probability (full [:max_T] grid — the entry
        # alone, no exit-horizon pairing).  Fraction of decision-eligible
        # stocks at each entry column t with a real entry open at t.
        entry_fill_prob_arr = compute_entry_fill_prob(
            decision_arr, arrays.entry,
        )

        # Sanitize + assemble.
        arrays.sanitize()

        # §T7: record a successful completion in run_manifest.json (the scratch
        # is normally removed by the finally below; the "done" stage matters
        # only when the dir survives, e.g. a hard kill right after the build).
        # The fingerprint is re-recorded so a surviving "done" manifest still
        # carries the switch set it was built under.
        _write_run_manifest(scratch, run_id, stage="done", resumed=resume,
                            feature_switches_hash=feature_hash)

        return arrays.assemble(
            global_dates, decision_arr, history_arr,
            universe_eligible_arr, fill_prob_arr, entry_fill_prob_arr,
            pk_cols_available, po_cols_available, valid_codes,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_panel_features(
    pipeline,
    panel: pd.DataFrame,
    target_col: str = "close",
    aux_data: dict[str, dict[str, pd.DataFrame]] | None = None,
    horizon: int = 1,
    prebuilt_dir: str | None = None,
    min_history: int | None = None,
    require_feature_manifest: bool = False,
    data_dir: str | None = None,
    daily_membership: pd.DataFrame | None = None,
    memmap_dir: str | None = None,
    scratch_dir: str | None = None,
    run_id: str | None = None,
    scratch_stale_days: int = _DEFAULT_SCRATCH_STALE_DAYS,
    scratch_cleanup_root: str | None = None,
    scratch_cleanup_prefix: str | None = None,
) -> dict:
    """Build panel-format features for VSN+xLSTM training from a multi-stock panel.

    The input panel must have columns: date, stock_code, open, high, low,
    close, volume (plus any auxiliary feature columns already merged).

    Args:
        panel: multi-stock DataFrame with columns date, stock_code, OHLCV.
        target_col: column name for close price.
        aux_data: optional dict stock_code → {aux_type: DataFrame}.
                  aux_type keys: "sentiment", "guba", "comment",
                  "announcement", "margin", "northbound", "dragon_tiger",
                  "fundamental", "etf_flow", "capital_flow", "block_trade",
                  "shareholder", "lockup", "dividend", "board", "sector", "concept".
        horizon: forward return horizon in days (1/5/20). Direction
                 threshold scales as 0.003 * sqrt(horizon).
        prebuilt_dir: optional dir of panel-mode feature parquets
                  (``save_features(panel_mode=True)``).  When set, per-stock
                  features are loaded from ``{prebuilt_dir}/{code}.parquet``
                  instead of being engineered live; ``aux_data`` is ignored.
                  Parquets must be built with the SAME ``--use-*`` flags, but
                  column SETS may legitimately differ per stock (merge
                  methods skip columns a stock has no data for) — those gaps
                  are reconciled by the all_cols ZI-alignment block below.
        require_feature_manifest: when True and prebuilt_dir is set, FAIL
                  (raise) instead of warning if any sidecar manifest is
                  missing/stale/schema-drifted or built by a different git
                  commit (or no ``.manifests/`` exists at all).  Formal
                  training passes the CLI's ``--require-feature-manifest``
                  (default on); legacy prebuilt dirs / unit tests pass False
                  to keep warn-only behavior.
        data_dir: data root used to fingerprint the source lineage when
                  validating sidecar manifests (``manifest_matches_detailed``
                  hashes the shared inputs + per-stock source files under it).
                  Defaults to the active config's ``project.data_dir``; pass
                  an explicit path in tests so the fake source files and the
                  manifest both resolve under the same temp root.
        daily_membership: optional long-form index-membership frame
                  ``(stock_code, in_date, out_date)`` (as returned by
                  ``stoke_ml.data.universe.load_index_membership``), ALREADY
                  filtered to the run's indices.  When set and non-empty, it
                  restricts the per-date cross-section STATISTICAL SET to only
                  those stocks that are members on that date (half-open
                  ``in_date <= date < out_date``; ``out_date`` NaT = still a
                  member).  Non-member stocks still get z-scored, but they do
                  NOT contribute to the mean/std (§T6 decision 2).  Default
                  None = the EXACT current all-stock behavior.
        memmap_dir: optional directory to sink the three large (N, T, D)
                  feature grids (static_features / past_known / past_observed)
                  directly to disk via ``np.lib.format.open_memmap``, so the
                  full dense grids never reside in RAM (T8, §七-P0).  When None
                  (default) the current all-dense behavior is preserved.  The
                  returned dict carries ``np.memmap`` objects for those keys;
                  the caller must flush + close them before re-writing the same
                  directory (Windows file-lock constraint — see panel_store.py
                  docstring).
        scratch_dir: §T7 optional dir for the STREAMING path's per-stock
                  Pass-1 pickles (only read when ``memmap_dir`` is set).  A
                  given dir is reused (created if missing); ``None`` (default)
                  falls back to a fresh OS-temp dir (legacy behavior).
        run_id: §T7 optional run identity recorded in ``run_manifest.json``.
                  ``None`` derives a fresh one unless the scratch dir already
                  holds a manifest, in which case the crashed run's id is
                  adopted and Pass 1 RESUMES by skipping pickles already on
                  disk (crash recovery).
        scratch_stale_days: §T7 orphan-scratch stale window (days) for the
                  startup sweep (default ``_DEFAULT_SCRATCH_STALE_DAYS``).
        scratch_cleanup_root: §T7 parent dir scanned at startup for stale
                  scratch dirs (None = skip the sweep — an explicit
                  ``--scratch-dir`` is never swept).
        scratch_cleanup_prefix: §T7 when set, the sweep removes only entries
                  with this name prefix (the ``panel_stream_scratch_`` temp
                  fallback).

    Returns:
        dict with numpy arrays: static_features, past_known, past_observed,
        y_direction, y_return, y_volatility.
    """
    codes = sorted(panel["stock_code"].unique())
    aux_data = aux_data or {}
    if min_history is None:
        min_history = pipeline.min_history
    # §十四-1: count WHY stocks drop out so an all-cleaned panel raises a
    # clear error instead of the misleading "Max timesteps (0)".
    input_stocks = len(codes)
    drop_reasons: Counter = Counter()
    drop_examples: dict[str, list[str]] = defaultdict(list)

    if prebuilt_dir:
        # A missing prebuilt parquet would otherwise surface as a bare
        # FileNotFoundError mid-loop (or an empty frame corrupting the
        # panel).  Drop missing stocks up front; fail loudly if the dir
        # holds nothing usable at all.
        prebuilt_paths = {
            c: os.path.join(prebuilt_dir, f"{c}.parquet") for c in codes
        }
        missing = [c for c, p in prebuilt_paths.items() if not os.path.isfile(p)]
        if len(missing) == len(codes):
            raise FileNotFoundError(
                f"No prebuilt feature parquets found in {prebuilt_dir}"
            )
        if missing:
            drop_reasons["prebuilt_missing_parquet"] += len(missing)
            drop_examples["prebuilt_missing_parquet"].extend(missing[:8])
            if require_feature_manifest:
                raise FileNotFoundError(
                    f"prebuilt_dir {prebuilt_dir}: {len(missing)}/{len(codes)} "
                    f"feature parquets missing (first 20: {missing[:20]}). "
                    f"Regenerate with build_features.py before a formal run "
                    f"(--no-require-feature-manifest to override)."
                )
            logger.warning(
                "prebuilt_dir missing %d/%d parquets (first 20: %s); "
                "dropping those stocks from the panel",
                len(missing), len(codes), missing[:20],
            )
            codes = [c for c in codes if c not in set(missing)]

        # Lineage guard: surface prebuilt features that lack a
        # sidecar manifest, or whose manifest no longer matches the file
        # (schema drift) or the current code (built by another git commit).
        # require_feature_manifest makes these FAIL — silently reusing
        # unverified/stale features would corrupt a formal experiment —
        # while warn-only keeps legacy un-manifested dirs trainable.
        manifest_dir = os.path.join(prebuilt_dir, ".manifests")
        missing_manifest: list[str] = []
        stale_manifest: list[str] = []
        stale_reasons: dict[str, list[str]] = {}
        if os.path.isdir(manifest_dir) and os.listdir(manifest_dir):
            commit = cache_manifest.git_head()
            # §十一-3: config.yaml can change under the SAME git commit
            # (or outside git entirely) — compare the recorded config_hash
            # against the current config snapshot too.  None when config
            # cannot load → comparison skipped.
            cfg_hash = cache_manifest.current_config_hash()
            # §六: the full lineage check (cache_manifest.manifest_matches_detailed)
            # — code tree + config + schema + daily range + every shared
            # input + every per-stock source channel — replaces the old
            # hand-rolled 4-field probe.  It is STRICTER than the manual
            # version (the code-tree hash is compared unconditionally, even
            # inside a repo where git_commit matches), which is the point:
            # an uncommitted source edit or a shared-data change must not
            # let a stale feature survive a formal run.
            if data_dir is None:
                try:
                    from stoke_ml.config import load_config as _load_cfg
                    data_dir = _load_cfg().project.data_dir
                except Exception:
                    data_dir = None
            mconfig = _manifest_check_config(pipeline.seq_len, horizon)
            for code in codes:
                mp = os.path.join(manifest_dir, f"{code}.json")
                if not os.path.isfile(mp):
                    missing_manifest.append(code)
                    continue
                ok, reasons = cache_manifest.manifest_matches_detailed(
                    mp, code, mconfig,
                    os.path.join(prebuilt_dir, f"{code}.parquet"),
                    data_dir or "", commit, cfg_hash,
                )
                if not ok:
                    stale_manifest.append(code)
                    stale_reasons[code] = reasons
        else:
            # No .manifests/ at all: every stock is unverifiable, so both
            # the warn path and the require path speak the same language.
            missing_manifest = list(codes)

        if require_feature_manifest and (missing_manifest or stale_manifest):
            reason_counts = Counter(
                r for rs in stale_reasons.values() for r in rs
            )
            raise RuntimeError(
                f"prebuilt_dir {prebuilt_dir}: feature-manifest check FAILED "
                f"({len(missing_manifest)} missing, {len(stale_manifest)} "
                f"stale — lineage mismatch; reason_counts="
                f"{dict(reason_counts)}; first stale (code → reasons): "
                f"{list(stale_reasons.items())[:5]}; first missing: "
                f"{missing_manifest[:5]}). "
                f"Regenerate with build_features.py --panel-mode before a "
                f"formal run (--no-require-feature-manifest to override)."
            )
        if missing_manifest:
            logger.warning(
                "prebuilt_dir %s: %d/%d stocks lack sidecar manifests "
                "(first 10: %s) — regenerate with build_features.py for "
                "verifiable lineage",
                prebuilt_dir, len(missing_manifest), len(codes),
                missing_manifest[:10],
            )
        if stale_manifest:
            reason_counts = Counter(
                r for rs in stale_reasons.values() for r in rs
            )
            logger.warning(
                "prebuilt_dir %s: %d/%d stocks have STALE manifests "
                "(reason_counts=%s; first 10 codes: %s) — rebuild features "
                "before trusting training output",
                prebuilt_dir, len(stale_manifest), len(codes),
                dict(reason_counts), stale_manifest[:10],
            )

    # ── Streaming / two-pass branch (§T5) ──
    if memmap_dir is not None:
        return _build_panel_streaming(
            pipeline, target_col, horizon,
            prebuilt_dir, panel, aux_data, data_dir,
            daily_membership, memmap_dir, min_history,
            codes, input_stocks, drop_reasons, drop_examples,
            scratch_dir=scratch_dir,
            run_id=run_id,
            scratch_stale_days=scratch_stale_days,
            scratch_cleanup_root=scratch_cleanup_root,
            scratch_cleanup_prefix=scratch_cleanup_prefix,
        )

    # ── Dense path (byte-identical) ──
    # Engineer features per stock (reuses existing pipeline)
    all_feat_dfs = []
    # §v12-P0: valid_codes tracks the codes whose features SURVIVED cleaning
    # (a stock with all-invalid dates or an emptied prebuilt parquet drops
    # out of all_feat_dfs).  Array row i MUST map to valid_codes[i], never to
    # the original `codes[i]` — a dropped stock would otherwise mislabel every
    # subsequent row (feature→stock, board one-hot, universe mask, delist day,
    # OOS artifact codes) without any error being raised.
    valid_codes: list[str] = []
    for code in codes:
        feats = _engineer_stock(
            pipeline, code, prebuilt_dir, panel, aux_data, data_dir,
            drop_reasons, drop_examples,
        )
        if feats is None:
            continue
        all_feat_dfs.append(feats)
        valid_codes.append(code)

    # ── Compute targets from RAW close BEFORE cross-sectional normalization ──
    # Cross-sectional z-score normalization mutates close (and all PK/PO
    # columns) to relative-value space.  Targets MUST be computed from raw
    # prices — using z-score changes as returns distorts the signal.
    # ── Global trading-calendar alignment ──
    # Every stock is aligned to the UNION of all stock dates (sorted), so
    # array column t is the SAME calendar date for every stock.  Without
    # this, a short-history stock would start at position 0 and its column
    # t would be a different date than a long-history stock's column t —
    # corrupting cross-sectional IC / Top-K / long-short evaluation (which
    # index by column) and walk-forward fold boundaries.
    all_dates = sorted({d for df in all_feat_dfs for d in pd.to_datetime(df["date"])})
    # §九-1 defensive invariant: the panel time axis MUST be a subset of the
    # official A-share trading calendar.  `_clean_calendar_dates` enforces
    # this per stock on both entry paths and the merge methods only left-join
    # aux data onto the K-line axis, so an off-calendar date surviving to the
    # UNION signals an upstream regression — fail loudly instead of silently
    # widening column t (the global calendar column) for every stock.
    if all_dates:
        _cal = _get_panel_calendar(data_dir)
        _official = set(_cal.get_trading_days(
            all_dates[0].date(), all_dates[-1].date()))
        _off = [d.strftime("%Y-%m-%d") for d in all_dates
                if d.date() not in _official]
        if _off:
            raise ValueError(
                "panel union axis contains dates that are not in the "
                f"official a_shares trading calendar: "
                f"{_off[:10]}{' ...' if len(_off) > 10 else ''}")
    max_T = len(all_dates)
    global_dates = np.array(all_dates, dtype="datetime64[ns]")
    # `all_dates` holds pandas Timestamps (which have .date()); iterating
    # the numpy global_dates array would yield datetime64 scalars instead.
    date_to_pos = {str(d.date()): i for i, d in enumerate(all_dates)}

    N_stocks = len(all_feat_dfs)

    # ── Targets & masks (per-stock labels / PIT-static raw inputs) ──
    arrays = PanelArrays(N_stocks, max_T, sink_dir=memmap_dir)
    target_builder = TargetBuilder(horizon, target_col)
    target_builder.compute(all_feat_dfs, valid_codes, max_T, date_to_pos, arrays)

    # §T13: per-date exit-fill probability — the fraction of stocks
    # entry-eligible at column t (open_valid[t]) that ALSO have a real exit
    # open at open[t+horizon].  NaN where no stock is entry-eligible at t, and
    # NaN for the tail columns (t+horizon >= max_T) where no exit window
    # exists.  Records the residual fill rate now that carried exits enter the
    # return label (see §十四-4 note above).
    fill_prob_arr = np.full(max_T, np.nan, dtype=np.float64)
    if max_T > horizon:
        denom = arrays.entry_counts[:-horizon]
        numer = arrays.filled_counts[:-horizon]
        fill_prob_arr[:max_T - horizon] = np.divide(
            numer, denom,
            out=np.full(max_T - horizon, np.nan),
            where=denom > 0,
        )

    # Align columns across all stocks — sparse data types (dragon_tiger,
    # block_trade, lockup, etc.) may have data for some stocks but not
    # others, producing different column sets. Missing columns get ZI fill.
    if all_feat_dfs:
        all_cols = set()
        for df in all_feat_dfs:
            all_cols.update(df.columns)
        for i, df in enumerate(all_feat_dfs):
            missing = all_cols - set(df.columns)
            if not missing:
                continue
            fill_data: dict[str, np.ndarray] = {}
            n = len(df)
            for col in missing:
                if col == "date":
                    continue
                elif col.startswith("has_"):
                    fill_data[col] = np.full(n, False)
                elif col.endswith("_count") or col.endswith("_streak"):
                    fill_data[col] = np.zeros(n, dtype=np.int16)
                else:
                    fill_data[col] = np.zeros(n, dtype=np.float32)
            if fill_data:
                fill_df = pd.DataFrame(fill_data, index=df.index)
                all_feat_dfs[i] = pd.concat([df, fill_df], axis=1)

    # ── Cross-sectional fundamental features ──
    # Sector-relative valuation, leverage warning, composite cheapness.
    # Computed on the full multi-stock panel so sector medians are meaningful.
    cs_fund_cols = ["date", "stock_code", "sector_code",
                    "pe_ttm", "pb_mrq", "ps_ttm", "debt_ratio",
                    "pe_percentile_252d", "pb_percentile_252d"]
    if (pipeline._fundamental_refiner is not None
            and any("sector_code" in df.columns for df in all_feat_dfs)):
        panel_parts: list[pd.DataFrame] = []
        for i, df in enumerate(all_feat_dfs):
            if len(df) == 0:
                continue
            avail = [c for c in cs_fund_cols if c in df.columns]
            if "sector_code" not in avail:
                continue
            part = df[avail].copy()
            panel_parts.append(part)
        if panel_parts:
            cs_panel = pd.concat(panel_parts, ignore_index=True)
            cs_panel = FundamentalRefiner.add_cross_sectional(cs_panel)
            new_cs_cols = [c for c in cs_panel.columns
                           if c not in set(cs_fund_cols)]
            if new_cs_cols:
                for i, df in enumerate(all_feat_dfs):
                    if len(df) == 0 or "sector_code" not in df.columns:
                        continue
                    # §v12-P0: index into valid_codes, not codes — a stock
                    # dropped during cleaning shifts all_feat_dfs vs codes.
                    stock_code = valid_codes[i]
                    stock_cs = cs_panel[cs_panel["stock_code"] == stock_code]
                    if stock_cs.empty:
                        continue
                    merge_df = stock_cs[["date"] + new_cs_cols].copy()
                    df = df.merge(merge_df, on="date", how="left")
                    for col in new_cs_cols:
                        if col not in df.columns:
                            df[col] = np.float32(0.0)
                        else:
                            df[col] = df[col].fillna(0.0).astype(np.float32)
                    all_feat_dfs[i] = df

    # §十四-1: ALL input stocks cleaned out — raise with drop stats instead
    # of the misleading "Max timesteps (0)" (max_T collapses to 0 when
    # all_feat_dfs is empty).  seq_len validation still runs below for any
    # non-empty panel.
    if not all_feat_dfs:
        raise ValueError(
            f"build_panel_features: every input stock was dropped — "
            f"{input_stocks} input stock(s), {len(valid_codes)} survived "
            f"cleaning.  drop_reason_counts={dict(drop_reasons)}; drop "
            f"examples (first reason → codes): "
            f"{dict(list(drop_examples.items())[:6])}.  Check the prebuilt "
            f"dir / panel data / calendar alignment before training."
        )

    if max_T < pipeline.seq_len + 5:
        raise ValueError(
            f"Max timesteps ({max_T}) must be > seq_len+5 ({pipeline.seq_len + 5})"
        )

    # ── Dynamic column discovery (replaces hardcoded _PAST_KNOWN/OBSERVED_COLS) ──
    first_df = all_feat_dfs[0]
    # PIT static columns: time-varying per-window
    # context derived from data available at each decision day.  Replaces the
    # leaky first-20-days permanent quantiles.  All are derivable from OHLCV
    # + date + stock code — see _PIT_STATIC_COLS.
    static_cols_available = list(_PIT_STATIC_COLS)
    pk_cols_available = pipeline._discover_pk_columns(first_df)
    pk_set = set(pk_cols_available)
    po_cols_available = pipeline._discover_po_columns(first_df, pk_set)
    # Dead-feature drop is deliberately NOT applied here: a column's
    # constancy over the FULL history (including future periods) must never
    # decide an earlier fold's feature set.  All engineered columns stay in
    # the grids; train_panel drops dead columns per-fold using only its own
    # training window (fold_dead_feature_columns).

    # ── Cross-section z-norm (per-date z-score + member-set restriction) ──
    normalizer = DateWiseZScoreNormalizer(daily_membership)
    norm_cols, date_stats = normalizer.normalize(
        all_feat_dfs, pk_cols_available, po_cols_available,
    )

    # ── Feature grids (static / pk / po) + quantile ranks ──
    arrays.alloc_features(
        len(static_cols_available), len(pk_cols_available), len(po_cols_available),
    )
    static_builder = StaticContextBuilder()
    static_builder.build(
        all_feat_dfs, valid_codes,
        static_cols_available, pk_cols_available, po_cols_available,
        arrays,
    )

    # ── Eligibility masks (decision / history / universe) ──
    elig_builder = EligibilityBuilder(pipeline.seq_len, min_history)
    decision_arr, history_arr, universe_eligible_arr = elig_builder.compute(
        arrays.obs, arrays.first_col, arrays.amt60_raw, arrays.has_amount,
    )

    # §十八: ENTRY-side fill probability (full [:max_T] grid — the entry
    # alone, no exit-horizon pairing).  Fraction of decision-eligible
    # stocks at each entry column t with a real entry open at t.
    entry_fill_prob_arr = compute_entry_fill_prob(
        decision_arr, arrays.entry,
    )

    # ── Sanitize + final assembly ──
    arrays.sanitize()

    return arrays.assemble(
        global_dates, decision_arr, history_arr, universe_eligible_arr,
        fill_prob_arr, entry_fill_prob_arr,
        pk_cols_available, po_cols_available, valid_codes,
    )
