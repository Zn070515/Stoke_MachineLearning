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
import logging
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from stoke_ml.config.feature_profile import CHANNEL_COLUMNS
from stoke_ml.features import cache_manifest
from stoke_ml.features.fundamental import FundamentalRefiner
from stoke_ml.features.panel_builders._arrays import PanelArrays
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
            # §T7/§十四: generic per-channel scrub.  build_features.py
            # --panel-mode bakes ALL channels in with all-True defaults, so a
            # restricted run (safe-only vintage / ablation) would otherwise
            # silently consume channels its pipeline does not request.  Drop
            # the EXACT CHANNEL_COLUMNS set of every channel whose use_* switch
            # is OFF (map channel → switch attr; "announcement" is the one
            # special-cased name).  Only the exact sets are used — NO
            # name-prefix matching, which is exactly the market_env-vs-macd /
            # market_env_refine collision trap.  fundamental_refine is coupled
            # to fundamental (pipeline forces it off with fundamental), so a
            # safe-only run drops its columns too.  topic_* is handled
            # separately above (prefix drop, frozen non-PIT topic model); a
            # prebuilt parquet built with use_topic=True is scrubbed there.
            _channel_switch_attr = {"announcement": "use_announcements"}
            _off_cols: list[str] = []
            for _channel, _cols in CHANNEL_COLUMNS.items():
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
            feats = pipeline._clean_calendar_dates(feats, code)
            if feats is None:
                drop_reasons["calendar_clean_dropped"] += 1
                drop_examples["calendar_clean_dropped"].append(code)
                continue
            # Calendar features are idempotent (overwrite in place); safe
            # to re-apply even though save_features(panel_mode=True) already
            # added them — guards against hand-built parquets.
            feats = add_calendar_features(feats)
            # Schema note: parquets must be built with the SAME --use-*
            # flags (build_features.py).  Column SETS still legitimately
            # differ per stock — merge methods skip columns when a stock
            # has no data for a sparse aux type (block_trade, dividend,
            # valuation, ...).  No strict equality check here: those gaps
            # are reconciled by the all_cols ZI-alignment block after the
            # loop, and PK/PO columns are discovered BY NAME (never by
            # position), so a missing column simply becomes all-zero.
        else:
            mask = panel["stock_code"] == code
            df_stock = panel[mask].sort_values("date").reset_index(drop=True)
            # Drop phantom/duplicate/out-of-calendar rows before
            # feature engineering so a bad bar neither pollutes the UNION
            # date axis nor corrupts the rolling indicators around it.
            df_stock = pipeline._clean_calendar_dates(df_stock, code)
            if df_stock is None:
                drop_reasons["calendar_clean_dropped"] += 1
                drop_examples["calendar_clean_dropped"].append(code)
                continue
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
        _cal = _get_panel_calendar()
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

    # ── Sanitize + final assembly ──
    arrays.sanitize()

    return arrays.assemble(
        global_dates, decision_arr, history_arr, universe_eligible_arr,
        fill_prob_arr, pk_cols_available, po_cols_available, valid_codes,
    )
