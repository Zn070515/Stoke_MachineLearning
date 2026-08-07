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
import shutil
import tempfile
from collections import Counter, defaultdict

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
    from stoke_ml.config.feature_profile import (
        CHANNEL_COLUMNS,
        market_env_account_is_verified,
    )

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
) -> dict:
    """Streaming / two-pass panel build (§T5).

    Eliminates the ``all_feat_dfs`` full-residence list: each stock's
    engineered feature frame is written to a scratch pickle in Pass 1,
    re-read per-pass, and the scratch directory is cleaned in ``finally``.
    The only bounded structure is the per-date normalizer-stats accumulator
    and the (optional) cross-sectional-fundamental panel (~9 cols x total
    rows).
    """
    scratch = tempfile.mkdtemp(prefix="panel_stream_scratch_")
    try:
        # ── Pass 1: engineer → disk + metadata ──────────────────────────
        valid_codes: list[str] = []
        all_cols: set = set()
        all_dates: set = set()
        has_sector_code = False
        N_stocks = 0

        for code in codes:
            feats = _engineer_stock(
                pipeline, code, prebuilt_dir, panel, aux_data, data_dir,
                drop_reasons, drop_examples,
            )
            if feats is None:
                continue
            # Collect metadata BEFORE writing to disk.
            valid_codes.append(code)
            all_cols.update(feats.columns)
            sdates = {pd.Timestamp(d).date() for d in feats["date"]}
            all_dates.update(sdates)
            if not has_sector_code and "sector_code" in feats.columns:
                has_sector_code = True
            # Serialize to scratch pickle.
            pkl_path = os.path.join(scratch, f"{code}.pkl")
            feats.to_pickle(pkl_path)
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

    # ── Streaming / two-pass branch (§T5) ──
    if memmap_dir is not None:
        return _build_panel_streaming(
            pipeline, target_col, horizon,
            prebuilt_dir, panel, aux_data, data_dir,
            daily_membership, memmap_dir, min_history,
            codes, input_stocks, drop_reasons, drop_examples,
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
