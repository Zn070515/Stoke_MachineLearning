"""Panel-format feature construction for VSN+xLSTM training.

``build_panel_features`` builds the panel-format arrays (static / past-known /
past-observed grids plus the direction/return/volatility target masks) from a
multi-stock panel.  Extracted from ``FeaturePipeline.build_panel_features``
(§二十一); it operates on a ``FeaturePipeline`` instance passed as the first
argument so the public method keeps delegating through it.  Leaf-safe: imports
nothing from ``stoke_ml.features.pipeline`` (only the leaf ``panel_helpers``
plus data-layer lazy imports), so ``pipeline`` can import this module without
an import cycle.
"""
import logging
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from stoke_ml.config.feature_profile import CHANNEL_COLUMNS
from stoke_ml.data.codes import normalize_stock_code_series
from stoke_ml.features import cache_manifest
from stoke_ml.features.fundamental import FundamentalRefiner
from stoke_ml.features.panel_helpers import (
    _min_vol_nobs,
    _PIT_STATIC_COLS,
    _trailing_mean,
    _not_long_suspended,
    _manifest_check_config,
    _get_panel_calendar,
    _CS_NORM_SKIP_COLS,
    _BOARD_ONEHOT_COLS,
    _board_index,
)
from stoke_ml.features.temporal import add_calendar_features

logger = logging.getLogger(__name__)


def _daily_member_flag(
    all_feat: pd.DataFrame, membership: pd.DataFrame,
) -> pd.Series:
    """Row-level per-stock index-membership flag for each row's date.

    ``all_feat`` carries ``date`` + ``stock_code``; ``membership`` is the
    long-form ``(stock_code, in_date, out_date)`` frame (already filtered to the
    run's indices).  A row is a member iff ``in_date <= date < out_date``
    (half-open; ``out_date`` NaT = still a member).  Returns a bool Series
    aligned to ``all_feat``'s index.

    Vectorized: row positions are grouped by code ONCE via a hash (O(rows)),
    then each member code's rows get a sorted-interval lookup via
    ``numpy.searchsorted`` after a per-code interval merge — NOT an
    O(rows x intervals) loop.  The full market panel is ~33M rows.  The merge
    keeps "any covering interval" exact even when a stock's intervals overlap
    across the indices of a csi800 universe (000300 + 000905 windows can
    overlap with different out_dates).
    """
    n = len(all_feat)
    is_member = np.zeros(n, dtype=bool)
    if membership is None or membership.empty or n == 0:
        return pd.Series(is_member, index=all_feat.index)
    mem = pd.DataFrame({
        "code": normalize_stock_code_series(membership["stock_code"]),
        "in": pd.to_datetime(membership["in_date"], errors="coerce"),
        "out": pd.to_datetime(membership["out_date"], errors="coerce"),
    })
    mem = mem.dropna(subset=["code", "in"])
    if mem.empty:
        return pd.Series(is_member, index=all_feat.index)
    row_codes = normalize_stock_code_series(all_feat["stock_code"])
    pos_by_code = row_codes.groupby(row_codes).indices
    row_dates = pd.to_datetime(all_feat["date"]).to_numpy(dtype="datetime64[ns]")
    for code, sub in mem.groupby("code", sort=True):
        rows = pos_by_code.get(code)
        if rows is None or rows.size == 0:
            continue
        in_i = sub["in"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        out_i = np.where(
            np.isnat(sub["out"].to_numpy(dtype="datetime64[ns]")),
            np.iinfo(np.int64).max,
            sub["out"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        )
        # Merge overlapping intervals so searchsorted on the last-starting
        # interval answers "any interval covers" exactly.
        order = np.argsort(in_i, kind="mergesort")
        starts: list[int] = []
        ends: list[int] = []
        cur_in = int(in_i[order[0]])
        cur_out = int(out_i[order[0]])
        for j in order[1:]:
            a, b = int(in_i[j]), int(out_i[j])
            if a <= cur_out:
                cur_out = max(cur_out, b)
            else:
                starts.append(cur_in)
                ends.append(cur_out)
                cur_in, cur_out = a, b
        starts.append(cur_in)
        ends.append(cur_out)
        s_arr = np.asarray(starts, dtype=np.int64)
        e_arr = np.asarray(ends, dtype=np.int64)
        rd = row_dates[rows].astype(np.int64)
        pos = np.searchsorted(s_arr, rd, side="right") - 1
        good = pos >= 0
        covered = np.zeros(rows.size, dtype=bool)
        covered[good] = rd[good] < e_arr[np.clip(pos[good], 0, e_arr.size - 1)]
        is_member[rows] |= covered
    return pd.Series(is_member, index=all_feat.index)


def _cross_section_stats(feat: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-date cross-sectional ``["mean", "std", "count"]`` for one column.

    ``feat`` is the frame restricted to the desired statistical set (all
    stocks, or a membership subset); ``col`` must be a column of ``feat``.
    Sparse dates fall back to expanding moments (see below).
    """
    stats = feat.groupby("date")[col].agg(["mean", "std", "count"])
    stats["std"] = stats["std"].fillna(1.0).clip(lower=1e-8)
    # Dates with very few listed stocks (the 2000-2015 backfill has
    # 1-5 stocks/day) give a degenerate cross-section: std→0 inflates
    # z-scores to ±hundreds, which dominates the loss.  Fall back to
    # the pooled global mean/std for those sparse dates — the global
    # moments are stable even when the daily cross-section is tiny.
    sparse = stats["count"] < 5
    if sparse.any():
        # Full-panel pooled moments would leak future dates' statistics
        # into early dates' z-scores (the exact bias the per-date
        # cross-section avoids).  Use expanding moments over dates <=
        # the sparse date — strictly point-in-time.  Cumulative sums
        # give O(dates) per column instead of O(dates^2).
        sdf = feat[["date", col]].sort_values("date")
        col_vals = sdf[col].to_numpy(dtype=np.float64)
        # Treat inf as invalid too (np.nanmean/np.nanstd choke on it
        # and would leak NaN through the z-score).
        valid_vals = np.isfinite(col_vals)
        x = np.where(valid_vals, col_vals, 0.0)
        ccount = np.cumsum(valid_vals.astype(np.float64))
        csum = np.cumsum(x)
        csq = np.cumsum(x * x)
        sdates = pd.to_datetime(sdf["date"]).to_numpy(dtype="datetime64[ns]")
        sparse_dates = pd.to_datetime(stats.index[sparse]).to_numpy(dtype="datetime64[ns]")
        pos = np.clip(
            np.searchsorted(sdates, sparse_dates, side="right") - 1,
            0, len(sdates) - 1,
        )
        cnt = np.maximum(ccount[pos], 1.0)
        mean = csum[pos] / cnt
        var = np.maximum(csq[pos] / cnt - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), 1e-8)
        # groupby agg returns float32 columns; the float64 arrays must
        # be cast back or pandas raises LossySetitemError.
        stats.loc[sparse, "mean"] = mean.astype(stats["mean"].dtype)
        stats.loc[sparse, "std"] = std.astype(stats["std"].dtype)
    return stats


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
    y_dir_arr = np.full((N_stocks, max_T), -100, dtype=np.int64)  # CE ignore_index
    y_ret_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
    y_vol_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
    # §十四-3: per-label metadata — count of VALID daily returns inside each
    # forward vol window (t, t+h] (0 for the tail where no window exists).
    # vol_tgt_arr requires this to reach _min_vol_nobs(horizon); callers use
    # it to weight/augment the vol loss with the label's true sample size.
    forward_vol_nobs = np.zeros((N_stocks, max_T), dtype=np.int32)
    stock_T = np.zeros(N_stocks, dtype=np.int32)
    # §T13: per-date exit-fill counts — entry_counts[t] = # stocks open-valid at
    # column t, filled_counts[t] = # of those that ALSO have a real exit open at
    # t+horizon.  The per-date ratio (fill_prob_arr) is computed once after the
    # loop so a study can quantify how much of each date's label set is carried.
    entry_counts = np.zeros(max_T, dtype=np.int64)
    filled_counts = np.zeros(max_T, dtype=np.int64)

    # ── Per-task target masks ──
    # One `y_direction != -100` cannot carry four distinct
    # jobs — "tradable today", "clean label exists", "loss applies here",
    # "portfolio P&L computable".  Split them:
    #   obs_arr        — real close at t (base observation / history count)
    #   entry_arr      — real open at t → can enter a position at open[t]
    #   ret_tgt_arr    — clean or carried forward return available (open[t+h]/open[t]-1, else carry to the last real close in (t, min(t+h, T-1)])
    #   vol_tgt_arr    — vol window (t, t+h] holds >= _min_vol_nobs(horizon)
    #                    valid daily returns (max(1, ceil(horizon/2)), hard
    #                    floor >=2) — see forward_vol_nobs metadata below
    #   realized_arr   — evaluation P&L: clean open return, else carry to the
    #                    last real close in (t, t+h], else 0 (flat) — so a
    #                    stock that suspends/delists AFTER entry still counts
    #                    and the Top-K pool never conditions on the future.
    obs_arr = np.zeros((N_stocks, max_T), dtype=bool)
    entry_arr = np.zeros((N_stocks, max_T), dtype=bool)
    ret_tgt_arr = np.zeros((N_stocks, max_T), dtype=bool)
    vol_tgt_arr = np.zeros((N_stocks, max_T), dtype=bool)
    realized_arr = np.zeros((N_stocks, max_T), dtype=np.float32)
    # Daily price paths for sleeve-account evaluation: the
    # realized_return array is per-ENTRY-day open-to-exit, which cannot
    # reconstruct a true daily mark-to-market series.  Expose the raw
    # close/open paths (NaN outside a stock's trading days) so evaluate.py
    # can build chronological sleeve daily returns + exit_status.
    close_price_arr = np.full((N_stocks, max_T), np.nan, dtype=np.float32)
    open_price_arr = np.full((N_stocks, max_T), np.nan, dtype=np.float32)

    # Row i of a stock's feature df → its column on the global calendar.
    # Computed once here and reused by the feature-array scatter below.
    stock_pos: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(N_stocks)]

    # Raw PIT-static inputs: trailing 60d means of turnover (canonical
    # `amount`), plus first-listed global column — captured HERE because
    # the per-date z-score normalization later mutates the feature dfs.
    # §五: price_60d_q (a qfq absolute price tier) was REMOVED — qfq levels
    # re-anchor with future corporate actions and would leak future
    # behaviour into historical decisions.
    amt60_raw = np.zeros((N_stocks, max_T), dtype=np.float32)
    first_col = np.full(N_stocks, -1, dtype=np.int32)
    # The canonical `amount` (real CNY turnover) is REQUIRED by the formal
    # daily contract — a panel without it raises above (§十一-5).  Every
    # stock here has it, so every stock qualifies for the liquidity floor.
    has_amount_arr = np.ones(N_stocks, dtype=bool)

    # Direction noise threshold — scale by sqrt(horizon)
    # (0.003 per day, 1.0% / 5-day, 1.3% / 20-day)
    dir_threshold = 0.003 * (horizon ** 0.5)

    for i, df in enumerate(all_feat_dfs):
        if len(df) == 0:
            continue
        df_sorted = df.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df_sorted["date"])
        pos = np.array([date_to_pos[str(d.date())] for d in dates], dtype=np.int32)
        stock_pos[i] = pos
        T_i = len(pos)
        stock_T[i] = T_i
        # Trading-time convention: features up to close[t]
        # (window's last column end-1) → signal after close[t] → ENTER at
        # open[end]=open[t+1] → hold h days → EXIT at open[end+h].  Labels
        # are therefore open-to-open; entry eligibility needs a real open.
        close_full = np.full(max_T, np.nan, dtype=np.float64)
        close_full[pos] = df_sorted[target_col].to_numpy(dtype=np.float64)
        open_col = "open" if "open" in df_sorted.columns else target_col
        open_full = np.full(max_T, np.nan, dtype=np.float64)
        open_full[pos] = df_sorted[open_col].to_numpy(dtype=np.float64)
        # Row-level quality = REPAIR/MASK, not stock
        # ejection.  A non-positive price is DATA-MISSING (a dead data row,
        # indistinguishable from a delisting) — mask it
        # like a suspension so it never becomes a training observation or an
        # entry, instead of ejecting the whole stock because of one bad row.
        close_valid = ~np.isnan(close_full) & (close_full > 0)
        open_valid = ~np.isnan(open_full) & (open_full > 0)
        obs_arr[i] = close_valid
        entry_arr[i] = open_valid
        close_price_arr[i] = close_full.astype(np.float32)
        open_price_arr[i] = open_full.astype(np.float32)
        # §T13 fill-probability accumulation — per-date counts of
        # entry-eligible days (open_valid[t]) and of those with a real exit
        # open at t+horizon; the ratio (fill_prob_arr) is computed after the loop.
        entry_counts[np.nonzero(open_valid)[0]] += 1
        if max_T > horizon:
            filled_counts[np.nonzero(
                open_valid[:-horizon] & open_valid[horizon:])[0]] += 1

        # PIT static raw inputs — trailing 60d means over the trading days
        # in each global-calendar window (NaNs from pre-listing/suspension
        # are skipped).  Computed here on the RAW df before z-scoring.
        # (price_60d_q removed §五 — see _PIT_STATIC_COLS.)
        # The formal daily contract requires canonical CNY turnover
        # (`amount`, real 成交额).  volume×qfq-close misstates historical
        # nominal turnover because qfq prices are rescaled while volume is
        # not.  Fail loudly rather than silently substituting a proxy that
        # is not a real turnover measure (§十一-5).
        if "amount" not in df_sorted.columns:
            raise ValueError(
                f"Stock {valid_codes[i]}: daily K-line lacks canonical `amount` — "
                "the formal daily contract requires it (§十一-5); no "
                "volume×close / price fallback."
            )
        has_amount_arr[i] = True
        amt_full = np.full(max_T, np.nan, dtype=np.float64)
        amt_full[pos] = df_sorted["amount"].to_numpy(dtype=np.float64)
        amt60_raw[i] = _trailing_mean(amt_full, 60).astype(np.float32)
        first_col[i] = int(pos[0]) if len(pos) else -1

        # Forward return (training label): clean open[t]->open[t+h] where a real
        # exit open exists, else carry to the LAST real close in (t, t+h]
        # (§T13 decision 3 — aligned with the evaluation realized path below).
        # Positions with no usable exit (no exit open AND no real close in the
        # window) stay NaN → direction -100 / return 0 with ret_tgt_arr False so
        # training ignores them.
        # §十四-4 (ENTRY-FILL SELECTION BIAS — research design choice, not a
        # bug): the OLD ``both`` condition (open_valid[t] AND open_valid[t+h])
        # required a FUTURE entry open at t+h, so a stock that is
        # decision-eligible at t but NOT fillable at the exit horizon
        # (suspended/delisted before t+h) was EXCLUDED from the training label
        # set.  The learned function was therefore "decision on stocks that will
        # stay tradeable for h days" — a subtly easier population than the full
        # decision pool, which evaluation never conditioned on.
        # §T13 decision 3 APPLIES mitigation #1: training now carries
        # non-fillable exits to the last real close in (t, t+h] — EXACTLY the
        # evaluation realized semantics — so a non-fillable pick is learned
        # (rewarded with its carry value) instead of being masked out of the
        # label population.  Label distribution shift vs the old clean-only
        # labels is expected (decision 3).  ``fill_prob_arr`` (per-date fraction
        # of entry-eligible stocks with a real open[t+h], computed after the
        # loop) records the residual per-date fill rate so a study can quantify
        # how much of each date's label set is carried.
        # Mitigations still NOT applied (leave for a controlled study):
        #   2) a return mask returned alongside labels so the loss can
        #      down-weight partial-window / carried exits;
        #   3) an explicit entry-fill head predicting whether open[t+h]
        #      will exist, and conditioning on it at inference.
        ret_fwd = np.full(max_T, np.nan, dtype=np.float32)
        if max_T > horizon:
            both = open_valid[:-horizon] & open_valid[horizon:]
            num = open_full[horizon:][both] - open_full[:-horizon][both]
            ret_fwd[:max_T - horizon][both] = (
                num / (open_full[:-horizon][both] + 1e-8)).astype(np.float32)
        # Carry non-fillable exits: the last real close at-or-before the
        # truncated window end hi = min(t+h, T-1), i.e. the last real close in
        # (t, hi].  Forward-fill the valid-close indices; k > t selects a real
        # close strictly after entry (in-window), k <= t means NO close in the
        # window → no label (NaN).
        last_close_idx = np.maximum.accumulate(
            np.where(close_valid, np.arange(max_T), -1))
        hi = np.minimum(np.arange(max_T) + horizon, max_T - 1)
        k = last_close_idx[hi]
        carry_ok = open_valid & (k > np.arange(max_T))
        carried = np.full(max_T, np.nan, dtype=np.float64)
        carried[carry_ok] = close_full[k[carry_ok]] / open_full[carry_ok] - 1.0
        missing_clean = open_valid & ~np.isfinite(ret_fwd)
        ret_fwd[missing_clean] = carried[missing_clean].astype(np.float32)
        ret_tgt_arr[i] = np.isfinite(ret_fwd)
        valid = ret_tgt_arr[i]
        y_dir_arr[i, valid] = np.where(
            ret_fwd[valid] > dir_threshold, 2,
            np.where(ret_fwd[valid] < -dir_threshold, 0, 1),
        )
        y_ret_arr[i] = np.nan_to_num(ret_fwd, nan=0.0)

        # Realized return for portfolio evaluation — defined for EVERY
        # entry-eligible (open-valid) day so the candidate pool never
        # depends on whether a future label exists:
        #   clean open[t]->open[t+h] where available; else carry to the last
        #   real close in (t, t+h] / open[t] - 1; else 0 (no exit → flat).
        # §T13: ret_fwd now carries non-fillable exits with the SAME value as
        # this path, so realized is just the finite part of ret_fwd, else 0 —
        # guaranteed bit-identical to the training label for carried days.
        realized = np.zeros(max_T, dtype=np.float32)
        finite_ret = open_valid & np.isfinite(ret_fwd)
        realized[finite_ret] = ret_fwd[finite_ret]
        realized_arr[i] = realized

        # FORWARD-looking realized volatility: std of the daily returns
        # realized over the NEXT `horizon` days (return[t+1 : t+horizon+1]),
        # spanning the same forward window as y_return.  The target is
        # strictly positive, matching VolatilityHead's softplus — train_panel
        # must NOT z-score it.  Suspended days get a 0 return and the
        # resumption day records the accumulated close gap, so a "5-day vol"
        # label uses all 5 days instead of silently collapsing to however
        # many days actually traded.  §十四-3: a window with fewer than
        # _min_vol_nobs(horizon) valid closes (max(1, ceil(h/2)), hard floor
        # >=2) sets vol_tgt_arr False so the vol loss never sees a
        # degenerate / non-comparable partial-window label; the raw valid
        # count is recorded in forward_vol_nobs for any window position.
        ret_daily = np.zeros(max_T, dtype=np.float32)
        last_valid = np.maximum.accumulate(
            np.where(close_valid, np.arange(max_T), -1))
        prev_close = np.full(max_T, -1)
        prev_close[1:] = last_valid[:-1]
        ok = close_valid & (prev_close >= 0)
        ret_daily[ok] = (
            close_full[ok] / close_full[prev_close[ok]] - 1.0
        )
        min_nobs = _min_vol_nobs(horizon)
        for t in range(max_T - horizon):
            win = ret_daily[t + 1:t + 1 + horizon]
            nobs = int(close_valid[t + 1:t + 1 + horizon].sum())
            forward_vol_nobs[i, t] = nobs
            if nobs < 2 or nobs < min_nobs:
                continue
            y_vol_arr[i, t] = float(np.std(win))
            vol_tgt_arr[i, t] = True

    # §T13: per-date exit-fill probability — the fraction of stocks
    # entry-eligible at column t (open_valid[t]) that ALSO have a real exit
    # open at open[t+horizon].  NaN where no stock is entry-eligible at t, and
    # NaN for the tail columns (t+horizon >= max_T) where no exit window
    # exists.  Records the residual fill rate now that carried exits enter the
    # return label (see §十四-4 note above).
    fill_prob_arr = np.full(max_T, np.nan, dtype=np.float64)
    if max_T > horizon:
        denom = entry_counts[:-horizon]
        numer = filled_counts[:-horizon]
        fill_prob_arr[:max_T - horizon] = np.divide(
            numer, denom,
            out=np.full(max_T - horizon, np.nan),
            where=denom > 0,
        )

    # ── Decision / history eligibility ──
    # decision_arr[t] = close[t-1] is real, so a signal computed after
    # close[t-1] (features through column t-1) can rank this stock and
    # ENTER at open[t].  Aligned to the ENTRY column t so the candidate
    # pool is decision & entry & history on one grid.
    decision_arr = np.zeros((N_stocks, max_T), dtype=bool)
    if max_T > 1:
        decision_arr[:, 1:] = obs_arr[:, :-1]
    # history_arr[t] = the seq_len input window ending at t-1 (columns
    # [t-seq_len, t-1]) holds >= min_history real observations — excludes
    # freshly-listed stocks whose window is mostly zero padding.
    if min_history <= 0:
        history_arr = np.ones((N_stocks, max_T), dtype=bool)
    else:
        obs_i = obs_arr.astype(np.int32)
        cum = np.concatenate(
            [np.zeros((N_stocks, 1), dtype=np.int32), np.cumsum(obs_i, axis=1)],
            axis=1,
        )
        t_idx = np.arange(max_T)
        lo = np.maximum(t_idx - pipeline.seq_len, 0)
        history_arr = (cum[:, t_idx] - cum[:, lo]) >= min_history

    # ── Research-universe eligibility (§七-3) ──
    # Data-derived PIT gates merged into the decision pool:
    #   已上市  (first_col) + 当日未长期停牌 + 符合研究流动性规则.
    # The 未退市 delist gate and the per-day index-membership gate need the
    # EXTERNAL universe status / membership records, so they are applied
    # per-fold in train_panel and ANDed into this same decision mask there.
    from stoke_ml.config import load_config
    uni_cfg = dict(load_config().get("universe", {}) or {})
    long_susp_thr = int(uni_cfg.get("long_suspension_days", 60))
    susp_lookback = int(uni_cfg.get("suspension_lookback", 60))
    min_amount_60d = float(uni_cfg.get("min_amount_60d", 5_000_000))
    universe_eligible_arr = _not_long_suspended(
        obs_arr, first_col, max_T, long_susp_thr, susp_lookback,
    )
    if min_amount_60d > 0 and has_amount_arr.any():
        # Causal trailing-60d turnover known at close[t-1] → entry day t:
        # shift amt60_raw (mean over [t-59, t]) right by one column.  Only
        # stocks with a canonical `amount` get the floor; the volume×close
        # / price proxies are not a real turnover measure.
        amt_causal = np.zeros_like(amt60_raw, dtype=np.float32)
        if max_T > 1:
            amt_causal[:, 1:] = amt60_raw[:, :-1]
        liquid = np.ones((N_stocks, max_T), dtype=bool)
        liquid[has_amount_arr] = amt_causal[has_amount_arr] >= min_amount_60d
        universe_eligible_arr &= liquid
    decision_arr &= universe_eligible_arr

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

    # ── Per-date cross-sectional z-score normalization ──
    # Normalize each feature across stocks within each date, so that
    # a feature's value is expressed relative to the cross-section that
    # day.  This avoids pooling future dates' statistics into today's
    # normalized value and is the standard panel-finance treatment.
    # §T6 decision 2: when daily_membership is set, the STATISTICAL SET for
    # each date is restricted to that day's index members (half-open
    # in_date <= date < out_date); non-member stocks are still z-scored but do
    # NOT contribute to the mean/std.  None/empty = the EXACT current all-stock
    # behavior (byte-for-byte for the default path).
    norm_cols = [c for c in pk_cols_available + po_cols_available
                 if c not in _CS_NORM_SKIP_COLS]
    membership_active = (
        daily_membership is not None and not daily_membership.empty
    )
    if membership_active:
        all_feat = pd.concat([
            df[["date", "stock_code"] + norm_cols]
            for df in all_feat_dfs
            if len(df) > 0
        ], ignore_index=True)
        is_member = _daily_member_flag(all_feat, daily_membership)
    else:
        all_feat = pd.concat([
            df[["date"] + norm_cols]
            for df in all_feat_dfs
            if len(df) > 0
        ], ignore_index=True)
        is_member = None
    # Strip non-finite BEFORE any cross-sectional statistic.
    # A single inf (e.g. a near-zero divisor in a factor) pollutes the
    # groupby mean/std, corrupting the whole date's z-score before the
    # final nan_to_num silently zeroes it out.
    finite_cols = [c for c in norm_cols if c in all_feat.columns]
    for c in finite_cols:
        vals = all_feat[c]
        # np.isfinite is undefined for bool (numpy 2.x raises TypeError)
        # and meaningless for non-numeric dtypes; a bool state flag
        # (has_ever_observed / is_stale) is always finite by construction.
        if vals.dtype.kind not in "biuf":
            continue
        if not np.isfinite(vals.to_numpy()).all():
            all_feat[c] = vals.replace([np.inf, -np.inf], np.nan)

    date_stats: dict[str, pd.DataFrame] = {}
    # §T6: the member subset (hence the dates missing from it) is
    # column-invariant — hoist the boolean mask and the missing-date set out of
    # the per-column loop so each column does NOT re-run a full-panel groupby
    # over ~33M rows just to find the same zero-member dates.
    if is_member is not None:
        member_feat = all_feat[is_member]
        missing_dates = sorted(set(all_feat["date"]) - set(member_feat["date"]))
    else:
        member_feat = all_feat
        missing_dates = []
    for col in norm_cols:
        if col not in all_feat.columns:
            continue
        stats = _cross_section_stats(member_feat, col)
        date_stats[col] = stats
        if is_member is not None and missing_dates:
            # §T6: a date present in the panel but with ZERO member rows must
            # still receive stats — otherwise the .map below yields NaN and the
            # post-processing nan_to_num zeroes the WHOLE date's features.  Fall
            # back to the all-stock cross-section stats for exactly those dates.
            # Restricting to the missing-date rows makes the groupby tiny AND
            # routes it through the sparse expanding-moments fallback — the
            # all-stock cross-section on those dates can itself be sparse, and
            # without the fallback its degenerate std→0 (clipped to 1e-8) would
            # blow up that date's z-scores.  all_stats carries ONLY the missing
            # dates, so no .loc[missing] filter is needed here.
            missing_feat = all_feat[all_feat["date"].isin(missing_dates)]
            all_stats = _cross_section_stats(missing_feat, col)
            date_stats[col] = pd.concat([stats, all_stats])

    for df in all_feat_dfs:
        for col in norm_cols:
            if col not in df.columns or col not in date_stats:
                continue
            aligned_mean = df["date"].map(date_stats[col]["mean"])
            aligned_std = df["date"].map(date_stats[col]["std"]).clip(lower=1e-8)
            df[col] = (df[col] - aligned_mean) / aligned_std

    static_dim = len(static_cols_available)
    pk_dim = len(pk_cols_available)
    po_dim = len(po_cols_available)

    # Pre-allocate feature arrays aligned to the GLOBAL calendar (column t
    # is the same date for every stock).  Days before listing or during a
    # suspension keep zero features; observation_mask / entry_eligible_mask
    # (returned below) tell training & evaluation which positions are real.
    static_arr = np.zeros((N_stocks, max_T, static_dim), dtype=np.float32)
    pk_arr = np.zeros((N_stocks, max_T, pk_dim), dtype=np.float32)
    po_arr = np.zeros((N_stocks, max_T, po_dim), dtype=np.float32)

    for i, df in enumerate(all_feat_dfs):
        if len(df) == 0:
            continue

        df_sorted = df.sort_values("date").reset_index(drop=True)
        pos = stock_pos[i]
        if len(pos) == 0:
            continue

        # PIT static — per-row series scattered onto global-calendar columns.
        # amt_60d_q holds the RAW trailing 60d mean (captured in the mask
        # loop before z-score); its cross-sectional per-date quantile is
        # computed over the whole (N, T) grid after the loop.
        if static_dim > 0:
            s = np.zeros((len(pos), static_dim), dtype=np.float32)
            sidx = {c: k for k, c in enumerate(static_cols_available)}
            if "amt_60d_q" in sidx:
                s[:, sidx["amt_60d_q"]] = amt60_raw[i][pos]
            if "listing_days" in sidx:
                glob_col = pos.astype(np.float32)
                if first_col[i] >= 0:
                    glob_col = np.maximum(glob_col - first_col[i], 0.0)
                s[:, sidx["listing_days"]] = glob_col / 250.0
            bid = _board_index(valid_codes[i])
            bcol = _BOARD_ONEHOT_COLS[bid]
            if bcol in sidx:
                s[:, sidx[bcol]] = 1.0
            static_arr[i, pos] = s

        # Past known / observed — scattered onto global-calendar columns.
        pk_arr[i, pos] = df_sorted[pk_cols_available].fillna(0.0).values.astype(np.float32)
        po_arr[i, pos] = df_sorted[po_cols_available].fillna(0.0).values.astype(np.float32)

    # Cross-sectional per-date quantile for the trailing-mean size/liquidity
    # features.  Rank within each column's cross-section of stocks that are
    # genuinely listed there (obs True) with a nonzero trailing mean.
    # PIT-safe: every value in column t uses only data through close t, and
    # the within-column rank is itself known at t.
    for qname in static_cols_available:
        if not qname.endswith("_60d_q"):
            continue
        if qname not in static_cols_available:
            continue
        qk = static_cols_available.index(qname)
        qcol = static_arr[:, :, qk]
        qlisted = obs_arr & (qcol > 0)
        for qt in range(max_T):
            qidxs = np.nonzero(qlisted[:, qt])[0]
            if qidxs.size < 2:
                if qidxs.size == 1:
                    qcol[qidxs, qt] = 0.5  # singleton cross-section → neutral rank
                continue
            qvals = qcol[qidxs, qt]
            # Average-rank ties — pandas rank(method="average",
            # pct=True).  argsort ordinal ranks would give equal values
            # different quantiles purely from stock array order.
            qorder = np.argsort(qvals, kind="mergesort")
            q0 = qvals[qorder]
            qsz = qidxs.size
            grp_start = np.concatenate([[0], np.nonzero(np.diff(q0))[0] + 1])
            grp_end = np.concatenate([grp_start[1:], [qsz]])
            grp_rank1 = (grp_start + grp_end + 1) / 2.0  # 1-based avg rank/group
            qranks = np.empty(qsz, dtype=np.float64)
            qranks[qorder] = np.repeat(grp_rank1, grp_end - grp_start)
            qcol[qidxs, qt] = (qranks / qsz).astype(np.float32)

    # ── Sanitize: replace NaN/Inf with zeros and clip extreme values ──
    # Alpha158 factors can produce Inf from near-zero divisors (e.g.
    # open0 = open/close with close≈0 for suspended stocks).  The z-score
    # normalization also amplifies tiny variance features.
    pk_arr = np.nan_to_num(pk_arr, nan=0.0, posinf=0.0, neginf=0.0)
    pk_arr = np.clip(pk_arr, -10.0, 10.0)
    po_arr = np.nan_to_num(po_arr, nan=0.0, posinf=0.0, neginf=0.0)
    po_arr = np.clip(po_arr, -10.0, 10.0)
    static_arr = np.nan_to_num(static_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_ret_arr = np.nan_to_num(y_ret_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_vol_arr = np.nan_to_num(y_vol_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # NOTE: Targets are NOT scaled here.  Per-stock z-score normalization
    # is applied in the training script (train_panel.py) using only training
    # statistics, which avoids look-ahead bias and gives each stock equal
    # weight regardless of its native return volatility.
    # After per-stock z-scoring (μ=0, σ=1), the MSE baseline ≈ 1.0,
    # which is naturally balanced with CE loss (~1.0).

    # Per-stock date indices — maps each time step to its absolute
    # position in the global trading calendar.  With global alignment the
    # mapping is IDENTICAL for every stock (column t == global_dates[t]),
    # so PairwiseRankingLoss groups true same-day cross-sections.
    date_idx_arr = np.tile(np.arange(max_T, dtype=np.int32), (N_stocks, 1))

    # Union trading calendar (datetime64[ns]) — lets callers convert a
    # panel column index back to the real trading date.
    return {
        "static_features": static_arr,
        "past_known": pk_arr,
        "past_observed": po_arr,
        "y_direction": y_dir_arr,
        "y_return": y_ret_arr,
        "y_volatility": y_vol_arr,
        "date_indices": date_idx_arr,
        "global_dates": global_dates,
        "observation_mask": obs_arr,
        "entry_eligible_mask": entry_arr,
        "return_target_mask": ret_tgt_arr,
        "vol_target_mask": vol_tgt_arr,
        # §十四-3: per-entry-day count of valid daily returns in the forward
        # vol window (t, t+h] — lets the vol loss weight labels by their
        # true sample size (vol_tgt_mask is already gated on this reaching
        # _min_vol_nobs(horizon)).
        "forward_vol_nobs": forward_vol_nobs,
        "realized_return": realized_arr,
        # §T13: per-date exit-fill probability — fraction of entry-eligible
        # stocks at column t that also have a real open[t+horizon] (NaN where
        # none, or in the tail where no exit window exists).  Records the
        # residual fill rate now that carried exits enter the return label.
        "fill_prob": fill_prob_arr,
        "decision_eligible_mask": decision_arr,
        "history_eligible_mask": history_arr,
        # Data-derived research-universe gate (已上市 + 未长期停牌 +
        # 流动性); the delist / index-membership halves are ANDed in per-fold
        # by train_panel (§七-3).
        "universe_eligible_mask": universe_eligible_arr,
        "close_price": close_price_arr,
        "open_price": open_price_arr,
        # Column order of the feature grids (axis 2) — lets consumers probe
        # per-channel presence via the has_* flags.  Full engineered order:
        # per-fold dead-column removal happens in train_panel after slicing.
        "past_known_cols": list(pk_cols_available),
        "past_observed_cols": list(po_cols_available),
        # §v12-P0: the stock identity of each array row — the pipeline's
        # valid_codes (survived cleaning), so callers must NOT re-derive the
        # code list from the raw panel (misaligns after a dropped stock).
        "stock_codes": list(valid_codes),
    }
