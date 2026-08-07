"""Module-level constants and pure helpers for the FeaturePipeline.

Holds the FeaturePipeline's module-level constants and pure helpers (column
specs, dead-feature folding, PK/PO grid constants, calendar singleton, panel
builder utilities) extracted for reuse without pulling in the full pipeline.
This is a LEAF module: it imports nothing from ``stoke_ml.features``.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code


# The official A-share trading calendar used to validate every stock's date axis
# before it joins the panel's UNION date axis.  Cached per DATA ROOT
# (realpath-keyed) so a config-resolved call and an explicit-data_dir call
# unify, and a formal flow that passes the data root it actually reads gets the
# frozen exchange_calendar artifact at THAT root (hash-bindable).  Lazy-loaded
# so the panel path pays for it only when used.
_panel_calendars = {}
# Legacy frozen snapshot, kept only for import-compat (pipeline.py imports the
# name).  NEVER assigned by _get_panel_calendar anymore — callers must use
# _get_panel_calendar(data_dir).
_panel_calendar = None


def _get_panel_calendar(data_dir: str | None = None):
    if data_dir is None:
        from stoke_ml.config import load_config
        data_dir = load_config()["project"]["data_dir"]
    key = os.path.normcase(os.path.realpath(str(data_dir)))
    if key not in _panel_calendars:
        from stoke_ml.data.calendar import get_research_calendar
        # Formal research path: the frozen exchange_calendar artifact at the
        # given data root, strict so any date past verified_until fails loudly
        # instead of guessing.
        _panel_calendars[key] = get_research_calendar(strict=True, data_dir=data_dir)
    return _panel_calendars[key]

# Sparse feature policy: drop structurally-dead columns — constant on every
# observed day of a stock's series (flat path, _prep_feature_df) or of a fold's
# training window (panel path, fold_dead_feature_columns) — from training.  The
# SPARSE_KEEP_PREFIXES families are exempt: they are genuinely-rare events that
# stay constant for most stocks but carry real signal on their activation days
# (market-state regimes, dragon-tiger presence, dividend growth, pledge plan
# stage, transfer ratio).  The judgment is per-stock / per-fold and uses only
# data already available at the decision point, so a future period never picks
# an earlier period's feature set (research-selection leakage).
DEAD_FEATURE_RATIO = 0.9
SPARSE_KEEP_PREFIXES = (
    "market_state_", "is_dt", "dividend_growth", "plan_stage_", "transfer_ratio",
)


def _constant_col_indices(arr: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-stock dead-feature mask: (N, D) const[stock, feature].

    const[i, f] is True when feature f's value is identical on every day of
    stock i whose observation_mask is True, or stock i has no observed days in
    the slice.  Zero-padded rows carry no evidence and are excluded from the
    scan.
    """
    n_stocks, _n_dates, n_feat = arr.shape
    listed = obs.sum(axis=1) > 0
    hi = np.float32(1e30)
    const = np.zeros((n_stocks, n_feat), dtype=bool)
    # Chunk over stocks so the two where() temporaries stay bounded.
    for s0 in range(0, n_stocks, 512):
        s1 = min(s0 + 512, n_stocks)
        m = obs[s0:s1, :, None]
        vmin = np.where(m, arr[s0:s1], hi).min(axis=1)   # (chunk, D)
        vmax = np.where(m, arr[s0:s1], -hi).max(axis=1)  # (chunk, D)
        const[s0:s1] = vmin == vmax
    const[~listed] = True
    return const


def fold_dead_feature_columns(
    train_data: dict,
    pk_cols: list[str],
    po_cols: list[str],
    ratio: float = DEAD_FEATURE_RATIO,
) -> tuple[list[int], list[int]]:
    """Axis-2 indices of dead columns in a fold's training window.

    A column is dead when >= `ratio` of the fold's eligible stocks show it
    time-constant over their observed days within the training period (the
    same per-stock constant_stock_ratio measure the old report used), EXCEPT
    the SPARSE_KEEP_PREFIXES rare-event families, which are kept — their
    signal lives on sparse activation days.  The judgment uses ONLY the
    training slice — never validation/test — so a future period can't decide
    an earlier fold's feature set.  Returned index lists are safe to np.delete
    from the past_known / past_observed grids of ALL fold slices (train/val/
    test share the column layout).
    """
    obs = train_data["observation_mask"]
    pk_const = _constant_col_indices(train_data["past_known"], obs)
    po_const = _constant_col_indices(train_data["past_observed"], obs)
    pk_dead = (pk_const.mean(axis=0) >= ratio) & ~_sparse_kept(pk_cols)
    po_dead = (po_const.mean(axis=0) >= ratio) & ~_sparse_kept(po_cols)
    pk_idx = [int(i) for i in np.where(pk_dead)[0]]
    po_idx = [int(i) for i in np.where(po_dead)[0]]
    return pk_idx, po_idx


def _sparse_kept(cols: list[str]) -> np.ndarray:
    """Per-column boolean: True for the exempt SPARSE_KEEP_PREFIXES families."""
    return np.array(
        [c.startswith(SPARSE_KEEP_PREFIXES) for c in cols], dtype=bool,
    )

ANNOUNCEMENT_COLS = [
    "ann_sentiment_mean", "ann_sentiment_std", "ann_count",
    "ann_positive_ratio", "ann_negative_ratio", "has_announce",
]

TEMPORAL_BASE_COLS = [
    "open", "high", "low", "close", "volume",
    "volume_ratio", "atr_14", "rsi_12",
]

# Rich text features from DailyAggregator (new preprocessing text chain).
# Per-source prefixes are applied by the benchmark/data-loading layer.
_AGGREGATOR_BASE_COLS = [
    "bipolar_sent", "agreement", "attention", "weighted_sent",
]

# ── New multi-shape preprocessing (spec §6) ──

FLOW_COLS = [
    "flow_intensity", "flow_z", "flow_momentum",
    "flow_persistence_5d", "flow_persistence_10d", "flow_persistence_20d",
    "flow_divergence", "flow_residual", "flow_spread_large_small",
]

BLOCK_TRADE_COLS = [
    "bt_count", "bt_total_amount", "bt_vwap_premium",
    "bt_deep_discount_count", "bt_permanent_impact", "bt_temporary_impact",
    "bt_volatility_6d",
]

SHAREHOLDER_COLS = [
    "sh_hnum_change_pct", "sh_hnum_zscore", "sh_pcrc",
    "sh_consecutive_neg", "sh_dual_concentration_signal", "sh_avg_shares_held",
]

LOCKUP_COLS = [
    "lu_pressure", "lu_ratio", "lu_days_until",
    "lu_event_count", "lu_is_vc_backed",
]

DIVIDEND_COLS = [
    "dv_yield", "dv_effective_yield", "dv_months_since_last",
]

BOARD_COLS = [
    "is_zt", "is_zb", "is_dt", "is_yzt",
    "consecutive_zt", "board_height_20d", "seal_strength", "seal_success",
    "net_zt_proportion", "break_rate", "advance_rate", "max_height",
]

SECTOR_COLS = [
    "sector_relative_strength", "sector_breadth_z",
    "sector_rrg_y", "sector_rrg_x", "sector_rrg_quadrant",
]

CONCEPT_COLS = [
    "board_count", "has_hot_board", "avg_concept_heat",
    "is_concept_leader", "board_overlap_score",
]

LIMIT_UP_COLS = [
    "zt_first_seal_hour", "zt_last_seal_hour", "zt_seal_fund_ratio",
    "zt_break_times", "zt_limit_days", "zt_pct",
    "zb_first_seal_hour", "zb_break_times", "zb_amplitude", "zb_speed",
    "dt_seal_fund_ratio", "dt_open_times", "dt_days", "dt_pe",
    "yzt_first_seal_hour", "yzt_limit_days",
    "has_zt", "has_zb", "has_dt", "has_yzt",
]  # DEFERRED (limit-up ecology family, top scope note) — defined for future re-enable, NOT wired

PLEDGE_COLS = [
    "pledge_ratio", "pledge_margin_dist", "pledge_risk",
    "pledge_count_20d", "has_pledge",
]

INDEX_MEMBER_COLS = [
    "is_index_member", "n_indexes", "idx_change_30d",
]  # no index_weight — Baostock has no historical weights (A4a scope note)

DRAGON_TIGER_SEAT_COLS = [
    "lhb_is_wave", "lhb_is_sustained", "lhb_is_drop", "lhb_count_5d",
]


# ── Panel model feature column definitions ──────────────────────────────────


# PIT-static features:
#   amt_60d_q       trailing 60d mean turnover → per-date cross-sectional rank
#   listing_days    (global col − first listed col) / 250
#   board_*         exchange-board one-hot derived from the stock code
# All eight are computed purely from data known at the decision day.
# §五 P0: `price_60d_q` (trailing 60d mean of qfq close) was REMOVED — qfq
# absolute prices re-anchor with each future corporate action (a 2026 2:1 split
# rewrites 2025 history to half), so an absolute qfq price tier makes historical
# decisions read future corporate behaviour.  `amount` is real CNY and is NOT
# re-anchored, so `amt_60d_q` is scale-invariant and retained.
# NOTE on size: a genuine PIT float market cap (real 流通市值) is NOT currently
# derivable from canonical on-disk data — valuation data begins 2015 and is
# PE/PB/PS/PCF only, the daily contract has no share counts, and fundamentals
# are quarterly without share counts.  `amt_60d_q` (trailing 60d turnover rank)
# is the size/liquidity axis.  Replacing it with real PIT float market cap
# requires new data acquisition (e.g. Sina backup `float_mcap_yi` or Baostock
# `turn`), not a derivation (§十一-5).
# `industry_code` is deliberately EXCLUDED: the only available stock→industry
# sources (sector_map.json / stock_sector_cache.csv) are current-snapshot maps
# with no point-in-time membership history, so a static industry_code would
# backfill today's classification onto historical rows — a present-backfill
# that must never happen.  Re-add only behind a genuine PIT membership source.
# The per-stock industry-relative features (ind_matched_return /
# stock_vs_industry) are likewise removed from _merge_industry for the same
# reason.
_BOARD_NAMES = ("unknown", "sh_main", "star", "sz_main", "chinext", "bse")
_BOARD_ONEHOT_COLS = [f"board_{n}" for n in _BOARD_NAMES]

# §五 P0: absolute qfq price columns must NEVER reach a model-input grid.
# Forward-adjusted (qfq) prices re-anchor with each future corporate action
# (a 2026 2:1 split rewrites 2025 history to half), so an absolute qfq level
# would leak future corporate behaviour into historical decisions.  The raw
# columns stay in the engineered frame — targets / entry / evaluation read
# close & open — but neither the past-known nor past-observed grid may carry
# them.  Only the EXACT OHLC names are excluded: scale-invariant relatives
# derived from them (open0/high0/low0, kmid/klen/... ratios) are legal inputs.
# volume/amount stay: amount is real CNY (not re-anchored); volume is unchanged
# under qfq adjustment.
_ABSOLUTE_PRICE_COLS = frozenset({"open", "high", "low", "close"})

_PIT_STATIC_COLS = [
    "amt_60d_q",       # trailing 60d mean turnover (canonical amount) → size/liquidity
    "listing_days",    # days since first bar (scaled by 250 → years)
    *_BOARD_ONEHOT_COLS,  # exchange-board one-hot derived from the stock code
]


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Point-in-time trailing mean over [t-window+1, t].

    Early rows use the partial window (mean of the days available so far);
    NaNs inside the window are skipped.  Rows with no valid value → 0.
    Vectorized via cumulative sums — O(n) per call.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    valid = np.isfinite(values)
    vals = np.where(valid, values, 0.0)
    scs = np.concatenate([[0.0], np.cumsum(vals)])
    vcs = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])
    t_idx = np.arange(n)
    lo = np.maximum(t_idx - window + 1, 0)
    cnt = vcs[1:] - vcs[lo]
    s = scs[1:] - scs[lo]
    return np.where(cnt >= 1.0, s / np.maximum(cnt, 1.0), 0.0)


def _not_long_suspended(
    obs_arr: np.ndarray,
    first_col: np.ndarray,
    max_T: int,
    threshold: int,
    lookback: int,
) -> np.ndarray:
    """Point-in-time long-suspension eligibility ``(n_stocks, T)`` (§七-3).

    A stock is disqualified from the first column its consecutive missing-close
    run reaches ``threshold``, through ``lookback`` trading columns after the
    run resumes.  Pre-listing columns are never "missing" (the stock is not yet
    listed, not suspended).  Fully vectorized: cumulative run lengths + a
    difference array for the active-window intervals.
    """
    n = obs_arr.shape[0]
    if n == 0 or max_T == 0:
        return np.zeros((n, max_T), dtype=bool)
    cols = np.arange(max_T)[None, :]
    listed = (first_col[:, None] >= 0) & (cols >= first_col[:, None])
    missing = (~obs_arr) & listed
    # Consecutive missing-run length ending at t: last non-missing column up to
    # t, so run[t] = t - last_ok[t] when missing, else 0.
    last_ok = np.maximum.accumulate(np.where(missing, -1, cols), axis=1)
    run = np.where(missing, cols - last_ok, 0)
    trig = run >= threshold
    if not trig.any():
        return np.ones((n, max_T), dtype=bool)
    trig_start = trig.copy()
    trig_start[:, 1:] &= ~trig[:, :-1]
    trig_end = trig.copy()
    trig_end[:, :-1] &= ~trig[:, 1:]
    rows_s, cols_s = np.where(trig_start)
    rows_e, cols_e = np.where(trig_end)
    diff = np.zeros((n, max_T + 1), dtype=np.int64)
    np.add.at(diff, (rows_s, cols_s), 1)
    end = np.minimum(cols_e + lookback + 1, max_T)
    np.add.at(diff, (rows_e, end), -1)
    active = np.cumsum(diff[:, :max_T], axis=1)
    return active == 0


def _board_index(code) -> int:
    """Map a 6-digit A-share code to an index into _BOARD_ONEHOT_COLS."""
    s = normalize_stock_code(code)
    if s is None:
        return 0
    if s.startswith("60"):
        return 1   # SH main board
    if s.startswith("68"):
        return 2   # STAR market
    if s.startswith("00"):
        return 3   # SZ main board (incl. 002 SME)
    if s.startswith("30"):
        return 4   # ChiNext
    if s[0] in ("8", "4"):
        return 5   # Beijing Stock Exchange
    return 0



# Alpha158 rolling-window factor name generator.
# Must stay in sync with _WINDOWS in stoke_ml/features/technical.py.
_ALPHA158_WINDOWS = [5, 10, 20, 30, 60]


def _alpha158_factor_names() -> list[str]:
    """Return Alpha158 rolling-window factor column names for all windows."""
    names: list[str] = []
    for d in _ALPHA158_WINDOWS:
        names.extend([
            f"max_{d}d", f"min_{d}d", f"qtlu_{d}d", f"qtld_{d}d",
            f"rank_{d}d", f"rsv_{d}d",
            f"corr_{d}d", f"cord_{d}d",
            f"beta_{d}d", f"rsqr_{d}d", f"resi_{d}d",
            f"vma_{d}d", f"vstd_{d}d",
            f"cntp_{d}d", f"cntn_{d}d", f"cntd_{d}d",
            f"sump_{d}d", f"sumn_{d}d", f"sumd_{d}d",
            f"imax_{d}d", f"imin_{d}d", f"imxd_{d}d",
            f"wvma_{d}d", f"vsump_{d}d", f"vsumn_{d}d", f"vsumd_{d}d",
        ])
    return names


# Features that are stock-invariant (same value for every stock on a given date).
# Cross-sectional z-score normalization would divide by near-zero std, producing
# saturated ±10.0 values with no signal. Skip them.
_CS_NORM_SKIP_COLS = frozenset({
    "minutes_from_open", "minutes_to_close", "is_am_session", "is_pm_session",
    "session_progress", "bar_of_day",
    "day_of_week", "day_of_month", "month", "quarter",
    # Macro (market-wide, identical for every stock on a date)
    "shibor_O_N", "shibor_1W", "shibor_2W", "shibor_1M",
    "shibor_3M", "shibor_6M", "shibor_9M", "shibor_1Y",
    "fx_usd_cny", "fx_eur_cny", "fx_jpy_cny", "fx_hkd_cny", "fx_gbp_cny",
    "bond_cn_2y", "bond_cn_5y", "bond_cn_10y", "bond_cn_30y",
    "bond_cn_10y2y_spread",
    "bond_us_2y", "bond_us_5y", "bond_us_10y", "bond_us_30y",
    "bond_us_10y2y_spread",
    "gdp_cn_yoy", "m2_yoy", "m1_yoy", "sf_total", "cpi_yoy",
    # Industry cross-sectional stats (row-wise over ALL industries — market-wide).
    # NOTE: ind_matched_return / stock_vs_industry are PER-STOCK and intentionally
    # excluded from this set (they carry cross-sectional signal).
    "ind_pct_up", "ind_return_mean", "ind_return_std",
    "ind_return_max", "ind_return_min", "ind_return_skew",
    "ind_dispersion_20d",
})


def _active_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the subset of *candidates* that exist in *df*."""
    return [c for c in candidates if c in df.columns]


def _manifest_check_config(seq_len: int, horizon: int) -> dict:
    """Config view used to validate a panel-mode feature sidecar manifest.

    Mirrors build_features.py's worker args so a same-day build → train
    handoff sees a MATCHING ``config``.  start/end/horizon/seq_len/panel_mode
    are compared directly inside ``manifest_matches_detailed`` (NOT via
    config_hash — build and train resolve them differently, §十一-3).
    ``start`` follows the config's research start date; ``end`` is "today",
    matching build_features.py's ``date_end = datetime.now().strftime(...)``.
    """
    from stoke_ml.config import load_config as _load_cfg
    try:
        start = _load_cfg().markets.a_shares.start_date
    except Exception:
        # Unit tests / unconfigured context: fall back to the research default
        # so a manifest written with the same fallback still matches.
        start = "2000-01-01"
    return {
        "seq_len": seq_len,
        "horizon": horizon,
        "panel_mode": True,
        "start": start,
        "end": datetime.now().strftime("%Y-%m-%d"),
    }


def _min_vol_nobs(horizon: int) -> int:
    """Minimum number of VALID daily returns a vol window must hold to label.

    §十四-3: ``max(1, ceil(0.5 * horizon))`` — a label computed from fewer
    valid returns than half the horizon is a partial-window estimate whose
    magnitude is not comparable across stocks (a 1-day vol read is not a
    5-day vol read).  Using the midpoint (not the full window) keeps the
    realistic case of one or two suspension days inside (t, t+h] still
    labelable while excluding degenerate <2-return windows.
    """
    return max(1, int(np.ceil(horizon / 2)))

