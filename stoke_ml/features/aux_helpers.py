"""Pure helper functions for aux-channel alignment (§十七-1).

Module-level functions extracted from ``stoke_ml.features.aux_aligner``:
staleness-feature recording (``_append_state_staleness``), vectorized
fill → shift → fill (``_batch_fill_shift``), generic daily-aux merge
(``_merge_daily_aux``), long→wide concept aggregation
(``_aggregate_concept_long``), and macro-feature loading
(``_load_macro_features``).  These are pure helpers: they reference no
``*_COLS`` column constants, so they import nothing from ``aux_cols``.
"""
import os

import numpy as np
import pandas as pd


def _append_state_staleness(df: pd.DataFrame, value_cols: list[str],
                            prefix: str, max_staleness: int) -> list[str]:
    """Record a state channel's observation pattern as staleness features.

    Must run BEFORE the ffill destroys the NaN pattern.  A row is an
    "observation" when ANY value column is non-NaN.  Four columns are appended:

    * ``{prefix}_staleness_days`` — calendar days since the last observation
      (the ``age`` feature, §8.2); 0 before the first observation.
    * ``{prefix}_has_ever_observed`` — True from the first observation onward;
      distinguishes pre-history "unknown" from a real zero (§8.3).
    * ``{prefix}_days_since_first_available`` — calendar days since the first
      observation (§8.3); 0 before it.
    * ``{prefix}_is_stale`` — ``has_ever_observed`` AND age > max_staleness.

    Returns the new column names so the caller can PIT-shift them with the
    value columns (the feature at t reflects the state as of t-1).
    """
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    obs = df[value_cols].notna().any(axis=1).reset_index(drop=True)
    n = len(df)
    has_ever = obs.cummax().astype(bool).to_numpy()
    staleness = np.zeros(n, dtype="int32")
    days_since_first = np.zeros(n, dtype="int32")
    is_stale = np.zeros(n, dtype=bool)
    if bool(obs.any()):
        obs_dates = dates.where(obs)
        last_obs = obs_dates.ffill()
        first_obs = dates[obs].iloc[0]
        age = (dates - last_obs).dt.days.fillna(0).to_numpy()
        staleness = np.where(has_ever, age, 0).astype("int32")
        days_since_first = np.where(
            has_ever, (dates - first_obs).dt.days.fillna(0).to_numpy(), 0
        ).astype("int32")
        is_stale = has_ever & (staleness > max_staleness)
    cols = [
        f"{prefix}_staleness_days",
        f"{prefix}_has_ever_observed",
        f"{prefix}_days_since_first_available",
        f"{prefix}_is_stale",
    ]
    df[cols[0]] = staleness
    df[cols[1]] = has_ever
    df[cols[2]] = days_since_first
    df[cols[3]] = is_stale
    return cols


def _batch_fill_shift(df: pd.DataFrame, cols: list[str],
                      lag: bool = True, policy: str = "zero",
                      prefix: str | None = None,
                      max_staleness: int | None = None) -> None:
    """Vectorized fill → shift → fill for merged aux columns.

    Groups columns by dtype and does each operation in a single block
    assignment — zero DataFrame fragmentation (no PerformanceWarning).
    Mutates *df* in-place.

    ``lag=False`` skips the PIT shift for sources whose storage layer already
    mapped events to their ``effective_trade_date`` (earnings/guba/news:
    post-close → next trading day).  Their date column IS the PIT-effective
    day, so an extra shift would double-lag the signal.

    ``policy`` is the channel's aux missingness policy (§九-4):

    * ``"zero"`` — event-type channels (news count, announcements, LHB, ...):
      a day with no record genuinely means "no event", so gaps are zero-filled
      (the historical ZI convention).
    * ``"ffill"`` — state-type channels (margin balance, valuation, macro
      rates, fundamentals, ...): the value persists between observations, so a
      missing day means "unchanged", never zero.  Gaps forward-fill the last
      known value; only the pre-history head (before the first record) is
      zero-filled as the conventional "unknown state" boundary.

    Counts / ``has_*`` / streak / quadrant indicators keep their zero/False
    fill under both policies — a count of 0 or an absent flag is meaningful
    regardless of channel type.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return

    # Partition by expected dtype
    float_cols = [c for c in available
                  if not c.startswith("has_")
                  and not c.endswith("_count")
                  and not c.endswith("_streak")
                  and not c.endswith("_quadrant")]
    int_cols = [c for c in available
                if (c.endswith("_count") or c.endswith("_streak")
                    or c.endswith("_quadrant"))
                and not c.startswith("has_")]
    bool_cols = [c for c in available if c.startswith("has_")]

    # §P1-7: state channels record their observation pattern BEFORE the fill
    # destroys it — staleness / has_ever_observed / days-since-first / is_stale
    # (only when the caller supplied a channel prefix + max_staleness).
    staleness_cols: list[str] = []
    if (policy == "ffill" and prefix and max_staleness is not None
            and "date" in df.columns and float_cols):
        staleness_cols = _append_state_staleness(df, float_cols, prefix,
                                                 max_staleness)

    # Pre-lag fill
    if float_cols:
        if policy == "ffill":
            df[float_cols] = df[float_cols].ffill().astype(np.float32)
        else:
            df[float_cols] = df[float_cols].fillna(0.0).astype(np.float32)
    if int_cols:
        df[int_cols] = df[int_cols].fillna(0).astype("int16")
    if bool_cols:
        df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

    # PIT lag: feature[t-1] paired with price[t]
    if lag:
        df[available] = df[available].shift(1)
        if staleness_cols:
            df[staleness_cols] = df[staleness_cols].shift(1)

    # Post-lag fill (first row becomes NaN after shift)
    if float_cols:
        df[float_cols] = df[float_cols].fillna(0.0).astype(np.float32)
    if int_cols:
        df[int_cols] = df[int_cols].fillna(0).astype("int16")
    if bool_cols:
        df[bool_cols] = df[bool_cols].fillna(False).astype(bool)
    if staleness_cols:
        int_s = [c for c in staleness_cols
                 if c.endswith(("_days", "_since_first_available"))]
        bool_s = [c for c in staleness_cols if c not in int_s]
        if int_s:
            df[int_s] = df[int_s].fillna(0).astype("int32")
        if bool_s:
            df[bool_s] = df[bool_s].fillna(False).astype(bool)


def _merge_daily_aux(df: pd.DataFrame, aux: pd.DataFrame,
                     policy: str = "zero", prefix: str | None = None,
                     max_staleness: int | None = None) -> pd.DataFrame:
    """Merge a preprocessed auxiliary DataFrame on date with fill + PIT lag.

    Any column that exists in *aux* (except date, stock_code, has_* flags and
    K-line derived columns) is merged and lagged by 1 trading day.  ``policy``
    selects the channel's aux missingness policy (§九-4): "zero" for event-type
    channels, "ffill" for state-type channels (see :func:`_batch_fill_shift`).
    ``prefix`` + ``max_staleness`` enable §P1-7 state-staleness tracking for
    state channels (forwarded to :func:`_batch_fill_shift`).
    """
    a = aux.copy()
    a["date"] = pd.to_datetime(a["date"])
    # Drop stock-level columns — we merge on date only
    a = a.drop(columns=["stock_code"], errors="ignore")
    a = a.drop_duplicates(subset="date", keep="last")

    # K-line derived columns must NEVER come from aux. technical.compute_all
    # drops pct_change/vol_change as intermediates, which would otherwise let
    # an aux (e.g. board/industry ranking) inject its own — possibly stale —
    # values as if they were the stock's daily return.
    skip = {"date", "stock_code", "pct_change", "vol_change"}
    available = [c for c in a.columns if c not in skip]
    # Drop aux columns that collide with existing df columns (e.g. block_trade
    # has 'volume'/'amount' which clash with K-line OHLCV). Colliding columns
    # would cause pandas merge to create _x/_y suffixes, breaking downstream
    # column name access.
    df_cols = set(df.columns)
    colliding = [c for c in available if c in df_cols]
    if colliding:
        available = [c for c in available if c not in df_cols]
    # Drop non-numeric columns (e.g. 'buyer'/'seller' in block_trade) — they
    # can't be ZI-filled or cast to float32.
    available = [c for c in available
                 if pd.api.types.is_numeric_dtype(a[c]) or c.startswith("has_")]
    if not available:
        return df

    df = df.merge(a[["date"] + available], on="date", how="left")
    _batch_fill_shift(df, available, policy=policy,
                      prefix=prefix, max_staleness=max_staleness)
    return df


def _aggregate_concept_long(concept_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concept data from long format to per-stock-per-date.

    ConceptBlockEncoder outputs one row per (date, stock_code, board_name).
    Multi-hot columns (cb_*) and per-board momentum columns need to be
    collapsed to a single row per (date, stock_code) before merging with
    the main feature DataFrame.
    """
    agg_spec = {}
    # Multi-hot: max works as OR (1 if any board has the flag)
    cb_cols = [c for c in concept_df.columns if c.startswith("cb_")]
    agg_spec.update({c: (c, "max") for c in cb_cols})
    # Per-board momentum: average across boards
    mom_cols = [c for c in concept_df.columns if c.startswith("concept_momentum_")]
    agg_spec.update({c: (c, "mean") for c in mom_cols})
    bmom_cols = [c for c in concept_df.columns if c.startswith("board_momentum_")]
    agg_spec.update({c: (c, "mean") for c in bmom_cols})
    # Per-stock columns: same value across rows (take first)
    static_cols = [
        c for c in concept_df.columns
        if c not in {"date", "stock_code", "board_name"}
        and c not in cb_cols
        and c not in mom_cols
        and c not in bmom_cols
    ]
    agg_spec.update({c: (c, "first") for c in static_cols})

    key_cols = ["date", "stock_code"]
    available = [c for c in key_cols if c in concept_df.columns]
    return (
        concept_df.groupby(available, as_index=False)
        .agg(**agg_spec)
        .reset_index(drop=True)
    )


def _load_macro_features(data_dir: str) -> pd.DataFrame | None:
    """Load macro features: generation layout first, legacy flat fallback.

    Raises GenerationStoreError on a torn generation layout (§十三-2).
    """
    from stoke_ml.data.generation_store import read_generation
    df = read_generation(data_dir, "a_shares/macro/macro_daily")
    if df is not None:
        return df
    legacy = os.path.join(data_dir, "a_shares", "macro", "macro_daily.parquet")
    if os.path.exists(legacy):
        return pd.read_parquet(legacy)
    return None
