"""Date-wise z-score normalizer + member-flag helpers extracted from panel_builder.py.

``_daily_member_flag`` and ``_cross_section_stats`` moved here from
``panel_builder.py`` (they are re-exported there for import-compat).
``DateWiseZScoreNormalizer`` encapsulates the per-date cross-sectional z-score
normalization including the member-mask statistical-set restriction (§T6
decision 2).  Named distinctly from the unrelated preprocessing
``stoke_ml.preprocessing.numeric.cross_section.CrossSectionNormalizer``.
"""
import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code_series
from stoke_ml.features.panel_helpers import _CS_NORM_SKIP_COLS


# ── module-level helpers (moved from panel_builder.py) ──────────────────

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
    # 1-5 stocks/day) give a degenerate cross-section: std->0 inflates
    # z-scores to +/-hundreds, which dominates the loss.  Fall back to
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


# ── DateWiseZScoreNormalizer ────────────────────────────────────────────

class DateWiseZScoreNormalizer:
    """Per-date cross-sectional z-score normalization (§T6 decision 2).

    For each date, each feature is re-expressed relative to that date's
    cross-section (mean/std).  When ``daily_membership`` is set and non-empty,
    the statistical set for each date is restricted to that day's index
    members; non-member stocks are still z-scored but do NOT contribute to the
    mean/std.

    Parameters
    ----------
    daily_membership : pd.DataFrame or None
        Long-form ``(stock_code, in_date, out_date)`` frame; None = all-stock
        behavior (every stock contributes equally).
    """

    def __init__(self, daily_membership: pd.DataFrame | None = None):
        self.daily_membership = daily_membership

    @property
    def membership_active(self) -> bool:
        m = self.daily_membership
        return m is not None and not m.empty

    def normalize(
        self,
        all_feat_dfs: list,
        pk_cols: list,
        po_cols: list,
    ) -> tuple[list, dict]:
        """Z-score normalize *all_feat_dfs* in place.

        Returns ``(norm_cols, date_stats)`` — the list of columns that were
        normalized and the per-column per-date stats dict.
        """
        norm_cols = [c for c in pk_cols + po_cols
                     if c not in _CS_NORM_SKIP_COLS]

        if self.membership_active:
            all_feat = pd.concat([
                df[["date", "stock_code"] + norm_cols]
                for df in all_feat_dfs
                if len(df) > 0
            ], ignore_index=True)
            is_member = _daily_member_flag(all_feat, self.daily_membership)
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
        # column-invariant — hoist the boolean mask and the missing-date set
        # out of the per-column loop so each column does NOT re-run a
        # full-panel groupby over ~33M rows just to find the same zero-member
        # dates.
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
                # §T6: a date present in the panel but with ZERO member rows
                # must still receive stats — otherwise the .map below yields
                # NaN and the post-processing nan_to_num zeroes the WHOLE
                # date's features.  Fall back to the all-stock cross-section
                # stats for exactly those dates.  Restricting to the
                # missing-date rows makes the groupby tiny AND routes it
                # through the sparse expanding-moments fallback — the
                # all-stock cross-section on those dates can itself be sparse,
                # and without the fallback its degenerate std->0 (clipped to
                # 1e-8) would blow up that date's z-scores.  all_stats carries
                # ONLY the missing dates, so no .loc[missing] filter is needed
                # here.
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

        return norm_cols, date_stats
