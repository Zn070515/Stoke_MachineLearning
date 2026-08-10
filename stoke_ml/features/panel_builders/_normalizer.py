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

    # ── Streaming / two-pass methods (§T5) ────────────────────────────────

    def init_stats_accumulator(self, all_dates: set):
        """Initialize per-date cross-sectional stats accumulators.

        Called once before the streaming accumulate pass.  *all_dates* is the
        global date axis as a ``set`` of ``datetime.date`` objects (the union
        of every chunk's dates); it fixes the row axis of the dense float64
        accumulator arrays.  Resets internal state so the same normalizer
        instance can be reused.

        The dense arrays are allocated lazily on the first
        :meth:`accumulate_stats_chunk` call because ``norm_cols`` (the column
        axis) is only known there.  ``date -> row`` and ``col -> position``
        maps anchor every scatter-add.
        """
        # Global date axis → dense row index.
        self._all_dates_sorted: list = sorted(all_dates)
        self._date_to_row: dict = {
            d: i for i, d in enumerate(self._all_dates_sorted)
        }
        # Dense float64 accumulators, shape (n_dates, n_norm_cols) — allocated
        # lazily by _ensure_arrays (norm_cols is unknown until the first chunk).
        self._all_cnt: np.ndarray | None = None
        self._all_sum: np.ndarray | None = None
        self._all_sq: np.ndarray | None = None
        self._member_cnt: np.ndarray | None = None
        self._member_sum: np.ndarray | None = None
        self._member_sq: np.ndarray | None = None
        self._norm_cols: list | None = None
        self._col_pos: dict = {}
        # Per-column source dtype (§T5): the dense path stores mean/std in the
        # concat frame's dtype (float64 if ANY engineered frame carries the col
        # as float64, else float32).  The streaming stats must cast to the SAME
        # dtype — a near-constant column (std clipped to 1e-8) amplifies a
        # float32-vs-float64 mean rounding (~2.5e-6 at magnitude ~126) into a
        # ±10 z-score flip.  Track whether any accumulated chunk is float64.
        self._col_is_f64: dict[str, bool] = {}

    def _ensure_arrays(self, norm_cols: list):
        """Allocate the dense float64 accumulator arrays on first use.

        ``norm_cols`` fixes the column axis (len = number of columns); the
        date axis was fixed at :meth:`init_stats_accumulator` time.  The real
        streaming caller passes the SAME ``norm_cols`` for every chunk, so a
        later call with a different column count is rejected rather than
        silently mis-accumulating.
        """
        if self._all_cnt is not None:
            if list(norm_cols) != self._norm_cols:
                raise RuntimeError(
                    "DateWiseZScoreNormalizer: accumulate_stats_chunk called "
                    f"with {len(norm_cols)} norm_cols after the dense arrays "
                    f"were allocated for {len(self._norm_cols)} — the "
                    "streaming build must use a single fixed norm_cols across "
                    "chunks")
            return
        T = len(self._all_dates_sorted)
        C = len(norm_cols)
        self._norm_cols = list(norm_cols)
        self._col_pos = {c: i for i, c in enumerate(norm_cols)}
        self._all_cnt = np.zeros((T, C), dtype=np.float64)
        self._all_sum = np.zeros((T, C), dtype=np.float64)
        self._all_sq = np.zeros((T, C), dtype=np.float64)
        if self.membership_active:
            self._member_cnt = np.zeros((T, C), dtype=np.float64)
            self._member_sum = np.zeros((T, C), dtype=np.float64)
            self._member_sq = np.zeros((T, C), dtype=np.float64)

    def _get_accum(self, date, col):
        """Return ``(count, sum, sumsq)`` for one ``(date, col)`` from the
        dense arrays.

        Mirrors the old ``dict.get((date, col), (0, 0.0, 0.0))`` contract so
        throwaway diagnostics that read raw accumulator values keep working
        after the dict → dense-array change.
        """
        r = self._date_to_row.get(date)
        cpos = self._col_pos.get(col)
        if r is None or cpos is None or self._all_cnt is None:
            return (0, 0.0, 0.0)
        return (
            float(self._all_cnt[r, cpos]),
            float(self._all_sum[r, cpos]),
            float(self._all_sq[r, cpos]),
        )

    def accumulate_stats_chunk(
        self,
        df: pd.DataFrame,
        norm_cols: list,
    ):
        """Accumulate per-date cross-sectional stats from one stock's frame.

        *df* must carry ``date`` (and ``stock_code`` when membership is
        active).  Non-finite values are skipped (matching the dense path's
        strip-inf step).  Accumulates both all-stock and (when membership is
        active) member-set aggregates into float64 running sums.

        Called once per stock in the streaming Pass 2stats.

        Vectorized: a panel chunk carries ONE row per date, so the
        accumulation adds exactly one float64 value per (date, col) per stock,
        in stock order — the same order the old per-cell dict loop used.  The
        dense arrays therefore reproduce the dict sums BIT-IDENTICALLY while
        replacing ~12M per-cell Python/numpy extractions per stock (~120 s)
        with a few whole-matrix numpy ops.
        """
        if df is None or len(df) == 0:
            return
        active_cols = [c for c in norm_cols if c in df.columns]
        if not active_cols:
            return
        self._ensure_arrays(norm_cols)
        active_cols = [c for c in active_cols if c in self._col_pos]
        if not active_cols:
            return

        for col in active_cols:
            if not self._col_is_f64.get(col, False) and df[col].dtype == np.float64:
                self._col_is_f64[col] = True

        # Map chunk rows onto the global date axis.  Panel chunks carry one
        # row per date, all of which lie on the axis (builder's all_dates union
        # + ZI-align).  A stray date would silently mis-accumulate under a
        # dense layout, so fail loudly instead of guessing.
        dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
        row_idx = np.fromiter(
            (self._date_to_row.get(d, -1) for d in dates),
            dtype=np.int64, count=len(dates),
        )
        if (row_idx == -1).any():
            bad = [d for d in dates if d not in self._date_to_row]
            raise ValueError(
                "accumulate_stats_chunk: chunk dates not on the global date "
                f"axis: {bad[:10]}{'...' if len(bad) > 10 else ''}"
            )

        col_pos = np.array([self._col_pos[c] for c in active_cols],
                           dtype=np.int64)

        # Whole-matrix extraction: (n_rows, n_active_cols) float64.
        mat = df[active_cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(mat)
        clean = np.where(finite, mat, 0.0)

        # Panel format gives unique dates per chunk; duplicate dates are
        # defensively handled via np.add.at (the per-date float-add order then
        # differs from the groupwise dict loop by ~ULP, but a well-formed
        # panel chunk cannot produce them).
        has_dup = (
            len(set(dates)) != len(dates)
            or np.unique(col_pos).size != col_pos.size
        )

        if not has_dup:
            # Unique row indices → plain advanced-indexed += is exact.
            self._all_cnt[np.ix_(row_idx, col_pos)] += finite.astype(np.float64)
            self._all_sum[np.ix_(row_idx, col_pos)] += clean
            self._all_sq[np.ix_(row_idx, col_pos)] += clean * clean
        else:
            n, m = len(row_idx), len(col_pos)
            flat_idx = (
                np.repeat(row_idx, m) * self._all_cnt.shape[1]
                + np.tile(col_pos, n)
            )
            f64 = finite.astype(np.float64).ravel()
            cr = clean.ravel()
            np.add.at(self._all_cnt.ravel(), flat_idx, f64)
            np.add.at(self._all_sum.ravel(), flat_idx, cr)
            np.add.at(self._all_sq.ravel(), flat_idx, cr * cr)

        # Member-set: same scatter-add restricted to member rows.  Adding the
        # zeroed rows when mcnt==0 is a no-op, exactly matching the old loop's
        # ``if mcnt > 0`` guard.
        if self.membership_active:
            is_member = _daily_member_flag(
                df[["date", "stock_code"]], self.daily_membership,
            ).to_numpy(dtype=bool)
            member_finite = finite & is_member[:, None]
            member_clean = np.where(member_finite, mat, 0.0)
            if not has_dup:
                self._member_cnt[np.ix_(row_idx, col_pos)] += (
                    member_finite.astype(np.float64))
                self._member_sum[np.ix_(row_idx, col_pos)] += member_clean
                self._member_sq[np.ix_(row_idx, col_pos)] += (
                    member_clean * member_clean)
            else:
                mf = member_finite.astype(np.float64).ravel()
                mcr = member_clean.ravel()
                np.add.at(self._member_cnt.ravel(), flat_idx, mf)
                np.add.at(self._member_sum.ravel(), flat_idx, mcr)
                np.add.at(self._member_sq.ravel(), flat_idx, mcr * mcr)

    def _build_stats_df(
        self,
        arrays: tuple,
        dates_subset: set,
        norm_cols: list,
    ) -> dict[str, pd.DataFrame]:
        """Build per-column ``(mean, std, count)`` DataFrames from dense
        accumulator arrays for the given date subset.

        *arrays* is the ``(cnt, sum, sumsq)`` tuple of dense float64 arrays,
        each shaped ``(n_dates_axis, n_norm_cols)``.  Reads the same
        per-(date, col) aggregates the old dict accumulator held and
        reproduces the expanding-moment sparse fallback from
        :func:`_cross_section_stats` exactly as before.

        The sparse fallback is NOT bit-exact against the dense path: the dense
        path cumsums individual rows in concat order while this path cumsums
        per-date sums (associative in exact arithmetic, but float rounding
        differs) — a controlled ~1e-13 float64 summation-order diff.

        §T5 amplification edge: on a NEAR-CONSTANT cross-section (every stock
        identical on a date) the per-date std is 0 and clips to the 1e-8
        floor, amplifying that ~1e-13 mean diff to a ~1e-5..1e-4 absolute
        z-diff.  Immaterial in production (real cross-sections are
        non-constant, std >> 1e-8), which is why the panel_builder test
        fixtures use seeded per-stock price noise to keep compared columns
        non-constant.
        """
        cnt2d, sum2d, sq2d = arrays
        all_dates_sorted = sorted(dates_subset)
        row_pos = np.array(
            [self._date_to_row[d] for d in all_dates_sorted], dtype=np.int64,
        )
        date_stats: dict[str, pd.DataFrame] = {}

        for col in norm_cols:
            # Match the dense path's stored dtype for this column (§T5).
            out_dtype = np.float64 if self._col_is_f64.get(col, False) else np.float32
            if col in self._col_pos:
                cpos = self._col_pos[col]
                cnt_arr = cnt2d[row_pos, cpos]
                sum_arr = sum2d[row_pos, cpos]
                sq_arr = sq2d[row_pos, cpos]
            else:
                # A norm col never accumulated (no chunk carried it): all-zero
                # rows, same as the old ``acc.get(..., (0, 0.0, 0.0))``.
                cnt_arr = np.zeros(len(all_dates_sorted), dtype=np.float64)
                sum_arr = np.zeros(len(all_dates_sorted), dtype=np.float64)
                sq_arr = np.zeros(len(all_dates_sorted), dtype=np.float64)
            stats_df = pd.DataFrame({
                "date": [pd.Timestamp(d) for d in all_dates_sorted],
                "count": cnt_arr,
                "sum": sum_arr,
                "sumsq": sq_arr,
            })
            if stats_df.empty:
                date_stats[col] = pd.DataFrame(
                    columns=["mean", "std", "count"],
                )
                continue

            stats_df = stats_df.set_index("date")

            # Per-date mean & sample std (ddof=1) via the sumsq identity.
            # Dates with no (or a single) observation divide by cnt=0/1 — the
            # np.where masks those to NaN, but the division is still evaluated
            # for every element, so silence the resulting divide-by-zero
            # RuntimeWarning noise.
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_arr = np.where(cnt_arr > 0, sum_arr / cnt_arr, np.nan)
                var_arr = np.where(
                    cnt_arr >= 2,
                    (sq_arr - sum_arr ** 2 / cnt_arr) / (cnt_arr - 1),
                    np.nan,
                )
                std_arr = np.sqrt(np.maximum(var_arr, 0.0))

            stats_df["mean"] = mean_arr.astype(out_dtype)
            stats_df["std"] = std_arr.astype(out_dtype)
            stats_df["count"] = cnt_arr

            # fillna(1.0) + clip(lower=1e-8) on std (matching dense path).
            stats_df["std"] = stats_df["std"].fillna(1.0).clip(lower=1e-8)

            # Sparse-date expanding-moment fallback (count < 5).
            sparse = stats_df["count"] < 5
            if sparse.any():
                cum_cnt = np.maximum(np.cumsum(cnt_arr), 1.0)
                cum_sum = np.cumsum(sum_arr)
                cum_sq = np.cumsum(sq_arr)
                cum_mean = cum_sum / cum_cnt
                cum_var = np.maximum(cum_sq / cum_cnt - cum_mean ** 2, 0.0)
                cum_std = np.maximum(np.sqrt(cum_var), 1e-8)
                stats_df.loc[sparse, "mean"] = (
                    cum_mean[sparse].astype(out_dtype)
                )
                stats_df.loc[sparse, "std"] = (
                    cum_std[sparse].astype(out_dtype)
                )

            date_stats[col] = stats_df[["mean", "std", "count"]]

        return date_stats

    def finalize_date_stats(
        self,
        norm_cols: list,
        all_dates: set,
    ) -> dict[str, pd.DataFrame]:
        """Convert accumulated per-date aggregates to mean/std DataFrames.

        Returns the same ``dict[col -> DataFrame]`` format as
        :meth:`normalize` so the downstream ``apply_zscore`` loop is
        identical for both paths.
        """
        self._ensure_arrays(norm_cols)
        if self.membership_active:
            # §T6: a date is a member date iff it received >=1 member FINITE
            # contribution — the old dict loop added a date exactly when some
            # (date, col) had mcnt > 0, i.e. member_cnt.sum(axis=1) > 0.
            member_mask = self._member_cnt.sum(axis=1) > 0
            member_dates = {
                d for d, m in zip(self._all_dates_sorted, member_mask) if m
            }
            missing_dates = all_dates - member_dates

            member_stats = self._build_stats_df(
                (self._member_cnt, self._member_sum, self._member_sq),
                member_dates, norm_cols,
            )

            if missing_dates:
                all_stats = self._build_stats_df(
                    (self._all_cnt, self._all_sum, self._all_sq),
                    missing_dates, norm_cols,
                )
                # Concat exactly as the dense path does.
                date_stats: dict[str, pd.DataFrame] = {}
                for col in norm_cols:
                    parts = []
                    if col in member_stats and len(member_stats[col]) > 0:
                        parts.append(member_stats[col])
                    if col in all_stats and len(all_stats[col]) > 0:
                        parts.append(all_stats[col])
                    if parts:
                        date_stats[col] = pd.concat(parts)
                    else:
                        date_stats[col] = pd.DataFrame(
                            columns=["mean", "std", "count"],
                        )
                return date_stats
            else:
                return member_stats
        else:
            return self._build_stats_df(
                (self._all_cnt, self._all_sum, self._all_sq),
                all_dates, norm_cols,
            )

    @staticmethod
    def apply_zscore(
        df: pd.DataFrame,
        norm_cols: list,
        date_stats: dict[str, pd.DataFrame],
    ):
        """Apply per-date z-score to *df* in place (exact same logic as the
        ``.map`` loop in :meth:`normalize`)."""
        for col in norm_cols:
            if col not in df.columns or col not in date_stats:
                continue
            aligned_mean = df["date"].map(date_stats[col]["mean"])
            aligned_std = (
                df["date"].map(date_stats[col]["std"]).clip(lower=1e-8)
            )
            df[col] = (df[col] - aligned_mean) / aligned_std
