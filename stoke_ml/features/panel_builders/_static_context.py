"""Static-context builder extracted from panel_builder.py (§二十一).

``StaticContextBuilder.build()`` populates the three feature grids
(static / past-known / past-observed) from the normalized feature DataFrames,
then computes per-date cross-sectional trailing-mean quantile ranks for the
``*_60d_q`` PIT-static columns.
"""
import numpy as np

from stoke_ml.features.panel_helpers import (
    _BOARD_ONEHOT_COLS,
    _PIT_STATIC_COLS,
    _board_index,
)


class StaticContextBuilder:
    """Populate the static / past-known / past-observed feature grids.

    Writes directly into the ``PanelArrays`` container (T8 memmap seam).
    """

    def build(
        self,
        all_feat_dfs: list,
        valid_codes: list,
        static_cols: list,
        pk_cols: list,
        po_cols: list,
        arrays,  # PanelArrays
    ):
        """Scatter per-stock features into arrays.static / .pk / .po.

        *arrays* must already have ``alloc_features()`` called so
        ``.static``, ``.pk``, ``.po`` exist.

        After the scatter, computes cross-sectional per-date trailing-mean
        quantile ranks (``*_60d_q`` columns) — mutates ``arrays.static``
        in place.
        """
        N_stocks, max_T = arrays.N, arrays.T
        static_cols_available = list(static_cols)
        pk_cols_available = list(pk_cols)
        po_cols_available = list(po_cols)

        for i, df in enumerate(all_feat_dfs):
            if len(df) == 0:
                continue

            df_sorted = df.sort_values("date").reset_index(drop=True)
            pos = arrays.stock_pos[i]
            if len(pos) == 0:
                continue

            # PIT static — per-row series scattered onto global-calendar
            # columns.  amt_60d_q holds the RAW trailing 60d mean (captured in
            # the target loop before z-score); its cross-sectional per-date
            # quantile is computed over the whole (N, T) grid after the loop.
            if len(static_cols_available) > 0:
                s = np.zeros((len(pos), len(static_cols_available)), dtype=np.float32)
                sidx = {c: k for k, c in enumerate(static_cols_available)}
                if "amt_60d_q" in sidx:
                    s[:, sidx["amt_60d_q"]] = arrays.amt60_raw[i][pos]
                if "listing_days" in sidx:
                    glob_col = pos.astype(np.float32)
                    if arrays.first_col[i] >= 0:
                        glob_col = np.maximum(glob_col - arrays.first_col[i], 0.0)
                    s[:, sidx["listing_days"]] = glob_col / 250.0
                bid = _board_index(valid_codes[i])
                bcol = _BOARD_ONEHOT_COLS[bid]
                if bcol in sidx:
                    s[:, sidx[bcol]] = 1.0
                arrays.static[i, pos] = s

            # Past known / observed — scattered onto global-calendar columns.
            arrays.pk[i, pos] = (
                df_sorted[pk_cols_available].fillna(0.0).values.astype(np.float32)
            )
            arrays.po[i, pos] = (
                df_sorted[po_cols_available].fillna(0.0).values.astype(np.float32)
            )

        # Cross-sectional per-date quantile for the trailing-mean
        # size/liquidity features.  Rank within each column's cross-section of
        # stocks that are genuinely listed there (obs True) with a nonzero
        # trailing mean.  PIT-safe: every value in column t uses only data
        # through close t, and the within-column rank is itself known at t.
        for qname in static_cols_available:
            if not qname.endswith("_60d_q"):
                continue
            if qname not in static_cols_available:
                continue
            qk = static_cols_available.index(qname)
            qcol = arrays.static[:, :, qk]
            qlisted = arrays.obs & (qcol > 0)
            for qt in range(max_T):
                qidxs = np.nonzero(qlisted[:, qt])[0]
                if qidxs.size < 2:
                    if qidxs.size == 1:
                        qcol[qidxs, qt] = 0.5  # singleton cross-section -> neutral
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
