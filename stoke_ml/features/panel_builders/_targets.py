"""Target/label builder extracted from panel_builder.py (§二十一).

``TargetBuilder.compute()`` runs the per-stock loop that produces all
direction/return/volatility targets, observation/entry masks, raw PIT-static
inputs, and fill-probability accumulators.  It writes directly into a
``PanelArrays`` container (T8 memmap seam).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stoke_ml.features.panel_helpers import _min_vol_nobs, _trailing_mean

if TYPE_CHECKING:
    from stoke_ml.features.panel_builders._arrays import PanelArrays


class TargetBuilder:
    """Compute per-stock targets, masks, and PIT-static raw inputs.

    Parameters
    ----------
    horizon : int
        Forward return horizon in days.
    target_col : str
        Column name for close price (default ``"close"``).
    """

    def __init__(self, horizon: int, target_col: str = "close"):
        self.horizon = horizon
        self.target_col = target_col
        # Direction noise threshold — scale by sqrt(horizon)
        # (0.003 per day, 1.0% / 5-day, 1.3% / 20-day)
        self.dir_threshold = 0.003 * (horizon ** 0.5)

    def compute(
        self,
        all_feat_dfs: list,
        valid_codes: list,
        max_T: int,
        date_to_pos: dict,
        arrays: PanelArrays,
    ):
        """Run the per-stock target loop, writing into *arrays*.

        Populates arrays.y_dir, .y_ret, .y_vol, .obs, .entry, .ret_tgt,
        .vol_tgt, .realized, .close_price, .open_price, .forward_vol_nobs,
        .entry_counts, .filled_counts, .amt60_raw, .first_col, .has_amount,
        .stock_pos, .stock_T.
        """
        N_stocks = len(all_feat_dfs)

        # Pre-size the stock_pos list so assignment by index works.
        arrays.stock_pos = [np.empty(0, dtype=np.int32) for _ in range(N_stocks)]

        for i, df in enumerate(all_feat_dfs):
            self.compute_stock(
                df, i, valid_codes[i], max_T, date_to_pos, arrays,
            )

    def compute_stock(
        self,
        df: pd.DataFrame,
        i: int,
        valid_code: str,
        max_T: int,
        date_to_pos: dict,
        arrays: PanelArrays,
    ):
        """Compute targets/masks for a single stock, writing into arrays row *i*.

        Extracted from ``compute()`` (§T5 streaming/two-pass) so the streaming
        path can call it directly with the real global stock index.
        """
        horizon = self.horizon
        target_col = self.target_col
        dir_threshold = self.dir_threshold

        if len(df) == 0:
            return
        df_sorted = df.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df_sorted["date"])
        pos = np.array([date_to_pos[str(d.date())] for d in dates], dtype=np.int32)
        arrays.stock_pos[i] = pos
        T_i = len(pos)
        arrays.stock_T[i] = T_i

        # Trading-time convention: features up to close[t]
        # (window's last column end-1) -> signal after close[t] -> ENTER at
        # open[end]=open[t+1] -> hold h days -> EXIT at open[end+h].  Labels
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
        arrays.obs[i] = close_valid
        arrays.entry[i] = open_valid
        arrays.close_price[i] = close_full.astype(np.float32)
        arrays.open_price[i] = open_full.astype(np.float32)

        # §T13 fill-probability accumulation — per-date counts of
        # entry-eligible days (open_valid[t]) and of those with a real exit
        # open at t+horizon; the ratio (fill_prob_arr) is computed after
        # the loop.
        arrays.entry_counts[np.nonzero(open_valid)[0]] += 1
        if max_T > horizon:
            arrays.filled_counts[np.nonzero(
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
                f"Stock {valid_code}: daily K-line lacks canonical "
                "`amount` — the formal daily contract requires it "
                "(§十一-5); no volume×close / price fallback."
            )
        arrays.has_amount[i] = True
        amt_full = np.full(max_T, np.nan, dtype=np.float64)
        amt_full[pos] = df_sorted["amount"].to_numpy(dtype=np.float64)
        arrays.amt60_raw[i] = _trailing_mean(amt_full, 60).astype(np.float32)
        arrays.first_col[i] = int(pos[0]) if len(pos) else -1

        # Forward return (training label): clean open[t]->open[t+h] where a
        # real exit open exists, else carry to the LAST real close in
        # (t, t+h] (§T13 decision 3 — aligned with the evaluation realized
        # path below).  Positions with no usable exit (no exit open AND no
        # real close in the window) stay NaN -> direction -100 / return 0
        # with ret_tgt_arr False so training ignores them.
        # §十四-4 (ENTRY-FILL SELECTION BIAS — research design choice, not a
        # bug): the OLD ``both`` condition (open_valid[t] AND
        # open_valid[t+h]) required a FUTURE entry open at t+h, so a stock
        # that is decision-eligible at t but NOT fillable at the exit
        # horizon (suspended/delisted before t+h) was EXCLUDED from the
        # training label set.  The learned function was therefore "decision
        # on stocks that will stay tradeable for h hours" — a subtly easier
        # population than the full decision pool, which evaluation never
        # conditioned on.
        # §T13 decision 3 APPLIES mitigation #1: training now carries
        # non-fillable exits to the last real close in (t, t+h] — EXACTLY
        # the evaluation realized semantics — so a non-fillable pick is
        # learned (rewarded with its carry value) instead of being masked
        # out of the label population.  Label distribution shift vs the old
        # clean-only labels is expected (decision 3).
        ret_fwd = np.full(max_T, np.nan, dtype=np.float32)
        if max_T > horizon:
            both = open_valid[:-horizon] & open_valid[horizon:]
            num = open_full[horizon:][both] - open_full[:-horizon][both]
            ret_fwd[:max_T - horizon][both] = (
                num / (open_full[:-horizon][both] + 1e-8)).astype(np.float32)
        # Carry non-fillable exits: the last real close at-or-before the
        # truncated window end hi = min(t+h, T-1), i.e. the last real close
        # in (t, hi].  Forward-fill the valid-close indices; k > t selects
        # a real close strictly after entry (in-window), k <= t means NO
        # close in the window -> no label (NaN).
        last_close_idx = np.maximum.accumulate(
            np.where(close_valid, np.arange(max_T), -1))
        hi = np.minimum(np.arange(max_T) + horizon, max_T - 1)
        k = last_close_idx[hi]
        carry_ok = open_valid & (k > np.arange(max_T))
        carried = np.full(max_T, np.nan, dtype=np.float64)
        carried[carry_ok] = close_full[k[carry_ok]] / open_full[carry_ok] - 1.0
        missing_clean = open_valid & ~np.isfinite(ret_fwd)
        ret_fwd[missing_clean] = carried[missing_clean].astype(np.float32)
        arrays.ret_tgt[i] = np.isfinite(ret_fwd)
        valid = arrays.ret_tgt[i]
        arrays.y_dir[i, valid] = np.where(
            ret_fwd[valid] > dir_threshold, 2,
            np.where(ret_fwd[valid] < -dir_threshold, 0, 1),
        )
        arrays.y_ret[i] = np.nan_to_num(ret_fwd, nan=0.0)

        # Realized return for portfolio evaluation — defined for EVERY
        # entry-eligible (open-valid) day so the candidate pool never
        # depends on whether a future label exists:
        #   clean open[t]->open[t+h] where available; else carry to the last
        #   real close in (t, t+h] / open[t] - 1; else 0 (no exit -> flat).
        # §T13: ret_fwd now carries non-fillable exits with the SAME value
        # as this path, so realized is just the finite part of ret_fwd,
        # else 0 — guaranteed bit-identical to the training label for
        # carried days.
        realized = np.zeros(max_T, dtype=np.float32)
        finite_ret = open_valid & np.isfinite(ret_fwd)
        realized[finite_ret] = ret_fwd[finite_ret]
        arrays.realized[i] = realized

        # FORWARD-looking realized volatility: std of the daily returns
        # realized over the NEXT `horizon` days (return[t+1 :
        # t+horizon+1]), spanning the same forward window as y_return.  The
        # target is strictly positive, matching VolatilityHead's softplus —
        # train_panel must NOT z-score it.  Suspended days get a 0 return
        # and the resumption day records the accumulated close gap, so a
        # "5-day vol" label uses all 5 days instead of silently collapsing
        # to however many days actually traded.  §十四-3: a window with
        # fewer than _min_vol_nobs(horizon) valid closes (max(1, ceil(h/2)),
        # hard floor >=2) sets vol_tgt_arr False so the vol loss never sees
        # a degenerate / non-comparable partial-window label; the raw valid
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
            arrays.forward_vol_nobs[i, t] = nobs
            if nobs < 2 or nobs < min_nobs:
                continue
            arrays.y_vol[i, t] = float(np.std(win))
            arrays.vol_tgt[i, t] = True
