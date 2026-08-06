"""Eligibility masks builder extracted from panel_builder.py (§二十一).

``EligibilityBuilder.compute()`` produces the decision / history / universe
eligibility masks that gate which panel positions are valid for training and
evaluation.
"""
import numpy as np

from stoke_ml.features.panel_helpers import _not_long_suspended


class EligibilityBuilder:
    """Compute decision, history, and universe-eligibility masks.

    Parameters
    ----------
    seq_len : int
        Sequence length for the history window.
    min_history : int
        Minimum number of real observations required in the lookback window.
    """

    def __init__(self, seq_len: int, min_history: int):
        self.seq_len = seq_len
        self.min_history = min_history

    def compute(
        self,
        obs_arr: np.ndarray,
        first_col: np.ndarray,
        amt60_raw: np.ndarray,
        has_amount_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(decision_arr, history_arr, universe_eligible_arr)``.

        All three are ``(N_stocks, max_T)`` bool arrays.
        """
        N_stocks, max_T = obs_arr.shape

        # -- Decision / history eligibility --
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
        if self.min_history <= 0:
            history_arr = np.ones((N_stocks, max_T), dtype=bool)
        else:
            obs_i = obs_arr.astype(np.int32)
            cum = np.concatenate(
                [np.zeros((N_stocks, 1), dtype=np.int32),
                 np.cumsum(obs_i, axis=1)],
                axis=1,
            )
            t_idx = np.arange(max_T)
            lo = np.maximum(t_idx - self.seq_len, 0)
            history_arr = (cum[:, t_idx] - cum[:, lo]) >= self.min_history

        # -- Research-universe eligibility (§七-3) --
        # Data-derived PIT gates merged into the decision pool:
        #   已上市 (first_col) + 当日未长期停牌 + 符合研究流动性规则.
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
            # Causal trailing-60d turnover known at close[t-1] -> entry day t:
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

        return decision_arr, history_arr, universe_eligible_arr
