"""BoardBroadcaster: market-wide limit-up pool → per-stock daily features.

5-layer transformation (spec §3.3):
  L1 — board membership booleans (is_zt/zb/dt/yzt)
  L2 — consecutive board tracking (consecutive_zt, board_height)
  L3 — seal strength classification (type × time × cycles)
  L4 — market-level sentiment indices (broadcast to all stocks)
  L5 — market state classification (strong/volatile/weak/ice/frenzy)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)

SEAL_BASE_SCORES = {"一字板": 1.0, "T字板": 0.7, "换手板": 0.5, "": 0.3}


class BoardBroadcaster(PreprocessingStep):
    """Broadcast board pool membership to per-stock daily features.

    Parameters:
        consecutive_lookback: window for board_height calculation.
    """

    def __init__(self, consecutive_lookback: int = 20):
        self.consecutive_lookback = consecutive_lookback

    def transform(
        self,
        df: pd.DataFrame,
        pools: Optional[dict[str, pd.DataFrame]] = None,
        sentiment: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Add board features to the per-stock OHLCV DataFrame.

        Args:
            df: per-stock daily DataFrame (must have date + stock_code).
            pools: dict with keys "zt","zb","dt","yzt", each a DataFrame
                   with at least columns [date, stock_code].
            sentiment: limit_up_sentiment DataFrame with market-level indices.
        """
        if df.empty:
            return df
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if pools is None:
            pools = {}

        # L1: membership booleans
        for pool_name, col_name in [
            ("zt", "is_zt"),
            ("zb", "is_zb"),
            ("dt", "is_dt"),
            ("yzt", "is_yzt"),
        ]:
            pool_df = pools.get(pool_name)
            if pool_df is not None and not pool_df.empty:
                df[col_name] = self._check_membership(df, pool_df).astype(np.int8)
            else:
                df[col_name] = 0

        # L2: consecutive ZT tracking
        if "is_zt" in df.columns and "stock_code" in df.columns:
            df = df.sort_values(["stock_code", "date"])
            df["consecutive_zt"] = (
                df.groupby("stock_code")["is_zt"]
                .transform(_count_consecutive)
                .astype(np.int16)
            )
            df["board_height_20d"] = (
                df.groupby("stock_code")["consecutive_zt"]
                .transform(
                    lambda s: s.rolling(self.consecutive_lookback, min_periods=1).max()
                )
                .astype(np.int16)
            )

        # L3: seal strength (from zt pool data)
        zt_pool = pools.get("zt")
        if zt_pool is not None and not zt_pool.empty:
            df = self._compute_seal_strength(df, zt_pool)

        # L4: market-level sentiment broadcast
        if sentiment is not None and not sentiment.empty:
            df = self._broadcast_sentiment(df, sentiment)

        # L5: market state classification
        df = self._classify_market_state(df, pools)

        return df

    # ── helpers ────────────────────────────────────────────────────────

    def _check_membership(self, df, pool_df):
        """Return boolean Series: True if this (date, stock_code) is in pool."""
        if "date" not in pool_df.columns or "stock_code" not in pool_df.columns:
            return pd.Series(False, index=df.index)
        pool_df = pool_df.copy()
        pool_df["date"] = pd.to_datetime(pool_df["date"], errors="coerce")
        pool_set = set(
            zip(
                pool_df["date"].dt.strftime("%Y-%m-%d"),
                pool_df["stock_code"].astype(str),
            )
        )
        df_key = zip(
            pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            df["stock_code"].astype(str),
        )
        return pd.Series([k in pool_set for k in df_key], index=df.index)

    def _compute_seal_strength(self, df, zt_pool):
        """Compute seal strength from zt pool metadata."""
        zt_pool = zt_pool.copy()
        if "date" in zt_pool.columns:
            zt_pool["date"] = pd.to_datetime(zt_pool["date"], errors="coerce")
        # Merge relevant columns
        seal_cols = ["date", "stock_code"]
        available = [c for c in ["seal_type", "seal_time", "seal_cycles"] if c in zt_pool.columns]
        seal_cols.extend(available)
        if len(available) == 0:
            return df
        zt_slim = zt_pool[seal_cols].drop_duplicates(subset=["date", "stock_code"])
        df = df.merge(zt_slim, on=["date", "stock_code"], how="left")

        if "seal_type" in df.columns:
            for stype, col in [
                ("一字板", "seal_type_one_price"),
                ("换手板", "seal_type_hand_change"),
                ("T字板", "seal_type_t_shape"),
            ]:
                df[col] = (df["seal_type"] == stype).astype(np.int8)
            df["_base_score"] = df["seal_type"].map(SEAL_BASE_SCORES).fillna(0.3)
        else:
            df["_base_score"] = 0.3

        # Time factor: morning=1.0, afternoon=0.6
        if "seal_time" in df.columns:
            t = pd.to_datetime(df["seal_time"], format="%H:%M:%S", errors="coerce")
            df["_time_factor"] = np.where(
                t.dt.hour < 11, 1.0,
                np.where(t.dt.hour < 14, 0.8, 0.6)
            )
            df["_time_factor"] = np.where(t.isna(), 0.7, df["_time_factor"])
        else:
            df["_time_factor"] = 0.7

        # Cycle penalty
        if "seal_cycles" in df.columns:
            cycles = df["seal_cycles"].fillna(1).clip(1, 10).astype(int)
            df["_cycle_penalty"] = 0.5 ** (cycles - 1)
        else:
            df["_cycle_penalty"] = 1.0

        df["seal_strength"] = (
            df["_base_score"] * df["_time_factor"] * df["_cycle_penalty"]
        ).astype(np.float32)
        df["seal_success"] = df["is_zb"].eq(0).astype(np.int8)

        # Cleanup temp columns
        for c in ["_base_score", "_time_factor", "_cycle_penalty",
                   "seal_type", "seal_time", "seal_cycles"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        return df

    def _broadcast_sentiment(self, df, sentiment):
        """Merge market-level sentiment indices to all rows."""
        sent = sentiment.copy()
        if "date" in sent.columns:
            sent["date"] = pd.to_datetime(sent["date"], errors="coerce")
        expected = ["break_rate", "advance_rate", "max_board_height"]
        available = [c for c in expected if c in sent.columns]
        if not available:
            return df
        keep = ["date"] + available
        sent = sent[keep]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.merge(sent, on="date", how="left")

    def _classify_market_state(self, df, pools):
        """Classify each day into market state based on pool sizes."""
        zt_pool = pools.get("zt")
        dt_pool = pools.get("dt")
        zb_pool = pools.get("zb")

        # Count pools per date
        n_zt = self._pool_counts_by_date(zt_pool) if zt_pool is not None else pd.Series(dtype=int)
        n_dt = self._pool_counts_by_date(dt_pool) if dt_pool is not None else pd.Series(dtype=int)
        n_zb = self._pool_counts_by_date(zb_pool) if zb_pool is not None else pd.Series(dtype=int)

        df["_n_zt"] = df["date"].map(n_zt).fillna(0).astype(int)
        df["_n_dt"] = df["date"].map(n_dt).fillna(0).astype(int)
        df["_n_zb"] = df["date"].map(n_zb).fillna(0).astype(int)

        # Break rate
        df["_break_rate"] = df["_n_zb"] / (df["_n_zt"] + df["_n_zb"] + 1).astype(float)

        # State classification
        df["market_state_strong"] = (
            (df["_n_zt"] > 80) & (df["_break_rate"] < 0.15)
        ).astype(np.int8)
        df["market_state_volatile"] = (df["_break_rate"] > 0.25).astype(np.int8)
        df["market_state_weak"] = (df["_n_zt"] < 20).astype(np.int8)
        df["market_state_normal"] = (
            ~df["market_state_strong"].astype(bool)
            & ~df["market_state_volatile"].astype(bool)
            & ~df["market_state_weak"].astype(bool)
        ).astype(np.int8)

        # Net proportions
        df["net_zt_proportion"] = (
            (df["_n_zt"] - df["_n_dt"]) / max(df["_n_zt"].max() + df["_n_dt"].max(), 1)
        ).astype(np.float32)

        # Cleanup
        df.drop(columns=["_n_zt", "_n_dt", "_n_zb", "_break_rate"], inplace=True)
        return df

    @staticmethod
    def _pool_counts_by_date(pool_df):
        if pool_df is None or pool_df.empty or "date" not in pool_df.columns:
            return pd.Series(dtype=int)
        pool_df = pool_df.copy()
        pool_df["date"] = pd.to_datetime(pool_df["date"], errors="coerce")
        return pool_df.groupby("date").size()


# ── helpers ────────────────────────────────────────────────────────────

def _count_consecutive(series: pd.Series) -> pd.Series:
    """Running count of consecutive 1s, resetting on 0. Per-group use."""
    result = pd.Series(0, index=series.index, dtype=np.int16)
    cnt = 0
    for i, v in enumerate(series.fillna(0).values):
        if v > 0:
            cnt += 1
        else:
            cnt = 0
        result.iloc[i] = cnt
    return result
