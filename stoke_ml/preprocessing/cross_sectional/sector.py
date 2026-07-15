"""SectorBroadcaster: industry ranking → per-stock daily features.

5-layer transformation (spec §3.4):
  L1 — stock-to-sector join via sector_map
  L2 — sector-level features per stock (rank, change_pct, breadth, leader)
  L3 — sector momentum (multi-timeframe: 5/20/60/252d)
  L4 — RRG framework (RS-Ratio × RS-Momentum, 252-bar z-score)
  L5 — sector rotation signals (rank_change, relative_strength, is_top5)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)


class SectorBroadcaster(PreprocessingStep):
    """Broadcast industry ranking to per-stock daily features.

    Parameters:
        momentum_windows: rolling windows for sector momentum (trading days).
        breadth_normalize_window: window for breadth z-score.
    """

    def __init__(
        self,
        momentum_windows: tuple[int, ...] = (5, 20, 60, 252),
        breadth_normalize_window: int = 252,
    ):
        self.momentum_windows = momentum_windows
        self.breadth_normalize_window = breadth_normalize_window

    def transform(
        self,
        df: pd.DataFrame,
        industry_ranking: Optional[pd.DataFrame] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Add sector features to the per-stock DataFrame.

        Args:
            df: per-stock daily DataFrame (date + stock_code).
            industry_ranking: daily industry ranking with columns
                [date, code, change_pct, up_count, down_count, leader, rank].
            sector_map: dict stock_code → industry_code.
        """
        if df.empty:
            return df
        if industry_ranking is None or industry_ranking.empty:
            return df
        if sector_map is None:
            sector_map = {}

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        ir = industry_ranking.copy()
        if "date" in ir.columns:
            ir["date"] = pd.to_datetime(ir["date"], errors="coerce")
        ir = ir.rename(columns={"code": "sector_code"})

        # Map stocks to sectors
        if sector_map:
            df["sector_code"] = df["stock_code"].astype(str).map(sector_map)
        else:
            return df

        # L1-2: join sector features
        df = df.merge(ir, on=["date", "sector_code"], how="left", suffixes=("", "_sec"))

        # L3: sector momentum
        self._add_sector_momentum(df, ir)

        # L4: breadth normalization
        if "up_count" in df.columns and "down_count" in df.columns:
            total = df["up_count"] + df["down_count"]
            df["sector_breadth_raw"] = (
                (df["up_count"] - df["down_count"]) / total.replace(0, np.nan)
            ).astype(np.float32)
            df["sector_breadth_z"] = _cross_sectional_zscore(
                df, "sector_breadth_raw", self.breadth_normalize_window
            )

        # L5: rotation signals
        if "rank" in df.columns:
            df["sector_rank_change"] = (
                df.groupby("stock_code")["rank"].diff().fillna(0).astype(np.int16)
            )

        if "change_pct" in df.columns:
            df["sector_relative_strength"] = (
                df["change_pct"] - df.groupby("date")["change_pct"].transform("mean")
            ).astype(np.float32)

        if "rank" in df.columns:
            df["is_top5_sector"] = df["rank"].le(5).astype(np.int8)

        # is_sector_leader
        if "leader" in df.columns:
            df["is_sector_leader"] = (
                df["leader"].astype(str) == df["stock_code"].astype(str)
            ).astype(np.int8)

        return df

    def _add_sector_momentum(self, df, industry_ranking):
        """Compute sector momentum for each window."""
        ir = industry_ranking.copy()
        if "date" not in ir.columns or "sector_code" not in ir.columns:
            return
        if "change_pct" not in ir.columns:
            return

        ir = ir.sort_values(["sector_code", "date"])
        for w in self.momentum_windows:
            ir[f"momentum_{w}d"] = (
                ir.groupby("sector_code")["change_pct"]
                .transform(lambda s: s.rolling(w, min_periods=max(5, w // 4)).sum())
            )
        # Merge back
        mom_cols = ["date", "sector_code"] + [
            f"momentum_{w}d" for w in self.momentum_windows
        ]
        ir_mom = ir[mom_cols]
        if "sector_code" in df.columns:
            df.drop(
                columns=[c for c in df.columns if c.startswith("momentum_")],
                inplace=True,
                errors="ignore",
            )
            df = df.merge(ir_mom, on=["date", "sector_code"], how="left")

        # RRG: 252-bar z-score of cumulative return (simplified)
        if "sector_code" in df.columns and 252 in self.momentum_windows:
            # RS-Ratio: cumulative return relative to benchmark
            df["_rs_ratio"] = df.groupby("stock_code")["change_pct"].transform(
                lambda s: s.rolling(252, min_periods=63).sum()
            )
            global_mean = df.groupby("date")["_rs_ratio"].transform("mean")
            global_std = df.groupby("date")["_rs_ratio"].transform("std")
            df["sector_rrg_y"] = (
                (df["_rs_ratio"] - global_mean) / (global_std + 1e-8)
            ).astype(np.float32)
            # RS-Momentum: rate of change of RS-Ratio
            df["sector_rrg_x"] = (
                df.groupby("stock_code")["sector_rrg_y"]
                .diff(10)
                .fillna(0)
                .astype(np.float32)
            )
            # Quadrant
            df["sector_rrg_quadrant"] = (
                (df["sector_rrg_x"] > 0).astype(int) * 2
                + (df["sector_rrg_y"] > 0).astype(int)
            ).astype(np.int8)
            df.drop(columns=["_rs_ratio"], inplace=True, errors="ignore")


def _cross_sectional_zscore(df, col, window):
    """Cross-sectional z-score: (value - date_mean) / date_std per date."""
    date_mean = df.groupby("date")[col].transform("mean")
    date_std = df.groupby("date")[col].transform("std")
    # Winsorize before z-scoring
    lo = df.groupby("date")[col].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("date")[col].transform(lambda s: s.quantile(0.99))
    clipped = df[col].clip(lo, hi)
    return ((clipped - date_mean) / (date_std + 1e-8)).astype(np.float32)
