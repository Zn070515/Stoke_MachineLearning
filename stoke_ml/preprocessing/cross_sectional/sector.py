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

        # L3-4: sector momentum + RRG (returns merged df)
        df = self._add_sector_momentum(df, ir)

        # L4: breadth normalization
        if "up_count" in df.columns and "down_count" in df.columns:
            total = df["up_count"] + df["down_count"]
            df["sector_breadth_raw"] = (
                (df["up_count"] - df["down_count"]) / total.replace(0, np.nan)
            ).astype(np.float32)
            df["sector_breadth_z"] = _cross_sectional_zscore(
                df, "sector_breadth_raw", self.breadth_normalize_window
            )

        # L5: rotation signals (require date-sorted df)
        if "rank" in df.columns:
            df = df.sort_values(["stock_code", "date"])
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

        # P1 #7: crowding indicators
        df = self._add_crowding(df)

        # P1 #8: residual momentum (strip market beta)
        df = self._add_residual_momentum(df)

        return df

    def _add_sector_momentum(self, df, industry_ranking):
        """Compute sector momentum for each window and RRG features.

        Returns the df mutated with new columns merged in.
        """
        ir = industry_ranking.copy()
        if "date" not in ir.columns or "sector_code" not in ir.columns:
            return df
        if "change_pct" not in ir.columns:
            return df

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

        # RRG: compute sector-level RS-Ratio from industry_ranking (not per-stock)
        ir_sector = ir.copy()
        if 252 in self.momentum_windows:
            ir_sector["_cum_return"] = (
                ir_sector.groupby("sector_code")["change_pct"]
                .transform(lambda s: s.rolling(252, min_periods=63).sum())
            )
            # RS-Momentum: cumulative return cross-sectional z-score per date
            date_mean = ir_sector.groupby("date")["_cum_return"].transform("mean")
            date_std = ir_sector.groupby("date")["_cum_return"].transform("std")
            ir_sector["_rrg_y"] = (
                (ir_sector["_cum_return"] - date_mean) / (date_std + 1e-8)
            )
            # RS-Momentum: rate of change of RS-Ratio over 10d
            ir_sector["_rrg_x"] = (
                ir_sector.groupby("sector_code")["_rrg_y"]
                .diff(10)
                .fillna(0)
            )
            ir_rrg = ir_sector[["date", "sector_code", "_rrg_y", "_rrg_x"]]
            df = df.merge(ir_rrg, on=["date", "sector_code"], how="left")
            df["sector_rrg_y"] = df["_rrg_y"].astype(np.float32)
            df["sector_rrg_x"] = df["_rrg_x"].astype(np.float32)
            df.drop(columns=["_rrg_y", "_rrg_x"], inplace=True, errors="ignore")
            # Quadrant: x>0=leading, x<0=lagging  ×  y>0=strong, y<0=weak
            df["sector_rrg_quadrant"] = (
                (df["sector_rrg_x"].gt(0).astype(int)) * 2
                + df["sector_rrg_y"].gt(0).astype(int)
            ).astype(np.int8)

        return df

    # ── P1 #7: crowding indicators ───────────────────────────────────

    def _add_crowding(self, df):
        """Sector-level crowding: volume volatility + turnover z-score.

        Literature: 2024 quant research consensus — crowding is the most
        important sector risk factor. High crowding → fragile sector
        leadership, increased reversal probability.
        """
        if "sector_code" not in df.columns:
            return df
        required = ["volume", "date", "stock_code"]
        if not all(c in df.columns for c in required):
            return df

        # Per-sector daily aggregate volume
        sector_vol = (
            df.groupby(["date", "sector_code"])["volume"]
            .sum()
            .reset_index(name="sector_volume")
        )
        # Rolling 20d coefficient of variation per sector
        sector_vol = sector_vol.sort_values(["sector_code", "date"])
        roll_mean = (
            sector_vol.groupby("sector_code")["sector_volume"]
            .rolling(20, min_periods=10).mean()
            .reset_index(level=0, drop=True)
        )
        roll_std = (
            sector_vol.groupby("sector_code")["sector_volume"]
            .rolling(20, min_periods=10).std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        sector_vol["sector_vol_volatility"] = (
            roll_std / (roll_mean.abs() + 1e-8)
        ).astype(np.float32)

        # Merge sector-level crowding back
        df = df.merge(
            sector_vol[["date", "sector_code", "sector_vol_volatility"]],
            on=["date", "sector_code"], how="left",
        )
        df["sector_vol_volatility"] = df["sector_vol_volatility"].fillna(0).astype(np.float32)

        # Turnover rate z-score (cross-sectional per date)
        if "turnover_rate" in df.columns:
            # Aggregate per sector
            sector_turn = (
                df.groupby(["date", "sector_code"])["turnover_rate"]
                .mean()
                .reset_index(name="sector_turnover")
            )
            date_mean = sector_turn.groupby("date")["sector_turnover"].transform("mean")
            date_std = sector_turn.groupby("date")["sector_turnover"].transform("std")
            sector_turn["sector_turnover_z"] = (
                (sector_turn["sector_turnover"] - date_mean) / (date_std.replace(0, np.nan).fillna(1e-8))
            ).clip(-5, 5).astype(np.float32)
            df = df.merge(
                sector_turn[["date", "sector_code", "sector_turnover_z"]],
                on=["date", "sector_code"], how="left",
            )
            df["sector_turnover_z"] = df["sector_turnover_z"].fillna(0).astype(np.float32)

        return df

    # ── P1 #8: residual momentum ─────────────────────────────────────

    def _add_residual_momentum(self, df):
        """Strip market beta from sector returns via cross-sectional regression.

        For each date, regresses sector return on market return and keeps
        the residual — purified sector alpha, orthogonal to market direction.
        """
        if "sector_code" not in df.columns or "change_pct" not in df.columns:
            return df
        if "close" not in df.columns or "stock_code" not in df.columns:
            return df

        # Market return: equal-weighted mean of stock returns per date
        if "close" in df.columns:
            df_sorted = df.sort_values(["stock_code", "date"])
            df_sorted["_ret"] = (
                df_sorted.groupby("stock_code")["close"].pct_change()
            )
            mkt_ret = (
                df_sorted.groupby("date")["_ret"]
                .mean()
                .reset_index(name="mkt_return")
            )
            df_sorted.drop(columns=["_ret"], inplace=True)
        else:
            return df

        # Sector-level daily return (from industry_ranking change_pct)
        if "change_pct" not in df.columns:
            return df

        # Merge market return and run per-date cross-sectional regression
        df = df.merge(mkt_ret, on="date", how="left")

        from numpy.polynomial import polynomial as P

        def _residualize_date(grp):
            # Dedup by sector_code so large sectors don't dominate regression
            dedup = grp.drop_duplicates(subset=["sector_code"])
            m = dedup["change_pct"].notna() & dedup["mkt_return"].notna()
            if m.sum() < 3:
                grp["sector_alpha"] = 0.0
                return grp
            c = P.polyfit(dedup.loc[m, "mkt_return"].values,
                          dedup.loc[m, "change_pct"].values, 1)
            fitted = c[0] + c[1] * grp["mkt_return"].fillna(0)
            grp["sector_alpha"] = (
                (grp["change_pct"].fillna(0) - fitted)
            ).astype(np.float32)
            return grp

        df = df.groupby("date", group_keys=False).apply(_residualize_date)
        df.drop(columns=["mkt_return"], inplace=True)
        return df


def _cross_sectional_zscore(df, col, window):
    """Cross-sectional z-score: (value - date_mean) / date_std per date.

    Uses rolling *window* of trading days to smooth both mean and std,
    falling back to expanding-window when fewer than *window* dates available.
    Winsorizes at 1%/99% within each cross-section before z-scoring.
    """
    date_mean = (
        df.groupby("date")[col].transform("mean")
        .rolling(window, min_periods=1)
        .mean()
    )
    date_std = (
        df.groupby("date")[col].transform("std")
        .rolling(window, min_periods=1)
        .mean()
    )
    # Winsorize within each date cross-section before z-scoring
    lo = df.groupby("date")[col].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("date")[col].transform(lambda s: s.quantile(0.99))
    clipped = df[col].clip(lo, hi)
    return ((clipped - date_mean) / (date_std.replace(0, np.nan).fillna(1e-8))).astype(np.float32)
