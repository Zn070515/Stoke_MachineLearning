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
        # Dedup insurance: producer guarantees unique (date, sector) rows today,
        # but guard against a left-merge row explosion if that ever regresses.
        ir = ir.drop_duplicates(subset=["date", "sector_code"], keep="last")

        # Map stocks to sectors
        if sector_map:
            df["sector_code"] = df["stock_code"].astype(str).map(sector_map)
        else:
            return df

        # L1-2: join sector features
        df = df.merge(ir, on=["date", "sector_code"], how="left", suffixes=("", "_sec"))

        # L3-4: sector momentum + RRG (returns merged df)
        df = self._add_sector_momentum(df, ir)

        # L4: breadth normalization.  ``sector_breadth_raw`` uses the merged
        # ir columns (correct), but the cross-sectional z collapses on a
        # per-stock df (1 row/date -> x - x = 0).  Compute the z on the full
        # ir PANEL (all sectors per date), then broadcast by date+sector_code.
        if (
            "up_count" in df.columns
            and "down_count" in df.columns
            and "up_count" in ir.columns
            and "down_count" in ir.columns
        ):
            total = df["up_count"] + df["down_count"]
            df["sector_breadth_raw"] = (
                (df["up_count"] - df["down_count"]) / total.replace(0, np.nan)
            ).astype(np.float32)

            panel = ir.copy()
            # Sort so the per-sector rolling smoothing in the z-score is
            # chronological within each sector.
            panel = panel.sort_values(["sector_code", "date"])
            panel["_breadth_raw"] = (
                (panel["up_count"] - panel["down_count"])
                / (panel["up_count"] + panel["down_count"]).replace(0, np.nan)
            ).astype(np.float32)
            # by="sector_code" makes the rolling-window smoothing per-sector
            # chronological (never crossing sector boundaries).
            panel["_breadth_z"] = _cross_sectional_zscore(
                panel, "_breadth_raw", self.breadth_normalize_window, by="sector_code"
            )
            df = df.merge(
                panel[["date", "sector_code", "_breadth_z"]],
                on=["date", "sector_code"],
                how="left",
            )
            df["sector_breadth_z"] = df["_breadth_z"].fillna(0).astype(np.float32)
            df.drop(columns=["_breadth_z"], inplace=True)

        # L5: rotation signals (require date-sorted df)
        if "rank" in df.columns:
            df = df.sort_values(["stock_code", "date"])
            df["sector_rank_change"] = (
                df.groupby("stock_code")["rank"].diff().fillna(0).astype(np.int16)
            )

        # Cross-sectional features must be computed on the PANEL (all sectors
        # from industry_ranking), then broadcast to each stock by date.
        # A per-stock groupby collapses to x - x = 0, so the sector mean must
        # come from the full cross-sector ``ir`` frame.  ``df["change_pct"]``
        # here is the sector's daily change_pct (the stock's own return is
        # ``pct_change``), so sector_relative_strength = sector − market-mean.
        if "date" in ir.columns and "change_pct" in ir.columns:
            panel = ir.groupby("date", as_index=False)["change_pct"].mean()
            panel = panel.rename(columns={"change_pct": "_sector_mean"})
            df = df.merge(panel, on=["date"], how="left")
            if "change_pct" in df.columns and "_sector_mean" in df.columns:
                df["sector_relative_strength"] = (
                    df["change_pct"] - df["_sector_mean"]
                ).fillna(0.0).astype(np.float32)
                df.drop(columns=["_sector_mean"], inplace=True)

        if "rank" in df.columns:
            df["is_top5_sector"] = df["rank"].le(5).astype(np.int8)

        # is_sector_leader
        if "leader" in df.columns:
            df["is_sector_leader"] = (
                df["leader"].astype(str) == df["stock_code"].astype(str)
            ).astype(np.int8)

        # P1 #7: crowding indicators
        df = self._add_crowding(df)

        # P1 #8: residual momentum (strip market beta) — computed on the ir panel
        df = self._add_residual_momentum(df, ir)

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

    def _add_residual_momentum(self, df, ir):
        """Strip market beta from sector returns via cross-sectional regression.

        Computed on the industry_ranking PANEL (all sectors per date), then
        broadcast to the per-stock df.  A per-stock df carries exactly one
        sector per date, so a per-date cross-sectional regression on ``df``
        dedups to a single sector and collapses to alpha = 0.

        For each date, regresses every sector's ``change_pct`` on that date's
        market return (polyfit degree 1) and keeps the residual — purified
        sector alpha, orthogonal to market direction.

        Market return = equal-weight mean of sector ``change_pct`` per date.
        NOTE: ``ir`` carries sector-level returns (not individual stocks), so
        the sector-equal-weight mean is the available "market" proxy.
        """
        if ir is None or ir.empty:
            return df
        if "sector_code" not in ir.columns or "change_pct" not in ir.columns:
            return df

        panel = ir.copy()

        # Market return: equal-weight mean of sector returns per date
        mkt = (
            panel.groupby("date", as_index=False)["change_pct"].mean().rename(
                columns={"change_pct": "_mkt_return"}
            )
        )
        panel = panel.merge(mkt, on="date", how="left")

        from numpy.polynomial import polynomial as P

        def _residualize_date(grp):
            # ir is already 1 row per sector per date (no dedup needed), but
            # keep a guard so dates with too few sectors get alpha = 0.
            # NaN rows (missing change_pct/_mkt_return) stay NaN here and are
            # zeroed by the panel-level fillna(0) below — never a fitted value.
            grp["sector_alpha"] = np.nan
            m = grp["change_pct"].notna() & grp["_mkt_return"].notna()
            if m.sum() < 3:
                grp.loc[m, "sector_alpha"] = 0.0
                return grp
            c = P.polyfit(grp.loc[m, "_mkt_return"].values,
                          grp.loc[m, "change_pct"].values, 1)
            fitted = c[0] + c[1] * grp.loc[m, "_mkt_return"]
            grp.loc[m, "sector_alpha"] = (
                (grp.loc[m, "change_pct"] - fitted)
            ).astype(np.float32)
            return grp

        # pandas >=3.0 drops groupby key columns from apply() results
        date_series = panel["date"].copy()
        panel = panel.groupby("date", group_keys=False).apply(_residualize_date)
        panel["date"] = date_series
        panel["sector_alpha"] = panel["sector_alpha"].fillna(0).astype(np.float32)

        # Broadcast residual alpha back onto the per-stock df
        df = df.merge(
            panel[["date", "sector_code", "sector_alpha"]],
            on=["date", "sector_code"],
            how="left",
        )
        df["sector_alpha"] = df["sector_alpha"].fillna(0).astype(np.float32)
        return df


def _cross_sectional_zscore(df, col, window, by):
    """Cross-sectional z-score: (value - date_mean) / date_std per date.

    Uses rolling *window* of trading days to smooth both mean and std,
    falling back to expanding-window when fewer than *window* dates available.
    Winsorizes at 1%/99% within each cross-section before z-scoring.

    ``by`` is the grouping column (e.g. ``by="sector_code"``) over which the
    rolling-window smoothing is applied in row order, so the window never
    crosses group boundaries.  ``df`` should be sorted by ``by`` then date
    for a chronological within-group smoothing.
    """
    date_mean_series = df.groupby("date")[col].transform("mean")
    date_std_series = df.groupby("date")[col].transform("std")
    keys = df[by].values
    date_mean = (
        date_mean_series.groupby(keys)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    date_std = (
        date_std_series.groupby(keys)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # Winsorize within each date cross-section before z-scoring
    lo = df.groupby("date")[col].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("date")[col].transform(lambda s: s.quantile(0.99))
    clipped = df[col].clip(lo, hi)
    return ((clipped - date_mean) / (date_std.replace(0, np.nan).fillna(1e-8))).astype(np.float32)
