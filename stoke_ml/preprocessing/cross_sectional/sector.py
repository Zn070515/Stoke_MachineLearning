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
        sector_features: Optional[pd.DataFrame] = None,
        sector_map_valid_from: Optional[str] = None,
    ) -> pd.DataFrame:
        """Add sector features to the per-stock DataFrame.

        Args:
            df: per-stock daily DataFrame (date + stock_code).
            industry_ranking: daily industry ranking with columns
                [date, code, change_pct, up_count, down_count, leader, rank].
                Used to (re)build ``sector_features`` when it is not supplied.
            sector_map: dict stock_code → industry_code.
            sector_features: precomputed panel from ``build_sector_features``.
                When provided, the stock-independent sector features
                (momentum, RRG, breadth_z, relative_strength, alpha) are
                broadcast straight from this panel instead of being
                recomputed once per stock.
            sector_map_valid_from: earliest date the static ``sector_map`` is
                valid from (YYYY-MM-DD).  Rows before it get NaN sector_code so
                their broadcast sector features read as unknown (zeroed) rather
                than backfilling today's classification onto history (§十-4).
                Ignored when ``df`` already carries a per-row ``sector_code``
                (genuine PIT membership).
        """
        if df.empty:
            return df
        if sector_map is None:
            sector_map = {}

        if sector_features is None:
            # Standalone recompute path: build the panel from the ranking.
            if industry_ranking is None or industry_ranking.empty:
                return df
            sector_features = self.build_sector_features(industry_ranking)
        if sector_features is None or sector_features.empty:
            return df

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Map stocks to sectors (§十-4 — no present-backfill).
        # A caller that already resolved a per-row sector_code (genuine PIT
        # membership) keeps it: re-mapping would stamp today's classification
        # onto the whole window.  Otherwise the static map is bounded to dates
        # >= sector_map_valid_from; older rows stay NaN so the merged sector
        # features read as unknown (zeroed) instead of being backfilled.
        if "sector_code" in df.columns:
            pass
        elif sector_map:
            assigned = df["stock_code"].astype(str).map(sector_map)
            if sector_map_valid_from is not None:
                assigned = assigned.where(
                    df["date"] >= pd.Timestamp(sector_map_valid_from)
                )
            df["sector_code"] = assigned
        else:
            return df

        # Drop any previously-broadcast sector columns so a re-run never
        # produces suffixed duplicates from the merge below.
        stale = [c for c in df.columns if c.startswith("momentum_")]
        stale += ["sector_rrg_y", "sector_rrg_x", "sector_rrg_quadrant",
                  "sector_relative_strength", "sector_breadth_raw",
                  "sector_breadth_z", "sector_alpha"]
        df.drop(columns=[c for c in stale if c in df.columns],
                inplace=True, errors="ignore")

        # L1-2: broadcast the precomputed sector panel to each stock row.
        df = df.merge(
            sector_features, on=["date", "sector_code"], how="left", suffixes=("", "_sec")
        )

        # Preserve OLD raw-output parity: on dates before the ranking's
        # coverage the left-merge leaves these NaN; the pre-refactor
        # transform 0-filled them, so keep that here.
        for col in ("sector_relative_strength", "sector_breadth_z", "sector_alpha"):
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)

        # L4: RRG quadrant (derived from the broadcast sector_rrg_x/y).
        if "sector_rrg_x" in df.columns and "sector_rrg_y" in df.columns:
            df["sector_rrg_quadrant"] = (
                (df["sector_rrg_x"].gt(0).astype(int)) * 2
                + df["sector_rrg_y"].gt(0).astype(int)
            ).astype(np.int8)

        # L5: rotation signals (per-stock; require date-sorted df).
        if "rank" in df.columns:
            df = df.sort_values(["stock_code", "date"])
            df["sector_rank_change"] = (
                df.groupby("stock_code")["rank"].diff().fillna(0).astype(np.int16)
            )

        if "rank" in df.columns:
            df["is_top5_sector"] = df["rank"].le(5).astype(np.int8)

        # is_sector_leader
        if "leader" in df.columns:
            df["is_sector_leader"] = (
                df["leader"].astype(str) == df["stock_code"].astype(str)
            ).astype(np.int8)

        # P1 #7: crowding indicators (per-stock volume).
        df = self._add_crowding(df)

        return df

    def build_sector_features(self, industry_ranking) -> pd.DataFrame:
        """Precompute all stock-independent sector features once.

        Runs the cross-sector computations (momentum, RRG, breadth z,
        relative strength, residual alpha) on the full industry-ranking
        panel — once per ranking instead of once per stock — and returns a
        de-duplicated, sorted panel that ``transform`` broadcasts by
        (date, sector_code).

        Returns columns:
          date, sector_code, change_pct, up_count, down_count, rank, leader,
          momentum_{w}d (each window), sector_rrg_y, sector_rrg_x,
          sector_relative_strength, sector_breadth_raw, sector_breadth_z,
          sector_alpha.
        """
        if industry_ranking is None or industry_ranking.empty:
            return industry_ranking

        ir = industry_ranking.copy()
        if "date" in ir.columns:
            ir["date"] = pd.to_datetime(ir["date"], errors="coerce")
        ir = ir.rename(columns={"code": "sector_code"})
        # Dedup insurance: producer guarantees unique (date, sector) rows.
        ir = ir.drop_duplicates(subset=["date", "sector_code"], keep="last")
        # Numeric coercion: upstream rows may arrive as object dtype (e.g.
        # after a ragged concat); the rolling/z/regression math needs numerics.
        for col in ("change_pct", "up_count", "down_count", "rank"):
            if col in ir.columns:
                ir[col] = pd.to_numeric(ir[col], errors="coerce")

        if "sector_code" not in ir.columns or "change_pct" not in ir.columns:
            return ir

        # L3-4: sector momentum + RRG (per-sector chronological).
        ir = self._panel_momentum_rrg(ir)

        # L4: breadth normalization — cross-sectional z per date.
        if "up_count" in ir.columns and "down_count" in ir.columns:
            ir["sector_breadth_raw"] = (
                (ir["up_count"] - ir["down_count"])
                / (ir["up_count"] + ir["down_count"]).replace(0, np.nan)
            ).astype(np.float32)
            ir["sector_breadth_z"] = _cross_sectional_zscore(
                ir, "sector_breadth_raw", self.breadth_normalize_window, by="sector_code"
            )

        # sector_relative_strength: sector − cross-sector mean (per date).
        if "change_pct" in ir.columns:
            mkt = ir.groupby("date", as_index=False)["change_pct"].mean()
            mkt = mkt.rename(columns={"change_pct": "_sector_mean"})
            ir = ir.merge(mkt, on="date", how="left")
            ir["sector_relative_strength"] = (
                ir["change_pct"] - ir["_sector_mean"]
            ).fillna(0.0).astype(np.float32)
            ir.drop(columns=["_sector_mean"], inplace=True)

        # P1 #8: residual momentum (strip market beta).
        ir = self._panel_alpha(ir)

        # Final column set, sorted, de-duplicated.
        cols = ["date", "sector_code", "change_pct", "up_count", "down_count",
                "rank", "leader"]
        cols += [f"momentum_{w}d" for w in self.momentum_windows]
        cols += ["sector_rrg_y", "sector_rrg_x", "sector_relative_strength",
                 "sector_breadth_raw", "sector_breadth_z", "sector_alpha"]
        cols = [c for c in cols if c in ir.columns]
        ir = ir[cols]
        ir = ir.sort_values(["sector_code", "date"])
        ir = ir.drop_duplicates(subset=["date", "sector_code"], keep="last")
        return ir

    def _panel_momentum_rrg(self, ir):
        """Add ``momentum_{w}d`` and ``sector_rrg_{x,y}`` columns to the panel."""
        if "date" not in ir.columns or "sector_code" not in ir.columns:
            return ir
        if "change_pct" not in ir.columns:
            return ir

        ir = ir.sort_values(["sector_code", "date"])
        for w in self.momentum_windows:
            ir[f"momentum_{w}d"] = (
                ir.groupby("sector_code")["change_pct"]
                .transform(lambda s: s.rolling(w, min_periods=max(5, w // 4)).sum())
            )

        if 252 in self.momentum_windows:
            ir["_cum_return"] = (
                ir.groupby("sector_code")["change_pct"]
                .transform(lambda s: s.rolling(252, min_periods=63).sum())
            )
            # RS-Momentum: cumulative return cross-sectional z-score per date
            date_mean = ir.groupby("date")["_cum_return"].transform("mean")
            date_std = ir.groupby("date")["_cum_return"].transform("std")
            ir["_rrg_y"] = (
                (ir["_cum_return"] - date_mean) / (date_std + 1e-8)
            )
            # RS-Momentum: rate of change of RS-Ratio over 10d
            ir["_rrg_x"] = (
                ir.groupby("sector_code")["_rrg_y"]
                .diff(10)
                .fillna(0)
            )
            ir["sector_rrg_y"] = ir["_rrg_y"].astype(np.float32)
            ir["sector_rrg_x"] = ir["_rrg_x"].astype(np.float32)
            ir.drop(columns=["_cum_return", "_rrg_y", "_rrg_x"],
                    inplace=True, errors="ignore")
        return ir

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

    def _panel_alpha(self, ir):
        """Add ``sector_alpha`` to the panel via per-date cross-sectional regression.

        Regresses every sector's ``change_pct`` on that date's market return
        (polyfit degree 1) and keeps the residual — purified sector alpha,
        orthogonal to market direction.

        Market return = equal-weight mean of sector ``change_pct`` per date.
        NOTE: ``ir`` carries sector-level returns (not individual stocks), so
        the sector-equal-weight mean is the available "market" proxy.
        """
        if ir is None or ir.empty:
            return ir
        if "sector_code" not in ir.columns or "change_pct" not in ir.columns:
            return ir

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
        panel.drop(columns=["_mkt_return"], inplace=True)
        return panel


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
