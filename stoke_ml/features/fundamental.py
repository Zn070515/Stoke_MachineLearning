"""Fundamental refinement features — quality, stability, trend, valuation.

Split into two execution phases:
- Per-stock (in _engineer_features): F-score, quality composite,
  stability, own-history valuation percentiles, growth trends.
- Cross-sectional (in build_panel_features): sector-relative valuation,
  leverage warning, composite cheapness.  These are computed in a
  separate class method that operates on the full multi-stock panel.
"""
import warnings

import numpy as np
import pandas as pd

from stoke_ml.features._rolling import (
    rolling_mean, rolling_std, rolling_slope, rolling_percentile_rank,
    expanding_zscore, zscore_cross_section,
)


class FundamentalRefiner:
    """Per-stock fundamental feature refinement.

    Operates on a single stock's daily DataFrame.  Requires forward-filled
    fundamental columns (roe, roa, eps, revenue_yoy, profit_yoy, debt_ratio,
    gross_margin, net_margin) and valuation columns (pe_ttm, pb_mrq, ps_ttm,
    pcf_ttm).
    """

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        result = self._compute_quality(result)
        result = self._compute_stability(result)
        result = self._compute_own_valuation(result)
        result = self._compute_trends(result)

        return result

    # ------------------------------------------------------------------
    # Quality composite
    # ------------------------------------------------------------------

    def _compute_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        roa = df.get("roa")
        roe = df.get("roe")
        debt = df.get("debt_ratio")
        margin = df.get("gross_margin")
        rev_yoy = df.get("revenue_yoy")
        prof_yoy = df.get("profit_yoy")

        if roa is not None:
            roa_v = roa.values.astype(np.float64)
            roa_pos = (roa_v > 0).astype(np.float64)
            roa_delta = np.full(len(roa_v), np.nan, dtype=np.float64)
            if len(roa_v) > 63:
                roa_delta[63:] = roa_v[63:] - roa_v[:-63]
            roa_improving = (roa_delta > 0).astype(np.float64)

            f_roa = roa_pos
            f_delta_roa = np.nan_to_num(roa_improving, 0.0)

            f_leverage = np.zeros(len(df), dtype=np.float64)
            if debt is not None:
                debt_v = debt.values.astype(np.float64)
                debt_delta = np.full(len(debt_v), np.nan, dtype=np.float64)
                if len(debt_v) > 63:
                    debt_delta[63:] = debt_v[63:] - debt_v[:-63]
                f_leverage = (debt_delta < 0).astype(np.float64)

            f_margin = np.zeros(len(df), dtype=np.float64)
            if margin is not None:
                margin_v = margin.values.astype(np.float64)
                margin_delta = np.full(len(margin_v), np.nan, dtype=np.float64)
                if len(margin_v) > 63:
                    margin_delta[63:] = margin_v[63:] - margin_v[:-63]
                f_margin = (margin_delta > 0).astype(np.float64)

            f_score = (
                np.nan_to_num(f_roa, 0.0)
                + np.nan_to_num(f_delta_roa, 0.0)
                + np.nan_to_num(f_leverage, 0.0)
                + np.nan_to_num(f_margin, 0.0)
            )
            df["f_score"] = f_score.astype(np.float32)

        # Quality composite: avg of expanding z(roe), z(margin), z(-debt)
        quality_parts = []
        eps = 1e-8
        if roe is not None:
            roe_v = roe.values.astype(np.float64)
            quality_parts.append(expanding_zscore(roe_v))
        if margin is not None:
            margin_v = margin.values.astype(np.float64)
            quality_parts.append(expanding_zscore(margin_v))
        if debt is not None:
            debt_v = debt.values.astype(np.float64)
            quality_parts.append(expanding_zscore(-debt_v))

        if len(quality_parts) >= 2:
            stacked = np.column_stack(quality_parts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                composite = np.nanmean(stacked, axis=1)
            df["quality_composite"] = np.nan_to_num(composite, 0.0).astype(np.float32)

        # Earnings quality: profit_yoy - revenue_yoy
        if prof_yoy is not None and rev_yoy is not None:
            df["earnings_quality"] = (
                prof_yoy.values.astype(np.float64) - rev_yoy.values.astype(np.float64)
            ).astype(np.float32)

        # Growth quality: revenue_yoy * gross_margin
        if rev_yoy is not None and margin is not None:
            df["growth_quality"] = (
                rev_yoy.values.astype(np.float64) * margin.values.astype(np.float64)
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def _compute_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-8
        roe = df.get("roe")
        margin = df.get("gross_margin")

        if roe is not None:
            roe_v = roe.values.astype(np.float64)
            roe_std = rolling_std(roe_v, 63)
            roe_mean = rolling_mean(roe_v, 63)
            df["profitability_stability"] = np.where(
                np.abs(roe_mean) > eps,
                1.0 - roe_std / (np.abs(roe_mean) + eps),
                0.0,
            ).astype(np.float32)

        if margin is not None:
            margin_v = margin.values.astype(np.float64)
            margin_std = rolling_std(margin_v, 63)
            margin_mean = rolling_mean(margin_v, 63)
            df["margin_stability"] = np.where(
                np.abs(margin_mean) > eps,
                1.0 - margin_std / (np.abs(margin_mean) + eps),
                0.0,
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Own-history valuation percentiles
    # ------------------------------------------------------------------

    def _compute_own_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        for val_col, prefix in [("pe_ttm", "pe"), ("pb_mrq", "pb")]:
            if val_col not in df.columns:
                continue
            v = df[val_col].values.astype(np.float64)
            window = 252
            df[f"{prefix}_percentile_252d"] = rolling_percentile_rank(v, window).astype(np.float32)

        # PE/PB divergence
        if "pe_percentile_252d" in df.columns and "pb_percentile_252d" in df.columns:
            df["pe_pb_divergence"] = (
                df["pe_percentile_252d"].values.astype(np.float64)
                - df["pb_percentile_252d"].values.astype(np.float64)
            ).astype(np.float32)

        # Deep value flag
        if "pe_percentile_252d" in df.columns and "pb_percentile_252d" in df.columns:
            df["deep_value"] = (
                (df["pe_percentile_252d"] < 0.2) & (df["pb_percentile_252d"] < 0.2)
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Growth trends (slopes over ~4 quarters)
    # ------------------------------------------------------------------

    def _compute_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        window = 63  # ~1 quarter in trading days

        for col, prefix in [
            ("roe", "roe"),
            ("revenue_yoy", "revenue"),
            ("gross_margin", "margin"),
        ]:
            if col not in df.columns:
                continue
            v = df[col].values.astype(np.float64)
            trend = rolling_slope(v, window)
            df[f"{prefix}_trend_4q"] = np.nan_to_num(trend, 0.0).astype(np.float32)

        # ROE acceleration: current_trend - prior_trend
        if "roe_trend_4q" in df.columns:
            t = df["roe_trend_4q"].values.astype(np.float64)
            prior = np.roll(t, window)
            prior[:window] = np.nan
            df["roe_accel"] = np.where(
                ~np.isnan(t) & ~np.isnan(prior), t - prior, 0.0
            ).astype(np.float32)

        # Earnings surprise: eps - ma4(eps) on daily data
        eps = df.get("eps")
        if eps is not None:
            eps_v = eps.values.astype(np.float64)
            eps_ma = rolling_mean(eps_v, window)
            df["earnings_surprise"] = np.where(
                ~np.isnan(eps_ma), eps_v - eps_ma, 0.0
            ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Cross-sectional phase (called from build_panel_features)
    # ------------------------------------------------------------------

    @staticmethod
    def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
        """Add sector-relative valuation features to a multi-stock panel.

        Requires columns: date, stock_code, sector_code, pe_ttm, pb_mrq,
        ps_ttm, debt_ratio.
        """
        if "sector_code" not in panel.columns or panel.empty:
            return panel

        result = panel.copy()

        # Sector medians per date
        gb = result.groupby(["date", "sector_code"], as_index=False)

        for val_col in ["pe_ttm", "pb_mrq", "ps_ttm"]:
            if val_col not in result.columns:
                continue
            prefix = val_col.split("_")[0]  # pe, pb, ps
            medians = gb[val_col].transform("median")
            result[f"{prefix}_sector_ratio"] = np.where(
                medians.abs() > 1e-8,
                result[val_col].values.astype(np.float64) / medians.values.astype(np.float64),
                1.0,
            ).astype(np.float32)

        # Leverage warning: debt_ratio > 80th percentile within sectorxdate
        if "debt_ratio" in result.columns:
            p80 = gb["debt_ratio"].transform(lambda x: x.quantile(0.8))
            result["leverage_warning"] = (
                result["debt_ratio"] > p80
            ).astype(np.float32)

        # Valuation composite z-score (per-date cross-sectional, no look-ahead)
        z_parts = []
        for pct_col in ["pe_percentile_252d", "pb_percentile_252d"]:
            if pct_col in result.columns:
                z = result.groupby("date")[pct_col].transform(
                    lambda x: (x - x.mean()) / max(x.std(), 1e-8)
                )
                z_parts.append(-z.values.astype(np.float64))
        if len(z_parts) >= 2:
            stacked = np.column_stack(z_parts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                composite = np.nanmean(stacked, axis=1)
            result["valuation_composite_z"] = np.nan_to_num(composite, 0.0).astype(np.float32)

        return result


# Rolling helpers imported from stoke_ml.features._rolling
