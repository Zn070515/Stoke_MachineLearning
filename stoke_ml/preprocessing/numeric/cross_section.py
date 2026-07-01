"""Cross-sectional normalization for multi-stock panel data.

Qlib-style: operates on a panel DataFrame with columns (date, stock_code,
sector, size_proxy, ...features...).  All stages work per-date across
all stocks at that timestamp.

Stages (applied in order):
  1. sector  — subtract sector-median within each (date, sector) group
  2. size    — regress each feature on log(size)+log²(size) per date,
               keep residual (removes market-cap effects)
  3. rank    — rank-normalize across stocks per date → N(0,1) via
               inverse-normal CDF (Qlib CSRankNorm)
  4. zscore  — (x - mean) / std across stocks per date (CSZScoreNorm)
  5. adaptive — amplify features in high-volatility regimes

META columns (preserved unchanged): date, stock_code, sector, size_proxy,
and any column whose name starts with "has_" (binary flags).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)

_META_PATTERNS = ("date", "stock_code", "sector", "size_proxy")


class CrossSectionNormalizer(PreprocessingStep):
    """Per-date cross-stock normalization for feature panels.

    Must receive a panel DataFrame from PanelBuilder or equivalent
    that contains at minimum 'date' and 'stock_code' columns.
    'sector' and 'size_proxy' columns are required for sector/size
    stages respectively.
    """

    def __init__(
        self,
        enabled: bool = True,
        stages: list[str] | None = None,
        columns: list[str] | None = None,
    ):
        self.enabled = enabled
        self.stages = stages or ["sector", "size", "rank"]
        self.columns = columns

    # ------------------------------------------------------------------
    # PreprocessingStep interface
    # ------------------------------------------------------------------

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if not self.enabled or df.empty:
            return df.copy()

        df = df.copy()
        feat_cols = self._feature_columns(df)

        if not feat_cols:
            logger.warning("CrossSectionNormalizer: no feature columns found")
            return df

        for stage in self.stages:
            df = self._apply_stage(stage, df, feat_cols)

        return df

    # ------------------------------------------------------------------
    # Stage dispatcher
    # ------------------------------------------------------------------

    def _apply_stage(
        self, stage: str, df: pd.DataFrame, feat_cols: list[str]
    ) -> pd.DataFrame:
        if stage == "sector" and "sector" in df.columns:
            return _sector_neutralize(df, feat_cols)
        elif stage == "size" and "size_proxy" in df.columns:
            return _size_neutralize(df, feat_cols)
        elif stage == "rank":
            return _rank_normalize(df, feat_cols)
        elif stage == "zscore":
            return _zscore_normalize(df, feat_cols)
        elif stage == "adaptive":
            return _adaptive_strength(df, feat_cols)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return columns to normalize: numeric, non-meta, non-binary."""
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]

        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not c.startswith(_META_PATTERNS)
            and not c.startswith("has_")
            and not c.startswith("is_")
        ]


# ------------------------------------------------------------------
# Stage implementations (module-level for testability)
# ------------------------------------------------------------------


def _sector_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subtract sector median within each (date, sector) group.

    Only applies when a sector has ≥2 stocks on that date;
    single-stock sectors are left unchanged to avoid zeroing out.
    """
    for col in cols:
        if col not in df.columns:
            continue
        grouped = df.groupby(["date", "sector"])[col]
        median = grouped.transform("median")
        count = grouped.transform("count")
        mask = count >= 2
        df[col] = np.where(mask, (df[col] - median).astype(np.float32), df[col].astype(np.float32))
    return df


def _size_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Regress each feature on log(size)+log²(size) per date, keep residual."""
    size = df["size_proxy"].values
    size2 = size * size
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        y = df[col].values.astype(np.float64)
        residual = np.full(n, np.nan, dtype=np.float64)

        for date, idx in df.groupby("date").groups.items():
            idx_list = list(idx)
            if len(idx_list) < 10:
                continue
            X = np.column_stack([
                np.ones(len(idx_list)),
                size[idx_list],
                size2[idx_list],
            ])
            y_sub = y[idx_list]
            valid = ~np.isnan(y_sub)
            if valid.sum() < 10:
                continue
            try:
                beta = np.linalg.lstsq(X[valid], y_sub[valid], rcond=None)[0]
                pred = X @ beta
                residual_arr = y_sub - pred
                # re-index back to original positions
                for j, orig_idx in enumerate(idx_list):
                    if valid[j]:
                        residual[orig_idx] = residual_arr[j]
            except np.linalg.LinAlgError:
                continue

        df[col] = np.where(np.isnan(residual), df[col].values, residual)
        df[col] = df[col].astype(np.float32)
    return df


def _rank_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Qlib CSRankNorm: rank → uniform → inverse normal CDF → N(0,1).

    For each date, rank each feature across stocks. Ties get the
    average rank. The result is ~N(0,1) per cross-section.
    """
    for col in cols:
        if col not in df.columns:
            continue

        def _rank_to_normal(series: pd.Series) -> pd.Series:
            ranked = series.rank(method="average", na_option="keep")
            n_valid = ranked.notna().sum()
            if n_valid < 2:
                return pd.Series(0.0, index=series.index)
            # Uniform [epsilon, 1-epsilon] to avoid ±∞ from ppf
            eps = 1.0 / (n_valid + 1)
            uniform = (ranked - 0.5) / n_valid
            uniform = uniform.clip(eps, 1.0 - eps)
            return pd.Series(norm.ppf(uniform.values), index=series.index)

        df[col] = df.groupby("date")[col].transform(_rank_to_normal).astype(np.float32)
    return df


def _zscore_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Qlib CSZScoreNorm: (x - mean) / std per date across stocks."""
    for col in cols:
        if col not in df.columns:
            continue
        grouped = df.groupby("date")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        denom = std.replace(0, 1.0)
        df[col] = ((df[col] - mean) / denom).astype(np.float32)
    return df


def _adaptive_strength(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Amplify features in high-volatility regimes.

    alpha = 1.0 + 0.5 * (sigma_short - sigma_long) / sigma_long
    alpha clipped to [0.75, 1.50].
    """
    if "close" not in df.columns:
        return df

    # Compute per-stock volatility ratio
    returns = df.groupby("stock_code")["close"].pct_change()
    sigma_short = returns.groupby(df["stock_code"]).rolling(
        20, min_periods=10
    ).std().reset_index(level=0, drop=True)
    sigma_long = returns.groupby(df["stock_code"]).rolling(
        60, min_periods=30
    ).std().reset_index(level=0, drop=True)

    rel_vol = (sigma_short - sigma_long) / sigma_long.replace(0, 1.0)
    alpha = (1.0 + 0.5 * rel_vol.clip(-0.5, 1.0)).fillna(1.0)

    for col in cols:
        if col not in df.columns:
            continue
        df[col] = (df[col].values * alpha.values).astype(np.float32)

    return df
