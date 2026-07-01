"""Cross-sectional normalization: sector neutralization → size neutralization → adaptive.

Three-stage pipeline:
1. Sector: X - median(X | sector, date) / MAD(X | sector, date)
2. Size: residual ~ log(mcap) + log²(mcap) → take residual
3. Adaptive: strengthen neutralization in high-volatility regimes
"""

import logging
import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)


class CrossSectionNormalizer(PreprocessingStep):
    """Remove market/sector/size effects from features.

    Each stage is optional and falls back gracefully if required
    columns are missing (e.g. no sector mapper available → skip sector).
    """

    def __init__(
        self,
        enabled: bool = True,
        stages: list[str] | None = None,
        columns: list[str] | None = None,
    ):
        self.enabled = enabled
        self.stages = stages or ["sector", "size", "adaptive"]
        self.columns = columns

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if not self.enabled or df.empty:
            return df.copy()
        df = df.copy()

        cols = self.columns
        if cols is None:
            cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("open", "high", "low", "close", "volume",
                                 "amount", "date_day", "date_month", "date_weekday")]

        for stage in self.stages:
            if stage == "sector" and "sector" in df.columns:
                df = self._sector_neutralize(df, cols)
            if stage == "size" and "market_cap" in df.columns:
                df = self._size_neutralize(df, cols)
            if stage == "adaptive":
                df = self._adaptive_strength(df, cols)

        return df

    @staticmethod
    def _sector_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Hybrid: within each (date, sector), subtract median and divide by MAD."""
        if "date" not in df.columns:
            return df
        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_raw"] = df[col].copy()
            grouped = df.groupby(["date", "sector"])[col]
            median = grouped.transform("median")
            mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
            denom = mad.replace(0, 1.0)
            df[col] = ((df[col] - median) / denom).astype(np.float32)
        return df

    @staticmethod
    def _size_neutralize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Regress each feature on log(mcap) + log²(mcap) cross-sectionally,
        take residual per day."""
        if "date" not in df.columns or "market_cap" not in df.columns:
            return df
        log_mcap = np.log(df["market_cap"].replace(0, np.nan))
        log_mcap2 = log_mcap ** 2

        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_pre_size"] = df[col].copy()
            result = pd.Series(np.nan, index=df.index)
            for date, idx in df.groupby("date").groups.items():
                subset = df.loc[idx]
                y = subset[col].dropna()
                if len(y) < 10:
                    continue
                X = pd.DataFrame({
                    "log_mcap": log_mcap.loc[y.index],
                    "log_mcap2": log_mcap2.loc[y.index],
                }).dropna()
                common = X.index.intersection(y.index)
                if len(common) < 10:
                    continue
                try:
                    beta = np.linalg.lstsq(
                        np.column_stack([np.ones(len(common)),
                                         X.loc[common, "log_mcap"].values,
                                         X.loc[common, "log_mcap2"].values]),
                        y.loc[common].values,
                        rcond=None,
                    )[0]
                    pred = (beta[0] + beta[1] * log_mcap.loc[common] +
                            beta[2] * log_mcap2.loc[common])
                    result.loc[common] = y.loc[common].values - pred.values
                except np.linalg.LinAlgError:
                    logger.warning(
                        "OLS failed for %s on %s (%d stocks), leaving NaN",
                        col, date, len(common),
                    )
            df[col] = result.astype(np.float32)
        return df

    @staticmethod
    def _adaptive_strength(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """In high-volatility regimes, strengthen neutralization.

        alpha = alpha_0 * (1 + beta * (sigma_short - sigma_long) / sigma_long)
        """
        if "close" not in df.columns:
            return df
        if "stock_code" in df.columns:
            returns = df.groupby("stock_code")["close"].pct_change()
        else:
            returns = df["close"].pct_change()
        sigma_short = returns.rolling(20, min_periods=10).std()
        sigma_long = returns.rolling(60, min_periods=30).std()
        rel_vol = (sigma_short - sigma_long) / sigma_long.replace(0, 1.0)
        alpha = 1.0 + 0.5 * rel_vol.clip(-0.5, 1.0)

        for col in cols:
            if col not in df.columns:
                continue
            df[col] = (df[col].values * alpha.values).astype(np.float32)

        return df
