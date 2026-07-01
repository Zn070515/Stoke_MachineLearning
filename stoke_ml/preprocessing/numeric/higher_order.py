"""Higher-order derived features from OHLCV data.

Adds: skew, kurtosis, realized volatility surface, Amihud illiquidity,
VWAP deviation, max drawdown, up-days ratio.
"""

import numpy as np
import pandas as pd
from stoke_ml.preprocessing.base import PreprocessingStep


class HigherOrderDeriver(PreprocessingStep):
    """Compute higher-order statistics from price/volume series.

    Only operates on raw OHLCV data, not on already-scaled features.
    All rolling windows are backward-looking (PIT-safe).
    """

    _VOL_WINDOWS = (5, 10, 20, 60)

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        if not self.enabled or df.empty:
            return df.copy()
        df = df.copy()

        if "close" in df.columns:
            close = df["close"]
            returns = close.pct_change()

            # Return distribution moments (20-day rolling)
            df["skew_20d"] = (
                returns.rolling(20, min_periods=10).skew().astype(np.float32)
            )
            df["kurt_20d"] = (
                returns.rolling(20, min_periods=10).kurt().astype(np.float32)
            )

            # Realized volatility surface (annualized approximation not applied)
            for w in self._VOL_WINDOWS:
                vol = returns.rolling(w, min_periods=max(3, w // 3)).std()
                df[f"realized_vol_{w}d"] = vol.astype(np.float32)

            # Max drawdown
            for w in (20, 60):
                roll_max = close.rolling(w, min_periods=w // 2).max()
                dd = (roll_max - close) / roll_max.replace(0, np.nan)
                df[f"max_drawdown_{w}d"] = dd.astype(np.float32)

            # Up-days ratio
            for w in (20,):
                up = (returns > 0).rolling(w, min_periods=w // 2).mean()
                df[f"up_days_ratio_{w}d"] = up.astype(np.float32)

        # Amihud illiquidity: |return| / (price * volume)
        if "close" in df.columns and "volume" in df.columns:
            volume = df["volume"].replace(0, np.nan)
            amihud = np.abs(returns) / (close * volume + 1)
            for w in (20,):
                avg_amihud = amihud.rolling(w, min_periods=w // 2).mean()
                df[f"amihud_illiq_{w}d"] = avg_amihud.astype(np.float32)

        # VWAP deviation
        if all(c in df.columns for c in ("high", "low", "close")):
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            vol = df.get("volume", pd.Series(1.0, index=df.index))
            vwap_num = (typical * vol).rolling(20, min_periods=5).sum()
            vwap_den = vol.rolling(20, min_periods=5).sum().replace(0, np.nan)
            df["vwap_deviation_20d"] = (
                (close - vwap_num / vwap_den) / (vwap_num / vwap_den).replace(0, np.nan)
            ).astype(np.float32)

        return df
