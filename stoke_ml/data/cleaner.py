"""Data cleaner — missing values, outliers, price adjustment validation."""
import pandas as pd


class DataCleaner:
    """Clean raw OHLCV data before storage."""

    def __init__(self, pct_change_limit: float = 11.0):
        self._pct_limit = pct_change_limit

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._fill_missing(df)
        df = self._remove_outliers(df)
        df = self._validate_ohlc(df)
        return df.reset_index(drop=True)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=1)
        for col in ["volume", "amount"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "pct_change" not in df.columns:
            return df
        mask = df["pct_change"].abs() <= self._pct_limit
        return df[mask].copy()

    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in ["high", "low", "open", "close"]):
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)
        return df
