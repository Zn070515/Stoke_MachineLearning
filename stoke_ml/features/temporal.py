"""Temporal features: lags, rolling windows, calendar features."""
import pandas as pd
import numpy as np


def add_lag_features(
    df: pd.DataFrame, cols: list[str], lags: list[int]
) -> pd.DataFrame:
    result = df.copy()
    new_cols = {}
    for col in cols:
        if col not in result.columns:
            continue
        for lag in lags:
            new_cols[f"{col}_lag{lag}"] = result[col].shift(lag)
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
    return result


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: list[int]
) -> pd.DataFrame:
    result = df.copy()
    new_cols = {}
    for col in cols:
        if col not in result.columns:
            continue
        for w in windows:
            new_cols[f"{col}_roll{w}_mean"] = result[col].rolling(w).mean()
            new_cols[f"{col}_roll{w}_std"] = result[col].rolling(w).std()
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result["date"])
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    return result
