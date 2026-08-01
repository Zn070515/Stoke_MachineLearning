"""Minute K-line data storage — Parquet partitioned by frequency/year/month.

Layout: {data_dir}/a_shares/minute/{frequency}/{year}/{month}/{code}.parquet

Each parquet file contains all bars for one stock in one month at one frequency.
The consolidated flat file at minute/{frequency}/{code}.parquet is also supported
for loading convenience (save routine writes to both).
"""
import os
from typing import Optional

import pandas as pd


class MinuteStorage:
    """Save and load minute K-line data as partitioned Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _base_dir(self, frequency: str, market: str = "a_shares") -> str:
        return os.path.join(self._root, market, "minute", frequency)

    def save(self, df: pd.DataFrame, frequency: str, market: str = "a_shares"):
        """Save minute bars, partitioning by year/month/stock.

        Args:
            df: DataFrame with columns [datetime, open, high, low, close,
                volume, amount, stock_code, bar_period].
            frequency: '5', '15', '30', or '60' (minutes).
        """
        if df.empty:
            return

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.time.astype(str)
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month

        base = self._base_dir(frequency, market)

        for (year, month, code), group in df.groupby(
            ["year", "month", "stock_code"]
        ):
            out_dir = os.path.join(base, str(year), f"{month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")

            save_df = group.drop(columns=["year", "month"])
            # If file exists, merge and deduplicate
            if os.path.isfile(out_path):
                existing = pd.read_parquet(out_path)
                existing["datetime"] = pd.to_datetime(existing["datetime"])
                combined = pd.concat([existing, save_df], ignore_index=True)
                combined.drop_duplicates(
                    subset=["datetime", "stock_code"], keep="last", inplace=True,
                )
                combined.sort_values("datetime", inplace=True)
                combined.to_parquet(out_path, index=False, compression='lz4')
            else:
                save_df.to_parquet(out_path, index=False, compression='lz4')

    def load(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        frequency: str = "60",
        market: str = "a_shares",
    ) -> pd.DataFrame:
        """Load minute bars for a stock within a date range.

        Prefers consolidated flat file, falls back to partitioned scan.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._base_dir(frequency, market)
        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer flat file
        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            mask = (df["datetime"] >= start) & (df["datetime"] <= end)
            return df[mask].sort_values("datetime").reset_index(drop=True)

        # Fallback: scan partitioned directories
        all_data = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
                    if mask.any():
                        all_data.append(df[mask])

        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values("datetime").reset_index(drop=True)

    def list_stocks(self, frequency: str = "60",
                    market: str = "a_shares") -> list[str]:
        """List stock codes that have minute data for a given frequency."""
        base = self._base_dir(frequency, market)
        if not os.path.exists(base):
            return []
        codes = set()
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    codes.add(f.replace(".parquet", ""))
        return sorted(codes)
