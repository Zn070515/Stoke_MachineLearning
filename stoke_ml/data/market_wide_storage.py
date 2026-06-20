"""Storage for market-wide data types (dragon-tiger, margin, northbound).

Partitions: data/a_shares/{data_type}/{year}/{month}/{stock_code}.parquet
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

MARKET_DATA_TYPES = ["dragon_tiger", "margin", "northbound"]


class MarketWideStorage:
    """Save/load market-wide data exploded to per-stock Parquet files."""

    def __init__(self, data_dir: str, data_type: str):
        if data_type not in MARKET_DATA_TYPES:
            raise ValueError(
                f"Unknown market data type: {data_type}. "
                f"Must be one of {MARKET_DATA_TYPES}"
            )
        self._root = data_dir
        self._data_type = data_type

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", self._data_type)
        os.makedirs(p, exist_ok=True)
        return p

    def save(self, df: pd.DataFrame) -> None:
        """Save per-stock market data partitioned by year/month/stock_code.

        Expects columns: date, stock_code, plus type-specific fields.
        """
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        base = self._base_dir()
        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(base, str(year), f"{month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load market data for a single stock in a date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = self._base_dir()

        if not os.path.exists(base):
            return pd.DataFrame()

        frames = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    df = pd.read_parquet(os.path.join(root, f))
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
