"""Storage for sector ETF flow data.

Partitions: data/a_shares/etf_flow/{year}/{month}/sector_{name}.parquet
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class ETFStorage:
    """Save/load sector ETF flow data."""

    def __init__(self, data_dir: str):
        self._root = data_dir

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "etf_flow")
        os.makedirs(p, exist_ok=True)
        return p

    def save(self, df: pd.DataFrame) -> None:
        """Save sector flow data partitioned by year/month/sector."""
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        base = self._base_dir()
        for (year, month, sector), group in df.groupby(
            ["year", "month", "sector_name"]
        ):
            out_dir = os.path.join(base, str(year), f"{month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"sector_{sector}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_sector_flow(
        self, sector_name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load sector flow data for a date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = self._base_dir()

        if not os.path.exists(base):
            return pd.DataFrame()

        frames = []
        target = f"sector_{sector_name}.parquet"
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == target:
                    df = pd.read_parquet(os.path.join(root, f))
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        mask = (df["date"] >= start) & (df["date"] <= end)
                        frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
