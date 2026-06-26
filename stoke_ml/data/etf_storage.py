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
        """Load sector flow data for a date range.

        Prefers consolidated flat file; falls back to year/month partitions.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = self._base_dir()

        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer consolidated flat file: etf_flow/sector_{name}.parquet
        flat_path = os.path.join(base, f"sector_{sector_name}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= start) & (df["date"] <= end)
                return df[mask].sort_values("date").reset_index(drop=True)
            return df

        # Fallback: partitioned etf_flow/{year}/{month}/sector_{name}.parquet
        target = f"sector_{sector_name}.parquet"
        frames = []
        for year in range(start.year, end.year + 1):
            year_dir = os.path.join(base, str(year))
            if not os.path.isdir(year_dir):
                continue
            for month in range(1, 13):
                if year == start.year and month < start.month:
                    continue
                if year == end.year and month > end.month:
                    continue
                file_path = os.path.join(year_dir, f"{month:02d}", target)
                if not os.path.exists(file_path):
                    continue
                df = pd.read_parquet(file_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
