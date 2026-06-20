"""Storage for quarterly fundamental data with forward-fill to daily.

Partitions: data/a_shares/fundamentals/{year}/{quarter}/{stock_code}.parquet
"""
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)


class FundamentalStorage:
    """Save/load quarterly fundamental data, forward-fill to daily."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "fundamentals")
        os.makedirs(p, exist_ok=True)
        return p

    def save(self, df: pd.DataFrame) -> None:
        """Save fundamental data partitioned by year/quarter/stock_code."""
        if df.empty:
            return
        df = df.copy()
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["year"] = df["report_date"].dt.year
        df["quarter"] = df["report_date"].dt.quarter

        base = self._base_dir()
        for (year, quarter, code), group in df.groupby(
            ["year", "quarter", "stock_code"]
        ):
            out_dir = os.path.join(base, str(year), f"Q{quarter}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "quarter"])
            save_df.to_parquet(out_path, index=False)

    def load(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load fundamental data for a stock in a date range.

        Returns raw quarterly data (no forward-fill). Uses direct
        path lookup by year/quarter partition.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = self._base_dir()

        if not os.path.exists(base):
            return pd.DataFrame()

        frames = []
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        for year in range(start.year, end.year + 1):
            for q in quarters:
                file_path = os.path.join(base, str(year), q,
                                         f"{stock_code}.parquet")
                if not os.path.exists(file_path):
                    continue
                df = pd.read_parquet(file_path)
                if "report_date" not in df.columns:
                    continue
                df["report_date"] = pd.to_datetime(df["report_date"])
                if "disclose_date" in df.columns:
                    df["disclose_date"] = pd.to_datetime(df["disclose_date"])
                mask = (df["report_date"] >= start) & (df["report_date"] <= end)
                frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            "report_date"
        ).reset_index(drop=True)

    def forward_fill_to_daily(
        self, stock_code: str, start_date: str, end_date: str,
        max_gap_days: int = 90,
    ) -> pd.DataFrame:
        """Load fundamentals and forward-fill to daily trading calendar.

        Uses disclose_date as the forward-fill start to prevent lookahead bias.
        Caps stale data at max_gap_days.
        """
        raw = self.load(stock_code, "2010-01-01", end_date)
        if raw.empty:
            return pd.DataFrame()

        # Get trading day range
        trading_days = self._calendar.get_trading_days(start_date, end_date)
        daily_df = pd.DataFrame({"date": trading_days})
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        # Use disclose_date for forward-fill origin (not report_date)
        fill_col = "disclose_date" if "disclose_date" in raw.columns else "report_date"
        raw["_fill_from"] = pd.to_datetime(raw[fill_col])

        # Get value columns (exclude metadata)
        value_cols = [
            c for c in raw.columns
            if c not in ("stock_code", "report_date", "disclose_date", "_fill_from")
        ]

        # Build daily forward-filled values
        result = daily_df.copy()
        result["stock_code"] = str(stock_code).zfill(6)

        for col in value_cols:
            result[col] = np.nan
            raw_sorted = raw.dropna(subset=[col]).sort_values("_fill_from")

            for _, row in raw_sorted.iterrows():
                fill_date = row["_fill_from"]
                val = row[col]
                mask = result["date"] >= fill_date
                # Cap staleness: set to NaN if too far from fill date
                if max_gap_days > 0:
                    stale = (result["date"] - fill_date).dt.days > max_gap_days
                    mask = mask & ~stale
                result.loc[mask, col] = val

            result[col] = result[col].ffill()

        return result.reset_index(drop=True)
