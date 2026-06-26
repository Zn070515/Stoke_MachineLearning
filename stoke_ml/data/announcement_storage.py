"""Storage for company announcements with daily sentiment aggregation.

Follows the same pattern as NewsStorage: raw Parquet per stock,
PIT alignment (post-15:00 CST → next trading day), daily aggregation.
"""
import logging
import os

import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

_COLS = ["sentiment_mean", "sentiment_std", "announce_count",
         "positive_ratio", "negative_ratio", "has_announce"]


class AnnouncementStorage:
    """Read/write announcement data partitioned by stock code."""

    def __init__(self, root_dir: str):
        self._root = root_dir
        self._base = os.path.join(root_dir, "a_shares", "announcements")
        os.makedirs(self._base, exist_ok=True)
        self._calendar = TradingCalendar()

    def save_raw(self, stock_code: str, df: pd.DataFrame) -> str:
        """Save raw announcements to {code}.parquet."""
        path = os.path.join(self._base, f"{stock_code}.parquet")
        df.to_parquet(path, index=False)
        return path

    def load_raw(self, stock_code: str) -> pd.DataFrame:
        """Load raw announcements for a stock."""
        path = os.path.join(self._base, f"{stock_code}.parquet")
        if not os.path.isfile(path):
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def build_daily_sentiment(
        self, stock_code: str, sentiment_col: str = "sentiment_title",
        save: bool = True,
    ) -> pd.DataFrame:
        """Compute daily sentiment aggregation from raw announcements.

        Returns DataFrame with date + SENTIMENT_COLS, saved to
        announcement_sentiment/{code}.parquet.
        """
        df = self.load_raw(stock_code)
        if df.empty or sentiment_col not in df.columns:
            return pd.DataFrame(columns=["date"] + _COLS)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["sentiment"] = pd.to_numeric(df[sentiment_col], errors="coerce").fillna(0)

        daily = df.groupby("date").agg(
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", lambda x: x.std() if len(x) > 1 else 0.0),
            announce_count=("sentiment", "count"),
            positive=("sentiment", lambda x: (x > 0.05).sum()),
            negative=("sentiment", lambda x: (x < -0.05).sum()),
        ).reset_index()

        daily["positive_ratio"] = daily["positive"] / daily["announce_count"]
        daily["negative_ratio"] = daily["negative"] / daily["announce_count"]
        daily["has_announce"] = daily["announce_count"] > 0
        daily = daily.drop(columns=["positive", "negative"])
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
        daily = daily.sort_values("date").reset_index(drop=True)

        if save:
            out_dir = os.path.join(self._base, "sentiment")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stock_code}.parquet")
            daily.to_parquet(out_path, index=False)

        return daily

    def load_daily_sentiment(
        self, stock_code: str, start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load precomputed daily announcement sentiment."""
        path = os.path.join(self._base, "sentiment", f"{stock_code}.parquet")
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["date"] + _COLS)

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def sentiment_columns() -> list[str]:
        return list(_COLS)
