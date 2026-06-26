"""Storage for AKShare market comment sentiment data.

Simpler than Guba/News storage — data is already numerical, no NLP pipeline needed.
Gold layer: partitioned daily snapshots + flat consolidated files.
"""
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

COMMENT_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend",
]

COMMENT_FEATURE_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend", "has_comment",
]


class CommentStorage:
    """Read/write market comment sentiment partitioned by stock code."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")
        os.makedirs(data_dir, exist_ok=True)

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "comment_sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    def save_snapshot(self, df: pd.DataFrame) -> None:
        """Save a full-market snapshot (from stock_comment_em)."""
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        date_str = df["date"].iloc[0].strftime("%Y-%m-%d")
        path = os.path.join(self._base_dir(), f"snapshot_{date_str}.parquet")
        cols = ["date", "stock_code"] + [
            c for c in COMMENT_COLS if c in df.columns
        ]
        df[cols].to_parquet(path, index=False)
        logger.info("Saved comment snapshot: %d stocks (%s)", len(df), date_str)

    def load_latest_snapshot(self) -> pd.DataFrame:
        """Load the most recent full-market snapshot."""
        base = self._base_dir()
        snapshots = sorted(
            [f for f in os.listdir(base) if f.startswith("snapshot_")],
            reverse=True,
        )
        if not snapshots:
            return pd.DataFrame()
        return pd.read_parquet(os.path.join(base, snapshots[0]))

    def save_daily(self, df: pd.DataFrame) -> None:
        """Save per-stock daily comment data (partitioned by year/month/stock)."""
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(self._base_dir(), str(year), f"{month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load daily comment data for a stock in a date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._base_dir()
        if not os.path.exists(base):
            return pd.DataFrame()

        frames = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values("date").reset_index(drop=True)

    def build_features(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Build COMMENT_FEATURE_COLS for FeaturePipeline merge.

        ZI method: trading days without comment data get zeros + has_comment=False.
        """
        daily = self.load_daily(stock_code, start_date, end_date)
        all_dates = self._calendar.get_trading_days(start_date, end_date)

        if daily.empty:
            df = pd.DataFrame({"date": all_dates})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["stock_code"] = stock_code
            for col in COMMENT_COLS:
                df[col] = 0.0
            df["has_comment"] = False
            return df[["date"] + COMMENT_FEATURE_COLS]

        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        date_df = pd.DataFrame({"date": all_dates})
        date_df["date"] = pd.to_datetime(date_df["date"]).dt.date

        merged = date_df.merge(daily, on="date", how="left")
        merged["stock_code"] = stock_code
        merged["has_comment"] = merged["comment_score"].notna()

        for col in COMMENT_COLS:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0).astype(np.float32)
            else:
                merged[col] = 0.0
        merged["has_comment"] = merged["has_comment"].fillna(False)

        return merged[["date"] + COMMENT_FEATURE_COLS].reset_index(drop=True)
