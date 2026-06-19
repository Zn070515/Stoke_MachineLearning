"""3-layer medallion storage for news and sentiment data.

Bronze: data/a_shares/news_raw/{stock_code}.parquet   — raw as-fetched
Silver: data/a_shares/news_silver/{stock_code}.parquet — PIT-aligned, deduped
Gold:   data/a_shares/sentiment/{year}/{month}/{stock_code}.parquet — daily aggregates
"""

import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)


class NewsStorage:
    """3-layer Parquet storage for financial news and daily sentiment."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")
        os.makedirs(data_dir, exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────

    def _raw_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "news_raw")
        os.makedirs(p, exist_ok=True)
        return p

    def _silver_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "news_silver")
        os.makedirs(p, exist_ok=True)
        return p

    def _sentiment_base(self) -> str:
        p = os.path.join(self._root, "a_shares", "sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    # ── Bronze: raw news ───────────────────────────────────────────

    def save_raw_news(self, stock_code: str, df: pd.DataFrame) -> None:
        """Save raw news for a stock. Appends if file already exists."""
        if df.empty:
            return
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        existing = self.load_raw_news(stock_code)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["title", "date"])
        combined = combined.sort_values("date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_raw_news(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_stocks_with_raw_news(self) -> list[str]:
        d = self._raw_dir()
        if not os.path.exists(d):
            return []
        return sorted(
            f.replace(".parquet", "")
            for f in os.listdir(d)
            if f.endswith(".parquet")
        )

    # ── Silver: PIT-aligned ────────────────────────────────────────

    def save_silver_news(self, stock_code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        existing = self.load_silver_news(stock_code)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["aligned_date"] = pd.to_datetime(combined["aligned_date"])
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["title", "aligned_date"])
        combined = combined.sort_values("aligned_date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_silver_news(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def bronze_to_silver(self, stock_code: str) -> pd.DataFrame:
        """PIT-align raw news: post-close news → next trading day.

        A-shares close at 15:00 CST. Sina news pages show dates only
        (no timestamps), so we treat all news as same-day.
        If timestamps are available later, news after 15:00 is bumped
        to the next trading day via TradingCalendar.next_trading_day().
        """
        raw = self.load_raw_news(stock_code)
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])

        # If we have a 'time' column, use it for PIT alignment
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str),
                errors="coerce",
            )
            cutoff = pd.Timestamp("15:00:00").time()
            post_close = df["datetime"].dt.time > cutoff
            # Bump post-close news to next trading day
            for idx in df[post_close].index:
                d = df.at[idx, "date"].date()
                df.at[idx, "aligned_date"] = pd.Timestamp(
                    self._calendar.next_trading_day(d)
                )
            # Same-day or before-close stays on original date
            df["aligned_date"] = df["aligned_date"].fillna(df["date"])
        else:
            # No timestamp: treat all as same-day
            df["aligned_date"] = df["date"]

        df["aligned_date"] = pd.to_datetime(df["aligned_date"])
        return df

    # ── Gold: daily sentiment ──────────────────────────────────────

    def save_daily_sentiment(self, df: pd.DataFrame) -> None:
        """Save daily sentiment partitioned by year/month/stock_code.

        Expects columns: date, stock_code, sentiment_mean, sentiment_std,
        news_count, positive_ratio, negative_ratio, has_news.
        """
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(
                self._sentiment_base(), str(year), f"{month:02d}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily_sentiment(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load daily sentiment for a stock in a date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._sentiment_base()
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

    def silver_to_gold(
        self,
        stock_code: str,
        analyzer: object | None = None,
    ) -> pd.DataFrame:
        """Aggregate silver news to daily sentiment features.

        Uses ZI method: days without news get zero-filled sentiment
        plus a has_news=False flag.
        """
        silver = self.load_silver_news(stock_code)
        if silver.empty:
            return pd.DataFrame()

        # If no sentiment scores yet, compute them from titles
        if "sentiment_title" not in silver.columns:
            if analyzer is not None:
                from stoke_ml.features.news_nlp import compute_raw_sentiment
                silver = compute_raw_sentiment(silver, analyzer)
            else:
                silver["sentiment_title"] = 0.0

        # Group by aligned_date
        silver["aligned_date"] = pd.to_datetime(silver["aligned_date"])
        daily = (
            silver.groupby("aligned_date")
            .agg(
                sentiment_mean=("sentiment_title", "mean"),
                sentiment_std=("sentiment_title", lambda x: x.std() if len(x) > 1 else 0.0),
                news_count=("sentiment_title", "count"),
                positive_ratio=("sentiment_title", lambda x: (x > 0.2).sum() / len(x)),
                negative_ratio=("sentiment_title", lambda x: (x < -0.2).sum() / len(x)),
            )
            .reset_index()
        )

        daily.rename(columns={"aligned_date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["stock_code"] = stock_code
        daily["has_news"] = True
        daily["sentiment_mean"] = daily["sentiment_mean"].astype(np.float32)
        daily["sentiment_std"] = daily["sentiment_std"].astype(np.float32)
        daily["news_count"] = daily["news_count"].astype("int16")
        daily["positive_ratio"] = daily["positive_ratio"].astype(np.float32)
        daily["negative_ratio"] = daily["negative_ratio"].astype(np.float32)

        # Fill missing trading days with zeros (ZI method)
        if len(daily) >= 2:
            all_dates = self._calendar.get_trading_days(
                daily["date"].min(), daily["date"].max()
            )
            date_df = pd.DataFrame({"date": all_dates})
            date_df["date"] = pd.to_datetime(date_df["date"]).dt.date
            daily = date_df.merge(daily, on="date", how="left")
            daily["stock_code"] = stock_code
            daily["has_news"] = daily["has_news"].fillna(False)
            for col in [
                "sentiment_mean", "sentiment_std", "positive_ratio", "negative_ratio",
            ]:
                daily[col] = daily[col].fillna(0.0).astype(np.float32)
            daily["news_count"] = daily["news_count"].fillna(0).astype("int16")

        cols = [
            "date", "stock_code", "sentiment_mean", "sentiment_std",
            "news_count", "positive_ratio", "negative_ratio", "has_news",
        ]
        return daily[cols]
