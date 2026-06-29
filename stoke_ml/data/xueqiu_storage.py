"""3-layer medallion storage for Xueqiu forum post sentiment data.

Bronze: data/a_shares/xueqiu_raw/{stock_code}.parquet   — raw as-fetched
Silver: data/a_shares/xueqiu_silver/{stock_code}.parquet — PIT-aligned, deduped
Gold:   data/a_shares/xueqiu_sentiment/{year}/{month}/{stock_code}.parquet — daily
"""

import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

XUEQIU_COLS = [
    "xueqiu_sentiment_mean", "xueqiu_sentiment_std", "xueqiu_post_count",
    "xueqiu_positive_ratio", "xueqiu_negative_ratio", "has_xueqiu_post",
]


class XueqiuStorage:
    """3-layer Parquet storage for Xueqiu forum posts and daily sentiment."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")
        os.makedirs(data_dir, exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────

    def _raw_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "xueqiu_raw")
        os.makedirs(p, exist_ok=True)
        return p

    def _silver_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "xueqiu_silver")
        os.makedirs(p, exist_ok=True)
        return p

    def _sentiment_base(self) -> str:
        p = os.path.join(self._root, "a_shares", "xueqiu_sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    # ── Bronze: raw posts ─────────────────────────────────────────

    def save_raw(self, stock_code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        existing = self.load_raw(stock_code)
        combined = pd.concat([df, existing], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])

        if "body" in combined.columns:
            combined["_body_len"] = combined["body"].str.len().fillna(0)
            sort_cols = ["_body_len"]
            if "sentiment_title" in combined.columns:
                combined["_has_sent"] = combined["sentiment_title"].notna().astype(int)
                sort_cols.append("_has_sent")
            combined = combined.sort_values(sort_cols, ascending=False)
            combined = combined.drop_duplicates(subset=["url"])
            drop_cols = ["_body_len"]
            if "_has_sent" in combined.columns:
                drop_cols.append("_has_sent")
            combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
        else:
            combined = combined.drop_duplicates(subset=["url"])

        combined = combined.sort_values("date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_raw(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_stocks_with_raw(self) -> list[str]:
        d = self._raw_dir()
        if not os.path.exists(d):
            return []
        return sorted(
            f.replace(".parquet", "")
            for f in os.listdir(d)
            if f.endswith(".parquet")
        )

    # ── Silver: PIT-aligned ────────────────────────────────────────

    def save_silver(self, stock_code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        existing = self.load_silver(stock_code)
        combined = pd.concat([df, existing], ignore_index=True)
        combined["aligned_date"] = pd.to_datetime(combined["aligned_date"])
        combined["date"] = pd.to_datetime(combined["date"])
        if "sentiment_title" in combined.columns:
            combined["_has_sent"] = combined["sentiment_title"].notna().astype(int)
            combined = combined.sort_values("_has_sent", ascending=False)
            combined = combined.drop_duplicates(subset=["url"])
            combined = combined.drop(columns=["_has_sent"])
        else:
            combined = combined.drop_duplicates(subset=["url"])
        combined = combined.sort_values("aligned_date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_silver(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def bronze_to_silver(self, stock_code: str) -> pd.DataFrame:
        """PIT-align raw posts: post-15:00 CST -> next trading day."""
        raw = self.load_raw(stock_code)
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["aligned_date"] = df["date"]
        df["aligned_date"] = pd.to_datetime(df["aligned_date"])
        return df

    # ── Gold: daily sentiment ──────────────────────────────────────

    def save_daily_sentiment(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(self._sentiment_base(), str(year), f"{month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily_sentiment(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._sentiment_base()
        if not os.path.exists(base):
            return pd.DataFrame()

        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            return df[mask].sort_values("date").reset_index(drop=True)

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
        """Aggregate silver posts to daily Xueqiu sentiment features."""
        silver = self.load_silver(stock_code)
        if silver.empty:
            return pd.DataFrame()

        if "sentiment_title" not in silver.columns:
            if analyzer is not None:
                from stoke_ml.features.news_nlp import compute_raw_sentiment
                silver = compute_raw_sentiment(silver, analyzer)
            else:
                silver["sentiment_title"] = 0.0

        silver["aligned_date"] = pd.to_datetime(silver["aligned_date"])
        daily = (
            silver.groupby("aligned_date")
            .agg(
                xueqiu_sentiment_mean=("sentiment_title", "mean"),
                xueqiu_sentiment_std=("sentiment_title", lambda x: x.std() if len(x) > 1 else 0.0),
                xueqiu_post_count=("sentiment_title", "count"),
                xueqiu_positive_ratio=("sentiment_title", lambda x: (x > 0.2).sum() / len(x)),
                xueqiu_negative_ratio=("sentiment_title", lambda x: (x < -0.2).sum() / len(x)),
            )
            .reset_index()
        )

        daily.rename(columns={"aligned_date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["stock_code"] = stock_code
        daily["has_xueqiu_post"] = True
        daily["xueqiu_sentiment_mean"] = daily["xueqiu_sentiment_mean"].astype(np.float32)
        daily["xueqiu_sentiment_std"] = daily["xueqiu_sentiment_std"].astype(np.float32)
        daily["xueqiu_post_count"] = daily["xueqiu_post_count"].astype("int16")
        daily["xueqiu_positive_ratio"] = daily["xueqiu_positive_ratio"].astype(np.float32)
        daily["xueqiu_negative_ratio"] = daily["xueqiu_negative_ratio"].astype(np.float32)

        if len(daily) >= 2:
            all_dates = self._calendar.get_trading_days(
                daily["date"].min(), daily["date"].max()
            )
            date_df = pd.DataFrame({"date": all_dates})
            date_df["date"] = pd.to_datetime(date_df["date"]).dt.date
            daily = date_df.merge(daily, on="date", how="left")
            daily["stock_code"] = stock_code
            daily["has_xueqiu_post"] = daily["has_xueqiu_post"].fillna(False)
            for col in [
                "xueqiu_sentiment_mean", "xueqiu_sentiment_std",
                "xueqiu_positive_ratio", "xueqiu_negative_ratio",
            ]:
                daily[col] = daily[col].fillna(0.0).astype(np.float32)
            daily["xueqiu_post_count"] = daily["xueqiu_post_count"].fillna(0).astype("int16")

        cols = [
            "date", "stock_code", "xueqiu_sentiment_mean", "xueqiu_sentiment_std",
            "xueqiu_post_count", "xueqiu_positive_ratio", "xueqiu_negative_ratio",
            "has_xueqiu_post",
        ]
        return daily[cols]
