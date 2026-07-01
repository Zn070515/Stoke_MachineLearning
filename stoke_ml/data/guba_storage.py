"""3-layer medallion storage for Guba forum post sentiment data.

Bronze: data/a_shares/guba_raw/{stock_code}.parquet   — raw as-fetched
Silver: data/a_shares/guba_silver/{stock_code}.parquet — PIT-aligned, deduped
Gold:   data/a_shares/guba_sentiment/{year}/{month}/{stock_code}.parquet — daily aggregates
"""

import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar

logger = logging.getLogger(__name__)

GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]


class GubaStorage:
    """3-layer Parquet storage for Guba forum posts and daily sentiment."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        self._calendar = calendar or TradingCalendar("a_shares")
        os.makedirs(data_dir, exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────

    def _raw_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_raw")
        os.makedirs(p, exist_ok=True)
        return p

    def _silver_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_silver")
        os.makedirs(p, exist_ok=True)
        return p

    def _sentiment_base(self) -> str:
        p = os.path.join(self._root, "a_shares", "guba_sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    # ── Bronze: raw posts ─────────────────────────────────────────

    def save_raw(self, stock_code: str, df: pd.DataFrame) -> None:
        """Save raw Guba posts for a stock. Appends if file already exists.

        Deduplication by post_id (NOT title+date since forum posts can
        share the same title).
        """
        if df.empty:
            return
        path = os.path.join(self._raw_dir(), f"{stock_code}.parquet")
        existing = self.load_raw(stock_code)
        # New data first + prefer rows with sentiment when body lengths tie.
        combined = pd.concat([df, existing], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])

        # Deduplicate by post_id — keep the row with the longest body,
        # breaking ties in favour of rows that have sentiment data.
        if "body" in combined.columns:
            sort_keys = ["_body_len"]
            combined["_body_len"] = combined["body"].str.len().fillna(0)
            if "sentiment_title" in combined.columns:
                combined["_has_sent"] = combined["sentiment_title"].notna().astype(int)
                sort_keys.append("_has_sent")
            combined = combined.sort_values(sort_keys, ascending=False)
            combined = combined.drop_duplicates(subset=["post_id"])
            combined = combined.drop(
                columns=[c for c in ["_body_len", "_has_sent"] if c in combined.columns]
            )
        else:
            combined = combined.drop_duplicates(subset=["post_id"])

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
        # New data first: drop_duplicates keeps first occurrence, so
        # new rows with updated sentiment win over stale existing rows.
        combined = pd.concat([df, existing], ignore_index=True)
        combined["aligned_date"] = pd.to_datetime(combined["aligned_date"])
        combined["date"] = pd.to_datetime(combined["date"])
        # Prefer rows with sentiment data (non-NaN sentiment_title)
        if "sentiment_title" in combined.columns:
            combined["_has_sent"] = combined["sentiment_title"].notna().astype(int)
            combined = combined.sort_values("_has_sent", ascending=False)
            combined = combined.drop_duplicates(subset=["post_id"])
            combined = combined.drop(columns=["_has_sent"])
        else:
            combined = combined.drop_duplicates(subset=["post_id"])
        combined = combined.sort_values("aligned_date", ascending=False)
        combined.to_parquet(path, index=False)

    def load_silver(self, stock_code: str) -> pd.DataFrame:
        path = os.path.join(self._silver_dir(), f"{stock_code}.parquet")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def bronze_to_silver(self, stock_code: str) -> pd.DataFrame:
        """PIT-align raw posts: post-15:00 CST -> next trading day.

        A-shares close at 15:00 CST. Posts published after 15:00 are
        bumped to the next trading day via TradingCalendar.next_trading_day()
        to prevent look-ahead bias.
        """
        raw = self.load_raw(stock_code)
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])

        if "time" in df.columns:
            # Build datetime from date + time columns
            df["datetime_str"] = (
                df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str)
            )
            df["datetime"] = pd.to_datetime(df["datetime_str"], errors="coerce")

            cutoff = pd.Timestamp("15:00:00").time()
            df["aligned_date"] = df["date"]  # default: same day

            post_close = df["datetime"].dt.time > cutoff
            for idx in df[post_close].index:
                d = df.at[idx, "date"].date()
                df.at[idx, "aligned_date"] = pd.Timestamp(
                    self._calendar.next_trading_day(d)
                )

            df["aligned_date"] = pd.to_datetime(df["aligned_date"])
            return df.drop(columns=["datetime_str", "datetime"])
        else:
            # No time column — fall back to same-day alignment
            df["aligned_date"] = df["date"]
            df["aligned_date"] = pd.to_datetime(df["aligned_date"])
            return df

    # ── Gold: daily sentiment ──────────────────────────────────────

    def save_daily_sentiment(self, df: pd.DataFrame) -> None:
        """Save daily sentiment partitioned by year/month/stock_code.

        Expects columns: date, stock_code, guba_sentiment_mean,
        guba_sentiment_std, guba_post_count, guba_positive_ratio,
        guba_negative_ratio, has_guba_post.
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
        """Load daily sentiment for a stock in a date range.

        Prefers consolidated flat file; falls back to year/month partitions.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._sentiment_base()
        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer consolidated flat file: guba_sentiment/{code}.parquet
        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            return df[mask].sort_values("date").reset_index(drop=True)

        # Fallback: partitioned guba_sentiment/{year}/{month}/{code}.parquet
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
        preprocessing_pipeline: object | None = None,
    ) -> pd.DataFrame:
        """Aggregate silver posts to daily Guba sentiment features.

        Uses ZI method: days without posts get zero-filled sentiment
        plus a has_guba_post=False flag.

        If *preprocessing_pipeline* is provided, runs the new text chain
        instead of the legacy simple mean/std aggregation.
        """
        silver = self.load_silver(stock_code)
        if silver.empty:
            return pd.DataFrame()

        # If no sentiment scores yet, compute them from titles/bodies
        if "sentiment_title" not in silver.columns:
            if analyzer is not None:
                from stoke_ml.features.news_nlp import compute_raw_sentiment
                silver = compute_raw_sentiment(silver, analyzer)
            else:
                silver["sentiment_title"] = 0.0

        if preprocessing_pipeline is not None:
            return self._silver_to_gold_new(silver, stock_code, preprocessing_pipeline)

        return self._silver_to_gold_legacy(silver, stock_code)

    def _silver_to_gold_new(
        self,
        silver: pd.DataFrame,
        stock_code: str,
        pp: object,
    ) -> pd.DataFrame:
        """New path: text preprocessing chain → daily aggregation."""
        silver = pp.run("text_pre", silver)

        tm = pp.topic_modeler
        if tm is not None and tm._enabled:
            silver = tm.transform(silver)

        gold = pp.run("text_aggregate", silver)
        gold["stock_code"] = stock_code

        # Map generic DailyAggregator column names to guba-specific legacy names
        _rename = {
            "sent_mean": "guba_sentiment_mean",
            "sent_std": "guba_sentiment_std",
            "post_count": "guba_post_count",
            "bull_ratio": "guba_positive_ratio",
            "bear_ratio": "guba_negative_ratio",
        }
        gold = gold.rename(
            columns={k: v for k, v in _rename.items() if k in gold.columns}
        )

        # Mark real-data rows before ZI merge (mirrors legacy path)
        gold["has_guba_post"] = True

        # ZI fill missing trading days — discover columns dynamically
        numeric_cols = [
            c for c in gold.columns
            if c not in ("date", "stock_code")
            and not c.startswith("has_")
            and not c.startswith("topic_")
            and gold[c].dtype in ("float32", "float64", "int16", "int32", "int64")
        ]
        bool_cols = [c for c in gold.columns if c.startswith("has_")]

        if len(gold) >= 2:
            all_dates = self._calendar.get_trading_days(
                gold["date"].min(), gold["date"].max()
            )
            date_df = pd.DataFrame({"date": all_dates})
            date_df["date"] = pd.to_datetime(date_df["date"]).dt.date
            gold["date"] = pd.to_datetime(gold["date"]).dt.date
            gold = date_df.merge(gold, on="date", how="left")
            gold["stock_code"] = stock_code

        # NaN-fill always runs (not gated by len >= 2)
        for col in bool_cols:
            if col in gold.columns:
                gold[col] = gold[col].fillna(False).astype(bool)
        for col in numeric_cols:
            if col in gold.columns:
                gold[col] = gold[col].fillna(0.0).astype(np.float32)

        # Topic columns need sentinel-aware ZI fill
        for col in [c for c in gold.columns if c.startswith("topic_")]:
            if col == "topic_dominant":
                gold[col] = gold[col].fillna(-1).astype("int16")
            else:
                gold[col] = gold[col].fillna(0.0).astype(np.float32)

        # Ensure standard columns exist
        for col in ("guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
                     "guba_positive_ratio", "guba_negative_ratio", "has_guba_post"):
            if col not in gold.columns:
                if col == "has_guba_post":
                    gold[col] = False
                elif col == "guba_post_count":
                    gold[col] = np.int16(0)
                else:
                    gold[col] = np.float32(0.0)

        return gold

    def _silver_to_gold_legacy(
        self, silver: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """Legacy path: simple mean/std aggregation (backward compatible)."""
        silver["aligned_date"] = pd.to_datetime(silver["aligned_date"])
        daily = (
            silver.groupby("aligned_date")
            .agg(
                guba_sentiment_mean=("sentiment_title", "mean"),
                guba_sentiment_std=("sentiment_title", lambda x: x.std() if len(x) > 1 else 0.0),
                guba_post_count=("sentiment_title", "count"),
                guba_positive_ratio=("sentiment_title", lambda x: (x > 0.2).sum() / len(x)),
                guba_negative_ratio=("sentiment_title", lambda x: (x < -0.2).sum() / len(x)),
            )
            .reset_index()
        )

        daily.rename(columns={"aligned_date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["stock_code"] = stock_code
        daily["has_guba_post"] = True
        daily["guba_sentiment_mean"] = daily["guba_sentiment_mean"].astype(np.float32)
        daily["guba_sentiment_std"] = daily["guba_sentiment_std"].astype(np.float32)
        daily["guba_post_count"] = daily["guba_post_count"].astype("int16")
        daily["guba_positive_ratio"] = daily["guba_positive_ratio"].astype(np.float32)
        daily["guba_negative_ratio"] = daily["guba_negative_ratio"].astype(np.float32)

        # Fill missing trading days with zeros (ZI method)
        if len(daily) >= 2:
            all_dates = self._calendar.get_trading_days(
                daily["date"].min(), daily["date"].max()
            )
            date_df = pd.DataFrame({"date": all_dates})
            date_df["date"] = pd.to_datetime(date_df["date"]).dt.date
            daily = date_df.merge(daily, on="date", how="left")
            daily["stock_code"] = stock_code
            daily["has_guba_post"] = daily["has_guba_post"].fillna(False)
            for col in [
                "guba_sentiment_mean", "guba_sentiment_std",
                "guba_positive_ratio", "guba_negative_ratio",
            ]:
                daily[col] = daily[col].fillna(0.0).astype(np.float32)
            daily["guba_post_count"] = daily["guba_post_count"].fillna(0).astype("int16")

        cols = [
            "date", "stock_code", "guba_sentiment_mean", "guba_sentiment_std",
            "guba_post_count", "guba_positive_ratio", "guba_negative_ratio",
            "has_guba_post",
        ]
        return daily[cols]
