"""Storage for company announcements with daily sentiment aggregation.

Follows the same pattern as NewsStorage: raw Parquet per stock,
PIT alignment (post-15:00 CST → next trading day), daily aggregation.
"""
import logging
import os

import pandas as pd

from stoke_ml.data.asset_contract import (
    AtomicCommit,
    DataAssetContract,
    check_asset_read,
    downloader_era_fields,
    write_asset_manifest,
)
from stoke_ml.data.calendar import get_research_calendar
from stoke_ml.data.date_normalize import as_date_us

logger = logging.getLogger(__name__)

ANNOUNCEMENT_ASSET = DataAssetContract(
    data_type="announcements",
    partition="stock_code",
    extent_column="date",
)

ANNOUNCEMENT_SENTIMENT_ASSET = DataAssetContract(
    data_type="announcement_sentiment",
    partition="stock_code",
    extent_column="date",
)

_COLS = ["sentiment_mean", "sentiment_std", "announce_count",
         "positive_ratio", "negative_ratio", "has_announce"]


class AnnouncementStorage:
    """Read/write announcement data partitioned by stock code."""

    def __init__(self, root_dir: str):
        self._root = root_dir
        self._base = os.path.join(root_dir, "a_shares", "announcements")
        os.makedirs(self._base, exist_ok=True)
        # Artifact-backed calendar from this storage's own data root.
        self._calendar = get_research_calendar(data_dir=self._root)

    def _manifest_dir(self) -> str:
        """The downloader's per-stock manifest dir for the ``announcement``
        channel.

        download_announcements.py writes ``mark_stock_result(a_shares/
        announcements, ...)`` — today the SAME dir raw announcements live in.
        The coupling to the downloader's output dir is made EXPLICIT here so
        ``downloader_era_fields`` reads the right per-stock manifest even if the
        raw/gold layout later diverges — change it in BOTH the downloader and
        here.
        """
        p = os.path.join(self._root, "a_shares", "announcements")
        os.makedirs(p, exist_ok=True)
        return p

    def save_raw(self, stock_code: str, df: pd.DataFrame) -> str:
        """Save raw announcements to {code}.parquet."""
        path = os.path.join(self._base, f"{stock_code}.parquet")
        with AtomicCommit(path) as ac:
            df.to_parquet(ac.tmp_path, index=False, compression='lz4')
        # §T8: stamp the raw manifest with the downloader manifest's
        # provider-era fields (read via _manifest_dir, the explicit coupling to
        # download_announcements.py).  `**{}` when there is no downloader
        # manifest → the stock is not_observed.
        write_asset_manifest(
            path, ANNOUNCEMENT_ASSET, df, entity=stock_code,
            **downloader_era_fields(self._manifest_dir(), stock_code),
        )
        return path

    def load_raw(
        self, stock_code: str, *, require_valid_manifest: bool = False,
    ) -> pd.DataFrame:
        """Load raw announcements for a stock.

        The file is cross-checked against its asset manifest
        (``check_asset_read``); pass ``require_valid_manifest=True`` to raise
        instead of read when the manifest is missing or mismatched.
        """
        path = os.path.join(self._base, f"{stock_code}.parquet")
        if not os.path.isfile(path):
            return pd.DataFrame()
        df = pd.read_parquet(path)
        check_asset_read(path, ANNOUNCEMENT_ASSET, df,
                         require_valid_manifest=require_valid_manifest)
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
            with AtomicCommit(out_path) as ac:
                daily.to_parquet(ac.tmp_path, index=False, compression='lz4')
            write_asset_manifest(
                out_path, ANNOUNCEMENT_SENTIMENT_ASSET, daily,
                entity=stock_code,
                **downloader_era_fields(self._manifest_dir(), stock_code),
            )

        return daily

    def load_daily_sentiment(
        self, stock_code: str, start_date: str | None = None,
        end_date: str | None = None,
        *, require_valid_manifest: bool = False,
    ) -> pd.DataFrame:
        """Load precomputed daily announcement sentiment.

        The file is cross-checked against its asset manifest
        (``check_asset_read``); pass ``require_valid_manifest=True`` to raise
        instead of read when the manifest is missing or mismatched.
        """
        path = os.path.join(self._base, "sentiment", f"{stock_code}.parquet")
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["date"] + _COLS)

        df = pd.read_parquet(path)
        check_asset_read(path, ANNOUNCEMENT_SENTIMENT_ASSET, df,
                         require_valid_manifest=require_valid_manifest)
        # §v19: canonical datetime64[us] coercion (ms/us mixed on disk).
        df = as_date_us(df)
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def sentiment_columns() -> list[str]:
        return list(_COLS)
