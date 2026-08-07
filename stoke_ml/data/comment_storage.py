"""Storage for AKShare market comment sentiment data.

Simpler than Guba/News storage — data is already numerical, no NLP pipeline needed.
Gold layer: partitioned daily snapshots + flat consolidated files.
"""
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.asset_contract import (
    check_asset_read,
    contract_for_channel,
    downloader_era_fields,
    write_asset_manifest,
)
from stoke_ml.data.calendar import TradingCalendar, get_research_calendar

logger = logging.getLogger(__name__)

COMMENT_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend",
]

# The per-stock daily comment files — the ``comment`` channel's file-level
# asset (§十七).  Flat per-stock files under a_shares/comment_sentiment/.
# (The ``snapshot_{date}.parquet`` full-market staging files are NOT part of
# this contract — ``load_latest_snapshot`` selects them by a ``snapshot_``
# name prefix that a ``.manifest.json`` sidecar would collide with.)
COMMENT_SENTIMENT_ASSET = contract_for_channel(
    "comment",
    data_type="comment_sentiment",
    partition="stock_code",
    extent_column="date",
    effective_date_policy="record_date",
)

COMMENT_FEATURE_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend", "has_comment",
]


class CommentStorage:
    """Read/write market comment sentiment partitioned by stock code."""

    def __init__(self, data_dir: str, calendar: TradingCalendar | None = None):
        self._root = data_dir
        # Artifact-backed calendar from this storage's own data root.
        self._calendar = calendar or get_research_calendar(data_dir=self._root)
        os.makedirs(data_dir, exist_ok=True)

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", "comment_sentiment")
        os.makedirs(p, exist_ok=True)
        return p

    def _manifest_dir(self) -> str:
        """The downloader's per-stock manifest dir for the ``comment`` channel.

        download_comment.py writes ``mark_stock_result(a_shares/comment_sentiment,
        ...)`` — today the SAME dir the gold daily files live in (comments are
        numeric, so gold IS the raw layer: there is no separate bronze).  The
        coupling to the downloader's output dir is made EXPLICIT here so
        ``downloader_era_fields`` reads the right per-stock manifest even if the
        gold layout later diverges — change it in BOTH the downloader and here.
        """
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
        df[cols].to_parquet(path, index=False, compression='lz4')
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
        """Save per-stock daily comment data to flat files."""
        if df.empty:
            return
        import tempfile

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        base = self._base_dir()
        for code, group in df.groupby("stock_code"):
            new_rows = group.sort_values("date")
            out_path = os.path.join(base, f"{code}.parquet")
            if os.path.isfile(out_path):
                existing = pd.read_parquet(out_path)
                existing["date"] = pd.to_datetime(existing["date"])
                if "stock_code" not in existing.columns:
                    existing["stock_code"] = code
                new_rows = pd.concat([existing, new_rows], ignore_index=True)
            new_rows = new_rows.drop_duplicates(subset=["date"], keep="last")
            new_rows = new_rows.sort_values("date")
            fd, tmp_path = tempfile.mkstemp(
                suffix=".parquet", dir=base, prefix=f".tmp_{code}_",
            )
            os.close(fd)
            try:
                new_rows.to_parquet(tmp_path, index=False, compression='lz4')
                os.replace(tmp_path, out_path)
            except Exception:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
                raise
            # §T8: stamp the gold manifest with the provider-era fields read
            # from the downloader's per-stock manifest via _manifest_dir (the
            # explicit coupling to download_comment.py).  `**{}` when there is
            # no downloader manifest → the stock is not_observed (no provider
            # era) — never counted as a zero-coverage event gap.
            write_asset_manifest(
                out_path, COMMENT_SENTIMENT_ASSET, new_rows, entity=code,
                **downloader_era_fields(self._manifest_dir(), code),
            )

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str,
        *, require_valid_manifest: bool = False,
    ) -> pd.DataFrame:
        """Load daily comment data for a stock in a date range.

        Prefers consolidated flat file; falls back to directory walk.
        Every file read is cross-checked against its asset manifest
        (``check_asset_read``); pass ``require_valid_manifest=True`` to raise
        instead of read when the manifest is missing or mismatched.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = self._base_dir()
        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer consolidated flat file
        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            check_asset_read(flat_path, COMMENT_SENTIMENT_ASSET, df,
                             require_valid_manifest=require_valid_manifest)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            return df[mask].sort_values("date").reset_index(drop=True)

        # Fallback: walk partition directories
        frames = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    check_asset_read(path, COMMENT_SENTIMENT_ASSET, df,
                                     require_valid_manifest=require_valid_manifest)
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
