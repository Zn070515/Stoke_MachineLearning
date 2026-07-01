"""Multi-stock panel builder for cross-sectional normalization.

Loads K-line data, sector, and size proxy for all stocks in a date range
into a single panel DataFrame, enabling per-date cross-stock operations
(sector neutralization, rank normalization, etc.).

Qlib-style: the panel is sorted by (date, stock_code) so groupby("date")
yields a cross-section of all stocks at each timestamp.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.data.stock_sector_mapper import StockSectorMapper

logger = logging.getLogger(__name__)


class PanelBuilder:
    """Build multi-stock panel with sector and size metadata."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._mapper = StockSectorMapper()
        self._sector_cache: dict[str, str] = {}
        self._fund_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        *,
        min_rows_per_stock: int = 100,
    ) -> pd.DataFrame:
        """Build panel with columns [date, stock_code, open, high, low,
        close, volume, amount, sector, size_proxy].

        Stocks with fewer than *min_rows_per_stock* rows are dropped.
        """
        frames: list[pd.DataFrame] = []
        skipped = 0

        for code in codes:
            df = self._load_one(code, start_date, end_date)
            if df is None or len(df) < min_rows_per_stock:
                skipped += 1
                continue
            frames.append(df)

        if skipped:
            logger.info("PanelBuilder: skipped %d/%d stocks (< %d rows)",
                        skipped, len(codes), min_rows_per_stock)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

        # Attach sector
        panel["sector"] = panel["stock_code"].map(self._get_sector_map(codes))

        # Attach size proxy from fundamentals
        panel["size_proxy"] = self._attach_size_proxy(panel, codes, start_date, end_date)

        logger.info("Panel built: %d stocks × %d rows, %d dates",
                     panel["stock_code"].nunique(), len(panel),
                     panel["date"].nunique())
        return panel

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_one(
        self, code: str, start: str, end: str
    ) -> pd.DataFrame | None:
        """Load K-line for a single stock. Returns None if not enough data."""
        base = os.path.join(self._data_dir, "a_shares", "daily")

        # Try flat file first
        flat = os.path.join(base, f"{code}.parquet")
        if os.path.isfile(flat):
            df = pd.read_parquet(flat)
        else:
            # Scan partitioned directories
            parts = []
            for root, _dirs, files in os.walk(base):
                for f in files:
                    if f == f"{code}.parquet":
                        parts.append(pd.read_parquet(os.path.join(root, f)))
            if not parts:
                return None
            df = pd.concat(parts, ignore_index=True)

        df["date"] = pd.to_datetime(df["date"])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        if len(df) < 50:
            return None

        df = df.sort_values("date").reset_index(drop=True)
        # Keep only OHLCV + amount columns
        keep = ["date", "stock_code", "open", "high", "low", "close", "volume"]
        if "amount" in df.columns:
            keep.append("amount")
        return df[[c for c in keep if c in df.columns]]

    def _get_sector_map(self, codes: list[str]) -> dict[str, str]:
        """Build code→sector mapping, using cache."""
        uncached = [c for c in codes if c not in self._sector_cache]
        for c in uncached:
            try:
                self._sector_cache[c] = self._mapper.get_sector(c) or "未知"
            except Exception:
                self._sector_cache[c] = "未知"
        return self._sector_cache

    def _attach_size_proxy(
        self, panel: pd.DataFrame, codes: list[str],
        start: str, end: str,
    ) -> np.ndarray:
        """Attach log-total-revenue as a size proxy for each (date, stock_code).

        Forward-filled from quarterly fundamental data. Falls back to
        log(close * volume) as a rough daily liquidity proxy when
        fundamental data is unavailable.
        """
        result = np.full(len(panel), np.nan, dtype=np.float64)

        fund_dir = os.path.join(self._data_dir, "a_shares", "fundamentals")
        if os.path.isdir(fund_dir):
            for code in codes:
                fpath = os.path.join(fund_dir, f"{code}.parquet")
                if not os.path.isfile(fpath):
                    continue
                try:
                    fd = pd.read_parquet(fpath)
                except Exception:
                    continue
                if "total_revenue" not in fd.columns or "disclose_date" not in fd.columns:
                    continue

                fd["disclose_date"] = pd.to_datetime(fd["disclose_date"])
                fd = fd.drop_duplicates(subset="disclose_date", keep="last")
                fd = fd.sort_values("disclose_date")
                fd = fd.set_index("disclose_date")
                # Forward-fill to daily
                full_idx = pd.date_range(
                    max(fd.index.min(), pd.Timestamp(start)),
                    min(fd.index.max(), pd.Timestamp(end)),
                    freq="D",
                )
                fd = fd.reindex(full_idx).ffill()
                fd = fd.dropna(subset=["total_revenue"])

                mask = panel["stock_code"] == code
                code_dates = panel.loc[mask, "date"]
                for i, idx in enumerate(code_dates.index):
                    d = code_dates.iloc[i]
                    if d in fd.index:
                        result[idx] = np.log(max(fd.loc[d, "total_revenue"], 1.0))

        # Fallback: daily liquidity proxy log(close * volume)
        nan_mask = np.isnan(result)
        if nan_mask.any():
            close_vals = panel["close"].values
            vol_vals = panel["volume"].values
            fallback = np.log(np.maximum(close_vals * vol_vals, 1.0))
            result[nan_mask] = fallback[nan_mask]

        return result.astype(np.float32)
