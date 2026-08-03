"""EarningsStorage — per-stock daily access to 业绩预告/业绩快报 snapshots.

The earnings source fetches market-wide snapshots (data/a_shares/earnings/
forecasts.parquet + express.parquet). Each row is an announcement with an
``announce_date``. This storage turns those event rows into a per-stock daily
series usable by the feature pipeline:

  - PIT: an announcement is only known from the trading day after announce_date
    (post-close convention, identical to the news-sentiment path). The pipeline
    merge adds one more shift(1), so a Tuesday announcement first shows up as a
    feature when predicting Thursday's return — no same-day leakage.
  - Persistence: the net-profit band stays active until superseded by a later
    announcement for the same stock — forward-filled across trading days.
  - Output columns: has_forecast, net_profit_yoy_low/high (%), net_profit_low/
    high (万元), forecast_age (calendar days since announcement, decay signal).

Both snapshot schemas are normalized here onto a common event schema, so the
pipeline merge only ever sees one shape.
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_FILES = ["forecasts.parquet", "express.parquet"]

# Columns the pipeline merge consumes (in the per-stock daily frame).
DAILY_COLS = [
    "has_forecast",
    "net_profit_yoy_low", "net_profit_yoy_high",
    "net_profit_low", "net_profit_high",
    "forecast_age",
]

# Common event schema after normalization of both snapshot types.
_COMMON = [
    "stock_code", "announce_date",
    "net_profit_yoy_low", "net_profit_yoy_high",
    "net_profit_low", "net_profit_high",
]


class EarningsStorage:
    """Per-stock daily view over accumulated earnings announcement snapshots."""

    def __init__(self, data_dir: str):
        from stoke_ml.data.calendar import TradingCalendar

        self._dir = os.path.join(data_dir, "a_shares", "earnings")
        self._calendar = TradingCalendar()
        self._snap: pd.DataFrame | None = None

    # ── snapshot loading / normalization ────────────────────────────────

    def _load_snapshot(self) -> pd.DataFrame:
        """Load + normalize both snapshot files to the common event schema."""
        if self._snap is not None:
            return self._snap
        frames = []
        for name in SNAPSHOT_FILES:
            p = os.path.join(self._dir, name)
            if not os.path.isfile(p):
                continue
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                logger.warning("earnings snapshot %s unreadable: %s", name, e)
                continue
            if df.empty:
                continue
            norm = self._normalize(df)
            if not norm.empty:
                frames.append(norm)
        self._snap = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._snap

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Map a snapshot onto the common event schema.

        Both sources emit point estimates (net_profit_yoy, net_profit); we widen
        them to low = high = point so both feed the same persistence logic. When
        a forecast_metric column exists (业绩预告), only net-profit rows carry
        signal — revenue/EPS forecasts are excluded.
        """
        if "stock_code" not in df.columns or "announce_date" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
        yoy = pd.to_numeric(df.get("net_profit_yoy"), errors="coerce")
        profit = pd.to_numeric(df.get("net_profit"), errors="coerce")
        if "forecast_metric" in df.columns:
            is_profit = df["forecast_metric"].astype(str).str.contains("净利", na=False)
            yoy = yoy.where(is_profit)
            profit = profit.where(is_profit)
        out = pd.DataFrame({
            "stock_code": df["stock_code"],
            "announce_date": df["announce_date"],
            "net_profit_yoy_low": yoy,
            "net_profit_yoy_high": yoy,
            "net_profit_low": profit,
            "net_profit_high": profit,
        })
        # Drop rows with no usable signal (e.g. revenue-only forecasts).
        out = out.dropna(subset=["net_profit_yoy_low", "net_profit_low"], how="all")
        return out.dropna(subset=["announce_date"])

    # ── per-stock daily view ────────────────────────────────────────────

    def load_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Return the stock's daily earnings-active frame over [start, end]."""
        snap = self._load_snapshot()
        if snap.empty:
            return pd.DataFrame()
        sub = snap[snap["stock_code"] == code].copy()
        if sub.empty:
            return pd.DataFrame()
        return self._to_daily(sub, start, end)

    def _to_daily(self, sub: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        sub = sub.dropna(subset=["announce_date"])
        if sub.empty:
            return pd.DataFrame()
        # PIT: signal becomes known on the next trading day after announce_date.
        sub["eff_date"] = sub["announce_date"].map(
            lambda d: pd.Timestamp(self._calendar.next_trading_day(d.date()))
        )
        sub = sub.dropna(subset=["eff_date"])
        # Latest announcement wins on a given effective date.
        sub = sub.sort_values("eff_date").drop_duplicates("eff_date", keep="last")

        idx = pd.DatetimeIndex(self._calendar.get_trading_days(start, end))
        if idx.empty:
            return pd.DataFrame()
        sub = sub.set_index("eff_date")

        band = sub[["net_profit_yoy_low", "net_profit_yoy_high",
                    "net_profit_low", "net_profit_high"]].reindex(idx).ffill()
        active = sub[["announce_date"]].reindex(idx).ffill()["announce_date"].notna()

        daily = pd.DataFrame({"date": idx})
        # Positional assignment (daily is RangeIndex, active is DatetimeIndex —
        # index-aligned assignment would silently produce all-NaN).
        daily["has_forecast"] = active.to_numpy(dtype=np.float32)
        for c in band.columns:
            daily[c] = band[c].to_numpy(dtype="float64")
        # Calendar days since the active announcement (0 when inactive).
        idx_s = pd.Series(idx, index=idx)
        ann_series = sub["announce_date"].reindex(idx).ffill()
        age = (idx_s - ann_series).dt.days
        daily["forecast_age"] = np.where(active.to_numpy(), age.to_numpy(), 0).astype(np.float32)
        return daily[["date"] + DAILY_COLS]
