"""Intraday features for minute-frequency K-line data.

A-share trading sessions (China Standard Time):
  Morning:  9:30 – 11:30  (2 hours, 2 × 60-min bars)
  Afternoon: 13:00 – 15:00 (2 hours, 2 × 60-min bars)

All features are computed per bar with zero forward-looking bias — each bar
only sees information available at or before its own timestamp.
"""
import pandas as pd
import numpy as np

_A_MARKET_OPEN = 9 * 60 + 30   # 9:30 in minutes-from-midnight
_A_MARKET_CLOSE = 15 * 60      # 15:00
_MORNING_END = 11 * 60 + 30    # 11:30
_AFTERNOON_START = 13 * 60     # 13:00
_MINS_PER_SESSION = 120        # 2 hours per session


class MinuteIntradayFeatures:
    """Compute session-aware intraday features from minute bar data.

    Requires a 'datetime' column of type datetime64[ns]. Uses the
    hour+minute to determine session position — no calendar date
    dependency, so the module works with any date range.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday feature columns to *df* (mutates in-place)."""
        if "datetime" not in df.columns:
            return df

        dt = pd.to_datetime(df["datetime"])
        minutes = dt.dt.hour * 60 + dt.dt.minute  # minutes from midnight

        # Session detection (bar timestamp = END of bar; use > open, <= close)
        is_am = (minutes > _A_MARKET_OPEN) & (minutes <= _MORNING_END)
        is_pm = (minutes > _AFTERNOON_START) & (minutes <= _A_MARKET_CLOSE)
        in_session = is_am | is_pm

        # Minutes from session open
        df["minutes_from_open"] = np.where(
            in_session,
            np.where(is_am, minutes - _A_MARKET_OPEN, minutes - _AFTERNOON_START),
            0,
        ).astype(np.float32)

        # Minutes to close (how much time remains in the trading day)
        df["minutes_to_close"] = np.where(
            in_session, _A_MARKET_CLOSE - minutes, 0,
        ).astype(np.float32)

        # Session flags
        df["is_am_session"] = is_am.astype(np.float32)
        df["is_pm_session"] = is_pm.astype(np.float32)

        # Session progress: 0.0 (open) → 1.0 (close)
        raw_progress = df["minutes_from_open"] / max(_MINS_PER_SESSION, 1)
        df["session_progress"] = np.where(in_session, raw_progress, 0.0).astype(np.float32)

        # Bar index within the trading day (1-indexed, computed per date group)
        df["date_key"] = pd.to_datetime(df.get("date", dt.dt.date))
        df["bar_of_day"] = (
            df.groupby("date_key").cumcount() + 1
        ).astype(np.int16)

        # Opening imbalance: first bar's O→C return, broadcast to all bars of the day.
        # Forward-filled so bars 2-4 see bar 1's value without look-ahead.
        close = df.get("close")
        _open = df.get("open")
        if close is not None and _open is not None:
            def _first_bar_ret(g):
                o = g["open"].iloc[0]
                c = g["close"].iloc[0]
                return (c - o) / o if o != 0 else 0.0
            first_ret = df.groupby("date_key").apply(
                _first_bar_ret, include_groups=False,
            )
            first_ret = first_ret.fillna(0.0)
            df["opening_imbalance"] = (
                df["date_key"].map(first_ret).fillna(0.0).astype(np.float32)
            )

        # Session position: where is the bar relative to day's high/low so far?
        if close is not None:
            high = df.get("high")
            low = df.get("low")
            if high is not None:
                day_high_sofar = df.groupby("date_key")["high"].cummax()
                day_low_sofar = df.groupby("date_key")["low"].cummin()
                hilo_range = day_high_sofar - day_low_sofar
                df["session_high_position"] = np.where(
                    hilo_range > 0,
                    (close - day_low_sofar) / hilo_range,
                    0.5,
                ).astype(np.float32)

        # Clean up temporary column
        if "date_key" in df.columns:
            df.drop(columns=["date_key"], inplace=True)

        return df
