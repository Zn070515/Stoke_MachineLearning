"""EventToDaily: convert sparse events to daily per-stock features.

Handles 4 event types (spec §3.2):
  block_trade — aggregate by date+stock, forward-fill, price impact
  shareholder — quarterly → daily, HN_z (8Q pre-fill), PCRC, dual-conc.
  lockup      — history + upcoming, unlock pressure with exponential decay
  dividend    — yield + decay, normalization pipeline, growth tracking
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)


class EventToDaily(PreprocessingStep):
    """Dispatch sparse events to daily features by event_type.

    Parameters:
        event_type: "block_trade" | "shareholder" | "lockup" | "dividend"
        decay_halflife_days: half-life for exponential decay (dividend/lockup).
        forward_fill_max: max consecutive days to forward-fill before ZI.
    """

    def __init__(
        self,
        event_type: str,
        decay_halflife_days: int = 90,
        forward_fill_max: int = 5,
    ):
        if event_type not in ("block_trade", "shareholder", "lockup", "dividend"):
            raise ValueError(f"Unknown event_type: {event_type}")
        self.event_type = event_type
        self.decay_halflife_days = decay_halflife_days
        self.forward_fill_max = forward_fill_max

    def transform(
        self,
        df: pd.DataFrame,
        close_prices: Optional[pd.DataFrame] = None,
        trading_dates: Optional[pd.DatetimeIndex] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        dispatch = {
            "block_trade": self._transform_block_trade,
            "shareholder": self._transform_shareholder,
            "lockup": self._transform_lockup,
            "dividend": self._transform_dividend,
        }
        return dispatch[self.event_type](df, close_prices, trading_dates, **kwargs)

    # ── block_trade ────────────────────────────────────────────────────

    def _transform_block_trade(self, df, close_prices, trading_dates, **kwargs):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        if df.empty or "stock_code" not in df.columns:
            return pd.DataFrame()

        # Build aggregation dict dynamically based on available columns
        agg_dict = {"trade_count": ("stock_code", "count")}

        if "premium_pct" in df.columns:
            agg_dict["premium_pct_mean"] = ("premium_pct", "mean")
        if "amount" in df.columns:
            agg_dict["total_amount"] = ("amount", "sum")
        if "volume" in df.columns:
            agg_dict["total_volume"] = ("volume", "sum")
        if "buyer" in df.columns:
            agg_dict["buyer_is_inst"] = (
                "buyer",
                lambda x: x.astype(str)
                .str.contains("机构|专用|瑞银|沪股通|深股通|QFII|社保|保险|基金|证券")
                .any(),
            )
            agg_dict["buyer_is_hot_money"] = (
                "buyer",
                lambda x: x.astype(str)
                .str.contains("拉萨|团结路|江苏路|深圳分公司|浙江分公司|中金财富")
                .any(),
            )
        if "seller" in df.columns:
            agg_dict["seller_is_inst"] = (
                "seller",
                lambda x: x.astype(str)
                .str.contains("机构|专用|瑞银|沪股通|深股通|QFII|社保|保险|基金|证券")
                .any(),
            )
            agg_dict["seller_is_hot_money"] = (
                "seller",
                lambda x: x.astype(str)
                .str.contains("拉萨|团结路|江苏路|深圳分公司|浙江分公司|中金财富")
                .any(),
            )

        # Compute VWAP premium if both premium and amount exist
        if "premium_pct" in df.columns and "amount" in df.columns:
            df["_weighted"] = df["premium_pct"] * df["amount"]
            agg_dict["premium_pct_wavg_sum"] = ("_weighted", "sum")

        grouped = df.groupby(["date", "stock_code"], as_index=False).agg(**agg_dict)

        if "premium_pct_wavg_sum" in grouped.columns and "total_amount" in grouped.columns:
            grouped["premium_pct_wavg"] = (
                grouped["premium_pct_wavg_sum"] / grouped["total_amount"].replace(0, np.nan)
            ).fillna(grouped.get("premium_pct_mean", 0)).astype(np.float32)
            grouped.drop(columns=["premium_pct_wavg_sum"], inplace=True)

        # Fill to daily calendar
        if trading_dates is not None and not grouped.empty:
            grouped = self._fill_to_daily(grouped, trading_dates, max_ffill=5)

        # 6-day amount volatility
        if "total_amount" in grouped.columns:
            grouped["amount_vol_6d"] = _grouped_rolling_cv(
                grouped, "total_amount", 6
            )

        # Price impact (if close_prices available)
        if close_prices is not None and "premium_pct_wavg" in grouped.columns:
            grouped = grouped.merge(
                close_prices, on=["date", "stock_code"], how="left", suffixes=("", "_cp")
            )
            close_col = "close" if "close" in grouped.columns else "close_cp"
            if close_col in grouped.columns:
                grouped["permanent_impact"] = (
                    grouped.groupby("stock_code")[close_col]
                    .pct_change()
                    .fillna(0)
                    .astype(np.float32)
                )
                grouped["temporary_impact"] = (
                    grouped["premium_pct_wavg"] - grouped["permanent_impact"]
                ).astype(np.float32)
                if close_col == "close_cp":
                    grouped.drop(columns=["close_cp"], inplace=True)

        # Deep discount flag
        if "premium_pct_mean" in grouped.columns:
            grouped["is_deep_discount"] = (
                grouped["premium_pct_mean"].lt(-8).astype(np.int8)
            )

        # Amount ratio: block trade amount / daily total turnover
        # (Guangda Securities: strongest block-trade alpha factor, 16.41% annual excess)
        daily_data = kwargs.get("daily_data")
        if daily_data is not None and not daily_data.empty and "total_amount" in grouped.columns:
            merge_on = [c for c in ["date", "stock_code"] if c in grouped.columns]
            if merge_on and all(c in daily_data.columns for c in merge_on):
                dd = daily_data.copy()
                dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
                # Get daily total amount (in yuan) from K-line data
                daily_amount_col = "amount" if "amount" in dd.columns else None
                if daily_amount_col:
                    merged = grouped.merge(
                        dd[merge_on + [daily_amount_col]], on=merge_on, how="left"
                    )
                    denom = merged[daily_amount_col].abs().replace(0, np.nan)
                    grouped["amount_ratio"] = (
                        merged["total_amount"] / denom
                    ).clip(0, 1).astype(np.float32)

        # Clean up temp columns
        if "_weighted" in df.columns:
            df.drop(columns=["_weighted"], inplace=True)

        return grouped

    # ── shareholder ────────────────────────────────────────────────────

    def _transform_shareholder(self, df, close_prices, trading_dates, **kwargs):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        if df.empty or "stock_code" not in df.columns:
            return pd.DataFrame()

        df = df.sort_values(["stock_code", "date"])
        required = ["holder_num", "change_ratio", "avg_shares"]
        available = [c for c in required if c in df.columns]
        if not available:
            return df

        # Compute HN_z on raw quarterly data BEFORE forward-fill (8-quarter window)
        if "holder_num" in df.columns:
            df["HN_z"] = (
                df.groupby("stock_code")["holder_num"]
                .transform(
                    lambda s: (
                        s - s.rolling(8, min_periods=4).mean()
                    )
                    / (s.rolling(8, min_periods=4).std(ddof=0) + 1e-8)
                )
                .astype(np.float32)
            )
            df["HN_z"] = -df["HN_z"]  # declining holders = positive factor

        # Consecutive quarter decline (per-group, before fill)
        if "change_ratio" in df.columns:
            df["consecutive_quarter_decline"] = (
                df.groupby("stock_code")["change_ratio"]
                .transform(_consecutive_neg)
                .astype(np.int16)
            )

        # PCRC: YoY change (4 quarters)
        if "avg_shares" in df.columns:
            df["PCRC"] = (
                df.groupby("stock_code")["avg_shares"]
                .transform(lambda s: s / s.shift(4).replace(0, np.nan) - 1)
                .fillna(0)
                .astype(np.float32)
            )

        # Forward-fill to daily
        if trading_dates is not None:
            df = self._fill_to_daily(df, trading_dates, max_ffill=90)

        # Dual-concentration signal (needs close prices from close_prices param)
        if close_prices is not None:
            cp = close_prices[["date", "stock_code", "close"]].copy()
            df = df.merge(cp, on=["date", "stock_code"], how="left")
            if "close" in df.columns and "change_ratio" in df.columns:
                sma60 = (
                    df.groupby("stock_code")["close"]
                    .transform(lambda s: s.rolling(60, min_periods=20).mean())
                )
                df["dual_concentration_signal"] = (
                    (df["close"] < sma60) & (df["change_ratio"] < 0)
                ).astype(np.int8)

        return df

    # ── lockup ─────────────────────────────────────────────────────────

    def _transform_lockup(self, df, close_prices, trading_dates, **kwargs):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        if df.empty or "stock_code" not in df.columns:
            return pd.DataFrame()

        df = df.sort_values(["stock_code", "date"])

        # Identify upcoming vs historical
        if "is_upcoming" not in df.columns:
            today = pd.Timestamp.now()
            df["is_upcoming"] = df["date"] > today

        # VC-backed flag — set BEFORE split so both upcoming+hist inherit it
        if "free_type" in df.columns:
            df["is_vc_backed"] = (
                df["free_type"]
                .astype(str)
                .str.contains("首发|IPO", na=False)
                .astype(np.int8)
            )

        upcoming = df[df["is_upcoming"]] if df["is_upcoming"].any() else pd.DataFrame()
        hist = df[~df["is_upcoming"]] if (~df["is_upcoming"]).any() else pd.DataFrame()

        lam = np.log(2) / max(self.decay_halflife_days, 1)

        if not upcoming.empty:
            today = pd.Timestamp.now()
            upcoming = upcoming.copy()
            upcoming["days_until_unlock"] = (
                pd.to_datetime(upcoming["date"]) - today
            ).dt.days.clip(lower=1)

            free_ratio = upcoming.get("free_ratio", 0)
            if isinstance(free_ratio, pd.Series):
                free_ratio = free_ratio.fillna(0).astype(float)
            upcoming["unlock_pressure"] = (
                free_ratio * np.exp(-lam * upcoming["days_until_unlock"])
            ).astype(np.float32)

            # Market-cap-normalized unlock impact (free_ratio × close)
            if close_prices is not None:
                cp = close_prices[["date", "stock_code", "close"]].copy()
                upcoming = upcoming.merge(cp, on=["date", "stock_code"], how="left")
                if "close" in upcoming.columns:
                    upcoming["unlock_pressure_mcap"] = (
                        upcoming["unlock_pressure"] * upcoming["close"].fillna(0)
                    ).astype(np.float32)

            # Aggregate per stock: total upcoming
            agg_dict = {
                "unlock_pressure": ("unlock_pressure", "sum"),
                "days_to_nearest_unlock": ("days_until_unlock", "min"),
                "unlock_count_upcoming": ("date", "count"),
            }
            if "free_ratio" in upcoming.columns:
                agg_dict["total_upcoming_ratio"] = ("free_ratio", "sum")
            if "unlock_pressure_mcap" in upcoming.columns:
                agg_dict["unlock_pressure_mcap"] = ("unlock_pressure_mcap", "sum")
            agg_upcoming = (
                upcoming.groupby("stock_code")
                .agg(**agg_dict)
                .reset_index()
            )
            # Merge aggregated stats back into per-date upcoming records
            if not agg_upcoming.empty:
                upcoming = upcoming.merge(
                    agg_upcoming, on="stock_code", how="left",
                    suffixes=("", "_agg"),
                )

        # Historical lockup return
        if not hist.empty and close_prices is not None:
            hist = hist.merge(
                close_prices, on=["date", "stock_code"], how="left", suffixes=("", "_cp")
            )
            close_col = "close" if "close" in hist.columns else "close_cp"
            if close_col in hist.columns:
                hist["unlock_return_30d"] = (
                    hist.groupby("stock_code")[close_col]
                    .transform(lambda s: s.shift(-30) / s - 1)
                    .fillna(0)
                    .astype(np.float32)
                )

        result_parts = []
        if not hist.empty:
            result_parts.append(hist)
        if not upcoming.empty:
            result_parts.append(upcoming)
        if not result_parts:
            return df

        result = pd.concat(result_parts, ignore_index=True)
        if trading_dates is not None:
            result = self._fill_to_daily(result, trading_dates, max_ffill=5)

        return result

    # ── dividend ───────────────────────────────────────────────────────

    def _transform_dividend(self, df, close_prices, trading_dates, **kwargs):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        if df.empty or "stock_code" not in df.columns:
            return pd.DataFrame()

        df = df.sort_values(["stock_code", "date"])

        # Merge close prices for yield computation
        if close_prices is not None and "bonus_rmb" in df.columns:
            df = df.merge(
                close_prices[["date", "stock_code", "close"]],
                on=["date", "stock_code"],
                how="left",
            )
            if "close" in df.columns:
                df["dividend_yield"] = (
                    df["bonus_rmb"] / df["close"].replace(0, np.nan)
                ).astype(np.float32)
        elif "bonus_rmb" in df.columns:
            df["dividend_yield"] = df["bonus_rmb"].astype(np.float32)

        # Forward-fill to daily
        if trading_dates is not None:
            df = self._fill_to_daily(df, trading_dates, max_ffill=30)

        # Effective yield with exponential decay (vectorized per group)
        if "dividend_yield" in df.columns:
            lam = np.log(2) / max(self.decay_halflife_days, 1)
            df["days_since_last_ex_div"] = (
                df.groupby("stock_code")["_has_div"]
                if "_has_div" in df.columns
                else pd.Series(0, index=df.index)
            )
            # Vectorized: mark rows with non-zero yield, compute cumulative days
            has_yield = df["dividend_yield"].notna() & (df["dividend_yield"] > 0)
            df["_has_yield"] = has_yield.astype(int)
            df["days_since_last_ex_div"] = (
                df.groupby("stock_code")["_has_yield"]
                .transform(_days_since_last_event, lam)
                .astype(np.int16)
            )
            df["effective_yield"] = (
                df["dividend_yield"].ffill()
                * np.exp(-lam * df["days_since_last_ex_div"])
            ).astype(np.float32)
            if "_has_yield" in df.columns:
                df.drop(columns=["_has_yield"], inplace=True)

        if "days_since_last_ex_div" in df.columns:
            df["has_recent_dividend"] = (
                df["days_since_last_ex_div"].le(90).astype(np.int8)
            )

        if "plan" in df.columns:
            stages = {"预案": 1, "决案": 2, "实施": 3}
            df["plan_stage_encoded"] = (
                df["plan"].astype(str).map(stages).fillna(0).astype(np.int8)
            )

        if "bonus_rmb" in df.columns:
            df["dividend_growth"] = (
                df.groupby("stock_code")["bonus_rmb"]
                .transform(lambda s: s.diff() / s.shift(1).replace(0, np.nan))
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
                .astype(np.float32)
            )

        return df

    # ── helpers ────────────────────────────────────────────────────────

    def _fill_to_daily(
        self, df: pd.DataFrame, trading_dates: pd.DatetimeIndex, max_ffill: int
    ) -> pd.DataFrame:
        """Reindex each stock to trading calendar, forward-fill up to max_ffill."""
        if df.empty or "stock_code" not in df.columns:
            return df

        def _fill_group(grp):
            grp = grp.set_index("date").sort_index()
            grp = grp.reindex(trading_dates)
            cols = [c for c in grp.columns if c not in ("stock_code",)]
            grp[cols] = grp[cols].ffill(limit=max_ffill)
            return grp

        result = (
            df.groupby("stock_code", group_keys=False)
            .apply(_fill_group)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        return result


# ── vectorized helpers ─────────────────────────────────────────────────

def _consecutive_neg(series: pd.Series) -> pd.Series:
    """Per-group: running count of consecutive negative values."""
    result = pd.Series(0, index=series.index, dtype=np.int16)
    cnt = 0
    vals = series.fillna(0).values
    for i, v in enumerate(vals):
        if v < 0:
            cnt += 1
        else:
            cnt = 0
        result.iloc[i] = cnt
    return result


def _days_since_last_event(series: pd.Series, lam: float = 0.0) -> pd.Series:
    """Compute days since last non-zero entry in a binary series, per group.

    Input series is 1 where event occurred, 0 otherwise. Output is days since
    the most recent 1.
    """
    result = pd.Series(0, index=series.index, dtype=np.int16)
    last_idx = -1
    vals = series.values
    for i, v in enumerate(vals):
        if v > 0:
            last_idx = i
        if last_idx >= 0:
            result.iloc[i] = i - last_idx
    return result


def _weighted_mean(values, weights):
    """Weighted mean, handling NaN and zero-sum weights."""
    w = np.nan_to_num(weights.values.astype(float), nan=0.0)
    v = np.nan_to_num(values.values.astype(float), nan=0.0)
    total = w.sum()
    if abs(total) < 1e-12:
        return float(np.mean(v)) if len(v) > 0 else 0.0
    return float(np.average(v, weights=w))


def _grouped_rolling_cv(df, col, window):
    """Grouped rolling coefficient of variation (std/mean)."""
    if "stock_code" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    roll_mean = (
        df.groupby("stock_code")[col]
        .rolling(window, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_std = (
        df.groupby("stock_code")[col]
        .rolling(window, min_periods=3)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    return (roll_std / (roll_mean.abs() + 1e-8)).astype(np.float32)
