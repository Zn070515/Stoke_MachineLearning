"""Technical indicators — MA, MACD, RSI, KDJ, BOLL, ATR, volume, momentum.

Inspired by Qlib Alpha158 factor set, covering momentum, volatility,
volume-price, and overbought/oversold dimensions.
"""
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Compute standard + extended technical indicators from OHLCV data."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        amount = df.get("amount", None)

        # ── Moving averages ─────────────────────────────────────────
        for period in [5, 10, 20, 60, 120]:
            result[f"ma_{period}"] = close.rolling(period).mean()

        # EMA
        result["ema_12"] = close.ewm(span=12, adjust=False).mean()
        result["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # ── MACD ────────────────────────────────────────────────────
        result["macd_dif"] = result["ema_12"] - result["ema_26"]
        result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

        # ── RSI ─────────────────────────────────────────────────────
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ── KDJ (Stochastic) ───────────────────────────────────────
        for period in [9, 14]:
            low_n = low.rolling(period).min()
            high_n = high.rolling(period).max()
            rsv = (close - low_n) / (high_n - low_n).replace(0, 1e-10) * 100
            k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
            d = k.ewm(alpha=1 / 3, adjust=False).mean()
            j = 3 * k - 2 * d
            result[f"kdj_k_{period}"] = k
            result[f"kdj_d_{period}"] = d
            result[f"kdj_j_{period}"] = j

        # ── Bollinger Bands ────────────────────────────────────────
        result["boll_mid"] = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        result["boll_upper"] = result["boll_mid"] + 2 * boll_std
        result["boll_lower"] = result["boll_mid"] - 2 * boll_std
        # Price position within BB band [0,1]
        bb_range = result["boll_upper"] - result["boll_lower"]
        result["boll_pct"] = (close - result["boll_lower"]) / bb_range.replace(0, 1e-10)

        # ── ATR ─────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean()

        # ── ROC (Rate of Change / momentum) ────────────────────────
        for period in [6, 12, 20]:
            result[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period).replace(0, 1e-10) * 100

        # ── WR (Williams %R) ───────────────────────────────────────
        for period in [10, 20]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            result[f"wr_{period}"] = (high_n - close) / (high_n - low_n).replace(0, 1e-10) * -100

        # ── CCI (Commodity Channel Index) ──────────────────────────
        for period in [14, 20]:
            tp = (high + low + close) / 3
            tp_ma = tp.rolling(period).mean()
            tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            result[f"cci_{period}"] = (tp - tp_ma) / tp_mad.replace(0, 1e-10) / 0.015

        # ── Historical volatility ──────────────────────────────────
        for period in [5, 20]:
            log_ret = np.log(close / close.shift(1))
            result[f"vol_{period}"] = log_ret.rolling(period).std() * np.sqrt(252)

        # ── Volume indicators ───────────────────────────────────────
        result["volume_ma5"] = volume.rolling(5).mean()
        result["volume_ratio"] = volume / result["volume_ma5"].replace(0, 1)
        result["obv"] = (
            volume * ((close.diff() > 0).astype(int) * 2 - 1)
        ).cumsum()

        # Volume trend: ratio of up-day volume to total volume (20d)
        up_vol = volume.where(close.diff() > 0, 0)
        down_vol = volume.where(close.diff() < 0, 0)
        result["vol_up_ratio_20"] = (
            up_vol.rolling(20).sum() / volume.rolling(20).sum().replace(0, 1e-10)
        )

        # ── Price-volume correlation ───────────────────────────────
        result["pct_change"] = close.pct_change()
        result["vol_change"] = volume.pct_change()

        # ── Max/Min over window ─────────────────────────────────────
        for period in [5, 20, 60]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            result[f"high_{period}d"] = (close - low_n) / close.replace(0, 1e-10)
            result[f"low_{period}d"] = (high_n - close) / close.replace(0, 1e-10)

        # ── Amount-based (if available) ────────────────────────────
        if amount is not None:
            amount = amount.astype(float)
            result["amount_ma5"] = amount.rolling(5).mean()
            result["amount_ratio"] = amount / result["amount_ma5"].replace(0, 1e-10)
            # Turnover approximation: amount / close (proxy for shares traded)
            result["turnover_proxy"] = amount / close.replace(0, 1e-10)

        # ── Clean up intermediate columns ──────────────────────────
        for col in ["pct_change", "vol_change"]:
            if col in result.columns:
                result = result.drop(columns=[col])

        return result
