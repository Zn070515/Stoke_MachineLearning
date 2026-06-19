"""Technical indicators — MA, MACD, RSI, BOLL, ATR, volume."""
import pandas as pd


class TechnicalIndicators:
    """Compute standard technical indicators from OHLCV data."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # Moving averages
        for period in [5, 10, 20, 60, 120]:
            result[f"ma_{period}"] = close.rolling(period).mean()

        # EMA
        result["ema_12"] = close.ewm(span=12, adjust=False).mean()
        result["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # MACD
        result["macd_dif"] = result["ema_12"] - result["ema_26"]
        result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

        # RSI
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2)
        result["boll_mid"] = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        result["boll_upper"] = result["boll_mid"] + 2 * boll_std
        result["boll_lower"] = result["boll_mid"] - 2 * boll_std

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean()

        # Volume indicators
        result["volume_ma5"] = volume.rolling(5).mean()
        result["volume_ratio"] = volume / result["volume_ma5"].replace(0, 1)
        result["obv"] = (
            volume * ((close.diff() > 0).astype(int) * 2 - 1)
        ).cumsum()

        return result
