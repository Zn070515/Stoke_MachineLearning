"""Rule-based trend and buy signal scoring.

Extracts structured signals from technical indicators to serve
as model input features. Not used as standalone trading signals.
"""
import pandas as pd
import numpy as np


class TrendScorer:
    """Score trend strength and generate buy/sell level features."""

    BIAS_THRESHOLD = 5.0
    VOLUME_SHRINK_RATIO = 0.7
    VOLUME_HEAVY_RATIO = 1.5

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result = self._classify_trend(result)
        result = self._compute_bias(result)
        result = self._classify_volume(result)
        result = self._compute_buy_signal(result)
        return result

    def _classify_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        ma5 = df["ma_5"].values
        ma10 = df["ma_10"].values
        ma20 = df["ma_20"].values
        ma60 = df["ma_60"].values
        close = df["close"].values

        trend = np.full(len(df), 3, dtype=int)
        for i in range(len(df)):
            if close[i] > ma5[i] > ma10[i] > ma20[i] > ma60[i]:
                trend[i] = 0  # strong_bull
            elif close[i] > ma5[i] > ma10[i] > ma20[i]:
                trend[i] = 1  # bull
            elif close[i] > ma20[i]:
                trend[i] = 2  # mild_bull
            elif close[i] < ma5[i] < ma10[i] < ma20[i] < ma60[i]:
                trend[i] = 6  # strong_bear
            elif close[i] < ma5[i] < ma10[i] < ma20[i]:
                trend[i] = 5  # bear
            elif close[i] < ma20[i]:
                trend[i] = 4  # mild_bear
            else:
                trend[i] = 3  # neutral
        df["trend_level"] = trend
        return df

    def _compute_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [5, 10, 20, 60]:
            ma_col = f"ma_{period}"
            if ma_col in df.columns:
                df[f"bias_ma{period}"] = (
                    (df["close"] - df[ma_col]) / df[ma_col] * 100
                )
        return df

    def _classify_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume_ratio" in df.columns:
            df["volume_shrink"] = df["volume_ratio"] < self.VOLUME_SHRINK_RATIO
            df["volume_heavy"] = df["volume_ratio"] > self.VOLUME_HEAVY_RATIO
        else:
            df["volume_shrink"] = False
            df["volume_heavy"] = False
        return df

    def _compute_buy_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        signal = np.full(len(df), 3, dtype=int)
        for i in range(len(df)):
            score = 0
            trend = df["trend_level"].iloc[i]
            if trend <= 1:
                score -= 2
            elif trend <= 2:
                score -= 1
            elif trend >= 5:
                score += 2
            elif trend >= 4:
                score += 1

            bias = df.get("bias_ma5", pd.Series(0)).iloc[i]
            if abs(bias) > self.BIAS_THRESHOLD:
                score += 1 if bias > 0 else -1

            if df.get("volume_shrink", pd.Series(False)).iloc[i]:
                score += 1
            if df.get("volume_heavy", pd.Series(False)).iloc[i]:
                score -= 1

            if score <= -3:
                signal[i] = 0  # strong_buy
            elif score <= -1:
                signal[i] = 1  # buy
            elif score == 0:
                signal[i] = 2  # mild_buy
            elif score == 1:
                signal[i] = 3  # mild_sell
            elif score <= 3:
                signal[i] = 4  # sell
            else:
                signal[i] = 5  # strong_sell
        df["buy_signal"] = signal
        return df
