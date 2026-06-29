"""Technical indicators — MA, MACD, RSI, KDJ, BOLL, ATR, volume, momentum.

Inspired by Qlib Alpha158 factor set (158 factors). Covers:
- K-bar statistics (9): KUP, KLOW, KSFT, KMID, KLEN (shadow/body ratios)
- Price standardization (3): OPEN0, HIGH0, LOW0
- Rolling window stats (5 windows × 20 types): MAX, MIN, QTL, RANK, RSV,
  CORR, CORD, BETA, RSQR, RESI, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD,
  VMA, VSTD, IMAX, IMIN, IMXD
- Momentum/volatility/volume-price dimensions
"""
import pandas as pd
import numpy as np

_WINDOWS = [5, 10, 20, 30, 60]


class TechnicalIndicators:
    """Compute Alpha158-style technical indicators from OHLCV data."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        amount = df.get("amount", None)

        # ── 1. K-bar features (Alpha158 K系列) ────────────────────────
        self._add_kbar_features(result, open_, high, low, close)

        # ── 2. Price standardization (Alpha158 价格系列) ──────────────
        self._add_price_features(result, open_, high, low, close)

        # ── 3. Moving averages ─────────────────────────────────────────
        for period in [5, 10, 20, 60, 120]:
            result[f"ma_{period}"] = close.rolling(period).mean()

        result["ema_12"] = close.ewm(span=12, adjust=False).mean()
        result["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # ── 4. MACD ────────────────────────────────────────────────────
        result["macd_dif"] = result["ema_12"] - result["ema_26"]
        result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

        # ── 5. RSI ─────────────────────────────────────────────────────
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ── 6. KDJ ────────────────────────────────────────────────────
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

        # ── 7. Bollinger Bands ────────────────────────────────────────
        result["boll_mid"] = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        result["boll_upper"] = result["boll_mid"] + 2 * boll_std
        result["boll_lower"] = result["boll_mid"] - 2 * boll_std
        bb_range = result["boll_upper"] - result["boll_lower"]
        result["boll_pct"] = (close - result["boll_lower"]) / bb_range.replace(0, 1e-10)

        # ── 8. ATR ─────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean()

        # ── 9. ROC ────────────────────────────────────────────────────
        for period in [6, 12, 20]:
            result[f"roc_{period}"] = (
                (close - close.shift(period)) / close.shift(period).replace(0, 1e-10) * 100
            )

        # ── 10. WR ─────────────────────────────────────────────────────
        for period in [10, 20]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            result[f"wr_{period}"] = (high_n - close) / (high_n - low_n).replace(0, 1e-10) * -100

        # ── 11. CCI ────────────────────────────────────────────────────
        for period in [14, 20]:
            tp = (high + low + close) / 3
            tp_ma = tp.rolling(period).mean()
            tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            result[f"cci_{period}"] = (tp - tp_ma) / tp_mad.replace(0, 1e-10) / 0.015

        # ── 12. Historical volatility ──────────────────────────────────
        for period in [5, 20]:
            log_ret = np.log(close / close.shift(1))
            result[f"vol_{period}"] = log_ret.rolling(period).std() * np.sqrt(252)

        # ── 13. Volume indicators ───────────────────────────────────────
        result["volume_ma5"] = volume.rolling(5).mean()
        result["volume_ratio"] = volume / result["volume_ma5"].replace(0, 1)
        result["obv"] = (
            volume * ((close.diff() > 0).astype(int) * 2 - 1)
        ).cumsum()

        up_vol = volume.where(close.diff() > 0, 0)
        down_vol = volume.where(close.diff() < 0, 0)
        result["vol_up_ratio_20"] = (
            up_vol.rolling(20).sum() / volume.rolling(20).sum().replace(0, 1e-10)
        )

        # ── 14. Amount-based (if available) ────────────────────────────
        if amount is not None:
            amount = amount.astype(float)
            result["amount_ma5"] = amount.rolling(5).mean()
            result["amount_ratio"] = amount / result["amount_ma5"].replace(0, 1e-10)
            result["turnover_proxy"] = amount / close.replace(0, 1e-10)

        # ── 15. Rolling window: position stats (Alpha158) ──────────────
        self._add_rolling_position(result, high, low, close)

        # ── 16. Rolling window: price-volume correlation (Alpha158) ────
        self._add_rolling_corr(result, close, volume)

        # ── 17. Rolling window: trend & momentum (Alpha158) ────────────
        self._add_rolling_trend(result, close, volume)

        # ── 18. Rolling window: up/down stats (Alpha158 CNT/SUM) ──────
        self._add_rolling_counts(result, close, volume)

        # ── 19. Rolling window: Aroon & volume stats (Alpha158) ───────
        self._add_rolling_aroon_vol(result, high, low, close, volume)

        # ── Clean up intermediate columns ──────────────────────────────
        for col in ["pct_change", "vol_change"]:
            if col in result.columns:
                result = result.drop(columns=[col])

        return result

    # ── Private helpers for Alpha158 factor groups ──────────────────

    @staticmethod
    def _add_kbar_features(df, open_, high, low, close):
        """K-bar microstructural features (Alpha158 K系列, 9 factors)."""
        hl_range = high - low
        df["kmid"] = (close - open_) / open_.replace(0, 1e-10)
        df["klen"] = hl_range / open_.replace(0, 1e-10)
        df["kmid2"] = (close - open_) / hl_range.replace(0, 1e-10)
        # Upper shadow
        upper_body = np.maximum(open_, close)
        df["kup"] = (high - upper_body) / open_.replace(0, 1e-10)
        df["kup2"] = (high - upper_body) / hl_range.replace(0, 1e-10)
        # Lower shadow
        lower_body = np.minimum(open_, close)
        df["klow"] = (lower_body - low) / open_.replace(0, 1e-10)
        df["klow2"] = (lower_body - low) / hl_range.replace(0, 1e-10)
        # Settlement position
        df["ksft"] = (2 * close - high - low) / open_.replace(0, 1e-10)
        df["ksft2"] = (2 * close - high - low) / hl_range.replace(0, 1e-10)

    @staticmethod
    def _add_price_features(df, open_, high, low, close):
        """Price standardization relative to close (Alpha158, 3 factors)."""
        df["open0"] = open_ / close
        df["high0"] = high / close
        df["low0"] = low / close

    @staticmethod
    def _add_rolling_position(df, high, low, close):
        """Rolling max/min/quantile/rank/RSV (Alpha158, 5 windows × 6 types)."""
        for d in _WINDOWS:
            high_n = high.rolling(d).max()
            low_n = low.rolling(d).min()
            # MAX/MIN — already partially present, now use standardized form
            df[f"max_{d}d"] = high_n / close
            df[f"min_{d}d"] = low_n / close
            # Quantiles
            df[f"qtlu_{d}d"] = close.rolling(d).quantile(0.8) / close
            df[f"qtld_{d}d"] = close.rolling(d).quantile(0.2) / close
            # Rank percentile
            df[f"rank_{d}d"] = close.rolling(d).apply(
                lambda x: (x.iloc[-1] > x).mean(), raw=False
            )
            # RSV (KDJ raw)
            df[f"rsv_{d}d"] = (close - low_n) / (high_n - low_n).replace(0, 1e-10)

    @staticmethod
    def _add_rolling_corr(df, close, volume):
        """Price-volume correlations (Alpha158 CORR/CORD, 5 windows × 2 types)."""
        log_vol = np.log(volume + 1)
        ret = close / close.shift(1) - 1
        vol_ret = volume / volume.shift(1).replace(0, 1e-10) - 1
        log_vol_ret = np.log(vol_ret.abs() + 1)

        for d in _WINDOWS:
            df[f"corr_{d}d"] = close.rolling(d).corr(log_vol)
            # CORD: correlation of daily returns with volume changes
            df[f"cord_{d}d"] = ret.rolling(d).corr(log_vol_ret)

    @staticmethod
    def _add_rolling_trend(df, close, volume):
        """Trend linearity & volume MA/STD (Alpha158, 5 windows × 5 types)."""
        log_ret = np.log(close / close.shift(1))
        for d in _WINDOWS:
            # BETA: slope / close
            x = np.arange(d)
            sx = x.sum()
            sxx = (x * x).sum()
            denom = d * sxx - sx * sx

            def _slope(win):
                sy = win.sum()
                sxy = (x * win.values).sum()
                return (d * sxy - sx * sy) / denom if denom != 0 else 0.0

            slope = close.rolling(d).apply(_slope, raw=False)
            df[f"beta_{d}d"] = slope / close.replace(0, 1e-10)

            # RSQR: R² of linear trend
            def _rsqr(win):
                y = win.values
                sst = ((y - y.mean()) ** 2).sum()
                if sst == 0:
                    return 1.0
                slope_val = (d * (x * y).sum() - sx * y.sum()) / denom if denom != 0 else 0
                intercept = (y.sum() - slope_val * sx) / d
                pred = slope_val * x + intercept
                ssr = ((y - pred) ** 2).sum()
                return 1 - ssr / sst

            df[f"rsqr_{d}d"] = close.rolling(d).apply(_rsqr, raw=False)

            # RESI: regression residual std
            def _resi_std(win):
                y = win.values
                slope_val = (d * (x * y).sum() - sx * y.sum()) / denom if denom != 0 else 0
                intercept = (y.sum() - slope_val * sx) / d
                pred = slope_val * x + intercept
                return (y[-1] - pred[-1]) / y[-1] if y[-1] != 0 else 0.0

            df[f"resi_{d}d"] = close.rolling(d).apply(_resi_std, raw=False)

            # VMA / VSTD
            df[f"vma_{d}d"] = volume.rolling(d).mean() / volume.replace(0, 1e-10)
            df[f"vstd_{d}d"] = volume.rolling(d).std() / volume.replace(0, 1e-10)

    @staticmethod
    def _add_rolling_counts(df, close, volume):
        """Up/down day counts & RSI-style sums (Alpha158, 5 windows × 6 types)."""
        ret = close.diff()
        up = ret > 0
        down = ret < 0
        abs_ret = ret.abs()

        for d in _WINDOWS:
            # CNTP/CNTN/CNTD
            df[f"cntp_{d}d"] = up.rolling(d).mean()
            df[f"cntn_{d}d"] = down.rolling(d).mean()
            df[f"cntd_{d}d"] = df[f"cntp_{d}d"] - df[f"cntn_{d}d"]
            # SUMP/SUMN/SUMD (smoothed RSI variant)
            sum_pos = ret.clip(lower=0).rolling(d).sum()
            sum_neg = (-ret).clip(lower=0).rolling(d).sum()
            sum_abs = abs_ret.rolling(d).sum()
            df[f"sump_{d}d"] = sum_pos / sum_abs.replace(0, 1e-10)
            df[f"sumn_{d}d"] = sum_neg / sum_abs.replace(0, 1e-10)
            df[f"sumd_{d}d"] = df[f"sump_{d}d"] - df[f"sumn_{d}d"]

    @staticmethod
    def _add_rolling_aroon_vol(df, high, low, close, volume):
        """Aroon & volume change stats (Alpha158, 5 windows × 6 types)."""
        vol_change = volume.diff()
        abs_ret = (close.diff() / close.shift(1)).abs()
        for d in _WINDOWS:
            # IMAX/IMIN: index position of high/low within window
            def _aroon_up(win):
                return np.argmax(win.values) / (len(win) - 1) if len(win) > 1 else 0.5
            def _aroon_down(win):
                return np.argmin(win.values) / (len(win) - 1) if len(win) > 1 else 0.5

            df[f"imax_{d}d"] = high.rolling(d).apply(_aroon_up, raw=False)
            df[f"imin_{d}d"] = low.rolling(d).apply(_aroon_down, raw=False)
            df[f"imxd_{d}d"] = df[f"imax_{d}d"] - df[f"imin_{d}d"]

            # WVMA: volume-weighted return volatility
            df[f"wvma_{d}d"] = (
                (abs_ret * volume).rolling(d).std()
                / (abs_ret * volume).rolling(d).mean().replace(0, 1e-10)
            )

            # VSUMP/VSUMN/VSUMD: volume up/down change ratios
            vol_up = vol_change.clip(lower=0)
            vol_down = (-vol_change).clip(lower=0)
            sum_vabs = vol_change.abs().rolling(d).sum()
            df[f"vsump_{d}d"] = vol_up.rolling(d).sum() / sum_vabs.replace(0, 1e-10)
            df[f"vsumn_{d}d"] = vol_down.rolling(d).sum() / sum_vabs.replace(0, 1e-10)
            df[f"vsumd_{d}d"] = df[f"vsump_{d}d"] - df[f"vsumn_{d}d"]
