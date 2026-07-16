"""Technical indicators — MA, MACD, RSI, KDJ, BOLL, ATR, volume, momentum.

Inspired by Qlib Alpha158 factor set (158 factors). Covers:
- K-bar statistics (9): KUP, KLOW, KSFT, KMID, KLEN (shadow/body ratios)
- Price standardization (3): OPEN0, HIGH0, LOW0
- Rolling window stats (5 windows × 20 types): MAX, MIN, QTL, RANK, RSV,
  CORR, CORD, BETA, RSQR, RESI, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD,
  VMA, VSTD, IMAX, IMIN, IMXD
- Momentum/volatility/volume-price dimensions

All new columns are built in dicts and batch-assigned via pd.concat at the end
to avoid DataFrame fragmentation (PerformanceWarning from frame.insert).
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

        new = {}  # collect all new columns here

        # ── 1. K-bar features (Alpha158 K系列) ────────────────────────
        new.update(_kbar_features(open_, high, low, close))

        # ── 2. Price standardization (Alpha158 价格系列) ──────────────
        new.update(_price_features(open_, high, low, close))

        # ── 3. Moving averages ─────────────────────────────────────────
        for period in [5, 10, 20, 60, 120]:
            new[f"ma_{period}"] = close.rolling(period).mean()

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        new["ema_12"] = ema_12
        new["ema_26"] = ema_26

        # ── 4. MACD ────────────────────────────────────────────────────
        macd_dif = ema_12 - ema_26
        macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
        new["macd_dif"] = macd_dif
        new["macd_dea"] = macd_dea
        new["macd_hist"] = 2 * (macd_dif - macd_dea)

        # ── 5. RSI ─────────────────────────────────────────────────────
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            new[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ── 6. KDJ ────────────────────────────────────────────────────
        for period in [9, 14]:
            low_n = low.rolling(period).min()
            high_n = high.rolling(period).max()
            rsv = (close - low_n) / (high_n - low_n).replace(0, 1e-10) * 100
            k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
            d = k.ewm(alpha=1 / 3, adjust=False).mean()
            new[f"kdj_k_{period}"] = k
            new[f"kdj_d_{period}"] = d
            new[f"kdj_j_{period}"] = 3 * k - 2 * d

        # ── 7. Bollinger Bands ────────────────────────────────────────
        boll_mid = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std
        new["boll_mid"] = boll_mid
        new["boll_upper"] = boll_upper
        new["boll_lower"] = boll_lower
        bb_range = boll_upper - boll_lower
        new["boll_pct"] = (close - boll_lower) / bb_range.replace(0, 1e-10)

        # ── 8. ATR ─────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        new["atr_14"] = tr.rolling(14).mean()

        # ── 9. ROC ────────────────────────────────────────────────────
        for period in [6, 12, 20]:
            new[f"roc_{period}"] = (
                (close - close.shift(period))
                / close.shift(period).replace(0, 1e-10) * 100
            )

        # ── 10. WR ─────────────────────────────────────────────────────
        for period in [10, 20]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            new[f"wr_{period}"] = (
                (high_n - close) / (high_n - low_n).replace(0, 1e-10) * -100
            )

        # ── 11. CCI ────────────────────────────────────────────────────
        for period in [14, 20]:
            tp = (high + low + close) / 3
            tp_ma = tp.rolling(period).mean()
            tp_mad = tp.rolling(period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            new[f"cci_{period}"] = (
                (tp - tp_ma) / tp_mad.replace(0, 1e-10) / 0.015
            )

        # ── 12. Historical volatility ──────────────────────────────────
        log_ret = np.log(np.maximum(close / close.shift(1), 1e-12))
        for period in [5, 20]:
            new[f"vol_{period}"] = (
                log_ret.rolling(period).std() * np.sqrt(252)
            )

        # ── 13. Volume indicators ───────────────────────────────────────
        volume_ma5 = volume.rolling(5).mean()
        new["volume_ma5"] = volume_ma5
        new["volume_ratio"] = volume / volume_ma5.replace(0, 1)
        new["obv"] = (
            volume * ((close.diff() > 0).astype(int) * 2 - 1)
        ).cumsum()
        up_vol = volume.where(close.diff() > 0, 0)
        new["vol_up_ratio_20"] = (
            up_vol.rolling(20).sum()
            / volume.rolling(20).sum().replace(0, 1e-10)
        )

        # ── 14. Amount-based (if available) ────────────────────────────
        if amount is not None:
            amount = amount.astype(float)
            amount_ma5 = amount.rolling(5).mean()
            new["amount_ma5"] = amount_ma5
            new["amount_ratio"] = amount / amount_ma5.replace(0, 1e-10)
            new["turnover_proxy"] = amount / close.replace(0, 1e-10)

        # ── 15. Rolling window: position stats (Alpha158) ──────────────
        new.update(_rolling_position(high, low, close))

        # ── 16. Rolling window: price-volume correlation (Alpha158) ────
        new.update(_rolling_corr(close, volume))

        # ── 17. Rolling window: trend & momentum (Alpha158) ────────────
        new.update(_rolling_trend(close, volume))

        # ── 18. Rolling window: up/down stats (Alpha158 CNT/SUM) ──────
        new.update(_rolling_counts(close, volume))

        # ── 19. Rolling window: Aroon & volume stats (Alpha158) ───────
        new.update(_rolling_aroon_vol(high, low, close, volume))

        # ── 20. ADX — Average Directional Index (trend strength) ──────
        new.update(_adx(high, low, close))

        # ── 21. MFI — Money Flow Index (volume-weighted RSI) ───────────
        new.update(_mfi(high, low, close, volume))

        # ── 22. CMO — Chande Momentum Oscillator ───────────────────────
        new.update(_cmo(close))

        # ── 23. TRIX — Triple exponential average ─────────────────────
        new.update(_trix(close))

        # ── Batch-assign all new columns at once (no fragmentation) ────
        new_df = pd.DataFrame(new, index=result.index)
        result = pd.concat([result, new_df], axis=1)

        # ── Clean up intermediate columns ──────────────────────────────
        for col in ["pct_change", "vol_change"]:
            if col in result.columns:
                result = result.drop(columns=[col])

        return result


# ── Module-level helpers (return dicts to avoid DataFrame fragmentation) ──


def _kbar_features(open_, high, low, close):
    """K-bar microstructural features (Alpha158 K系列, 9 factors)."""
    hl_range = high - low
    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)
    return {
        "kmid": (close - open_) / open_.replace(0, 1e-10),
        "klen": hl_range / open_.replace(0, 1e-10),
        "kmid2": (close - open_) / hl_range.replace(0, 1e-10),
        "kup": (high - upper_body) / open_.replace(0, 1e-10),
        "kup2": (high - upper_body) / hl_range.replace(0, 1e-10),
        "klow": (lower_body - low) / open_.replace(0, 1e-10),
        "klow2": (lower_body - low) / hl_range.replace(0, 1e-10),
        "ksft": (2 * close - high - low) / open_.replace(0, 1e-10),
        "ksft2": (2 * close - high - low) / hl_range.replace(0, 1e-10),
    }


def _price_features(open_, high, low, close):
    """Price standardization relative to close (Alpha158, 3 factors)."""
    return {"open0": open_ / close, "high0": high / close, "low0": low / close}


def _rolling_position(high, low, close):
    """Rolling max/min/quantile/rank/RSV (Alpha158, 5 windows × 6 types)."""
    out = {}
    for d in _WINDOWS:
        high_n = high.rolling(d).max()
        low_n = low.rolling(d).min()
        out[f"max_{d}d"] = high_n / close
        out[f"min_{d}d"] = low_n / close
        out[f"qtlu_{d}d"] = close.rolling(d).quantile(0.8) / close
        out[f"qtld_{d}d"] = close.rolling(d).quantile(0.2) / close
        out[f"rank_{d}d"] = close.rolling(d).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        out[f"rsv_{d}d"] = (close - low_n) / (high_n - low_n).replace(0, 1e-10)
    return out


def _rolling_corr(close, volume):
    """Price-volume correlations (Alpha158 CORR/CORD, 5 windows × 2 types)."""
    log_vol = np.log(volume + 1)
    ret = close / close.shift(1) - 1
    vol_ret = volume / volume.shift(1).replace(0, 1e-10) - 1
    log_vol_ret = np.log(vol_ret.abs() + 1)
    out = {}
    for d in _WINDOWS:
        out[f"corr_{d}d"] = close.rolling(d).corr(log_vol)
        out[f"cord_{d}d"] = ret.rolling(d).corr(log_vol_ret)
    return out


def _rolling_trend(close, volume):
    """Trend linearity & volume MA/STD (Alpha158, 5 windows × 5 types)."""
    out = {}
    for d in _WINDOWS:
        x = np.arange(d)
        sx = x.sum()
        sxx = (x * x).sum()
        denom = d * sxx - sx * sx

        def _slope(win):
            sy = win.sum()
            sxy = (x * win.values).sum()
            return (d * sxy - sx * sy) / denom if denom != 0 else 0.0

        slope = close.rolling(d).apply(_slope, raw=False)
        out[f"beta_{d}d"] = slope / close.replace(0, 1e-10)

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

        out[f"rsqr_{d}d"] = close.rolling(d).apply(_rsqr, raw=False)

        def _resi_std(win):
            y = win.values
            slope_val = (d * (x * y).sum() - sx * y.sum()) / denom if denom != 0 else 0
            intercept = (y.sum() - slope_val * sx) / d
            pred = slope_val * x + intercept
            return (y[-1] - pred[-1]) / y[-1] if y[-1] != 0 else 0.0

        out[f"resi_{d}d"] = close.rolling(d).apply(_resi_std, raw=False)

        out[f"vma_{d}d"] = volume.rolling(d).mean() / volume.replace(0, 1e-10)
        out[f"vstd_{d}d"] = volume.rolling(d).std() / volume.replace(0, 1e-10)
    return out


def _rolling_counts(close, volume):
    """Up/down day counts & RSI-style sums (Alpha158, 5 windows × 6 types)."""
    ret = close.diff()
    up = ret > 0
    down = ret < 0
    abs_ret = ret.abs()
    out = {}
    for d in _WINDOWS:
        cntp = up.rolling(d).mean()
        cntn = down.rolling(d).mean()
        out[f"cntp_{d}d"] = cntp
        out[f"cntn_{d}d"] = cntn
        out[f"cntd_{d}d"] = cntp - cntn
        sum_pos = ret.clip(lower=0).rolling(d).sum()
        sum_neg = (-ret).clip(lower=0).rolling(d).sum()
        sum_abs = abs_ret.rolling(d).sum()
        sump = sum_pos / sum_abs.replace(0, 1e-10)
        sumn = sum_neg / sum_abs.replace(0, 1e-10)
        out[f"sump_{d}d"] = sump
        out[f"sumn_{d}d"] = sumn
        out[f"sumd_{d}d"] = sump - sumn
    return out


def _rolling_aroon_vol(high, low, close, volume):
    """Aroon & volume change stats (Alpha158, 5 windows × 6 types)."""
    vol_change = volume.diff()
    abs_ret = (close.diff() / close.shift(1).replace(0, 1e-10)).abs()
    out = {}
    for d in _WINDOWS:
        def _aroon_up(win):
            return np.argmax(win.values) / (len(win) - 1) if len(win) > 1 else 0.5
        def _aroon_down(win):
            return np.argmin(win.values) / (len(win) - 1) if len(win) > 1 else 0.5

        imax = high.rolling(d).apply(_aroon_up, raw=False)
        imin = low.rolling(d).apply(_aroon_down, raw=False)
        out[f"imax_{d}d"] = imax
        out[f"imin_{d}d"] = imin
        out[f"imxd_{d}d"] = imax - imin

        out[f"wvma_{d}d"] = (
            (abs_ret * volume).rolling(d).std()
            / (abs_ret * volume).rolling(d).mean().replace(0, 1e-10)
        )

        vol_up = vol_change.clip(lower=0)
        vol_down = (-vol_change).clip(lower=0)
        sum_vabs = vol_change.abs().rolling(d).sum()
        vsump = vol_up.rolling(d).sum() / sum_vabs.replace(0, 1e-10)
        vsumn = vol_down.rolling(d).sum() / sum_vabs.replace(0, 1e-10)
        out[f"vsump_{d}d"] = vsump
        out[f"vsumn_{d}d"] = vsumn
        out[f"vsumd_{d}d"] = vsump - vsumn
    return out


def _adx(high, low, close):
    """ADX — Average Directional Index (14-day, trend strength indicator)."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di14 = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, 1e-10)
    minus_di14 = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14.replace(0, 1e-10)
    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, 1e-10)
    adx14 = dx.ewm(alpha=1 / 14, adjust=False).mean()
    adxr = (adx14 + adx14.shift(14)) / 2

    return {
        "adx": adx14,
        "adxr": adxr,
        "pdi": plus_di14,
        "mdi": minus_di14,
    }


def _mfi(high, low, close, volume):
    """MFI — Money Flow Index (14-day, volume-weighted RSI)."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    tp_diff = tp.diff()
    pos_flow = raw_mf.where(tp_diff > 0, 0.0)
    neg_flow = raw_mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    mr = pos_sum / neg_sum.replace(0, 1e-10)
    mfi14 = 100 - (100 / (1 + mr))
    return {"mfi_14": mfi14}


def _cmo(close):
    """CMO — Chande Momentum Oscillator (14-day)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    sum_gain = gain.rolling(14).sum()
    sum_loss = loss.rolling(14).sum()
    cmo14 = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss).replace(0, 1e-10)
    return {"cmo_14": cmo14}


def _trix(close):
    """TRIX — Triple exponential average oscillator (15-day)."""
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, 1e-10) * 100
    return {"trix": trix}
