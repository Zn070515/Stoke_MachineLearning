"""Shared vectorized rolling-window helpers (numpy, O(n) cumsum-based).

Used by TemporalTransformer, EmotionRefiner, and FundamentalRefiner.
Centralised here to avoid duplicated implementations drifting apart.
"""
import numpy as np


def rolling_mean(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Rolling mean with expanding fallback for early positions (look-back only)."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n < min_periods:
        return out
    cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
    if n >= window:
        out[window - 1] = cumsum[window - 1] / window  # first full window
        out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(min_periods - 1, min(window - 1, n)):
        out[i] = np.nanmean(arr[:i + 1])
    return out


def rolling_std(arr: np.ndarray, window: int, min_periods: int = 2) -> np.ndarray:
    """Rolling std with expanding fallback for early positions (look-back only)."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n < min_periods:
        return out
    for i in range(min_periods - 1, min(window - 1, n)):
        out[i] = np.nanstd(np.nan_to_num(arr[:i + 1], 0.0))
    if n >= window:
        cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
        cumsum2 = np.cumsum(np.nan_to_num(arr, 0.0) ** 2)
        n_full = n - window + 1
        win_sum = np.empty(n_full, dtype=np.float64)
        win_sum2 = np.empty(n_full, dtype=np.float64)
        win_sum[0] = cumsum[window - 1]
        win_sum2[0] = cumsum2[window - 1]
        win_sum[1:] = cumsum[window:] - cumsum[:-window]
        win_sum2[1:] = cumsum2[window:] - cumsum2[:-window]
        var = win_sum2 / window - (win_sum / window) ** 2
        var = np.maximum(var, 0.0)
        out[window - 1:] = np.sqrt(var)
    return out


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum (look-back only)."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return out
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.nanmin(arr[start:i + 1])
    return out


def rolling_quantile(arr: np.ndarray, window: int, q: float,
                     min_periods: int = 5) -> np.ndarray:
    """Rolling quantile via sliding_window_view with expanding fallback."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < min_periods:
        return out
    expanding_end = min(window - 1, n)
    for i in range(min_periods - 1, expanding_end):
        out[i] = np.quantile(arr[:i + 1], q, method="linear")
    if n >= window:
        win = sliding_window_view(arr, window)
        qt = np.quantile(win, q, axis=1, method="linear")
        out[window - 1:] = qt
    return out


def rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Linear regression slope over rolling window."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_demean = x - x_mean
    x_ss = (x_demean ** 2).sum()
    if x_ss < 1e-10:
        return out
    win = sliding_window_view(arr, window)
    y_mean = win.mean(axis=1)
    slope = ((win - y_mean[:, None]) * x_demean[None, :]).sum(axis=1) / x_ss
    out[window - 1:] = slope
    return out


def rolling_percentile_rank(arr: np.ndarray, window: int,
                            min_periods: int = 63) -> np.ndarray:
    """Rolling percentile rank: fraction of values in window < current.

    Uses sliding window for full *window* observations; falls back to
    expanding window when min_periods <= available < window.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < min_periods:
        return out
    expanding_end = min(window - 1, n)
    for i in range(min_periods - 1, expanding_end):
        past = arr[:i + 1]
        out[i] = (past < arr[i]).mean()
    if n >= window:
        win = sliding_window_view(arr, window)
        current = arr[window - 1:]
        ranks = (win < current[:, None]).mean(axis=1)
        out[window - 1:] = ranks
    return out


def sign_streak(arr: np.ndarray) -> np.ndarray:
    """Consecutive days with same sign (0 = neutral reset)."""
    n = len(arr)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if arr[i] * arr[i - 1] > 0:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    return streak


def skew_proxy(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling (mean - median) / (std + eps) as skew proxy."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    eps = 1e-8
    if n < window:
        return out
    win = sliding_window_view(arr, window)
    mu = win.mean(axis=1)
    md = np.median(win, axis=1)
    sd = win.std(axis=1, ddof=1)
    out[window - 1:] = (mu - md) / (sd + eps)
    return out


def expanding_zscore(arr: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Expanding-window z-score: (v - expanding_mean) / expanding_std.

    Only uses data up to position i — no look-ahead bias.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < min_periods:
        return out
    cumsum = np.cumsum(np.nan_to_num(arr, 0.0))
    cumsum2 = np.cumsum(np.nan_to_num(arr, 0.0) ** 2)
    eps = 1e-8
    for i in range(min_periods - 1, n):
        count = i + 1
        mean = cumsum[i] / count
        var = cumsum2[i] / count - mean * mean
        sd = np.sqrt(max(var, 0.0))
        out[i] = (arr[i] - mean) / max(sd, eps)
    return out


def zscore_cross_section(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization (mean 0, std 1) for a single series."""
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd < 1e-10:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def accel(arr: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Acceleration: ma_fast - ma_slow."""
    ma_fast = rolling_mean(arr, fast)
    ma_slow = rolling_mean(arr, slow)
    out = ma_fast - ma_slow
    out[np.isnan(ma_fast) | np.isnan(ma_slow)] = np.nan
    return out


def zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Z-score: (v - ma_window) / std_window with expanding fallback."""
    ma = rolling_mean(arr, window)
    std = rolling_std(arr, window)
    valid = std > 1e-10
    out = np.full(len(arr), np.nan, dtype=np.float64)
    out[valid] = (arr[valid] - ma[valid]) / std[valid]
    return out
