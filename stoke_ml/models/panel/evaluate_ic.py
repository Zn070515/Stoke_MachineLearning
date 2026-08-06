"""Information-coefficient (IC) diagnostics for the panel evaluator (§二十一).

Extracted from ``stoke_ml.models.panel.evaluate`` — per-day Spearman rank IC,
the shared clean-IC definition (``_raw_clean_rank_ic``) used BOTH by the formal
report and by checkpoint selection, and the candidate-pool eligibility mask.
``evaluate`` re-exports these for backward compatibility.
"""
import numpy as np
import torch
from scipy.stats import spearmanr


def _compute_daily_ic(
    preds_np: np.ndarray,
    actuals_np: np.ndarray,
    mask_np: np.ndarray | None = None,
    min_stocks: int = 10,
) -> list[float]:
    """Per-day Spearman rank IC.

    mask_np: (n_stocks, n_windows) bool — candidate pool (entry-eligible) on
    each day. Without it, zero-feature padded rows (short listing history) get
    garbage predictions that drag down the rank correlation.  A day is skipped
    unless it holds >= `min_stocks` eligible stocks — the unified threshold
    so checkpoint selection and the formal report agree.
    Days whose predictions are constant (std < eps) have a degenerate Spearman
    and are skipped.
    """
    daily_ics = []
    n_windows = preds_np.shape[1]
    for t in range(n_windows):
        p = preds_np[:, t]
        a = actuals_np[:, t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask_np is not None:
            mask = mask & mask_np[:, t]
        if mask.sum() >= min_stocks:
            pv = p[mask]
            if pv.std() < 1e-8:
                continue  # constant predictions → rank correlation undefined
            ic, _ = spearmanr(pv, a[mask])
            if np.isfinite(ic):
                daily_ics.append(ic)
    return daily_ics


def _raw_clean_rank_ic(
    val_data: dict,
    preds_np: np.ndarray,
    n_windows: int,
    seq_len: int,
    min_stocks: int = 10,
    diag: dict | None = None,
) -> tuple[list[float] | None, torch.Tensor | None]:
    """Per-day Spearman IC of predictions vs the RAW open->open return.

    The single shared clean-IC definition used BOTH by the formal report
    (evaluate_portfolio) and by checkpoint selection (train._compute_val_loss).  Ranking against the
    z-scored + clipped [-5,5] model
    target manufactures ties and can select a different checkpoint than the
    report; both must rank the raw clean return (y_return_raw, saved before
    the fold z-score; falls back to y_return for data without
    a raw copy).

    The candidate pool is decision & history & return-target,
    so weak cross-sections are filtered identically to the report.  Returns
    (None, None) when val_data has no y_return_raw / y_return — the caller
    falls back to the legacy realized-return reconstruction.

    `diag` (optional mutable dict) receives the per-window pool statistics the
    failure path needs: valid_days / avg_stocks_per_day /
    mask_retention.
    """
    pool = _candidate_pool(val_data, n_windows, seq_len).numpy()
    if "return_target_mask" in val_data:
        rt = np.asarray(val_data["return_target_mask"], dtype=bool)
        pool = pool & rt[:, seq_len:seq_len + n_windows]
    if diag is not None:
        per_day = pool.sum(axis=0)
        diag["valid_days"] = int((per_day >= min_stocks).sum())
        diag["avg_stocks_per_day"] = float(
            per_day[per_day > 0].mean()) if per_day.any() else 0.0
        diag["mask_retention"] = float(pool.mean())
    clean_key = "y_return_raw" if "y_return_raw" in val_data else "y_return"
    if clean_key not in val_data:
        return None, None
    clean_actuals = torch.as_tensor(val_data[clean_key], dtype=torch.float32)[
        :, seq_len:seq_len + n_windows]
    daily_ics = _compute_daily_ic(
        preds_np, clean_actuals.numpy(), pool, min_stocks=min_stocks)
    return daily_ics, clean_actuals


def _newey_west_t(series: np.ndarray, lag: int) -> float:
    """Autocorrelation-robust t-stat (Newey & West 1987, Bartlett kernel)."""
    n = len(series)
    if n < 2:
        return float("nan")
    mean = float(series.mean())
    x = series - mean
    gamma0 = float(np.dot(x, x) / n)
    if gamma0 < 1e-12:
        if abs(mean) < 1e-12:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    var = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(x[:-k], x[k:]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    var = max(var, gamma0 * 1e-8)  # guard against negative NW variance
    return float(mean / np.sqrt(var / n))


def compute_ic_summary(daily_ics: list[float], horizon: int = 1) -> dict:
    """IC mean, std, IR, positivity rate, and a Newey-West t-stat.

    The plain mean/std ratio ignores serial correlation —
    with overlapping-horizon labels the per-day IC series is autocorrelated,
    so a naive IR overstates signal.  `ic_newey_west_t` uses the NW-1994
    automatic lag truncation (Bartlett kernel), floored to horizon-1 so the
    overlap is actually captured.
    """
    if not daily_ics:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
                "ic_pos_rate": 0.0, "ic_newey_west_t": 0.0}
    arr = np.array(daily_ics, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    n = len(arr)
    lag = max(horizon - 1, int(np.floor(4 * (n / 100.0) ** (2 / 9.0))))
    lag = min(lag, n - 1)
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 1e-8 else 0.0,
        "ic_pos_rate": float((arr > 0).mean()),
        "ic_newey_west_t": _newey_west_t(arr, lag),
    }


def _candidate_pool(data: dict, n_windows: int, seq_len: int) -> torch.Tensor:
    """Selection pool for window day t (column t ↔ panel column seq_len+t).

    Rank over the DECISION pool — close[t-1] is real
    (signal after close[t-1]) AND the seq_len-window ending at t-1 holds >=
    min_history real observations.  Both masks are aligned to the ENTRY column
    t.  Falls back to entry-eligibility, then to future-label validity, for
    synthetic/legacy data without the decision/history masks.
    """
    has_dh = "decision_eligible_mask" in data and "history_eligible_mask" in data
    if has_dh:
        dec = torch.as_tensor(data["decision_eligible_mask"])
        hist = torch.as_tensor(data["history_eligible_mask"])
        return dec[:, seq_len:seq_len + n_windows] & hist[:, seq_len:seq_len + n_windows]
    if "entry_eligible_mask" in data:
        entry = torch.as_tensor(data["entry_eligible_mask"])
        return entry[:, seq_len:seq_len + n_windows]
    y_dir = torch.as_tensor(data["y_direction"])
    return (y_dir[:, seq_len:seq_len + n_windows] != -100)
