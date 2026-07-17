"""Model evaluation metrics — classification and financial."""
import numpy as np


def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-class Matthews Correlation Coefficient.

    Uses the sklearn implementation when available, falls back to a
    confusion-matrix-based computation for 3-class case.
    """
    try:
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true, y_pred))
    except ImportError:
        pass
    # Manual multi-class MCC via confusion matrix
    classes = np.unique(np.concatenate([y_true, y_pred]))
    k = len(classes)
    if k < 2:
        return 0.0
    C = np.zeros((k, k), dtype=np.int64)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        C[class_to_idx[t], class_to_idx[p]] += 1
    # MCC = (N * tr(C) - sum(t_k * p_k)) / sqrt((N^2 - sum(t_k^2)) * (N^2 - sum(p_k^2)))
    N = C.sum()
    t_k = C.sum(axis=1)
    p_k = C.sum(axis=0)
    cov = N * np.trace(C) - np.dot(t_k, p_k)
    denom = np.sqrt((N**2 - np.dot(t_k, t_k)) * (N**2 - np.dot(p_k, p_k)))
    if denom == 0:
        return 0.0
    return float(cov / denom)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Multi-class classification metrics for 3-class direction labels.

    Labels: 0=DOWN, 1=FLAT, 2=UP. Filtered for -100 (masked) positions.
    """
    mask = (y_true != -100) & (y_pred != -100)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "mcc": 0.0}

    classes = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    per_class_precision = []
    per_class_recall = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        per_class_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        per_class_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    accuracy = np.mean(y_true == y_pred)
    precision = float(np.mean(per_class_precision))
    recall = float(np.mean(per_class_recall))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc_score(y_true, y_pred)),
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a summary statistic.

    Args:
        values: 1-D array of per-stock/per-fold metric values.
        statistic: "mean" only for now.
        n_boot: number of bootstrap resamples.
        alpha: 1 - confidence level (0.05 → 95% CI).
        random_state: seed for reproducibility.

    Returns:
        (ci_low, ci_high) tuple.
    """
    rng = np.random.RandomState(random_state)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(len(values), size=len(values), replace=True)
        if statistic == "mean":
            boot_stats[i] = np.mean(values[idx])
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
    boot_stats = np.sort(boot_stats)
    lo = int(n_boot * alpha / 2)
    hi = int(n_boot * (1 - alpha / 2))
    return float(boot_stats[lo]), float(boot_stats[hi])


def compute_financial_metrics(
    prices: np.ndarray, predictions: np.ndarray
) -> dict:
    n_returns = len(prices) - 1
    price_returns = np.diff(prices) / prices[:-1]
    preds_aligned = predictions[:n_returns]
    # 3-class direction mapping: 0=DOWN→short(-1), 1=FLAT→neutral(0), 2=UP→long(+1)
    strategy_returns = price_returns * (preds_aligned - 1)

    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns))
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_dd = float(np.min(drawdowns))

    wins = np.sum(strategy_returns > 0)
    total = len(strategy_returns)
    win_rate = wins / total if total > 0 else 0.0

    gross_profit = float(np.sum(strategy_returns[strategy_returns > 0]))
    gross_loss = float(abs(np.sum(strategy_returns[strategy_returns < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }
