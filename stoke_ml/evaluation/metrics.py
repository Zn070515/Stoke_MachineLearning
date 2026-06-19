"""Model evaluation metrics — classification and financial."""
import numpy as np


def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    n = len(y_true)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc_score(y_true, y_pred)),
    }


def compute_financial_metrics(
    prices: np.ndarray, predictions: np.ndarray
) -> dict:
    price_returns = np.diff(prices) / prices[:-1]
    strategy_returns = price_returns * (2 * predictions - 1)

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
