"""Walk-forward (expanding window) data splitter.

Time series data cannot be randomly shuffled. Walk-forward validation
trains on expanding historical windows and validates on subsequent
periods, preventing lookahead bias.
"""
import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """Generate train/validation splits respecting time order."""

    def __init__(
        self,
        train_years: int = 2,
        val_months: int = 3,
        step_months: int = 3,
    ):
        self.train_days = train_years * 252
        self.val_days = val_months * 21
        self.step_days = step_months * 21

    def split(self, dates: pd.DatetimeIndex | np.ndarray):
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.values
        n = len(dates)

        start = 0
        while True:
            train_end = start + self.train_days
            val_end = train_end + self.val_days
            if val_end > n:
                break
            train_idx = np.arange(start, train_end)
            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx
            start += self.step_days
