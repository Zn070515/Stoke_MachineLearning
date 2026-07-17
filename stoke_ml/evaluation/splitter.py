"""Walk-forward (fixed-size sliding window) data splitter.

Time series data cannot be randomly shuffled. Walk-forward validation
trains on fixed-size historical windows and validates on subsequent
periods, stepping forward to prevent lookahead bias.

A purge gap separates training and validation to prevent information
leakage when inputs are sliding windows whose targets overlap the
boundary (e.g. a 60-day window predicting 5-day returns).
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
        purge_days: int = 0,
    ):
        self.train_days = train_years * 252
        self.val_days = val_months * 21
        self.step_days = step_months * 21
        self.purge_days = purge_days

    def split(self, dates: pd.DatetimeIndex | np.ndarray):
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.values
        n = len(dates)

        start = 0
        while True:
            train_end = start + self.train_days
            val_start = train_end + self.purge_days
            val_end = val_start + self.val_days
            if val_end > n:
                break
            train_idx = np.arange(start, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx
            start += self.step_days
