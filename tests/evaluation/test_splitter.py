"""Tests for WalkForwardSplitter — chronological-only cross-validation."""
import numpy as np
import pytest
from stoke_ml.evaluation.splitter import WalkForwardSplitter


class TestWalkForwardSplitter:

    def test_default_split_produces_at_least_one_window(self):
        splitter = WalkForwardSplitter(train_years=1, val_months=3, step_months=3)
        X = np.arange(500).reshape(-1, 1)  # 500 trading days
        splits = list(splitter.split(X))
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            # Chronological: all train indices < all val indices
            assert train_idx[-1] < val_idx[0]

    def test_small_data_returns_empty(self):
        splitter = WalkForwardSplitter(train_years=1, val_months=3, step_months=3)
        X = np.arange(50).reshape(-1, 1)  # too short
        splits = list(splitter.split(X))
        assert len(splits) == 0

    def test_multiple_windows_no_overlap_between_validation_folds(self):
        splitter = WalkForwardSplitter(train_years=1, val_months=1, step_months=3)
        X = np.arange(1000).reshape(-1, 1)  # ~4 years
        splits = list(splitter.split(X))
        assert len(splits) >= 2
        # Validation sets should not overlap
        val_sets = [set(v) for _, v in splits]
        for i in range(len(val_sets) - 1):
            assert len(val_sets[i] & val_sets[i + 1]) == 0

    def test_indices_are_monotonic(self):
        """Train and validation indices should increase monotonically across windows."""
        splitter = WalkForwardSplitter(train_years=1, val_months=3, step_months=3)
        X = np.arange(800).reshape(-1, 1)
        splits = list(splitter.split(X))
        prev_train_start = -1
        prev_val_start = -1
        for train_idx, val_idx in splits:
            assert train_idx[0] > prev_train_start  # train window slides forward
            assert val_idx[0] > prev_val_start     # val window slides forward
            prev_train_start = train_idx[0]
            prev_val_start = val_idx[0]

    def test_custom_parameters(self):
        splitter = WalkForwardSplitter(
            train_years=2, val_months=6, step_months=6,
        )
        X = np.arange(1000).reshape(-1, 1)
        splits = list(splitter.split(X))
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            # Train set = 2 years * 252 = 504 trading days
            assert len(train_idx) == 504
