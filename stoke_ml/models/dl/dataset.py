"""PyTorch Dataset for stock time series data."""
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Time series dataset for stock prediction.

    Args:
        X: Feature tensor of shape (n_samples, seq_len, n_features)
        y: Target tensor of shape (n_samples,) or (n_samples, n_tasks)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = (
            torch.from_numpy(y).long()
            if y.ndim == 1
            else torch.from_numpy(y).float()
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
