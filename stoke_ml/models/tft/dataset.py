import torch
from torch.utils.data import Dataset
import numpy as np


class PanelDataset(Dataset):
    """Panel dataset for TFT training.

    Pre-built tensor data organized as (N_stocks, T_total, D_features).
    Each __getitem__ returns a single stock's sequence window.

    For panel training, the collate function groups same-date stocks
    across different sequences — see panel_collate().
    """

    def __init__(
        self,
        data: dict,
        seq_len: int = 252,
        stride: int = 1,
    ):
        self.static_features = data["static_features"]  # (N, S)
        self.past_known = data["past_known"]  # (N, T, P)
        self.past_observed = data["past_observed"]  # (N, T, O)
        self.y_direction = data["y_direction"]  # (N, T)
        self.y_return = data["y_return"]  # (N, T)
        self.y_volatility = data["y_volatility"]  # (N, T)
        self.dates = data.get("dates", None)
        self.stock_codes = data.get("stock_codes", None)

        self.seq_len = seq_len
        self.stride = stride
        self.n_stocks = self.past_known.shape[0]
        self.n_timesteps = self.past_known.shape[1]
        self.n_windows = self.n_timesteps - seq_len

        if self.n_windows <= 0:
            raise ValueError(
                f"n_timesteps ({self.n_timesteps}) must be > seq_len ({seq_len})"
            )

    def __len__(self) -> int:
        return self.n_stocks * self.n_windows

    def __getitem__(self, idx: int) -> tuple:
        stock_idx = idx // self.n_windows
        window_idx = idx % self.n_windows

        start = window_idx
        end = start + self.seq_len

        static = self.static_features[stock_idx]  # (S,)
        pk = self.past_known[stock_idx, start:end]  # (T, P)
        po = self.past_observed[stock_idx, start:end]  # (T, O)
        y_dir = self.y_direction[stock_idx, end]  # target at t+1
        y_ret = self.y_return[stock_idx, end]
        y_vol = self.y_volatility[stock_idx, end]

        # Convert targets to tensors (handle both np and torch inputs)
        if isinstance(y_dir, torch.Tensor):
            y_dir = y_dir.clone().detach().long()
        else:
            y_dir = torch.tensor(y_dir, dtype=torch.long)
        if isinstance(y_ret, torch.Tensor):
            y_ret = y_ret.clone().detach().float()
        else:
            y_ret = torch.tensor(y_ret, dtype=torch.float32)
        if isinstance(y_vol, torch.Tensor):
            y_vol = y_vol.clone().detach().float()
        else:
            y_vol = torch.tensor(y_vol, dtype=torch.float32)

        return static, pk, po, y_dir, y_ret, y_vol


def panel_collate(batch: list) -> tuple:
    """Collate panel samples into batch tensors."""
    statics = torch.stack([b[0] for b in batch])
    past_knowns = torch.stack([b[1] for b in batch])
    past_observeds = torch.stack([b[2] for b in batch])
    y_dirs = torch.stack([b[3] for b in batch])
    y_rets = torch.stack([b[4] for b in batch]).unsqueeze(-1)
    y_vols = torch.stack([b[5] for b in batch]).unsqueeze(-1)
    return statics, past_knowns, past_observeds, y_dirs, y_rets, y_vols
