import torch
from torch.utils.data import Dataset


class PanelDataset(Dataset):
    """Panel dataset for VSN+xLSTM model training.

    Pre-built tensor data organized as (N_stocks, T_total, D_features).
    Each __getitem__ returns a single stock's sequence window.

    All numpy inputs are converted to tensors once in __init__ so that
    __getitem__ is a pure indexing operation — no per-sample conversion
    overhead.
    """

    def __init__(
        self,
        data: dict,
        seq_len: int = 60,
    ):
        self.seq_len = seq_len

        def _to_tensor(arr, dtype):
            if isinstance(arr, torch.Tensor):
                return arr.clone().detach().to(dtype)
            return torch.from_numpy(arr).to(dtype)

        self.static_features = _to_tensor(data["static_features"], torch.float32)
        self.past_known = _to_tensor(data["past_known"], torch.float32)
        self.past_observed = _to_tensor(data["past_observed"], torch.float32)
        self.y_direction = _to_tensor(data["y_direction"], torch.long)
        self.y_return = _to_tensor(data["y_return"], torch.float32)
        self.y_volatility = _to_tensor(data["y_volatility"], torch.float32)

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

        return (
            self.static_features[stock_idx],
            self.past_known[stock_idx, start:end],
            self.past_observed[stock_idx, start:end],
            self.y_direction[stock_idx, end],
            self.y_return[stock_idx, end],
            self.y_volatility[stock_idx, end],
        )


def panel_collate(batch: list) -> tuple:
    """Collate panel samples into batch tensors."""
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        torch.stack([b[5] for b in batch]),
    )
