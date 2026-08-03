import torch
from torch.utils.data import Dataset, Sampler


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

        # valid_mask[stock, window] = 1.0 iff the target at end of that window
        # is a real observation.  Stocks with shorter listing history are padded
        # with y_direction == -100; windows whose target lands in the pad region
        # must never be sampled (they are all-zero features → garbage gradients).
        self.valid_mask = (self.y_direction[:, self.seq_len:] != -100)

        # date_idx[t] = t for each window position — used by PairwiseRankingLoss
        # to group samples from the same calendar date for cross-sectional ranking.
        date_indices = data.get("date_indices")
        if date_indices is not None:
            self.date_indices = _to_tensor(date_indices, torch.long)
        else:
            self.date_indices = None

        if self.date_indices is not None and self.date_indices.shape[1] < self.n_timesteps:
            raise ValueError(
                f"date_indices width ({self.date_indices.shape[1]}) < n_timesteps "
                f"({self.n_timesteps}) — cannot index target date at window end"
            )

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

        # Target is at `end` (the step after the window [start, end)), so the
        # date used to group this sample cross-sectionally is the TARGET date,
        # not the last feature date (`end - 1`).  Ranking pairs must compare
        # stocks' outcomes on the SAME future day.
        date_idx = (self.date_indices[stock_idx, end].item()
                     if self.date_indices is not None else 0)

        return (
            self.static_features[stock_idx],
            self.past_known[stock_idx, start:end],
            self.past_observed[stock_idx, start:end],
            self.y_direction[stock_idx, end],
            self.y_return[stock_idx, end],
            self.y_volatility[stock_idx, end],
            date_idx,
        )


class DateGroupedSampler(Sampler):
    """Groups samples by calendar date so each batch has cross-sectional diversity.

    Standard random shuffle produces batches where almost no two samples share
    the same date (batch_size << n_stocks). This sampler instead shuffles dates,
    then within each date shuffles stocks, so consecutive indices all belong to
    the same date.  When DataLoader batches these consecutive indices, every
    batch naturally contains multiple stocks from the same date(s) —
    PairwiseRankingLoss then has meaningful same-date pairs to compare.

    Only (stock, window) pairs with a valid target are emitted — padded windows
    whose target is -100 (short listing history) are skipped, so batches never
    contain all-zero feature rows for the ranking loss to trip on.

    Args:
        valid_mask: (N_stocks, N_windows) bool tensor — True where the window's
            target is a real observation.  Built by PanelDataset.
    """

    def __init__(self, valid_mask: torch.Tensor):
        self.valid_mask = valid_mask.bool()
        self.n_stocks, self.n_windows = self.valid_mask.shape

    def __len__(self) -> int:
        return int(self.valid_mask.sum().item())

    def __iter__(self):
        # Shuffle dates
        date_order = torch.randperm(self.n_windows).tolist()
        indices = []
        for window_idx in date_order:
            stock_order = torch.randperm(self.n_stocks).tolist()
            for stock_idx in stock_order:
                if self.valid_mask[stock_idx, window_idx]:
                    indices.append(stock_idx * self.n_windows + window_idx)
        return iter(indices)


def panel_collate(batch: list) -> tuple:
    """Collate panel samples into batch tensors (includes date_idx)."""
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        torch.stack([b[5] for b in batch]),
        torch.tensor([b[6] for b in batch], dtype=torch.long),
    )
