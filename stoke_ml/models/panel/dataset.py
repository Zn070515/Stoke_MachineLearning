import torch
from torch.utils.data import Dataset, Sampler


def _window_history_counts(obs_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Real-observation count inside each length-seq_len input window.

    obs_mask: (N, T) bool.  Returns (N, T-seq_len) where out[:, w] =
    obs_mask[:, w : w+seq_len].sum() — how many genuinely-observed days the
    model saw in window w (w = 0..T-seq_len-1, one per trainable target at
    column w+seq_len), used to reject mostly-padded new listings.
    """
    n_stocks, n_timesteps = obs_mask.shape
    cum = torch.cumsum(obs_mask.to(torch.int64), dim=1)
    cum = torch.cat([torch.zeros(n_stocks, 1, dtype=torch.int64), cum], dim=1)
    # cum[:, i] = sum of obs_mask[:, :i] → window count = cum[w+seq] - cum[w];
    # drop the last column whose window [T-seq, T) has no in-bounds target day.
    return (cum[:, seq_len:] - cum[:, :cum.shape[1] - seq_len])[:, :-1].to(torch.float32)


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
        min_history: int = 50,
    ):
        self.seq_len = seq_len
        self.min_history = min_history

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

        # Per-task target masks (review v3 §八): each loss applies its own mask
        # instead of one y_direction != -100 ruling all tasks.  Optional for
        # backward-compat — synthetic test data without masks falls back to the
        # old single-mask behaviour.
        self.obs_mask = (
            _to_tensor(data["observation_mask"], torch.bool)
            if "observation_mask" in data else None
        )
        self.entry_eligible = (
            _to_tensor(data["entry_eligible_mask"], torch.bool)
            if "entry_eligible_mask" in data else None
        )
        self.ret_target = (
            _to_tensor(data["return_target_mask"], torch.bool)
            if "return_target_mask" in data else None
        )
        self.vol_target = (
            _to_tensor(data["vol_target_mask"], torch.bool)
            if "vol_target_mask" in data else None
        )

        self.n_stocks = self.past_known.shape[0]
        self.n_timesteps = self.past_known.shape[1]
        self.n_windows = self.n_timesteps - seq_len

        if self.entry_eligible is not None and self.obs_mask is not None:
            # Window [w, w+seq_len) is trainable iff the target day w+seq_len is
            # entry-eligible, the input window holds >= min_history real
            # observations (new listings with mostly zero-padded history are
            # excluded — review v3 §四), and at least one target mask is set.
            target_any = (self.y_direction[:, self.seq_len:] != -100)
            if self.ret_target is not None:
                target_any = target_any | self.ret_target[:, self.seq_len:]
            if self.vol_target is not None:
                target_any = target_any | self.vol_target[:, self.seq_len:]
            hist_count = _window_history_counts(self.obs_mask, self.seq_len)
            self.valid_mask = (
                self.entry_eligible[:, self.seq_len:]
                & (hist_count >= self.min_history)
                & target_any
            )
        else:
            # Backward-compat fallback: target-day label validity only.
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
        # stocks' outcomes on the SAME future day.  Under the open-entry
        # convention (review v3 §六) `end` is the ENTRY date — buy at open[end],
        # exit at open[end+horizon].
        date_idx = (self.date_indices[stock_idx, end].item()
                     if self.date_indices is not None else 0)

        # Per-task target masks (review v3 §八): each loss applies its own mask.
        dir_mask = (self.y_direction[stock_idx, end] != -100)
        ret_mask = (self.ret_target[stock_idx, end]
                    if self.ret_target is not None else dir_mask)
        vol_mask = (self.vol_target[stock_idx, end]
                    if self.vol_target is not None else dir_mask)

        return (
            self.static_features[stock_idx],
            self.past_known[stock_idx, start:end],
            self.past_observed[stock_idx, start:end],
            self.y_direction[stock_idx, end],
            self.y_return[stock_idx, end],
            self.y_volatility[stock_idx, end],
            date_idx,
            dir_mask,
            ret_mask,
            vol_mask,
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
    """Collate panel samples into batch tensors (includes date_idx + per-task masks)."""
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        torch.stack([b[5] for b in batch]),
        torch.tensor([b[6] for b in batch], dtype=torch.long),
        torch.stack([b[7] for b in batch]),
        torch.stack([b[8] for b in batch]),
        torch.stack([b[9] for b in batch]),
    )
