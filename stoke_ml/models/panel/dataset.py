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
    """Date-centric panel dataset for VSN+xLSTM model training.

    §七/§十六 refactoring: the primary index is now DATE (window), not
    stock×date.  ``__len__`` = n_windows; ``__getitem__(window_idx)`` returns
    ALL (or a controlled sample of) valid stocks for that date.

    This gives PairwiseRankingLoss a complete cross-section per batch —
    no intra-date pairs lost to batch boundaries — and eliminates the
    materialized (n_stocks × n_windows) flat-index grid.

    All numpy inputs are converted to tensors once in __init__ so that
    __getitem__ is indexing + sampling — no per-sample conversion overhead.

    Parameters
    ----------
    data : dict
        Panel-format data from panel_builder.build_panel_features.
    seq_len : int
        Input window length in trading days.
    min_history : int
        Minimum real observations in the input window for a stock to be
        trainable on that date.
    max_stocks_per_date : int | None
        Cap on stocks per date in __getitem__.  When a date has more valid
        stocks, a random subset is sampled each call.  ``None`` = no cap
        (use all valid stocks — val default).
    training : bool
        If True, sampling is applied (within max_stocks_per_date).
        If False, all valid stocks are returned (used for val/eval).
    """

    def __init__(
        self,
        data: dict,
        seq_len: int = 60,
        min_history: int = 50,
        max_stocks_per_date: int | None = None,
        training: bool = True,
    ):
        self.seq_len = seq_len
        self.min_history = min_history
        self.max_stocks_per_date = max_stocks_per_date
        self.training = training

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

        # Per-task target masks: each loss applies its own mask
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
        self.decision_eligible = (
            _to_tensor(data["decision_eligible_mask"], torch.bool)
            if "decision_eligible_mask" in data else None
        )

        self.n_stocks = self.past_known.shape[0]
        self.n_timesteps = self.past_known.shape[1]
        self.n_windows = self.n_timesteps - seq_len

        if self.entry_eligible is not None and self.obs_mask is not None:
            # Window [w, w+seq_len) is trainable iff the target day w+seq_len is
            # entry-eligible, the input window holds >= min_history real
            # observations (new listings with mostly zero-padded history are
            # excluded), and at least one target mask is set.
            target_any = (self.y_direction[:, self.seq_len:] != -100)
            if self.ret_target is not None:
                target_any = target_any | self.ret_target[:, self.seq_len:]
            if self.vol_target is not None:
                target_any = target_any | self.vol_target[:, self.seq_len:]
            hist_count = _window_history_counts(self.obs_mask, self.seq_len)
            # decision_eligible[t] = close[t-1] real (signal computable after
            # close[t-1]) — adds the guard that we actually rank
            # at a known point in time, not the first day after a suspension.
            decision = (
                self.decision_eligible[:, self.seq_len:]
                if self.decision_eligible is not None else True
            )
            self.valid_mask = (
                self.entry_eligible[:, self.seq_len:]
                & (hist_count >= self.min_history)
                & target_any
                & decision
            )
            # §七 date-centric eval mask: broader than valid_mask — includes
            # all decision- & history-eligible stocks so the evaluation pool
            # (which does NOT require entry_eligible) is fully covered.
            # Training only sees entry-eligible stocks (correct for loss);
            # evaluation predicts for the full candidate pool and lets the
            # sleeve-sim / IC filter downstream (matches old stock-centric
            # behaviour where the model predicted for every stock).
            history_eligible = (
                _to_tensor(data["history_eligible_mask"], torch.bool)[:, self.seq_len:]
                if "history_eligible_mask" in data else None
            )
            history_mask = (
                history_eligible if history_eligible is not None else True
            )
            self.eval_mask = (hist_count >= self.min_history) & decision & history_mask
        else:
            # Backward-compat fallback: target-day label validity only.
            self.valid_mask = (self.y_direction[:, self.seq_len:] != -100)
            self.eval_mask = self.valid_mask

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

        # §七/§十六 date-centric index: for each window, pre-compute the list
        # of stock indices.  Training uses valid_mask (entry-eligible only);
        # evaluation uses eval_mask (broader: decision & history — matches the
        # candidate pool so no pool-eligible stock goes unpredicted).
        _mask = self.valid_mask if self.training else self.eval_mask
        self._date_to_stocks: list[torch.Tensor] = []
        for w in range(self.n_windows):
            valid_stocks = torch.where(_mask[:, w])[0]
            self._date_to_stocks.append(valid_stocks)

    def __len__(self) -> int:
        """Number of date-windows (primary index)."""
        return self.n_windows

    def __getitem__(self, window_idx: int) -> tuple:
        """Return all (or a controlled sample of) valid stocks for one date.

        Returns an 11-tuple where each tensor's batch-dim = M (stocks in this
        date), instead of the old per-stock scalars/vectors.  ``stock_indices``
        (element 10) lets val/eval consumers reconstruct the (N, W) prediction
        grid.
        """
        start = window_idx
        end = start + self.seq_len

        stock_indices = self._date_to_stocks[window_idx]
        n_valid = stock_indices.numel()

        # Controlled sampling for training when a date has more valid stocks
        # than max_stocks_per_date.  The RNG call order — randperm then index
        # select — preserves seed reproducibility (global torch RNG state).
        if (self.training and self.max_stocks_per_date is not None
                and n_valid > self.max_stocks_per_date):
            perm = torch.randperm(n_valid)[:self.max_stocks_per_date]
            stock_indices = stock_indices[perm]
            n_valid = stock_indices.numel()

        # Static context: 2D (N, D) for backward-compat synthetic data, or
        # 3D (N, T, D) PIT.  For 3D take the DECISION column
        # end-1 (the last feature day — known before entering at open[end]).
        if self.static_features.dim() == 3:
            static = self.static_features[stock_indices, end - 1]
        else:
            static = self.static_features[stock_indices]

        # Feature windows: (M, seq_len, D)
        pk = self.past_known[stock_indices, start:end]
        po = self.past_observed[stock_indices, start:end]

        # Targets: (M,)
        y_dir = self.y_direction[stock_indices, end]
        y_ret = self.y_return[stock_indices, end]
        y_vol = self.y_volatility[stock_indices, end]

        # Date index: (M,) — all stocks share the same target date for this
        # window.  PairwiseRankingLoss groups by this value; with date-centric
        # batches it always gets exactly one date per batch.
        if self.date_indices is not None:
            date_idx = self.date_indices[stock_indices, end]
        else:
            date_idx = torch.full((n_valid,), end, dtype=torch.long)

        # Per-task target masks: each loss applies its own mask.
        # Same semantics as before — y_direction validity for dir, per-channel
        # masks for ret/vol — but now returned as (M,) tensors.
        dir_mask = (self.y_direction[stock_indices, end] != -100)
        ret_mask = (self.ret_target[stock_indices, end]
                    if self.ret_target is not None else dir_mask)
        vol_mask = (self.vol_target[stock_indices, end]
                    if self.vol_target is not None else dir_mask)

        return (
            static,          # 0: (M, D)
            pk,              # 1: (M, seq_len, D_pk)
            po,              # 2: (M, seq_len, D_po)
            y_dir,           # 3: (M,)
            y_ret,           # 4: (M,)
            y_vol,           # 5: (M,)
            date_idx,        # 6: (M,)
            dir_mask,        # 7: (M,) bool
            ret_mask,        # 8: (M,) bool
            vol_mask,        # 9: (M,) bool
            stock_indices,   # 10: (M,) long — for grid reconstruction
        )


class DateSampler(Sampler):
    """Streaming date-window sampler for date-centric training (§七/§十六).

    Shuffles window indices each epoch, yielding one at a time.  The
    PanelDataset then returns all (or a controlled sample of) valid stocks
    for that window.  DataLoader ``batch_size=1`` gives one date cross-section
    per batch.

    Dates with zero valid stocks are skipped (they contribute nothing to
    training and would produce empty batches).

    This is a streaming generator — no materialized list of millions of
    (stock, window) flat indices.  The RNG call order (one randperm call)
    is deterministic for a given seed, preserving epoch-to-epoch
    reproducibility.
    """

    def __init__(self, valid_mask: torch.Tensor):
        self.valid_mask = valid_mask.bool()
        self.n_stocks, self.n_windows = self.valid_mask.shape
        # Per-date valid stock count (n_windows,)
        self._date_counts = valid_mask.sum(dim=0)

    def __len__(self) -> int:
        return int((self._date_counts > 0).sum().item())

    def __iter__(self):
        # One randperm call per epoch — deterministic for the global RNG state.
        date_order = torch.randperm(self.n_windows).tolist()
        for w in date_order:
            if self._date_counts[w] > 0:
                yield w


# ── Backward-compatible stock-centric classes (kept for reference / legacy) ──

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

    .. deprecated::
        Use ``DateSampler`` + date-centric ``PanelDataset`` instead (§七/§十六).
        This class is retained for backward compatibility only.

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
        # §七 P0: streamed generator instead of materializing the full
        # (up to n_stocks * n_windows) Python int list in memory.  The RNG
        # call order — one randperm per window, then one per stock inside —
        # is preserved exactly, so the sampling distribution and seed
        # reproducibility are unchanged.
        # Shuffle dates
        date_order = torch.randperm(self.n_windows).tolist()
        for window_idx in date_order:
            stock_order = torch.randperm(self.n_stocks).tolist()
            for stock_idx in stock_order:
                if self.valid_mask[stock_idx, window_idx]:
                    yield stock_idx * self.n_windows + window_idx


def panel_collate(batch: list) -> tuple:
    """Collate date-centric panel samples into batch tensors.

    In date-centric mode, each ``__getitem__`` already returns per-date
    (M, ...) tensors.  ``batch_size=1`` means a single-element list — this
    function passes it through unchanged.  ``batch_size>1`` concatenates
    multiple dates along the stock (dim-0) axis so the model sees a mixed
    batch.

    The 11-element tuple matches ``PanelDataset.__getitem__``:
    (static, pk, po, y_dir, y_ret, y_vol, date_idx, dir_mask, ret_mask,
     vol_mask, stock_indices).
    """
    if len(batch) == 1:
        return batch[0]
    return tuple(torch.cat([b[i] for b in batch], dim=0) for i in range(11))
