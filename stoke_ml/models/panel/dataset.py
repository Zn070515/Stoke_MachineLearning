import numpy as np
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

    §十六 lazy storage: the big per-timestep arrays (``static_features`` when
    3D, ``past_known``, ``past_observed`` and the three target grids) are kept
    as array references — dense ``np.ndarray`` / ``torch.Tensor`` inputs are
    eagerly converted to tensors as before, but ``np.memmap`` inputs are left
    lazy and sliced+converted per-window inside ``__getitem__`` (a "lazy
    window gather" that page-faults only the rows/columns actually read).  A
    memmap-backed dataset therefore never holds the whole dense (N, T, D)
    grid in RAM, and produces outputs elementwise-identical to the dense one.
    The masks and ``date_indices`` are always eager tensors (they are small
    bool/int grids needed at init to build the valid/eval masks).

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
            # A read-only source (e.g. a memmap view) must be copied — torch
            # can't wrap a non-writable numpy buffer without undefined-behavior
            # writes.  Writable dense arrays keep the zero-copy from_numpy path.
            if not arr.flags.writeable:
                arr = arr.copy()
            return torch.from_numpy(arr).to(dtype)

        # §十六: the big per-timestep arrays stay lazy when fed as np.memmap
        # (sliced+converted per window in __getitem__); dense ndarray/tensor
        # inputs keep the eager conversion so the default path is unchanged.
        self.static_features = self._init_big_array(data["static_features"], torch.float32)
        self.past_known = self._init_big_array(data["past_known"], torch.float32)
        self.past_observed = self._init_big_array(data["past_observed"], torch.float32)
        self.y_direction = self._init_big_array(data["y_direction"], torch.long)
        self.y_return = self._init_big_array(data["y_return"], torch.float32)
        self.y_volatility = self._init_big_array(data["y_volatility"], torch.float32)

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
            # _row_slice_tensor handles a lazy (memmap) y_direction so
            # target_any stays a tensor for the eager-mask AND/OR below.
            target_any = (self._row_slice_tensor(self.y_direction, self.seq_len) != -100)
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
            # §: eval_mask must agree with the downstream _candidate_pool
            # (evaluate_ic.py), which uses ONLY the panel-builder's
            # history_eligible_mask (decision & history).  When that mask is
            # present it already encodes the builder's min_history window rule,
            # so re-applying ``hist_count >= self.min_history`` here would
            # DIVERGE whenever config.min_history differs from the builder's —
            # pool-eligible stocks would get NaN preds and silently drop out of
            # IC.  hist_count stays as the fallback for legacy data that has no
            # history_eligible_mask.
            if history_eligible is not None:
                self.eval_mask = decision & history_eligible
            else:
                self.eval_mask = (hist_count >= self.min_history) & decision
        else:
            # Backward-compat fallback: target-day label validity only.
            # Kept a tensor even for a lazy (memmap) y_direction so valid_mask /
            # eval_mask stay tensor consumers' contract (DateSampler, _rank_pool_stats).
            self.valid_mask = (
                self._row_slice_tensor(self.y_direction, self.seq_len) != -100
            )
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

    @staticmethod
    def _init_big_array(arr, dtype):
        """Keep an np.memmap input lazy; eagerly tensorize a dense input.

        The lazy path (memmap, §十六) keeps the array reference and gathers
        per-window rows inside __getitem__, so a full (N, T, D) grid is never
        materialized as a tensor up front.  Dense ndarray / torch.Tensor
        inputs keep the original eager conversion — behaviour identical to
        the pre-§十六 dataset.
        """
        if isinstance(arr, np.memmap):
            return arr
        if isinstance(arr, torch.Tensor):
            return arr.clone().detach().to(dtype)
        return torch.from_numpy(arr).to(dtype)

    @staticmethod
    def _row_slice_tensor(arr, start):
        """torch tensor of ``arr[:, start:]`` — works for tensor / ndarray /
        memmap.  Used for the (N, T-seq_len) columns needed to build the
        valid/eval masks at init, where the result must stay a tensor even
        when the source is a lazy memmap."""
        if isinstance(arr, torch.Tensor):
            return arr[:, start:]
        out = np.asarray(arr[:, start:])
        if not out.flags.writeable:
            out = out.copy()
        return torch.from_numpy(out)

    @staticmethod
    def _lazy_slice(arr, idx, dtype):
        """Slice ``arr[idx]`` and return a tensor.

        Eager tensors index directly (zero conversion); lazy memmap references
        are gathered (advanced-index read of just the needed rows/columns —
        page-faulting only what the window touches) then converted.  Both paths
        yield elementwise-identical values.
        """
        if isinstance(arr, torch.Tensor):
            return arr[idx]
        out = np.asarray(arr[idx])
        if not out.flags.writeable:
            out = out.copy()
        return torch.from_numpy(out).to(dtype)

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
        # (.ndim works for both the eager tensor and the lazy memmap path.)
        if self.static_features.ndim == 3:
            static = self._lazy_slice(
                self.static_features, (stock_indices, end - 1), torch.float32)
        else:
            static = self._lazy_slice(
                self.static_features, (stock_indices,), torch.float32)

        # Feature windows: (M, seq_len, D) — lazy memmap rows are gathered
        # here (only the window's rows/columns page-fault in, §十六).
        pk = self._lazy_slice(
            self.past_known, (stock_indices, slice(start, end)), torch.float32)
        po = self._lazy_slice(
            self.past_observed, (stock_indices, slice(start, end)), torch.float32)

        # Targets: (M,)
        y_dir = self._lazy_slice(self.y_direction, (stock_indices, end), torch.long)
        y_ret = self._lazy_slice(self.y_return, (stock_indices, end), torch.float32)
        y_vol = self._lazy_slice(self.y_volatility, (stock_indices, end), torch.float32)

        # Date index: (M,) — all stocks share the same target date for this
        # window.  PairwiseRankingLoss groups by this value; with date-centric
        # batches it always gets exactly one date per batch.
        if self.date_indices is not None:
            date_idx = self._lazy_slice(self.date_indices, (stock_indices, end), torch.long)
        else:
            date_idx = torch.full((n_valid,), end, dtype=torch.long)

        # Per-task target masks: each loss applies its own mask.
        # Same semantics as before — y_direction validity for dir, per-channel
        # masks for ret/vol — but now returned as (M,) tensors.
        dir_mask = (y_dir != -100)
        ret_mask = (self._lazy_slice(self.ret_target, (stock_indices, end), torch.bool)
                    if self.ret_target is not None else dir_mask)
        vol_mask = (self._lazy_slice(self.vol_target, (stock_indices, end), torch.bool)
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

    def __init__(self, valid_mask: torch.Tensor,
                 generator: torch.Generator | None = None):
        self.valid_mask = valid_mask.bool()
        self.generator = generator
        self.n_stocks, self.n_windows = self.valid_mask.shape
        # Per-date valid stock count (n_windows,)
        self._date_counts = valid_mask.sum(dim=0)

    def __len__(self) -> int:
        return int((self._date_counts > 0).sum().item())

    def __iter__(self):
        # One randperm call per epoch — deterministic for the supplied
        # generator (falls back to the global RNG when None).
        date_order = torch.randperm(
            self.n_windows, generator=self.generator).tolist()
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
