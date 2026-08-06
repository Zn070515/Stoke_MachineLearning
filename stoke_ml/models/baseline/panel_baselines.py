"""Panel baselines: Ridge / LightGBM / MLP / naive momentum.

The VSN+xLSTM panel model is complex; simple baselines are
trained on the SAME prebuilt features, SAME non-overlapping walk-forward folds,
and scored by the SAME ``evaluate_portfolio`` sleeve-account evaluator, so
"xLSTM wins" is a real claim instead of an assumption.  The naive momentum
baseline ("过去收益均值") is the floor to beat: if the complex model
cannot beat it, the complexity adds no value.

Every baseline reduces to a (N, n_windows) cross-sectional score grid consumed
by the evaluator through the standard ``model(static, pk, po)`` interface.

Feature contract (point-in-time): a window ending at entry column ``e`` uses the
feature cross-section at column ``e-1`` — the last feature day before buying at
``open[e]``.  ``PanelDataset`` already hands the model ``static`` at the decision
column and ``pk/po`` windows whose last column is that same day, so a fitted
baseline can extract its feature vector straight from the batch (stateless,
robust to any batching).  The naive momentum baseline has no fit; its grid is
built from prices outside the model and replayed through a position-tracking
adapter that mirrors the evaluator DataLoader's stock-major order.
"""
import numpy as np
import torch
import torch.nn as nn


def entry_column_features(
    static: torch.Tensor, pk: torch.Tensor, po: torch.Tensor,
    with_seq: bool = False,
) -> torch.Tensor:
    """(B, D) point-in-time cross-section at the decision column.

    ``static`` may be 2D (N, D) legacy or 3D (N, T, D) PIT; the
    Dataset returns it already sliced to the decision column in the 3D case, so
    only the 3D form needs its last column taken.  ``pk``/``po`` are (B, T, D)
    input windows; column -1 is the decision day (the day before entry at
    ``open[e]``), known before the entry fill.  ``with_seq`` appends the same
    ``sequence_summary`` the training path uses, so a
    baseline fitted with history features sees identical vectors at eval time.
    """
    s = static[:, -1, :] if static.dim() == 3 else static
    out = [s, pk[:, -1, :], po[:, -1, :]]
    if with_seq:
        seq = sequence_summary(
            pk.detach().cpu().numpy(), po.detach().cpu().numpy())
        out.append(torch.from_numpy(seq).to(pk.dtype))
    return torch.cat(out, dim=1)


class FittedScoreAdapter(nn.Module):
    """Wrap a fitted sklearn-style regressor (``predict(X) -> (n,)``).

    ``forward`` returns the panel-model's (dir, ret, vol) triple so the
    evaluator is unchanged; only ``ret`` (the cross-sectional score) is used for
    ranking / IC / quintiles.
    """

    def __init__(self, model, with_seq: bool = False):
        super().__init__()
        self.model = model
        self.with_seq = with_seq

    def reset(self):
        """Stateless adapter: no replay position to clear (API parity with
        ``PrecomputedScoreAdapter`` so the benchmark loop can reset() both)."""
        return self

    def forward(self, static, pk, po):
        X = entry_column_features(
            static, pk, po, with_seq=self.with_seq).detach().cpu().numpy()
        score = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1)
        B = score.shape[0]
        return (
            torch.zeros(B, 1),
            torch.from_numpy(score).unsqueeze(-1),
            torch.zeros(B, 1),
        )


class PrecomputedScoreAdapter(nn.Module):
    """Replay a precomputed (N, n_windows) score grid through the evaluator.

    §七/§十六 date-centric: the evaluator iterates windows sequentially
    (shuffle=False) and for each window returns all eval-eligible stocks in
    ascending index order.  The flat grid is therefore stored in
    **window-major** order (window 0 stock 0..N-1, window 1 stock 0..N-1, …)
    so that sequential calls to ``forward`` consume the correct per-window
    chunks.  ``reset()`` must be called before each evaluation; the position
    buffer makes the alignment explicit and assertable instead of silently
    depending on iteration order.
    """

    def __init__(self, grid: np.ndarray):
        super().__init__()
        # grid is (N, W).  Transpose to (W, N) → flatten yields window-major
        # order: [w0_s0, w0_s1, …, w0_sN-1, w1_s0, …] — matches date-centric
        # sequential iteration where each fwd call receives one window's stocks
        # in ascending index order.
        grid_np = np.asarray(grid, dtype=np.float32)
        if grid_np.ndim != 2:
            raise ValueError(
                f"PrecomputedScoreAdapter expects 2D (N, W) grid, "
                f"got shape {grid_np.shape}"
            )
        flat = np.ascontiguousarray(grid_np.T).reshape(-1)
        self.register_buffer("_grid", torch.from_numpy(flat))
        self.register_buffer("_pos", torch.zeros((), dtype=torch.long))

    def forward(self, static, pk, po):
        B = static.shape[0]
        pos = int(self._pos.item())
        if pos + B > self._grid.shape[0]:
            raise RuntimeError(
                f"PrecomputedScoreAdapter ran past its grid "
                f"({pos}+{B} > {self._grid.shape[0]}); call reset() per eval"
            )
        chunk = self._grid[pos:pos + B].unsqueeze(-1)
        self._pos += B
        return (
            torch.zeros(B, 1),
            chunk,
            torch.zeros(B, 1),
        )

    def reset(self):
        self._pos.zero_()


class ScaledPredictor:
    """Model wrapper that scales features before calling the fitted model.

    The scaler is fit on the fold's training rows ONLY (a
    full-history fit would leak future cross-sectional statistics into early
    test days).  The same wrapper is used at eval time so train and test see
    identical preprocessing.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))


SEQ_LAGS = (1, 3, 5)


def sequence_summary_dim(pk_dim: int, po_dim: int, lags=SEQ_LAGS) -> int:
    """Width of ``sequence_summary``: per-channel mean/std/slope + lags."""
    return (3 + len(lags)) * (pk_dim + po_dim)


def sequence_summary(
    pk: np.ndarray, po: np.ndarray, lags=SEQ_LAGS,
) -> np.ndarray:
    """(N, T, D) feature windows → (N, D_seq) trailing-history summary.

    For every channel of ``pk`` and ``po``, summarize the window ending at the
    decision day (column T-1) — the SAME ``seq_len`` context the xLSTM sees:
    trailing mean, std, and linear-trend slope over the window, plus the
    flattened values ``lags`` days before the decision day.  Zero-padded
    pre-listing columns contribute zeros, matching what the model actually
    sees.  ``build_flat_samples`` and ``FittedScoreAdapter`` share this exact
    function so train-time and eval-time vectors are identical by construction
    (give baselines history information so a comparison can
    isolate sequence MODELING from "just saw more history").
    """
    parts: list[np.ndarray] = []
    for x in (pk, po):
        arr = np.asarray(x, dtype=np.float64)
        n, T, D = arr.shape
        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        t = np.arange(T, dtype=np.float64)
        t_c = t - t.mean()
        denom = float((t_c * t_c).sum())
        if denom > 0.0:
            xbar = arr.mean(axis=1, keepdims=True)
            slope = ((arr - xbar) * t_c[None, :, None]).sum(axis=1) / denom
        else:
            slope = np.zeros((n, D), dtype=np.float64)
        parts.extend([
            mean, std,
            np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0),
        ])
        for lag in lags:
            if lag < T:
                parts.append(arr[:, -1 - lag])
            else:
                parts.append(np.zeros((n, D), dtype=np.float64))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _stratified_quotas(counts, max_rows: int) -> np.ndarray:
    """Allocate ``max_rows`` across entry columns proportional to each column's
    valid-row count (largest-remainder), so the cap samples date-stratified
    instead of greedily taking the earliest columns."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= max_rows:
        return counts
    base = max_rows / total
    quotas = np.floor(counts * base).astype(np.int64)
    quotas[counts == 0] = 0
    rem = int(max_rows - quotas.sum())
    frac = counts * base - np.floor(counts * base)
    cand = quotas < counts
    order = np.argsort(-frac, kind="stable")
    for i in order:
        if rem <= 0:
            break
        if cand[i]:
            quotas[i] += 1
            rem -= 1
    return quotas


def fit_mlp_with_early_stopping(
    Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
    seed: int = 0, max_epochs: int = 60, patience: int = 8, min_delta: float = 1e-5,
):
    """Train an MLP with chronological early stopping on a held-out val split.

    sklearn's ``MLPRegressor`` only early-stops on a RANDOM fraction of the
    given training set.  Baselines tune on the same
    chronological inner_val region the deep model uses for checkpoint selection;
    ``partial_fit`` runs one epoch per call, so we score the val split after
    every epoch and keep the best state.
    """
    import copy

    from sklearn.neural_network import MLPRegressor

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
        max_iter=1, random_state=seed, shuffle=True,
    )
    best = None
    best_val = float("inf")
    bad = 0
    for _ in range(max_epochs):
        mlp.partial_fit(Xtr, ytr)
        val = float(np.mean((mlp.predict(Xval) - yval) ** 2))
        if val < best_val - min_delta:
            best_val = val
            best = copy.deepcopy(mlp)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best if best is not None else mlp


def build_flat_samples(
    data: dict,
    entry_start: int,
    entry_end: int,
    label_gate: str = "return_target_mask",
    extra_gates: tuple[str, ...] = ("decision_eligible_mask",),
    max_rows: int | None = None,
    seq_features: bool = False,
    seq_len: int | None = None,
    sample_rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-in-time (X, y) samples for a panel slice.

    For each entry column ``e`` in [entry_start, entry_end): the feature vector
    is the cross-section at column ``e-1`` (static / past_known / past_observed
    concatenated), and the target is ``y_return[:, e]`` — the same open-to-open
    forward-return label the xLSTM fits, already cross-sectionally z-scored by
    the fold loop.  Rows are kept only where ``label_gate`` AND every
    ``extra_gate`` are True at column ``e``; all-zero feature rows (ZI padding
    of untraded stocks) are dropped so the model does not learn from empty
    padding.

    ``max_rows`` caps the sample count DATE-STRATIFIED: the
    budget is allocated proportionally across entry columns so recent and early
    samples both stay represented — the old greedy stop sampled only the
    earliest columns.  ``seq_features`` appends the shared ``sequence_summary``
    of the trailing ``seq_len`` window so baselines see the
    same history the xLSTM sees; the window must equal the one the evaluator
    hands ``FittedScoreAdapter`` for train/eval consistency.
    """
    static = data["static_features"]  # (N, T, D) PIT
    pk = data["past_known"]           # (N, T, D)
    po = data["past_observed"]        # (N, T, D)
    y = data["y_return"]              # (N, T)
    gate = data[label_gate]           # (N, T)
    n, T = y.shape
    pk_dim = int(pk.shape[2])
    po_dim = int(po.shape[2])
    n_feat = int(static.shape[2] + pk_dim + po_dim)
    n_seq = sequence_summary_dim(pk_dim, po_dim) if seq_features else 0
    out_dim = n_feat + n_seq

    def _col_mask(e: int, X_e: np.ndarray) -> np.ndarray:
        keep = gate[:, e] & (y[:, e] == y[:, e])  # finite target
        for g in extra_gates:
            keep &= data[g][:, e]
        if keep.any():
            keep &= np.linalg.norm(X_e, axis=1) > 0.0
        return keep

    # Phase 1 — count valid rows per column (gates + norm) without storing every
    # column, so the stratified cap knows each date's budget before collecting.
    counts: list[int] = []
    for e in range(entry_start, min(entry_end, T)):
        fcol = e - 1
        X_e = np.concatenate([static[:, fcol], pk[:, fcol], po[:, fcol]], axis=1)
        counts.append(int(_col_mask(e, X_e).sum()))
    if not counts or sum(counts) == 0:
        return np.empty((0, out_dim), dtype=np.float32), np.empty(0, dtype=np.float32)
    if max_rows is not None:
        quotas = _stratified_quotas(np.asarray(counts), max_rows)
    else:
        quotas = np.asarray(counts)
    rng = sample_rng if sample_rng is not None else np.random.RandomState(0)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for c, q in enumerate(quotas):
        if q <= 0:
            continue
        e = entry_start + c
        fcol = e - 1
        X_e = np.concatenate([static[:, fcol], pk[:, fcol], po[:, fcol]], axis=1)
        keep = _col_mask(e, X_e)
        idx = np.where(keep)[0]
        if idx.size == 0:
            continue
        pick = idx if idx.size <= q else rng.choice(idx, size=int(q), replace=False)
        if seq_features:
            win = min(seq_len or T, e)  # at most seq_len cols ending at decision day
            if e - win < 0:
                pad = np.zeros((n, win - e, pk_dim), dtype=pk.dtype)
                wnd_pk = np.concatenate([pad, pk[:, :e]], axis=1)
                pad = np.zeros((n, win - e, po_dim), dtype=po.dtype)
                wnd_po = np.concatenate([pad, po[:, :e]], axis=1)
            else:
                wnd_pk = pk[:, e - win:e]
                wnd_po = po[:, e - win:e]
            seq = sequence_summary(wnd_pk, wnd_po)
            X_part = np.concatenate([X_e[pick], seq[pick]], axis=1)
        else:
            X_part = X_e[pick]
        X_parts.append(X_part)
        y_parts.append(y[pick, e])
    if not X_parts:
        return np.empty((0, out_dim), dtype=np.float32), np.empty(0, dtype=np.float32)
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    yv = np.concatenate(y_parts, axis=0).astype(np.float32)
    return X, yv


def build_momentum_grid(
    data: dict,
    entry_start: int,
    entry_end: int,
    lookback: int,
) -> np.ndarray:
    """Naive "past return mean" score grid: trailing mean daily return.

    Grid entry (i, d) for entry column ``e = entry_start + d`` is the mean simple
    daily close-to-close return over [e-lookback, e).  No fit and no features —
    the ultimate floor to beat.  Only data before entry is used
    (PIT); missing closes are forward-filled and gaps treated as 0 return.
    """
    close = np.asarray(data["close_price"], dtype=np.float64)  # (N, T) padded
    c = np.array(close, dtype=np.float64)
    mask = np.isfinite(c)
    idx = np.where(mask, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = np.take_along_axis(c, idx, axis=1)
    row_ok = mask.any(axis=1)
    first_valid = np.where(row_ok, np.argmax(mask, axis=1), c.shape[1])
    col = np.arange(c.shape[1])[None, :]
    filled = np.where(col < first_valid[:, None], 0.0, filled)
    filled = np.nan_to_num(filled, nan=0.0)

    daily = np.zeros_like(filled)
    valid = (filled[:, 1:] > 0) & (filled[:, :-1] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        daily[:, 1:] = np.where(
            valid, filled[:, 1:] / filled[:, :-1] - 1.0, 0.0)

    n_days = entry_end - entry_start
    grid = np.zeros((filled.shape[0], n_days), dtype=np.float32)
    for d in range(n_days):
        e = entry_start + d
        lo = max(1, e - lookback)
        grid[:, d] = daily[:, lo:e].mean(axis=1)
    return grid
