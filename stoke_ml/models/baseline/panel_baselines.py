"""Panel baselines (review v8 四-2): Ridge / LightGBM / MLP / naive momentum.

The VSN+xLSTM panel model is complex; the v8 review requires simple baselines
trained on the SAME prebuilt features, SAME non-overlapping walk-forward folds,
and scored by the SAME ``evaluate_portfolio`` sleeve-account evaluator, so
"xLSTM wins" is a real claim instead of an assumption.  The naive momentum
baseline ("过去收益均值") is the floor the review names: if the complex model
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
    static: torch.Tensor, pk: torch.Tensor, po: torch.Tensor
) -> torch.Tensor:
    """(B, D) point-in-time cross-section at the decision column.

    ``static`` may be 2D (N, D) legacy or 3D (N, T, D) PIT (review v4 §五); the
    Dataset returns it already sliced to the decision column in the 3D case, so
    only the 3D form needs its last column taken.  ``pk``/``po`` are (B, T, D)
    input windows; column -1 is the decision day (the day before entry at
    ``open[e]``), known before the entry fill.
    """
    s = static[:, -1, :] if static.dim() == 3 else static
    return torch.cat([s, pk[:, -1, :], po[:, -1, :]], dim=1)


class FittedScoreAdapter(nn.Module):
    """Wrap a fitted sklearn-style regressor (``predict(X) -> (n,)``).

    ``forward`` returns the panel-model's (dir, ret, vol) triple so the
    evaluator is unchanged; only ``ret`` (the cross-sectional score) is used for
    ranking / IC / quintiles.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def reset(self):
        """Stateless adapter: no replay position to clear (API parity with
        ``PrecomputedScoreAdapter`` so the benchmark loop can reset() both)."""
        return self

    def forward(self, static, pk, po):
        X = entry_column_features(static, pk, po).detach().cpu().numpy()
        score = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1)
        B = score.shape[0]
        return (
            torch.zeros(B, 1),
            torch.from_numpy(score).unsqueeze(-1),
            torch.zeros(B, 1),
        )


class PrecomputedScoreAdapter(nn.Module):
    """Replay a precomputed (N, n_windows) score grid through the evaluator.

    The evaluator's DataLoader emits samples in stock-major order (``idx =
    stock * n_windows + window``), so a flat grid laid out in the same order
    lines up when ``forward`` is called sequentially.  ``reset()`` must be
    called before each evaluation; the position buffer makes the alignment
    explicit and assertable instead of silently depending on iteration order.
    """

    def __init__(self, grid: np.ndarray):
        super().__init__()
        flat = np.asarray(grid, dtype=np.float32).reshape(-1)
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

    The scaler is fit on the fold's training rows ONLY (review v8 三-1: a
    full-history fit would leak future cross-sectional statistics into early
    test days).  The same wrapper is used at eval time so train and test see
    identical preprocessing.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))


def build_flat_samples(
    data: dict,
    entry_start: int,
    entry_end: int,
    label_gate: str = "return_target_mask",
    extra_gates: tuple[str, ...] = ("decision_eligible_mask",),
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-in-time (X, y) samples for a panel slice.

    For each entry column ``e`` in [entry_start, entry_end): the feature vector
    is the cross-section at column ``e-1`` (static / past_known / past_observed
    concatenated), and the target is ``y_return[:, e]`` — the same open-to-open
    forward-return label the xLSTM fits, already cross-sectionally z-scored by
    the fold loop.  Rows are kept only where ``label_gate`` AND every
    ``extra_gate`` are True at column ``e``; all-zero feature rows (ZI padding
    of untraded stocks) are dropped so the model does not learn from empty
    padding.  ``max_rows`` stops column collection early once enough rows are
    gathered (benchmark memory cap — biases toward earlier train columns only).
    """
    static = data["static_features"]  # (N, T, D) PIT
    pk = data["past_known"]           # (N, T, D)
    po = data["past_observed"]        # (N, T, D)
    y = data["y_return"]              # (N, T)
    gate = data[label_gate]           # (N, T)
    n, T = y.shape
    n_feat = int(static.shape[2] + pk.shape[2] + po.shape[2])
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total = 0
    for e in range(entry_start, min(entry_end, T)):
        keep = gate[:, e] & (y[:, e] == y[:, e])  # finite target
        for g in extra_gates:
            keep &= data[g][:, e]
        if not keep.any():
            continue
        fcol = e - 1
        X_e = np.concatenate([static[:, fcol], pk[:, fcol], po[:, fcol]], axis=1)
        keep &= np.linalg.norm(X_e, axis=1) > 0.0
        if not keep.any():
            continue
        X_parts.append(X_e[keep])
        y_parts.append(y[keep, e])
        total += int(keep.sum())
        if max_rows is not None and total >= max_rows:
            break
    if not X_parts:
        return np.empty((0, n_feat), dtype=np.float32), np.empty(0, dtype=np.float32)
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
    the ultimate threshold the review names.  Only data before entry is used
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
