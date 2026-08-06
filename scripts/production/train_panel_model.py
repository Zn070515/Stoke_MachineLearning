"""Model training helpers for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — ``_weight_hash`` (trained-
parameter fingerprint), ``_predict_outer`` (deployed-checkpoint outer-test
prediction), and ``_best_eval_metrics`` (inner-val eval nearest the deployed
checkpoint).  ``train_panel`` re-exports these names for backward
compatibility.
"""
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.production.train_panel_oos import _state_dict_hash
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate

logger = logging.getLogger(__name__)


def _weight_hash(model) -> str:
    """Content hash of a model's TRAINED parameters (float32, CPU).

    The version dict's `model_hash` only fingerprints config + architecture
    source — every fold shares it.  This one hashes the actual state_dict so
    an OOS tape row / checkpoint can be tied to the exact weights that
    produced it, and two differently-trained folds get different digests.
    """
    return _state_dict_hash(model.state_dict())

def _predict_outer(model, outer_data, config, device) -> np.ndarray | None:
    """Run the deployed checkpoint over the outer-test panel.

    Date-centric (§七/§十六): uses the same eval-mode no-sampler DataLoader as
    evaluate_portfolio (training=False → eval_mask / full candidate pool,
    max_stocks_per_date=None, batch_size=1).  Each ``__getitem__`` returns one
    date's (M, ...) tensors; the return prediction is placed directly at
    ``preds[stock_indices, window_idx]`` so the sparse grid is reconstructed
    without a flat cat+reshape (which would mismatch when sum(M_i) !=
    n_stocks*n_windows).  Cells for ineligible stocks/windows stay NaN.  Window
    d enters at panel column seq_len + d — i.e. global column val_start + d of
    the full panel.  Returns None only when the outer panel has no windows
    (all-NaN grid).
    """
    n_stocks = outer_data["static_features"].shape[0]
    val_ds = PanelDataset(outer_data, seq_len=config.seq_len,
                          min_history=config.min_history,
                          max_stocks_per_date=None, training=False)
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )
    n_windows = val_ds.n_windows
    seq_len = val_ds.seq_len
    model.eval()
    preds = torch.full((n_stocks, n_windows), float("nan"))
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_y, date_idx_t, _dm, _rm, _vm, stock_indices = batch
            if stock_indices.numel() == 0:
                continue
            # Per-stock window indices (supports mixed-date batches).
            window_idx = date_idx_t - seq_len
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            preds[stock_indices, window_idx] = pred_ret.cpu().squeeze(-1)
    if torch.isnan(preds).all():
        return None
    return preds.numpy()

def _best_eval_metrics(history: dict) -> tuple[dict, int]:
    """Metrics of the inner-val eval nearest the deployed checkpoint.

    Returns (metrics_dict, eval_epoch) for the evaluation whose 1-based epoch
    sits closest to best_epoch_idx+1 — NOT the post-hoc max, which would
    double-count hindsight.  Histories without val_eval_epochs (legacy) are
    assumed to have evaluated on the 5,10,15,... grid.  Empty histories yield
    ({}, 0).
    """
    metrics = history.get("val_metrics") or []
    if not metrics:
        return {}, 0
    best = history.get("best_epoch_idx", 0) + 1  # 1-based deployed epoch
    eval_epochs = history.get("val_eval_epochs")
    if not eval_epochs:
        eval_epochs = [5 + 5 * i for i in range(len(metrics))]
    nearest = min(range(len(metrics)), key=lambda i: abs(eval_epochs[i] - best))
    return metrics[nearest], eval_epochs[nearest]
