import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.loss import UncertaintyLoss, AdjMSELoss
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from stoke_ml.models.panel.evaluate import evaluate_portfolio

logger = logging.getLogger(__name__)


def _set_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _compute_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    ce_loss: nn.Module,
    ret_loss: nn.Module,
    loss_fn: UncertaintyLoss,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Quick val loss pass every epoch for best-model selection."""
    model.eval()
    total, n = 0.0, 0
    nan_batches = 0
    skipped_batches = 0
    with torch.no_grad():
        for static, pk, po, y_dir, y_ret, y_vol in val_loader:
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                # Guard against model producing NaN (e.g. bad weights after
                # training with corrupted data).
                if torch.isnan(pred_dir).any() or torch.isnan(pred_ret).any() or torch.isnan(pred_vol).any():
                    nan_batches += 1
                    continue
                mask = (y_dir != -100).float()
                # Entire batch has no valid labels (e.g. all positions are
                # tail-padded for short-history stocks).  CrossEntropyLoss
                # returns NaN when all targets are ignore_index, so skip.
                if mask.sum() == 0:
                    skipped_batches += 1
                    continue
                l_ce = ce_loss(torch.clamp(pred_dir, -5, 5), y_dir)
                # AdjMSE for returns (sign-aware), MSE for volatility
                l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                 y_ret[mask > 0])
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum()
                loss = loss_fn([l_ce, l_ret, l_vol])
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            total += loss.item()
            n += 1
    if nan_batches > 0 or skipped_batches > 0:
        logger.warning(
            "%d NaN + %d empty / %d total val batches",
            nan_batches, skipped_batches, n + nan_batches + skipped_batches,
        )
    model.train()
    return total / max(n, 1) if n > 0 else float("inf")


def _log_gradient_norms(model: nn.Module, epoch: int) -> None:
    """Log per-layer-gradient norms — detects gradient collapse.

    Collapse signal: output head norms < 0.1 × encoder norms AND decreasing.
    """
    head_patterns = ("direction_head", "return_head", "volatility_head")
    encoder_norms, head_norms = [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gnorm = param.grad.norm().item()
        if any(p in name for p in head_patterns):
            head_norms.append(gnorm)
        else:
            encoder_norms.append(gnorm)

    enc_avg = sum(encoder_norms) / max(len(encoder_norms), 1)
    head_avg = sum(head_norms) / max(len(head_norms), 1)
    ratio = head_avg / max(enc_avg, 1e-12)

    logger.info("Epoch %d grad norms: encoder=%.6f head=%.6f ratio=%.3f",
                epoch, enc_avg, head_avg, ratio)

    if ratio < 0.1 and epoch > 3:
        logger.warning(
            "Possible gradient collapse: head/encoder gradient ratio=%.3f < 0.1."
            " Heads may be underfitting. Consider increasing head_grad_clip "
            "or decreasing backbone_grad_clip.",
            ratio,
        )


def train_panel(
    config: PanelConfig,
    train_data: dict,
    val_data: dict,
    device: torch.device,
    raw_val_returns: np.ndarray | None = None,
) -> tuple[PanelModel, dict]:
    """Train VSN+xLSTM panel model with purged walk-forward fold.

    Args:
        raw_val_returns: (N_stocks, T_val) raw forward returns (percent).
            Passed through to evaluate_portfolio so Sharpe/IC are computed
            from real returns, not z-scored targets.

    Returns:
        model: best model (by validation Sharpe).
        history: dict of training metrics per epoch.
    """
    _set_seed(config.seed)

    model = PanelModel(config).to(device)
    if config.compile_model and device.type == "cuda":
        try:
            import triton  # noqa: F401
            model = torch.compile(model, mode="default")
        except ImportError:
            logger.info("Triton not available on this platform, skipping torch.compile")
        except Exception:
            logger.warning("torch.compile failed, continuing without compilation")

    loss_fn = UncertaintyLoss(num_tasks=3).to(device)
    ce_loss = nn.CrossEntropyLoss()
    ret_loss = AdjMSELoss(gamma=0.1)  # sign-aware: wrong-sign → 11× penalty

    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": loss_fn.parameters(), "weight_decay": 0.0},
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    train_ds = PanelDataset(train_data, seq_len=config.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=config.num_workers > 0,
    )

    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    # Cosine annealing with linear warmup — transformer-training standard.
    # Warmup prevents early-epoch gradient spikes; cosine decay gives smooth
    # convergence to a low final lr.
    # Epoch-based scheduling: scheduler.step() is called once per epoch.
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.lr_warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs - config.lr_warmup_epochs,
        eta_min=config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.lr_warmup_epochs],
    )

    # Per-layer param groups for stratified gradient clipping
    head_param_names = {"direction_head", "return_head", "volatility_head"}
    head_params = [
        p for n, p in model.named_parameters()
        if any(head_n in n for head_n in head_param_names)
    ]
    backbone_params = [
        p for n, p in model.named_parameters()
        if not any(head_n in n for head_n in head_param_names)
    ]

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ls_sharpe": [], "val_ic": []}
    use_amp = config.use_amp and device.type == "cuda"

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        accum_count = 0  # actual backward steps counted (skip NaN)

        for batch_idx, (static, pk, po, y_dir, y_ret, y_vol) in enumerate(train_loader):
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)

            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                mask = (y_dir != -100).float()
                if mask.sum() == 0:
                    continue
                l_ce = ce_loss(torch.clamp(pred_dir, -5, 5), y_dir)
                # AdjMSE: sign-aware loss — wrong-sign predictions cost 11× more
                l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                 y_ret[mask > 0])
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum()
                total_loss = loss_fn([l_ce, l_ret, l_vol])

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    "NaN/Inf loss at epoch %d batch %d — skipping update",
                    epoch + 1, batch_idx,
                )
                continue

            total_loss = total_loss / config.grad_accum_steps
            scaler.scale(total_loss).backward()
            accum_count += 1

            if accum_count % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                # Stratified gradient clipping: heads get looser bounds
                # to prevent gradient collapse (FinFusion 2024 finding).
                if backbone_params:
                    torch.nn.utils.clip_grad_norm_(
                        backbone_params, config.backbone_grad_clip,
                    )
                if head_params:
                    torch.nn.utils.clip_grad_norm_(
                        head_params, config.head_grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * config.grad_accum_steps

        # Apply trailing accumulated gradients (from valid batches only)
        remaining = accum_count % config.grad_accum_steps
        if remaining != 0:
            scaler.unscale_(optimizer)
            scale = config.grad_accum_steps / remaining
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        p.grad.mul_(scale)
            if backbone_params:
                torch.nn.utils.clip_grad_norm_(backbone_params, config.backbone_grad_clip)
            if head_params:
                torch.nn.utils.clip_grad_norm_(head_params, config.head_grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        # Gradient-flow monitoring (expensive — enable for debugging collapse)
        if config.log_gradient_flow:
            _log_gradient_norms(model, epoch + 1)

        val_loss = _compute_val_loss(
            model, val_loader, ce_loss, ret_loss, loss_fn, device, use_amp,
        )
        history["val_loss"].append(val_loss)

        # Cosine annealing steps every epoch (no-val-metric variant)
        scheduler.step()

        # Save best model by val_loss (more stable than Sharpe with short
        # validation windows — Sharpe has only ~12 non-overlapping samples).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Report evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            m = evaluate_portfolio(
                model, val_data, config, device,
                horizon=config.horizon, raw_returns=raw_val_returns,
            )
            ls_sharpe = m["ls_sharpe"]
            ic_mean = m["ic_mean"]
            history["val_ls_sharpe"].append(ls_sharpe)
            history["val_ic"].append(ic_mean)
            history.setdefault("val_metrics", [])
            history["val_metrics"].append(m)
            logger.info(
                "Epoch %d/%d: loss=%.4f val_loss=%.4f "
                "IC=%.4f(IR=%.2f) LS_Sharpe=%.2f[%.1f,%.1f] "
                "Long_Sharpe=%.2f q5-q1=%.1fbp lr=%.2e",
                epoch + 1, config.max_epochs, avg_loss, val_loss,
                ic_mean, m["ic_ir"],
                ls_sharpe, m["ls_sharpe_lo"], m["ls_sharpe_hi"],
                m["long_sharpe"], m["q5mq1_ret"] * 10000,
                optimizer.param_groups[0]["lr"])
        else:
            logger.info("Epoch %d/%d: loss=%.4f val_loss=%.4f lr=%.2e",
                        epoch + 1, config.max_epochs, avg_loss, val_loss,
                        optimizer.param_groups[0]["lr"])

        if patience_counter >= config.early_stop_patience:
            logger.info("Early stopping at epoch %d (best val_loss=%.4f)",
                        epoch + 1, best_val_loss)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
