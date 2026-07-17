import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.model import TFTModel
from stoke_ml.models.tft.loss import UncertaintyLoss, AdjMSELoss, RankICLoss
from stoke_ml.models.tft.dataset import PanelDataset, panel_collate
from stoke_ml.models.tft.evaluate import evaluate_sharpe

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
    rank_ic_loss: nn.Module | None = None,
    rank_ic_weight: float = 0.0,
) -> float:
    """Quick val loss pass every epoch — drives ReduceLROnPlateau."""
    model.eval()
    total, n = 0.0, 0
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
                l_ce = ce_loss(torch.clamp(pred_dir, -10, 10), y_dir)
                mask = (y_dir != -100).float()
                # AdjMSE for returns (sign-aware), MSE for volatility
                l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                 y_ret[mask > 0]) if mask.sum() > 1 else torch.tensor(0.0, device=device)
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum().clamp(min=1)
                loss = loss_fn([l_ce, l_ret, l_vol])
                if rank_ic_loss is not None and mask.sum() > 4:
                    valid_pred = pred_ret.squeeze(-1)[mask > 0]
                    valid_target = y_ret[mask > 0]
                    l_rankic = rank_ic_loss(valid_pred, valid_target)
                    if torch.isfinite(l_rankic):
                        loss = loss + rank_ic_weight * l_rankic
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_tft(
    config: TFTConfig,
    train_data: dict,
    val_data: dict,
    device: torch.device,
) -> tuple[TFTModel, dict]:
    """Train TFT model with purged walk-forward fold.

    Returns:
        model: best model (by validation Sharpe).
        history: dict of training metrics per epoch.
    """
    _set_seed(config.seed)

    model = TFTModel(config).to(device)
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
    rank_ic_loss = RankICLoss(temperature=0.5)  # cross-sectional ranking objective
    rank_ic_weight = 0.1  # small weight — ranking signal is complementary

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_reduce_factor,
        patience=config.lr_reduce_patience,
        threshold=config.lr_reduce_threshold,
        threshold_mode="rel",
        min_lr=config.min_lr,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_sharpe": [], "val_ic": []}
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
                l_ce = ce_loss(torch.clamp(pred_dir, -10, 10), y_dir)
                mask = (y_dir != -100).float()
                # AdjMSE: sign-aware loss — wrong-sign predictions cost 11× more
                if mask.sum() > 1:
                    l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                     y_ret[mask > 0])
                else:
                    l_ret = torch.tensor(0.0, device=device)
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum().clamp(min=1)
                total_loss = loss_fn([l_ce, l_ret, l_vol])
                # RankICLoss: cross-sectional ranking objective (soft Spearman)
                # applied across valid positions in the batch as a proxy for
                # true date-level cross-section.
                if mask.sum() > 4:
                    valid_pred = pred_ret.squeeze(-1)[mask > 0]
                    valid_target = y_ret[mask > 0]
                    l_rankic = rank_ic_loss(valid_pred, valid_target)
                    if torch.isfinite(l_rankic):
                        total_loss = total_loss + rank_ic_weight * l_rankic

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * config.grad_accum_steps

        # Apply trailing accumulated gradients (from valid batches only)
        remaining = accum_count % config.grad_accum_steps
        if remaining != 0:
            scaler.unscale_(optimizer)
            scale = config.grad_accum_steps / remaining
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        # Compute validation loss EVERY epoch (drives ReduceLROnPlateau).
        # pytorch-forecasting monitors val_loss, not train_loss — without
        # this the scheduler never fires because train_loss always drops.
        val_loss = _compute_val_loss(
            model, val_loader, ce_loss, ret_loss, loss_fn, device, use_amp,
            rank_ic_loss=rank_ic_loss, rank_ic_weight=rank_ic_weight,
        )
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        # Save best model by val_loss (more stable than Sharpe with short
        # validation windows — Sharpe has only ~12 non-overlapping samples).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Report Sharpe/IC every 5 epochs (informational only)
        if (epoch + 1) % 5 == 0:
            sharpe, val_metrics = evaluate_sharpe(
                model, val_data, config, device,
                return_metrics=True, horizon=config.horizon,
            )
            history["val_sharpe"].append(sharpe)
            ic = val_metrics.get("ic", 0.0)
            history["val_ic"].append(ic)
            logger.info("Epoch %d/%d: loss=%.4f val_loss=%.4f "
                        "sharpe=%.4f IC=%.4f lr=%.2e",
                        epoch + 1, config.max_epochs, avg_loss, val_loss,
                        sharpe, ic,
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
