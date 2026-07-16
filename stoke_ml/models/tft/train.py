import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from stoke_ml.models.tft.config import TFTConfig
from stoke_ml.models.tft.model import TFTModel
from stoke_ml.models.tft.loss import UncertaintyLoss
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_reduce_factor,
        patience=config.lr_reduce_patience,
        threshold=config.lr_reduce_threshold,
        threshold_mode="rel",
        min_lr=config.min_lr,
    )

    best_sharpe = -float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_sharpe": [], "val_ic": []}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        last_batch_idx = 0
        backward_done = False

        for batch_idx, (static, pk, po, y_dir, y_ret, y_vol) in enumerate(train_loader):
            last_batch_idx = batch_idx
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)

            use_amp = config.use_amp and device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                l_ce = ce_loss(torch.clamp(pred_dir, -10, 10), y_dir)
                mask = (y_dir != -100).float()
                ret_err = (pred_ret.squeeze(-1) - y_ret).pow(2) * mask
                l_ret = ret_err.sum() / mask.sum().clamp(min=1)
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum().clamp(min=1)
                total_loss = loss_fn([l_ce, l_ret, l_vol])

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    "NaN/Inf loss at epoch %d batch %d — skipping update",
                    epoch + 1, batch_idx,
                )
                continue

            total_loss = total_loss / config.grad_accum_steps
            scaler.scale(total_loss).backward()
            backward_done = True

            if (batch_idx + 1) % config.grad_accum_steps == 0 and backward_done:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * config.grad_accum_steps

        # Apply trailing accumulated gradients
        num_batches = last_batch_idx + 1
        remaining = num_batches % config.grad_accum_steps
        if remaining != 0 and backward_done:
            scaler.unscale_(optimizer)
            scale = config.grad_accum_steps / remaining
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        logger.info("Epoch %d/%d: loss=%.4f (ce=%.4f ret=%.4f vol=%.4f | "
                    "lv=[%.6f,%.6f,%.6f]) lr=%.2e",
                    epoch + 1, config.max_epochs, avg_loss,
                    l_ce.item(), l_ret.item(), l_vol.item(),
                    loss_fn.log_vars[0].item(),
                    loss_fn.log_vars[1].item(),
                    loss_fn.log_vars[2].item(),
                    optimizer.param_groups[0]["lr"])

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            sharpe, val_metrics = evaluate_sharpe(
                model, val_data, config, device, return_metrics=True,
            )
            history["val_sharpe"].append(sharpe)
            ic = val_metrics.get("ic", 0.0)
            history["val_ic"].append(ic)
            logger.info("Epoch %d/%d: loss=%.4f, val_sharpe=%.4f, val_IC=%.4f",
                        epoch + 1, config.max_epochs, avg_loss, sharpe, ic)

            # ReduceLROnPlateau on validation loss
            scheduler.step(avg_loss)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stop_patience // 5:  # checks every 5 epochs
                logger.info("Early stopping at epoch %d (best Sharpe=%.4f)",
                            epoch + 1, best_sharpe)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
