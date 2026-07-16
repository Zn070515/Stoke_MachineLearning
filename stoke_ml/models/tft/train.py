import logging
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
    model = TFTModel(config).to(device)
    if config.compile_model and device.type == "cuda":
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            logger.warning("torch.compile failed, continuing without compilation")

    loss_fn = UncertaintyLoss(num_tasks=3).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    train_ds = PanelDataset(train_data, seq_len=config.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=True,
    )

    # total_steps counts optimizer steps (one per grad_accum_steps batches),
    # not raw batches, so OneCycleLR's warmup/anneal spans the full training run.
    steps_per_epoch = max(
        (len(train_loader) + config.grad_accum_steps - 1) // config.grad_accum_steps, 1
    )
    total_steps = config.max_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=min(config.warmup_steps / max(total_steps, 1), 0.3),
        anneal_strategy="cos",
    )

    best_sharpe = -float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_sharpe": []}

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        last_batch_idx = 0

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
                l_ret = mse_loss(pred_ret.squeeze(-1), y_ret)
                l_vol = mse_loss(pred_vol.squeeze(-1), y_vol)
                total_loss = loss_fn([l_ce, l_ret, l_vol])

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    "NaN/Inf loss at epoch %d batch %d — skipping update",
                    epoch + 1, batch_idx,
                )
                continue

            total_loss = total_loss / config.grad_accum_steps
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0 and scaler._scale is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += total_loss.item() * config.grad_accum_steps

        # Apply trailing accumulated gradients (don't waste the last batches).
        # Guard: if every batch in the epoch was skipped (NaN), scaler._scale
        # is None and unscale_() would crash.
        num_batches = last_batch_idx + 1
        remaining = num_batches % config.grad_accum_steps
        if remaining != 0 and scaler._scale is not None:
            scaler.unscale_(optimizer)
            # Rescale gradients: each batch was divided by grad_accum_steps,
            # but only |remaining| batches contributed — restore proper scale.
            scale = config.grad_accum_steps / remaining
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        # Log every epoch with per-task breakdown so we can spot which
        # head dominates the loss (helps diagnose UncertaintyLoss dynamics).
        logger.info("Epoch %d/%d: loss=%.4f (ce=%.4f ret=%.4f vol=%.4f | "
                    "lv=[%.2f,%.2f,%.2f])",
                    epoch + 1, config.max_epochs, avg_loss,
                    l_ce.item(), l_ret.item(), l_vol.item(),
                    loss_fn.log_vars[0].item(),
                    loss_fn.log_vars[1].item(),
                    loss_fn.log_vars[2].item())

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            sharpe = evaluate_sharpe(model, val_data, config, device)
            history["val_sharpe"].append(sharpe)
            logger.info("Epoch %d/%d: loss=%.4f, val_sharpe=%.4f",
                        epoch + 1, config.max_epochs, avg_loss, sharpe)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 2:  # 2 checks × 5 epochs = 10 epoch patience
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
