import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from stoke_ml.models.panel.config import PanelConfig
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.loss import UncertaintyLoss, AdjMSELoss, PairwiseRankingLoss
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate, DateGroupedSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from stoke_ml.models.panel.evaluate import evaluate_portfolio

logger = logging.getLogger(__name__)

# SequentialLR emits a spurious warning on first step() — the internal
# _step_count check fires before any optimizer.step() is registered,
# even though our gradient-accumulation loop has already stepped the
# optimizer multiple times.  This is a known PyTorch issue (#118894).
warnings.filterwarnings(
    "ignore",
    message="Detected call of .*lr_scheduler.step.* before .*optimizer.step",
)


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
) -> tuple[float, float, float, float]:
    model.eval()
    total, ce_sum, ret_sum, vol_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    nan_batches = 0
    skipped_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, y_dir, y_ret, y_vol, _date_idx = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                if torch.isnan(pred_dir).any() or torch.isnan(pred_ret).any() or torch.isnan(pred_vol).any():
                    nan_batches += 1
                    continue
                mask = (y_dir != -100).float()
                if mask.sum() == 0:
                    skipped_batches += 1
                    continue
                l_ce = ce_loss(torch.clamp(pred_dir, -5, 5), y_dir)
                l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                 y_ret[mask > 0])
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum()
                loss = loss_fn([l_ce, l_ret, l_vol])
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            total += loss.item()
            ce_sum += l_ce.item()
            ret_sum += l_ret.item()
            vol_sum += l_vol.item()
            n += 1
    if nan_batches > 0 or skipped_batches > 0:
        logger.warning(
            "%d NaN + %d empty / %d total val batches",
            nan_batches, skipped_batches, n + nan_batches + skipped_batches,
        )
    model.train()
    if n == 0:
        return float("inf"), float("inf"), float("inf"), float("inf")
    return total / n, ce_sum / n, ret_sum / n, vol_sum / n


def _log_gradient_norms(model: nn.Module, epoch: int) -> None:
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
    ret_loss = AdjMSELoss(gamma=0.1)
    rank_loss = PairwiseRankingLoss(margin=0.0, tau=0.1)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": loss_fn.parameters(), "weight_decay": 0.0},
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    train_ds = PanelDataset(train_data, seq_len=config.seq_len)
    train_sampler = DateGroupedSampler(train_ds.n_stocks, train_ds.n_windows)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        sampler=train_sampler, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=config.num_workers > 0,
    )

    val_ds = PanelDataset(val_data, seq_len=config.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.lr_warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.max_epochs - config.lr_warmup_epochs),
        eta_min=config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.lr_warmup_epochs],
    )

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
        epoch_rank_loss = 0.0
        optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            static, pk, po, y_dir, y_ret, y_vol, date_idx = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            date_idx = date_idx.to(device)

            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                mask = (y_dir != -100).float()
                if mask.sum() == 0:
                    continue
                l_ce = ce_loss(torch.clamp(pred_dir, -5, 5), y_dir)
                l_ret = ret_loss(pred_ret.squeeze(-1)[mask > 0],
                                 y_ret[mask > 0])
                vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * mask
                l_vol = vol_err.sum() / mask.sum()

                # Pairwise ranking loss — directly optimises for cross-sectional
                # ordering (the same signal IC and Sharpe evaluate on).
                l_rank = rank_loss(
                    pred_ret.squeeze(-1), y_ret,
                    mask.squeeze(-1) if mask.dim() > 1 else mask,
                    date_idx,
                )

                total_loss = loss_fn([l_ce, l_ret, l_vol])
                total_loss = total_loss + config.rank_loss_weight * l_rank

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
            epoch_rank_loss += l_rank.item()

        # Apply trailing accumulated gradients
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

        n_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        if config.log_gradient_flow:
            _log_gradient_norms(model, epoch + 1)

        val_loss, v_ce, v_ret, v_vol = _compute_val_loss(
            model, val_loader, ce_loss, ret_loss, loss_fn, device, use_amp,
        )
        history["val_loss"].append(val_loss)

        # Step scheduler AFTER optimizer updates (PyTorch >=1.1 requirement).
        # Called here at epoch end since this is an epoch-level scheduler.
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        eval_start = 5
        do_eval = ((epoch + 1) >= eval_start and (epoch + 1) % 5 == 0)

        if do_eval:
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
                "Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.3f) rank=%.6f "
                "IC=%.4f(IR=%.2f) LS_Sharpe=%.2f[%.1f,%.1f] "
                "Long_Sharpe=%.2f q5-q1=%.1fbp lr=%.2e",
                epoch + 1, config.max_epochs, avg_loss, val_loss,
                v_ce, v_ret, v_vol,
                epoch_rank_loss / n_batches,
                ic_mean, m["ic_ir"],
                ls_sharpe, m["ls_sharpe_lo"], m["ls_sharpe_hi"],
                m["long_sharpe"], m["q5mq1_ret"] * 10000,
                optimizer.param_groups[0]["lr"])
        else:
            logger.info("Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.3f) rank=%.6f lr=%.2e",
                        epoch + 1, config.max_epochs, avg_loss, val_loss,
                        v_ce, v_ret, v_vol,
                        epoch_rank_loss / n_batches,
                        optimizer.param_groups[0]["lr"])

        if patience_counter >= config.early_stop_patience:
            logger.info("Early stopping at epoch %d (best val_loss=%.4f)",
                        epoch + 1, best_val_loss)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
