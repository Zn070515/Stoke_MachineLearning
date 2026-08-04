import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ret_loss: AdjMSELoss,
    loss_fn: UncertaintyLoss,
    device: torch.device,
    use_amp: bool,
    vol_enabled: bool = True,
) -> tuple[float, float, float, float]:
    """Validation loss accumulated per VALID SAMPLE, not per batch.

    Review v4 §九: sparse batches (few clean targets) must not weigh the same
    as dense ones — the checkpoint-selection metric v_ret is a sample-weighted
    mean over all valid return targets across the whole validation loader.
    """
    model.eval()
    n_batches = 0
    ce_sum = ce_cnt = 0.0
    ret_sum = ret_cnt = 0.0
    vol_sum = vol_cnt = 0.0
    uncer_num = uncer_den = 0.0
    nan_batches = 0
    skipped_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, y_dir, y_ret, y_vol, _date_idx, dir_mask, ret_mask, vol_mask = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            dir_mask = dir_mask.to(device).float()
            ret_mask = ret_mask.to(device).float()
            vol_mask = vol_mask.to(device).float()
            n_batches += 1
            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                if (torch.isnan(pred_dir).any() or torch.isnan(pred_ret).any()
                        or torch.isnan(pred_vol).any()):
                    nan_batches += 1
                    continue
                dir_valid = dir_mask > 0
                ret_valid = ret_mask > 0
                vol_valid = vol_mask > 0

                # Per-task elementwise losses (review v4 §八/§九): each loss is
                # computed ONLY over its valid samples, and a batch whose
                # direction labels are all -100 must not produce NaN CE.
                losses_elem: list[torch.Tensor] = []
                counts: list[int] = []
                active: list[bool] = []

                if dir_valid.any():
                    losses_elem.append(F.cross_entropy(
                        torch.clamp(pred_dir, -5, 5)[dir_valid], y_dir[dir_valid],
                        reduction="none",
                    ))
                    counts.append(int(dir_valid.sum().item()))
                    active.append(True)
                else:
                    losses_elem.append(torch.zeros((), device=device))
                    counts.append(0)
                    active.append(False)

                if ret_valid.any():
                    losses_elem.append(ret_loss(
                        pred_ret.squeeze(-1)[ret_valid], y_ret[ret_valid],
                        reduction="none",
                    ))
                    counts.append(int(ret_valid.sum().item()))
                    active.append(True)
                else:
                    losses_elem.append(torch.zeros((), device=device))
                    counts.append(0)
                    active.append(False)

                if vol_enabled:
                    if vol_valid.any():
                        losses_elem.append(
                            (pred_vol.squeeze(-1)[vol_valid] - y_vol[vol_valid]).pow(2)
                        )
                        counts.append(int(vol_valid.sum().item()))
                        active.append(True)
                    else:
                        losses_elem.append(torch.zeros((), device=device))
                        counts.append(0)
                        active.append(False)

                if not any(active):
                    skipped_batches += 1
                    continue

                # Per-task sums over valid samples — sample-weighted averages.
                if active[0]:
                    ce_sum += float(losses_elem[0].sum())
                    ce_cnt += counts[0]
                if active[1]:
                    ret_sum += float(losses_elem[1].sum())
                    ret_cnt += counts[1]
                if vol_enabled and active[2]:
                    vol_sum += float(losses_elem[2].sum())
                    vol_cnt += counts[2]

                # Uncertainty-weighted total, accumulated per valid sample so
                # the combined val_loss is independent of batch boundaries.
                log_vars = torch.clamp(loss_fn.log_vars, -2.0, 10.0)
                num = torch.tensor(0.0, device=device, dtype=log_vars.dtype)
                den = 0
                for i, (l_elem, cnt, act) in enumerate(zip(losses_elem, counts, active)):
                    if not act:
                        continue
                    precision = torch.exp(-log_vars[i])
                    num = num + 0.5 * (precision * l_elem.sum() + log_vars[i] * cnt)
                    den += cnt
                uncer_num += float(num)
                uncer_den += den
    if nan_batches > 0 or skipped_batches > 0:
        logger.warning(
            "%d NaN + %d empty / %d total val batches",
            nan_batches, skipped_batches, n_batches,
        )
    model.train()
    if uncer_den == 0:
        return float("inf"), float("inf"), float("inf"), float("inf")
    v_ce = ce_sum / ce_cnt if ce_cnt > 0 else float("inf")
    v_ret = ret_sum / ret_cnt if ret_cnt > 0 else float("inf")
    v_vol = vol_sum / vol_cnt if vol_cnt > 0 else float("inf")
    return uncer_num / uncer_den, v_ce, v_ret, v_vol


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

    # horizon==1 leaves no room for an intra-window vol estimate (vol window
    # needs >= 2 daily returns), so the vol task is dropped entirely rather than
    # learning a degenerate uncertainty weight on an all-zero target.
    vol_enabled = config.horizon != 1
    loss_fn = UncertaintyLoss(num_tasks=3 if vol_enabled else 2).to(device)
    ce_loss = nn.CrossEntropyLoss()
    ret_loss = AdjMSELoss(gamma=0.1)
    rank_loss = PairwiseRankingLoss(
        margin=0.0, tau=0.1, spread_target=1.0, spread_weight=0.5,
    )

    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": loss_fn.parameters(), "weight_decay": 0.0},
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    train_ds = PanelDataset(train_data, seq_len=config.seq_len,
                            min_history=config.min_history)
    train_sampler = DateGroupedSampler(train_ds.valid_mask)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        sampler=train_sampler, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=config.num_workers > 0,
    )

    val_ds = PanelDataset(val_data, seq_len=config.seq_len,
                          min_history=config.min_history)
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

    # Checkpoint selection uses the FIXED return loss, not the learned-weighted
    # total — the uncertainty log_vars are trainable parameters the model can
    # inflate to shrink the total without improving return prediction.
    best_val_ret = float("inf")
    best_state = None
    best_epoch_idx = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_ls_sharpe": [], "val_ic": [],
        "val_eval_epochs": [],  # 1-based epoch of each val_metrics entry
    }
    use_amp = config.use_amp and device.type == "cuda"

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_rank_pairs = 0
        epoch_rank_batches = 0
        optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            static, pk, po, y_dir, y_ret, y_vol, date_idx, dir_mask, ret_mask, vol_mask = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            date_idx = date_idx.to(device)
            dir_mask = dir_mask.to(device).float()
            ret_mask = ret_mask.to(device).float()
            vol_mask = vol_mask.to(device).float()

            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                # Per-task masks (review v4 §八): each loss runs ONLY over its
                # valid samples.  A batch whose direction labels are all -100
                # must not produce NaN CrossEntropy, and an inactive task must
                # not update its uncertainty log-var.
                dir_valid = dir_mask > 0
                ret_valid = ret_mask > 0
                vol_valid = vol_mask > 0
                l_ce = (
                    ce_loss(torch.clamp(pred_dir, -5, 5)[dir_valid], y_dir[dir_valid])
                    if dir_valid.any()
                    else torch.zeros((), device=device)
                )
                losses = [l_ce]
                task_active = [bool(dir_valid.any().item())]
                if ret_valid.any():
                    losses.append(ret_loss(pred_ret.squeeze(-1)[ret_valid],
                                           y_ret[ret_valid]))
                    task_active.append(True)
                else:
                    losses.append(torch.zeros((), device=device))
                    task_active.append(False)
                if vol_enabled:
                    if vol_valid.any():
                        vol_err = (pred_vol.squeeze(-1) - y_vol).pow(2) * vol_mask
                        losses.append(vol_err.sum() / vol_mask.sum())
                        task_active.append(True)
                    else:
                        losses.append(torch.zeros((), device=device))
                        task_active.append(False)

                # Pairwise ranking loss — directly optimises for cross-sectional
                # ordering (the same signal IC and Sharpe evaluate on).  Ranks
                # clean return targets only.
                batch_rank_stats: list[dict] = []
                l_rank = rank_loss(
                    pred_ret.squeeze(-1), y_ret,
                    ret_mask, date_idx, stats=batch_rank_stats,
                )

                if not any(task_active):
                    logger.debug("Epoch %d batch %d: no active task — skipping",
                                 epoch + 1, batch_idx)
                    continue

                total_loss = loss_fn(losses, task_active_mask=task_active)
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
                # Snapshot gradients BEFORE zero_grad — the old epoch-end call
                # read already-zeroed grads and reported meaningless norms.
                # Log once per epoch (on its final optimizer step) so a long
                # run doesn't flood the log or pay a device-sync every batch.
                if config.log_gradient_flow and batch_idx == len(train_loader) - 1:
                    _log_gradient_norms(model, epoch + 1)
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * config.grad_accum_steps
            epoch_rank_loss += l_rank.item()
            if batch_rank_stats:
                epoch_rank_pairs += batch_rank_stats[0]["n_pairs"]
                epoch_rank_batches += 1

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
            if config.log_gradient_flow:
                _log_gradient_norms(model, epoch + 1)
            optimizer.zero_grad()

        n_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        val_loss, v_ce, v_ret, v_vol = _compute_val_loss(
            model, val_loader, ret_loss, loss_fn, device, use_amp,
            vol_enabled=vol_enabled,
        )
        history["val_loss"].append(val_loss)
        history.setdefault("val_ret", []).append(v_ret)

        # Step scheduler AFTER optimizer updates (PyTorch >=1.1 requirement).
        # Called here at epoch end since this is an epoch-level scheduler.
        scheduler.step()

        if v_ret < best_val_ret:
            best_val_ret = v_ret
            best_epoch_idx = epoch
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
            history["val_eval_epochs"].append(epoch + 1)
            logger.info(
                "Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.5f) rank=%.6f "
                "pairs=%d IC=%.4f(IR=%.2f) LS_Sharpe=%.2f[%.1f,%.1f] "
                "Long_Sharpe=%.2f q5-q1=%.1fbp lr=%.2e",
                epoch + 1, config.max_epochs, avg_loss, val_loss,
                v_ce, v_ret, v_vol,
                epoch_rank_loss / n_batches,
                epoch_rank_pairs,
                ic_mean, m["ic_ir"],
                ls_sharpe, m["ls_sharpe_lo"], m["ls_sharpe_hi"],
                m["long_sharpe"], m["q5mq1_ret"] * 10000,
                optimizer.param_groups[0]["lr"])
        else:
            logger.info("Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.5f) rank=%.6f pairs=%d lr=%.2e",
                        epoch + 1, config.max_epochs, avg_loss, val_loss,
                        v_ce, v_ret, v_vol,
                        epoch_rank_loss / n_batches,
                        epoch_rank_pairs,
                        optimizer.param_groups[0]["lr"])

        if patience_counter >= config.early_stop_patience:
            logger.info("Early stopping at epoch %d (best val_ret=%.6f)",
                        epoch + 1, best_val_ret)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch_idx"] = best_epoch_idx
    # Exact portfolio evaluation on the deployed best-val-loss checkpoint.
    # The in-loop val_metrics snapshots are taken at each eval epoch from
    # whatever model was current then, which may differ from best_state —
    # reporting those is the "nearest epoch proxy" the review flags.  Compute
    # the true IC/Sharpe of the checkpoint that is actually returned/saved.
    try:
        history["best_metrics"] = evaluate_portfolio(
            model, val_data, config, device,
            horizon=config.horizon, raw_returns=raw_val_returns,
        )
    except Exception:
        logger.exception("best-checkpoint portfolio evaluation failed")
    return model, history
