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
from stoke_ml.models.panel.loss import (
    UncertaintyLoss, FixedTaskWeights, AdjMSELoss, PairwiseRankingLoss,
)
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate, DateSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from stoke_ml.models.panel.evaluate import evaluate_portfolio, _raw_clean_rank_ic

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


def _configure_reproducibility(config: PanelConfig) -> dict:
    """Configure the CUDA determinism knobs §十二-5 lists; declare the level.

    A random seed alone does NOT give CUDA bitwise reproducibility.  These
    flags are set unconditionally and are cheap/safe on this model (no conv
    layers, so cudnn.benchmark=False costs nothing): cudnn deterministic,
    autotune off, TF32 disabled (its truncated mantissa makes reductions
    order-dependent).  Strict `use_deterministic_algorithms(True)` is gated
    behind config.deterministic_algorithms because one non-deterministic op
    (e.g. a CUDA atomic reduction) then raises mid-training instead of
    silently diverging.  torch.compile is a JIT and is NOT bitwise
    reproducible, so a compiled run is declared statistical-only regardless.

    Returns the declared reproducibility level for history / the log.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    strict = bool(config.deterministic_algorithms)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    compile_enabled = bool(getattr(config, "compile_model", False))
    bitwise = strict and not compile_enabled
    report = {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "strict_deterministic_algorithms": strict,
        "compile_enabled": compile_enabled,
        "reproducibility_level": "bitwise" if bitwise else "statistical-only",
    }
    return report


def _rank_pool_stats(ds: PanelDataset) -> tuple[list[int], int]:
    """Per-date rank-eligible stock counts + total possible same-date pairs.

    PairwiseRankingLoss can only form pairs between stocks that share a target
    date AND both carry a valid return target.  When one date's eligible stock
    count exceeds batch_size the DateGroupedSampler splits it across sub-batches
    and pairs between sub-batches are never compared — so the true coverage is
    the ratio of pairs actually formed to the C(n,2) theoretical total below.
    """
    if ds.ret_target is None:
        pool = ds.valid_mask
    else:
        # valid_mask is (N, N_windows) over target windows; ret_target is
        # (N, T) over absolute days.  Window w's target day is w + seq_len.
        pool = ds.valid_mask & ds.ret_target[:, ds.seq_len:]
    per_date = pool.sum(dim=0).tolist()
    stocks_per_date = [n for n in per_date if n > 0]
    possible_pairs = sum(n * (n - 1) // 2 for n in per_date)
    return stocks_per_date, possible_pairs


def _entry_bias_report(train_data: dict, val_data: dict, seq_len: int) -> dict:
    """Train vs eval candidate-distribution gap (§十二-4).

    Training samples require a REAL entry open (`entry_eligible`), so the model
    never sees decision-selectable days where the NEXT day suspends / has no
    open.  Evaluation selects on decision & history and THEN checks fill — those
    unfilled candidates ARE in its pool.  `selectable` mirrors evaluate.py's
    candidate pool (decision & history); `fill_rate` is the fraction of that
    pool that also carries a next-day open, i.e. what training could see.
    """
    def _span(data: dict) -> dict:
        entry_full = data.get("entry_eligible_mask")
        if entry_full is None:
            # No eligibility masks (legacy synthetic data) — nothing to compare.
            return {
                "selectable": 0, "with_entry_open": 0, "unfilled": 0,
                "fill_rate": float("nan"), "n_dates": 0,
            }
        entry = entry_full[:, seq_len:]
        dec = (data.get("decision_eligible_mask")[:, seq_len:]
               if data.get("decision_eligible_mask") is not None
               else np.ones_like(entry))
        hist = (data.get("history_eligible_mask")[:, seq_len:]
                if data.get("history_eligible_mask") is not None
                else np.ones_like(entry))
        selectable = dec & hist
        n_selectable = int(selectable.sum())
        if n_selectable == 0:
            return {
                "selectable": 0, "with_entry_open": 0, "unfilled": 0,
                "fill_rate": float("nan"), "n_dates": 0,
            }
        with_entry = selectable & entry
        n_entry = int(with_entry.sum())
        return {
            "selectable": n_selectable,
            "with_entry_open": n_entry,
            "unfilled": n_selectable - n_entry,
            "fill_rate": n_entry / n_selectable,
            "n_dates": int((selectable.sum(axis=0) > 0).sum()),
        }

    return {"train": _span(train_data), "val": _span(val_data)}


def _compute_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    val_data: dict,
    config: PanelConfig,
    ret_loss: AdjMSELoss,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    vol_enabled: bool = True,
    dir_enabled: bool = True,
    diag: dict | None = None,
) -> tuple[float, float, float, float, float]:
    """Validation metrics accumulated per VALID SAMPLE, not per batch.

    Sparse batches (few clean targets) must not weigh the same
    as dense ones — each task loss is a sample-weighted mean over all valid
    targets across the whole validation loader.  The final element is the
    per-date cross-sectional Spearman RankIC of predicted vs RAW clean returns:
    the PRIMARY checkpoint-selection metric.
    It shares the evaluate.py raw-clean-IC definition and its
    min_stocks_per_day threshold with the formal report, so the metric that
    selects a checkpoint is exactly the metric the report prints.

    With date-centric batches (§七/§十六), the prediction grid is
    reconstructed by placing each stock's prediction at
    ``preds[stock_idx, window_idx]`` using the ``stock_indices`` and
    ``date_idx`` fields from each batch.  This is independent of batch
    boundaries — same as the old flat-order reshape but robust to date
    skipping and varying stocks-per-date.
    """
    model.eval()
    n_batches = 0
    ce_sum = ce_cnt = 0.0
    ret_sum = ret_cnt = 0.0
    vol_sum = vol_cnt = 0.0
    uncer_num = uncer_den = 0.0
    nan_batches = 0
    skipped_batches = 0
    n_stocks = val_data["static_features"].shape[0]
    n_windows = val_loader.dataset.n_windows
    seq_len = val_loader.dataset.seq_len
    # Pre-allocate the full (n_stocks, n_windows) grid for direct placement —
    # no torch.cat+reshape needed (§七 date-centric).
    preds = torch.full((n_stocks, n_windows), float("nan"))
    with torch.no_grad():
        for batch in val_loader:
            (static, pk, po, y_dir, y_ret, y_vol,
             date_idx, dir_mask, ret_mask, vol_mask, stock_indices) = batch
            if stock_indices.numel() == 0:
                skipped_batches += 1
                continue
            n_batches += 1
            # Per-stock window indices — may differ when batch_size > 1 and
            # collate concatenated multiple dates (date_idx varies per stock).
            window_idx = date_idx - seq_len  # (M,)
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            y_dir = y_dir.to(device)
            y_ret = y_ret.to(device)
            y_vol = y_vol.to(device)
            dir_mask = dir_mask.to(device).float()
            ret_mask = ret_mask.to(device).float()
            vol_mask = vol_mask.to(device).float()
            with autocast("cuda", enabled=use_amp):
                pred_dir, pred_ret, pred_vol = model(static, pk, po)
                # Place predictions directly in the (N, W) grid using
                # per-stock stock_indices + window_idx (supports mixed-date
                # batches when batch_size > 1).
                preds[stock_indices, window_idx] = (
                    pred_ret.detach().cpu().squeeze(-1)
                )
                if (torch.isnan(pred_dir).any() or torch.isnan(pred_ret).any()
                        or torch.isnan(pred_vol).any()):
                    nan_batches += 1
                    continue
                ret_valid = ret_mask > 0
                vol_valid = vol_mask > 0

                # Per-task elementwise losses: each loss is
                # computed ONLY over its valid samples, and a batch whose
                # direction labels are all -100 must not produce NaN CE.
                # Task slots are positional (dir first, ret, vol) so the
                # uncertainty log-var indices stay aligned with train().
                losses_elem: list[torch.Tensor] = []
                counts: list[int] = []
                active: list[bool] = []

                if dir_enabled:
                    dir_valid = dir_mask > 0
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
                i_ret = 1 if dir_enabled else 0
                i_vol = (2 if dir_enabled else 1) if vol_enabled else -1
                if dir_enabled and active[0]:
                    ce_sum += float(losses_elem[0].sum())
                    ce_cnt += counts[0]
                if active[i_ret]:
                    ret_sum += float(losses_elem[i_ret].sum())
                    ret_cnt += counts[i_ret]
                if vol_enabled and active[i_vol]:
                    vol_sum += float(losses_elem[i_vol].sum())
                    vol_cnt += counts[i_vol]

                # Combined total, accumulated so the val_loss is independent of
                # batch boundaries.  UncertaintyLoss re-weights each task by its
                # learned log-var; FixedTaskWeights is a plain equal-weight mean
                # over active tasks (its forward() has no log_vars).
                if hasattr(loss_fn, "log_vars"):
                    log_vars = torch.clamp(loss_fn.log_vars, -2.0, 10.0)
                    num = torch.tensor(0.0, device=device, dtype=log_vars.dtype)
                    den = 0
                    for i, (l_elem, cnt, act) in enumerate(zip(losses_elem, counts, active)):
                        if not act:
                            continue
                        precision = torch.exp(-log_vars[i])
                        num = num + 0.5 * (precision * l_elem.sum() + log_vars[i] * cnt)
                        den += cnt
                else:
                    num = torch.tensor(0.0, device=device, dtype=torch.float32)
                    den = 0
                    for l_elem, cnt, act in zip(losses_elem, counts, active):
                        if not act:
                            continue
                        num = num + l_elem.sum() / cnt  # cnt >= 1 when act
                        den += 1
                uncer_num += float(num)
                uncer_den += den
    if nan_batches > 0 or skipped_batches > 0:
        logger.warning(
            "%d NaN + %d empty / %d total val batches",
            nan_batches, skipped_batches, n_batches,
        )
    model.train()
    if uncer_den == 0:
        return float("inf"), float("inf"), float("inf"), float("inf"), float("nan")
    # The prediction grid is already reconstructed in (n_stocks, n_windows)
    # via direct placement.  RankIC is the SAME quantity as the formal report.
    daily_ics, _ = _raw_clean_rank_ic(
        val_data, preds.numpy(), n_windows, config.seq_len,
        min_stocks=config.min_stocks_per_day, diag=diag,
    )
    v_rankic = float(np.mean(daily_ics)) if daily_ics else float("nan")
    v_ce = ce_sum / ce_cnt if ce_cnt > 0 else float("inf")
    v_ret = ret_sum / ret_cnt if ret_cnt > 0 else float("inf")
    v_vol = vol_sum / vol_cnt if vol_cnt > 0 else float("inf")
    return uncer_num / uncer_den, v_ce, v_ret, v_vol, v_rankic


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
    # §十二-5: a seed alone does not give CUDA bitwise reproducibility — set
    # the determinism knobs explicitly and DECLARE the level achieved.
    reproducibility = _configure_reproducibility(config)
    # Worker processes re-seed from this generator, so multi-worker DataLoader
    # order is reproducible for the same seed.
    loader_generator = (
        torch.Generator().manual_seed(config.seed)
        if config.seed is not None else None
    )

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
    # §十一.3 ablation: the dir/vol heads can be disabled (the model then emits
    # zero tensors) — their tasks leave the multi-task pool entirely so an
    # ablated head neither updates a learned log-var nor dilutes an equal-weight
    # mean.  Task slot order is positional (dir, ret, vol) and must match
    # _compute_val_loss, which re-derives the same indices from dir_enabled.
    dir_enabled = config.use_dir_head
    vol_enabled = config.use_vol_head and config.horizon != 1
    num_tasks = (1 if dir_enabled else 0) + 1 + (1 if vol_enabled else 0)
    if config.fixed_task_weights:
        loss_fn = FixedTaskWeights(num_tasks=num_tasks).to(device)
    else:
        loss_fn = UncertaintyLoss(num_tasks=num_tasks).to(device)
    ce_loss = nn.CrossEntropyLoss()
    ret_loss = AdjMSELoss(gamma=0.1)
    rank_loss = PairwiseRankingLoss(
        margin=0.0, tau=0.1, spread_target=1.0, spread_weight=0.5,
    )

    # FixedTaskWeights carries no learnable parameters — drop its (empty) param
    # group so AdamW never holds a no-op group.  The uncertainty scheme keeps
    # weight_decay=0 on its log_vars.
    loss_groups = [{"params": model.parameters()}]
    if list(loss_fn.parameters()):
        loss_groups.append({"params": loss_fn.parameters(), "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        loss_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    train_ds = PanelDataset(
        train_data, seq_len=config.seq_len, min_history=config.min_history,
        max_stocks_per_date=config.max_stocks_per_date, training=True,
    )
    train_sampler = DateSampler(train_ds.valid_mask)
    train_loader = DataLoader(
        train_ds, batch_size=1,
        sampler=train_sampler, collate_fn=panel_collate,
        num_workers=config.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=config.num_workers > 0,
        generator=loader_generator,
    )

    val_ds = PanelDataset(
        val_data, seq_len=config.seq_len, min_history=config.min_history,
        max_stocks_per_date=None, training=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )

    # §十二-3: theoretical ranking pool — per-date eligible stock counts and
    # the C(n,2) pair total they imply.  Fixed for the dataset, so computed
    # once here and reused as the coverage denominator every epoch.
    rank_pool_sizes, rank_possible_pairs = _rank_pool_stats(train_ds)

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

    # Checkpoint selection runs on validation RANKIC, never
    # the total loss: the uncertainty log_vars and rank-loss weight are
    # tunable, so a loss-weighted selection drifts as those weights change.
    # The per-task losses are logged as auxiliary traces only.
    best_val_rankic = float("-inf")
    best_state = None
    best_epoch_idx = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_ls_sharpe": [], "val_ic": [],
        "val_eval_epochs": [],  # 1-based epoch of each val_metrics entry
    }
    # §十二-4: entry-selection bias — training only ever sees candidates with a
    # real next-day open, while evaluation also ranks selectable-but-unfillable
    # days.  Record the train/eval candidate distribution difference once.
    history["entry_bias"] = _entry_bias_report(train_data, val_data, config.seq_len)
    for span, st in history["entry_bias"].items():
        if st["selectable"] > 0:
            logger.info(
                "  Entry bias [%s]: selectable=%d fillable=%d (%.1f%%) "
                "unfilled=%d dates=%d",
                span, st["selectable"], st["with_entry_open"],
                100 * st["fill_rate"], st["unfilled"], st["n_dates"])

    # §十二-5: declared reproducibility level (bitwise vs statistical-only).
    history["reproducibility"] = reproducibility
    logger.info(
        "  Reproducibility: %s (cudnn_deterministic=%s benchmark=%s "
        "tf32_matmul=%s tf32_cudnn=%s strict=%s compile=%s)",
        reproducibility["reproducibility_level"],
        reproducibility["cudnn_deterministic"],
        not reproducibility["cudnn_benchmark"],
        reproducibility["tf32_matmul"],
        reproducibility["tf32_cudnn"],
        reproducibility["strict_deterministic_algorithms"],
        reproducibility["compile_enabled"])
    use_amp = config.use_amp and device.type == "cuda"
    # The validation pool (candidate & return-target masks)
    # can be degenerate — too few eligible stocks per day, or near-total mask
    # retention loss.  Track whether ANY epoch produced a finite RankIC so a
    # silently empty signal fails loudly with the pool diagnostics instead of
    # picking a checkpoint on NaN.
    val_diag: dict = {}
    finite_rankic_count = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_rank_pairs = 0
        epoch_rank_batches = 0
        # Batches that actually contributed a finite loss (passed both the
        # no-active-task and the NaN/Inf guards) — the only correct denominator
        # for the epoch average, since skipped batches add 0 loss but would
        # otherwise inflate the divisor and understate avg_loss.
        epoch_valid_batches = 0
        optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            static, pk, po, y_dir, y_ret, y_vol, date_idx, dir_mask, ret_mask, vol_mask, _stock_idx = batch
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
                # Per-task masks: each loss runs ONLY over its
                # valid samples.  A batch whose direction labels are all -100
                # must not produce NaN CrossEntropy, and an inactive task must
                # not update its uncertainty log-var.
                ret_valid = ret_mask > 0
                vol_valid = vol_mask > 0
                # Task slots are positional (dir first, ret, vol) and must
                # match loss_fn's num_tasks + _compute_val_loss's indices.
                losses = []
                task_active = []
                if dir_enabled:
                    dir_valid = dir_mask > 0
                    l_ce = (
                        ce_loss(torch.clamp(pred_dir, -5, 5)[dir_valid], y_dir[dir_valid])
                        if dir_valid.any()
                        else torch.zeros((), device=device)
                    )
                    losses.append(l_ce)
                    task_active.append(bool(dir_valid.any().item()))
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
                # clean return targets only.  Computed every batch so the
                # monitoring trace stays populated even when the ablation drops
                # it from the optimised loss.
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
                if config.use_ranking_loss:
                    total_loss = total_loss + config.rank_loss_weight * l_rank

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    "NaN/Inf loss at epoch %d batch %d — skipping update",
                    epoch + 1, batch_idx,
                )
                continue

            epoch_valid_batches += 1
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

        # §十二-2: average over the batches that actually produced a finite
        # loss — skipped/NaN batches add 0 loss and must not inflate the
        # divisor (the old len(train_loader) denominator understated avg_loss).
        n_batches = max(epoch_valid_batches, 1)
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        # Ranking loss is normalized by rank-ACTIVE batches only (a batch with
        # no pairs contributes 0 and is not a fair divisor member either).
        rank_divisor = max(epoch_rank_batches, 1)

        # §十二-3: ranking-loss coverage for the epoch.  When a date's eligible
        # stock count exceeds batch_size the sampler splits it across
        # sub-batches and pairs are never formed across sub-batches — record how
        # much of the theoretical C(n,2) pair space the epoch actually covered
        # so a silently degraded ranking signal is visible in history.
        rank_active_rate = epoch_rank_batches / max(len(train_loader), 1)
        pair_coverage = epoch_rank_pairs / max(rank_possible_pairs, 1)
        history.setdefault("rank_coverage", []).append({
            "stocks_per_date": rank_pool_sizes,
            "pair_coverage": pair_coverage,
            "rank_active_rate": rank_active_rate,
            "pairs_per_epoch": epoch_rank_pairs,
            "possible_pairs": rank_possible_pairs,
            "rank_active_batches": epoch_rank_batches,
        })

        val_loss, v_ce, v_ret, v_vol, v_rankic = _compute_val_loss(
            model, val_loader, val_data, config, ret_loss, loss_fn, device,
            use_amp, vol_enabled=vol_enabled, dir_enabled=dir_enabled,
            diag=val_diag,
        )
        if np.isfinite(v_rankic):
            finite_rankic_count += 1
        history["val_loss"].append(val_loss)
        history.setdefault("val_ret", []).append(v_ret)
        history.setdefault("val_rankic", []).append(v_rankic)

        # Step scheduler AFTER optimizer updates (PyTorch >=1.1 requirement).
        # Called here at epoch end since this is an epoch-level scheduler.
        scheduler.step()

        # Primary selection metric = validation RankIC (maximize).  A NaN
        # RankIC (degenerate val set: <2 valid samples per day) is not a
        # regression — keep the last best without advancing the patience
        # counter so early stopping never fires on missing signal.
        if not np.isfinite(v_rankic):
            pass
        elif v_rankic > best_val_rankic:
            best_val_rankic = v_rankic
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
                # Formal training requires price paths — no
                # silent fallback to the legacy phase-concatenation estimator.
                require_price_path=True,
            )
            ls_sharpe = m["ls_sharpe"]
            ic_mean = m["ic_mean"]
            history["val_ls_sharpe"].append(ls_sharpe)
            history["val_ic"].append(ic_mean)
            history.setdefault("val_metrics", [])
            history["val_metrics"].append(m)
            history["val_eval_epochs"].append(epoch + 1)
            logger.info(
                "Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.5f) "
                "RankIC=%.4f rank=%.6f pairs=%d IC=%.4f(IR=%.2f) "
                "LS_Sharpe=%.2f[%.1f,%.1f] Long_Sharpe=%.2f q5-q1=%.1fbp lr=%.2e",
                epoch + 1, config.max_epochs, avg_loss, val_loss,
                v_ce, v_ret, v_vol, v_rankic,
                epoch_rank_loss / rank_divisor,
                epoch_rank_pairs,
                ic_mean, m["ic_ir"],
                ls_sharpe, m["ls_sharpe_lo"], m["ls_sharpe_hi"],
                m["long_sharpe"], m["q5mq1_ret"] * 10000,
                optimizer.param_groups[0]["lr"])
        else:
            logger.info(
                "Epoch %d/%d: loss=%.4f val=%.4f(CE=%.3f R=%.3f V=%.5f) "
                "RankIC=%.4f rank=%.6f pairs=%d lr=%.2e",
                epoch + 1, config.max_epochs, avg_loss, val_loss,
                v_ce, v_ret, v_vol, v_rankic,
                epoch_rank_loss / rank_divisor,
                epoch_rank_pairs,
                optimizer.param_groups[0]["lr"])

        if rank_pool_sizes:
            logger.info(
                "  Rank coverage: pairs=%d/%d (%.1f%%) active_batches=%d/%d "
                "(%.1f%%) stocks/date max=%d mean=%.1f",
                epoch_rank_pairs, rank_possible_pairs, 100 * pair_coverage,
                epoch_rank_batches, len(train_loader), 100 * rank_active_rate,
                max(rank_pool_sizes), float(np.mean(rank_pool_sizes)))

        if patience_counter >= config.early_stop_patience:
            logger.info("Early stopping at epoch %d (best val_rankic=%.4f)",
                        epoch + 1, best_val_rankic)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch_idx"] = best_epoch_idx
    # No epoch produced a finite validation RankIC — the
    # candidate pool is degenerate (too few eligible stocks per day).  Fail
    # loudly with the pool diagnostics instead of silently shipping a
    # checkpoint selected on NaN.
    history["val_diag"] = val_diag
    if finite_rankic_count == 0:
        raise ValueError(
            "inner validation produced NO finite RankIC across all "
            f"{config.max_epochs} epochs — refusing to select a checkpoint. "
            "Pool diagnostics: valid_days="
            f"{val_diag.get('valid_days', 'n/a')}, "
            "avg_stocks_per_day="
            f"{val_diag.get('avg_stocks_per_day', 'n/a')}, "
            "mask_retention="
            f"{val_diag.get('mask_retention', 'n/a')}. "
            "Check the fold's candidate/return-target masks and "
            "config.min_stocks_per_day."
        )
    # Exact portfolio evaluation on the deployed best-val-RankIC checkpoint.
    # The in-loop val_metrics snapshots are taken at each eval epoch from
    # whatever model was current then, which may differ from best_state —
    # reporting those is only a "nearest epoch proxy" for the true metrics.  Compute
    # the true IC/Sharpe of the checkpoint that is actually returned/saved.
    try:
        history["best_metrics"] = evaluate_portfolio(
            model, val_data, config, device,
            horizon=config.horizon, raw_returns=raw_val_returns,
            require_price_path=True,
        )
    except Exception:
        logger.exception("best-checkpoint portfolio evaluation failed")
    return model, history
