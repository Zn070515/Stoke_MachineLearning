"""Per-fold panel slicing and universe gates for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — per-fold PIT stock
eligibility, stock masking, the whole-run universe gates (delist / index
membership), candidate-pool gate application, cross-sectional normalization,
panel slicing, date formatting, sequence augmentation, and the experiment
artifacts writer.  ``train_panel`` re-exports these names for backward
compatibility.
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from stoke_ml.data.universe import (
    delist_global_index,
    index_membership_mask,
    load_index_membership,
    load_universe_status,
    not_delisted_mask,
)
from stoke_ml.features.pipeline import (
    _PIT_STATIC_COLS, fold_dead_feature_columns,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import evaluate_portfolio
from stoke_ml.models.panel.train import train_panel
from scripts.production.train_panel_model import (
    _best_eval_metrics,
    _predict_outer,
    _weight_hash,
)

logger = logging.getLogger(__name__)


def _quality_fail_reason(close: np.ndarray) -> str | None:
    """Structural quality check on a stock's CLOSE prefix.

    `close` must be ONLY the rows up to the fold's train_end (caller slices the
    panel) — the check is inherently PIT.  Returns a reason string if the stock
    is unusable on that prefix, else None.  Row-level badness (a non-positive
    close, a dead row) is NOT a reason to eject the stock — the pipeline masks
    those rows; only structural corruption (all-NaN prefix,
    >50 % daily vol, >1000 % forward move) excludes the stock from THAT fold.
    """
    if close.size == 0 or np.isnan(close).all():
        return "all_nan"
    ret = np.diff(close) / (close[:-1] + 1e-8)
    if np.nanstd(ret) > 0.50:  # >50 % daily vol = data error
        return "hi_vol"
    if len(close) > 5:
        fwd_ret = (close[5:] - close[:-5]) / (close[:-5] + 1e-8)
        if np.nanmax(np.abs(fwd_ret)) > 10.0:
            return "extreme_fwd"
    return None

def _fold_eligible_stocks(panel_data: dict, train_end: int) -> np.ndarray:
    """Per-fold PIT stock-level eligibility.

    A stock is eligible for a fold iff its close path is structurally clean on
    columns [0, train_end) — ONLY data before the fold's train boundary.  The
    old global `_filter_quality` loaded 2015→2099 once and ejected a stock from
    EVERY fold if a 2025 row was bad; this judges each fold on its own past, so
    real-market volatility can no longer masquerade as a "bad stock".

    Uses the panel's close_price grid (N, T), aligned to panel_stocks order, so
    no per-fold DataFrame regroup is needed.
    """
    close_price = panel_data["close_price"]  # (N, T) float32, NaN = no data
    n_stocks = close_price.shape[0]
    keep = np.ones(n_stocks, dtype=bool)
    for i in range(n_stocks):
        if _quality_fail_reason(close_price[i, :train_end]) is not None:
            keep[i] = False
    return keep

def _mask_stocks(data: dict, keep: np.ndarray) -> dict:
    """Drop ineligible stocks (axis 0) from every panel slice array."""
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[keep]
        else:
            out[k] = v
    return out

def _require_universe_artifacts(
    data_dir: str, universe_name: str, formal: bool,
) -> None:
    """§P0-7: a FORMAL experiment must never silently no-op its universe gates.

    ``exploratory`` runs (``--no-formal``) may degrade with a prominent marker;
    a formal run REFUSES to start when a gate's required artifact is missing:

      - csi300/csi500/csi800 require PIT ``membership.parquet`` intervals
        (without them the "per-day member" gate collapses to the historical
        union — the exact silent no-op §P0-7 calls out);
      - every universe requires delisting records (``delisted.parquet``) — the
        sleeve's force-sell policy (§七-1) is part of the executed task;
      - ``all`` additionally requires IPO records so the delisted merge is real.

    Returns normally when everything present; exits 1 with a precise list when
    a required artifact is missing in formal mode.
    """
    missing: list[str] = []

    def _present(*relparts: str) -> bool:
        path = os.path.join(data_dir, *relparts)
        if not os.path.isfile(path):
            return False
        try:
            return not pd.read_parquet(path).empty
        except Exception as exc:  # noqa: BLE001 — a corrupt artifact is as bad as absent
            logger.warning("universe artifact unreadable: %s (%s)", path, exc)
            return False

    if universe_name in ("csi300", "csi500", "csi800"):
        if not _present("a_shares", "index_constituents_hist", "membership.parquet"):
            missing.append("membership.parquet (PIT index-membership gate)")
    if not _present("a_shares", "universe", "delisted.parquet"):
        missing.append("delisted.parquet (delisting force-sell policy)")
    if universe_name == "all":
        if not _present("a_shares", "universe", "ipo.parquet"):
            missing.append("ipo.parquet (delisted-stock universe merge)")

    if not missing:
        return
    if formal:
        logger.error(
            "universe=%s: required PIT artifacts missing — %s.  A formal "
            "experiment must NOT silently no-op its universe gates (that would "
            "measure a different task than intended); rerun with --no-formal "
            "for an explicitly-degraded exploratory run (§P0-7).",
            universe_name, "; ".join(missing),
        )
        sys.exit(1)
    logger.warning(
        "[exploratory] universe=%s: %s missing — universe gate DEGRADED "
        "(silent no-op); formal runs refuse to start in this state (§P0-7).",
        universe_name, "; ".join(missing),
    )

def _fold_universe_gates(
    global_dates: np.ndarray,
    panel_stocks: list[str],
    universe_name: str,
    data_dir: str,
    formal: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, pd.DataFrame]:
    """§七-1/§七-3: compute the whole-run universe gates ONCE, before the fold loop.

    Both the deep model (train_panel.py) and the baselines
    (train_baselines_panel.py) must consume the SAME candidate-pool gates
    (§P0-5), so this is the single shared construction.  Returns, all in the
    panel's (N_stocks, T) grid space:

      nd_mask       — 未退市: blocks ENTRY from a known delisting column on.
      mem_mask      — per-day index membership (in_date <= date < out_date) for
                      csi300/csi500/csi800; None for other universes so fold
                      gates become a no-op for the "all A" / stratified studies.
      delist_global — per-stock delisting day in global panel-column space
                      (each fold's delist_day for the sleeve simulator).
      universe_status — the raw universe frame; returned so callers can hash
                      it into their artifacts (§P0-6 universe_status_hash).

    Missing universe parquets → empty status → delist_global all -1 and
    nd_mask all True (no force-sell, no entry gate), so a data-dir without
    records never crashes — the strict formal-mode failure for missing
    artifacts is enforced here up front (§P0-7).
    """
    _require_universe_artifacts(data_dir, universe_name, formal)
    universe_status = load_universe_status(data_dir)
    delist_global = delist_global_index(
        global_dates, universe_status, panel_stocks,
    )
    nd_mask = not_delisted_mask(global_dates, panel_stocks, universe_status)
    universe_index_codes = {
        "csi300": {"000300"},
        "csi500": {"000905"},
        "csi800": {"000300", "000905"},
    }.get(universe_name, set())
    mem_mask = None
    if universe_index_codes:
        membership_df = load_index_membership(data_dir, sorted(universe_index_codes))
        if membership_df.empty:
            logger.warning(
                "universe=%s: no membership.parquet intervals — per-day "
                "index-membership gate is a no-op (candidate pool keeps the "
                "full historical-member union)", universe_name,
            )
        else:
            mem_mask = index_membership_mask(global_dates, panel_stocks, membership_df)
    return nd_mask, mem_mask, delist_global, universe_status

def _apply_candidate_gates(
    dd: dict,
    tslice: slice,
    rows: np.ndarray,
    nd_mask: np.ndarray,
    mem_mask: np.ndarray | None,
) -> None:
    """§七-3: merge the universe gates into ONE evaluation candidate pool.

    ``dd`` is a fold slice already masked to eligible stocks; ``rows`` maps its
    row axis back to original panel-stock rows and ``tslice`` maps its columns
    to global panel columns, so the gates AND into
    ``dd["decision_eligible_mask"]`` in this fold's row/column space and
    ``_candidate_pool`` picks them up automatically.  §八.3: inner_train is by
    default left ungated — the model still learns from the broad
    historical-member union; only what gets RANKED as a tradable candidate is
    restricted.  ``--strict-index-training`` instead ANDs the per-day
    membership gate into inner_train's ``entry_eligible_mask`` (the dataset
    valid_mask) so the training loss matches the evaluation candidate pool.
    """
    cols = np.arange(tslice.start, tslice.stop)
    gate = nd_mask[np.ix_(rows, cols)]
    if mem_mask is not None:
        gate = gate & mem_mask[np.ix_(rows, cols)]
    dd["decision_eligible_mask"] &= gate

def _gate_inner_train_membership(
    inner_train: dict,
    mem_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    """§八.3 strict mode: AND per-day index membership into the inner-TRAIN
    ``entry_eligible_mask`` (which feeds the dataset valid_mask, the per-sample
    training-loss mask) so the model learns only from index-member days —
    matching the evaluation candidate pool.  ``rows`` maps the fold's row axis
    back to original panel-stock rows and ``cols`` are the inner_train grid
    columns (both aligned by the fold's slicing order).
    """
    inner_train["entry_eligible_mask"] &= mem_mask[np.ix_(rows, cols)]

def _gate_descriptions(
    consumes_membership: bool, strict_index_training: bool,
) -> tuple[str, str]:
    """§八.3: human-readable ``(eval_gate, train_gate)`` descriptions for the
    summary.  eval_gate always gates 未退市, plus per-day membership for
    universes that consume membership.parquet.  train_gate is the broad
    historical-member union (ungated) unless strict_index_training gates it by
    per-day membership.
    """
    eval_gate = "not_delisted" + (
        " + per-day-membership" if consumes_membership else "")
    train_gate = (
        "per-day-membership"
        if strict_index_training and consumes_membership
        else "union (ungated)"
    )
    return eval_gate, train_gate

def _fold_delist_day(
    delist_global: np.ndarray,
    fold_eligible: np.ndarray,
    val_start: int,
    Wp: int,
) -> np.ndarray:
    """Delist-day index in the fold's simulation column space.

    Sim column d ↔ global column val_start+d (evaluate_portfolio slices prices
    at seq_len within the outer_test window starting at val_context_start =
    val_start - seq_len), so subtract val_start.  Values outside [0, Wp) clamp
    to -1 — the force-sell never fires within this window.
    """
    dd_global = delist_global[fold_eligible]
    return np.where(
        (dd_global >= val_start) & (dd_global < val_start + Wp),
        dd_global - val_start, -1,
    )

def _universe_artifact_hashes(
    universe_status: pd.DataFrame,
    data_dir: str,
    universe_name: str,
) -> dict:
    """§P0-6: content hashes of the universe artifacts a run's gates consumed.

    ``universe_status_hash`` covers the delist/list records that drive
    ``delist_global`` and ``nd_mask``; ``membership_hash`` covers the
    index-membership intervals that drive ``mem_mask`` for csi300/csi500/csi800
    (None for universes where membership is not consumed).  Every fold tape and
    summary embeds these so a later replay can prove it used the SAME universe
    records — a delist-file or membership edit between runs invalidates the
    OOS tape instead of passing silently.
    """
    status_hash = hashlib.sha1()
    if universe_status is not None and not universe_status.empty:
        status_hash.update(universe_status.to_csv(index=False).encode("utf-8"))
    membership_hash: str | None = None
    universe_index_codes = {
        "csi300": {"000300"},
        "csi500": {"000905"},
        "csi800": {"000300", "000905"},
    }.get(universe_name, set())
    if universe_index_codes:
        path = os.path.join(
            data_dir, "a_shares", "index_constituents_hist", "membership.parquet")
        h = hashlib.sha1()
        if os.path.isfile(path):
            h.update(pd.read_parquet(path).to_csv(index=False).encode("utf-8"))
        membership_hash = h.hexdigest()[:16]
    return {
        "universe_status_hash": status_hash.hexdigest()[:16],
        "membership_hash": membership_hash,
    }

def _cross_sectional_normalize(
    y_arr: np.ndarray,
    mask_arr: np.ndarray,
    min_stocks: int = 5,
) -> np.ndarray:
    """Z-score normalize returns across stocks within each date.

    Preserves cross-sectional ordering while giving each date's return
    distribution zero mean and unit variance.  Dates with too few valid
    stocks are left unchanged.

    Returns a new array (does not mutate input).
    """
    y_out = y_arr.copy()
    n_stocks, n_dates = y_arr.shape
    for t in range(n_dates):
        valid = mask_arr[:, t] if mask_arr is not None else np.ones(n_stocks, dtype=bool)
        if valid.sum() < min_stocks:
            continue
        vals = y_arr[valid, t]
        mean_t = float(np.nanmean(vals))
        std_t = max(float(np.nanstd(vals)), 1e-8)
        y_out[valid, t] = (y_arr[valid, t] - mean_t) / std_t
    return y_out

def _slice_panel(panel_data: dict, tslice: slice, price_pad: int = 0) -> dict:
    """Slice every time-axis array of the panel by `tslice`.

    Static features are (N, T, D) PIT — sliced on the time
    axis like every other panel array.  Arrays that downstream code mutates in
    place (y_return z-score + clip, and their neighbours) are copied so one
    fold's normalization never corrupts the shared panel for later folds.

    `price_pad`: extend the close/open price columns by this many beyond
    `tslice.stop` (capped at the panel end).  The sleeve-account evaluation
    needs open[t+h] to liquidate a position entered at open[t],
    so the last `price_pad` sleeves get a real exit instead of a forced carry.

    `y_return_raw` is a copy of the RAW open-to-open return saved BEFORE the
    caller z-scores/clips `y_return`: clean IC and quintile
    spreads must be computed on raw returns, not on the normalized model target.
    """
    stop = tslice.stop
    out = {
        "static_features": panel_data["static_features"][:, tslice, :],
        "past_known": panel_data["past_known"][:, tslice],
        "past_observed": panel_data["past_observed"][:, tslice],
        "y_direction": panel_data["y_direction"][:, tslice],
        "y_return_raw": panel_data["y_return"][:, tslice].copy(),
        "y_return": panel_data["y_return"][:, tslice].copy(),
        "y_volatility": panel_data["y_volatility"][:, tslice].copy(),
        "observation_mask": panel_data["observation_mask"][:, tslice],
        "entry_eligible_mask": panel_data["entry_eligible_mask"][:, tslice],
        "return_target_mask": panel_data["return_target_mask"][:, tslice],
        "vol_target_mask": panel_data["vol_target_mask"][:, tslice],
        "realized_return": panel_data["realized_return"][:, tslice].copy(),
        # REBASE date_indices to LOCAL column space.  panel_builder emits the
        # GLOBAL calendar position (0..max_T-1); a fold slice with start > 0
        # must restart at 0 so date-centric consumers' window placement
        # ``window_idx = date_idx - seq_len`` stays inside the (N, n_windows)
        # grid.  Same-date stocks keep equal local indices within a dataset,
        # so PairwiseRankingLoss grouping is unchanged.
        "date_indices": (
            panel_data["date_indices"][:, tslice].copy() - (tslice.start or 0)
        ),
        "decision_eligible_mask": panel_data["decision_eligible_mask"][:, tslice],
        "history_eligible_mask": panel_data["history_eligible_mask"][:, tslice],
    }
    # Price paths feed the sleeve-account evaluation; a stale prebuilt panel
    # without them just falls back to the legacy path in evaluate_portfolio.
    if "close_price" in panel_data and "open_price" in panel_data:
        max_T = panel_data["close_price"].shape[1]
        pstop = min(stop + price_pad, max_T) if price_pad > 0 else stop
        out["close_price"] = panel_data["close_price"][:, tslice.start:pstop]
        out["open_price"] = panel_data["open_price"][:, tslice.start:pstop]
    return out

def _fmt_date(global_dates, idx):
    """Global-calendar position → 'YYYY-MM-DD'.  Out of range → None."""
    if global_dates is None or idx < 0 or idx >= len(global_dates):
        return None
    return str(np.datetime_as_string(global_dates[idx], unit="D"))

def _augment_sequence(
    pk: np.ndarray,
    po: np.ndarray,
    obs_mask: np.ndarray | None = None,
    noise_std: float = 0.01,
    mask_prob: float = 0.05,
    feat_dropout: float = 0.02,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FIXED per-fold corruption pass on a training panel (§十一.1).

    This is NOT online, per-sample augmentation.  It corrupts a whole
    (n_stocks, T, F) block once and returns a fixed copy that every epoch
    reuses verbatim, so the model never sees re-sampled noise:

    1. Gaussian noise ~ N(0, noise_std) — per-element independent, gated by
       `obs_mask` (True = real observation) so zero-padded history of new
       listings stays exactly zero instead of gaining fake noise the model
       would read as real data.
    2. Time masking — ONE global contiguous segment is zeroed for EVERY
       stock in the block (a single ``start``/``mask_len`` shared across the
       stock axis), not an independent segment per stock.
    3. Feature dropout — ONE global feature subset is zeroed for EVERY
       stock (a single boolean mask over the feature axis), not an
       independent subset per stock.

    Conservative magnitudes are the point: this exists to probe robustness,
    not to expand the effective sample size.  Because the corruption is
    global and static across epochs, it is opt-in ablation only and OFF by
    default in the formal baseline.
    """
    if rng is None:
        rng = np.random.RandomState()

    pk_aug = pk.copy()
    po_aug = po.copy()

    # 1. Gaussian noise (per-element, independent), only on real-observation days
    if noise_std > 0:
        noise_pk = rng.randn(*pk.shape).astype(np.float32) * noise_std
        noise_po = rng.randn(*po.shape).astype(np.float32) * noise_std
        if obs_mask is not None:
            obs_b = obs_mask[..., None].astype(np.float32)
            noise_pk *= obs_b
            noise_po *= obs_b
        pk_aug += noise_pk
        po_aug += noise_po

    # 2. Time masking: zero out a random contiguous block of length 1-5
    if mask_prob > 0 and pk.shape[1] >= 3:
        T = pk.shape[1]
        mask_len = rng.randint(1, min(6, T // 2 + 1))
        if rng.random() < mask_prob:
            start = rng.randint(0, T - mask_len)
            pk_aug[:, start:start + mask_len, :] = 0.0
            po_aug[:, start:start + mask_len, :] = 0.0

    # 3. Feature dropout: zero out random feature dimensions
    if feat_dropout > 0:
        for arr in [pk_aug, po_aug]:
            if arr.shape[2] > 0:
                mask = rng.random(arr.shape[2]) < feat_dropout
                arr[:, :, mask] = 0.0

    return pk_aug, po_aug

def _save_artifacts(
    outdir: str,
    args: argparse.Namespace,
    resolved: list[str],
    used: list[str],
    universe_desc: str,
    summary: dict | None,
    channel_manifest: dict | None = None,
    version: dict | None = None,
) -> str:
    """Persist the experiment: args, resolved/used universes, fold summary,
    the channel-coverage manifest, and the frozen data/code
    versions."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if version is not None:
        with open(os.path.join(outdir, "version.json"), "w", encoding="utf-8") as f:
            json.dump(version, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "universe_resolved.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n# n={len(resolved)}\n")
        f.write("\n".join(resolved))
        f.write("\n")
    with open(os.path.join(outdir, "universe_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n"
                f"# n={len(used)} (per-fold PIT eligibility applied inside the "
                f"fold loop)\n")
        f.write("\n".join(used))
        f.write("\n")
    if channel_manifest is not None:
        with open(os.path.join(outdir, "channel_coverage.json"),
                  "w", encoding="utf-8") as f:
            json.dump(channel_manifest, f, indent=2, ensure_ascii=False)
    if summary is not None:
        if channel_manifest:
            summary["channel_coverage"] = {
                k: {"status": v.get("status"),
                    "coverage": v.get("coverage")}
                for k, v in sorted(channel_manifest.items())
                if not k.startswith("_")
            }
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Experiment artifacts saved to %s", outdir)
    return outdir


@dataclasses.dataclass(frozen=True)
class _FoldLoopResult:
    """Accumulated fold-loop outputs consumed by the OOS/summary finalization.

    ``oos_*`` arrays are one row per (stock, entry-day) across all
    non-overlapping folds; ``oos_ledgers`` the per-fold position ledgers;
    ``all_sharpes`` / ``fold_histories`` the per-fold outer Sharpe and the
    fold history dicts.  Everything ``_finalize_summary`` needs from the loop.
    """
    oos_preds_all: list[np.ndarray]
    oos_dates_all: list[str]
    oos_stocks_all: list[str]
    oos_pool_all: list[np.ndarray]
    oos_fold_all: list[int]
    oos_weight_hash_all: list[str]
    oos_ledgers: list[pd.DataFrame]
    all_sharpes: list[float]
    fold_histories: list[dict]


def _run_fold_loop(
    panel_data: dict,
    panel_stocks: list[str],
    config: PanelConfig,
    device: torch.device,
    args: argparse.Namespace,
    global_dates: np.ndarray,
    n_timesteps: int,
    val_len: int,
    step: int,
    purge: int,
    lockbox_len: int,
    nd_mask: np.ndarray,
    mem_mask: np.ndarray | None,
    delist_global: np.ndarray,
    universe_hashes: dict,
    version_info: dict,
    oos_dir: str,
    n_trials: int,
) -> _FoldLoopResult:
    """Walk BACKWARD from the lockbox boundary running the per-fold train/eval loop.

    §二十一 extracted from ``train_panel.main`` — each fold slices the panel,
    judges per-fold PIT stock eligibility, drops dead features, applies the
    universe gates to the evaluation candidate pools (plus optional inner-TRAIN
    membership gating), trains the deployed checkpoint, evaluates it ONCE on
    the held-out outer test, and writes the fold's tape artifacts
    (``fold_NNN_model.pt`` / ``fold_NNN.npz`` / ``fold_NNN_ledger.parquet``)
    into ``oos_dir``.  Accumulates the per-(stock, entry-day) OOS arrays,
    per-position ledgers, outer Sharpes and fold histories that the OOS/summary
    finalization consumes.

    Returns a :class:`_FoldLoopResult` carrying every accumulator.  ``args`` /
    ``config`` / ``panel_data`` / ``panel_stocks`` / ``global_dates`` /
    ``n_timesteps`` and the whole-run universe gates (``nd_mask`` /
    ``mem_mask`` / ``delist_global``), the universe artifact hashes, the frozen
    ``version_info``, ``n_trials`` (DSR multiplicity) and ``oos_dir`` are the
    fold-independent context built once in ``main``.
    """
    all_sharpes = []
    fold_histories = []
    oos_dates_all: list[str] = []
    oos_stocks_all: list[str] = []
    oos_preds_all: list[np.ndarray] = []
    oos_pool_all: list[np.ndarray] = []
    oos_ledgers: list[pd.DataFrame] = []
    oos_fold_all: list[int] = []
    oos_weight_hash_all: list[str] = []
    rng = np.random.RandomState(args.seed)
    fold = 0
    # Walk BACKWARD from the lockbox boundary so the (max_folds) validation
    # windows cover the newest period instead of the earliest.  The training
    # window GROWS from position 0 out to (val_start - purge) each fold, so
    # the 2000-2015 history is genuinely in the training set — the old
    # fixed-width 756-day scheme left [0, n_timesteps-train_len-purge-val_len)
    # permanently unused and put short-history stocks' data entirely before
    # every fold window.
    # Reserve the `horizon` steps before the lockbox as a
    # settlement buffer — the last outer entry <= lockbox_start - horizon, so
    # its liquidation reads prices that end BEFORE the lockbox opens.  Without
    # this the final fold's sleeve would settle inside the untouched lockbox
    # and those gains would be counted as "evaluated" OOS performance.
    last_val_start = n_timesteps - config.horizon - val_len - lockbox_len
    val_start = last_val_start
    while val_start >= 0:
        if args.max_folds and fold >= args.max_folds:
            break
        train_end = val_start - purge
        if train_end < config.seq_len + 1:
            # The panel dataset needs at least seq_len+1 rows for one window
            # (n_windows = n_timesteps - seq_len must be >= 1).
            break
        fold += 1
        train_start = 0
        val_end = min(val_start + val_len, n_timesteps)

        # Carve the last ~15% of the trainable span as inner_val — used ONLY
        # for checkpoint selection inside train_panel.  The outer test (the old
        # val window) is fully held out of training and evaluated exactly once
        # on the deployed checkpoint.
        n_train_targets = train_end - config.seq_len
        inner_val_len = max(1, int(round(0.15 * n_train_targets)))
        inner_val_context_start = train_end - inner_val_len - config.seq_len
        if inner_val_context_start < config.seq_len + 1:
            # not enough rows left for one inner_train + one inner_val window
            break
        inner_train_end = inner_val_context_start
        val_context_start = val_start - config.seq_len

        inner_train_slice = slice(0, inner_train_end)
        inner_val_slice = slice(inner_val_context_start, train_end)
        outer_test_slice = slice(val_context_start, val_end)

        inner_train_data = _slice_panel(panel_data, inner_train_slice, price_pad=config.horizon)
        inner_val_data = _slice_panel(panel_data, inner_val_slice, price_pad=config.horizon)
        outer_test_data = _slice_panel(panel_data, outer_test_slice, price_pad=config.horizon)

        # Per-fold PIT stock-level eligibility: judge a stock
        # ONLY on data before train_end, never the full 2000→2099 history.  The
        # old global _filter_quality ejected a stock from EVERY fold because of
        # one bad 2025 row; now each fold judges its own past.  Row-level
        # badness remains masked (pipeline), not a reason to eject.
        fold_eligible = _fold_eligible_stocks(panel_data, train_end)
        fold_stocks = [panel_stocks[i] for i in np.where(fold_eligible)[0]]
        if len(fold_stocks) < 20:
            logger.warning("Fold %d: only %d stocks eligible PIT (need >= 20) — "
                           "skipping fold", fold, len(fold_stocks))
            val_start -= step
            continue
        inner_train_data = _mask_stocks(inner_train_data, fold_eligible)
        inner_val_data = _mask_stocks(inner_val_data, fold_eligible)
        outer_test_data = _mask_stocks(outer_test_data, fold_eligible)

        # Per-fold dead-feature drop (§十一-4): a column constant across every
        # observed day of THIS fold's training window is dropped from all three
        # slices (they share the column layout).  Judged only on the training
        # period — never validation/test — so a future fold can't decide an
        # earlier fold's feature set.  The full-history sparsity report is NOT
        # used for selection.  Config dims shrink by the same count so the
        # model's VSN input widths match the sliced grids.
        pk_drop, po_drop = fold_dead_feature_columns(
            inner_train_data,
            panel_data["past_known_cols"],
            panel_data["past_observed_cols"],
        )
        if pk_drop or po_drop:
            for dd in (inner_train_data, inner_val_data, outer_test_data):
                if pk_drop:
                    dd["past_known"] = np.delete(dd["past_known"], pk_drop, axis=2)
                if po_drop:
                    dd["past_observed"] = np.delete(dd["past_observed"], po_drop, axis=2)
            fold_config = dataclasses.replace(
                config,
                past_known_dim=config.past_known_dim - len(pk_drop),
                past_observed_dim=config.past_observed_dim - len(po_drop),
            )
            logger.info("Fold %d: dropped %d dead past_known + %d past_observed "
                        "columns (train-window constancy)",
                        fold, len(pk_drop), len(po_drop))
        else:
            fold_config = config

        # Merge the §七-3 universe gates into the EVALUATION candidate pools:
        # 未退市 for every universe, plus 当日是该指数成员 (per-day index
        # membership) for csi300/csi500/csi800.  §八.3: inner_train is by
        # DEFAULT left ungated — the model learns from the broad
        # historical-member union; only what gets RANKED as a tradable
        # candidate (inner_val/outer_test) is restricted.  That asymmetry is
        # recorded in the summary (train_gate/eval_gate).  Applied in this
        # fold's row/column space (rows = surviving original stock rows, cols
        # = the slice's global columns) so _candidate_pool picks the gates up
        # automatically.
        rows = np.where(fold_eligible)[0]
        for name, tslice, dd in (
            ("inner_val", inner_val_slice, inner_val_data),
            ("outer_test", outer_test_slice, outer_test_data),
        ):
            _apply_candidate_gates(dd, tslice, rows, nd_mask, mem_mask)

        # §八.3 strict mode: also gate the inner-TRAIN loss by per-day index
        # membership (see _gate_inner_train_membership).  Default: off —
        # inner_train learns from the broad historical-member union.
        if args.strict_index_training and mem_mask is not None:
            _gate_inner_train_membership(
                inner_train_data, mem_mask, rows, np.arange(0, inner_train_end))

        # y_return: cross-sectional z-score per date — preserves relative
        # ordering across stocks so ranking loss and IC evaluation work on a
        # consistent scale.  Normalize using the RETURN-target mask (clean
        # open-to-open returns) so dirty/missing positions don't skew the
        # z-score.  y_volatility: kept as the raw positive future-vol target
        # (std of the next-horizon daily returns).  VolatilityHead outputs
        # softplus > 0, so z-scoring the target would reintroduce the
        # negative-target-vs-positive-output contradiction — it must stay in
        # original units.
        inner_train_data["y_return"] = _cross_sectional_normalize(
            inner_train_data["y_return"], inner_train_data["return_target_mask"],
        )
        inner_val_data["y_return"] = _cross_sectional_normalize(
            inner_val_data["y_return"], inner_val_data["return_target_mask"],
        )
        outer_test_data["y_return"] = _cross_sectional_normalize(
            outer_test_data["y_return"], outer_test_data["return_target_mask"],
        )
        # Clip normalized targets to [-5, 5] — same band for train and val so
        # the model is never asked to fit z-scores beyond the eval range.
        # (Only y_return is z-scored; y_volatility stays in original units,
        # well below 5, so the clip applies to y_return only.)
        for dd in (inner_train_data, inner_val_data, outer_test_data):
            np.clip(dd["y_return"], -5.0, 5.0, out=dd["y_return"])

        # §十一.1: OPTIONAL fixed corruption pass on the inner-training data.
        # OFF by default.  This is NOT online per-sample augmentation — the
        # Gaussian noise is per-element independent (gated by observation_mask
        # so zero-padded history of new listings stays exactly zero), but the
        # time mask zeroes the SAME global time segment and the feature dropout
        # the SAME feature set for every stock; the pass runs once per fold and
        # every epoch reuses the identical corrupted copy.  Use --augment for
        # ablation only.
        if args.augment:
            pk_aug, po_aug = _augment_sequence(
                inner_train_data["past_known"],
                inner_train_data["past_observed"],
                obs_mask=inner_train_data["observation_mask"],
                noise_std=0.005,
                mask_prob=0.03,
                feat_dropout=0.01,
                rng=rng,
            )
            inner_train_data["past_known"] = pk_aug
            inner_train_data["past_observed"] = po_aug

        logger.info("Fold %d/%d: inner_train [%d:%d] inner_val [%d:%d] "
                    "outer_test [%d:%d]",
                    fold, args.max_folds or "∞",
                    0, inner_train_end,
                    inner_val_context_start, train_end,
                    val_context_start, val_end)

        t0 = time.time()
        # Checkpoint selection runs on inner_val inside train_panel; the
        # returned model is the best-inner-val checkpoint.
        model, history = train_panel(
            fold_config, inner_train_data, inner_val_data, device,
            raw_val_returns=inner_val_data["realized_return"],
        )
        elapsed = time.time() - t0

        # §十二-1: persist the best-inner-val checkpoint per fold so a fold's
        # OOS tape is reproducible / deployable, not only an in-memory state.
        # version_info["model_hash"] fingerprints config + architecture source
        # (shared by every fold); weight_hash below fingerprints the actual
        # trained parameters so an OOS row maps to exactly one set of weights.
        weight_hash = _weight_hash(model)
        if pk_drop or po_drop:
            pk_cols = [c for j, c in enumerate(panel_data["past_known_cols"])
                       if j not in pk_drop]
            po_cols = [c for j, c in enumerate(panel_data["past_observed_cols"])
                       if j not in po_drop]
        else:
            pk_cols = list(panel_data["past_known_cols"])
            po_cols = list(panel_data["past_observed_cols"])
        model_path = os.path.join(oos_dir, f"fold_{fold:03d}_model.pt")
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": dataclasses.asdict(fold_config),
            "feature_schema": {
                "static_cols": list(_PIT_STATIC_COLS)[:fold_config.static_dim],
                "past_known_cols": pk_cols,
                "past_observed_cols": po_cols,
            },
            "fold_range": {
                "train_start": _fmt_date(global_dates, 0),
                "train_end": _fmt_date(global_dates, inner_train_end - 1),
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "context_start": _fmt_date(global_dates, val_context_start),
                "signal_start": _fmt_date(global_dates, val_start - 1),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "exit_end": _fmt_date(
                    global_dates,
                    min(val_start + val_len - 1 + config.horizon,
                        n_timesteps - 1)),
                "test_start": _fmt_date(global_dates, val_context_start),
                "test_end": _fmt_date(global_dates, val_end - 1),
            },
            "weight_hash": weight_hash,
            "model_source_hash": version_info["model_source_hash"],
            "model_config_hash": version_info["model_config_hash"],
            "best_epoch": history.get("best_epoch_idx", 0) + 1,
            "data_version": version_info["data_manifest_hash"],
            "feature_schema_hash": version_info["feature_schema_hash"],
            "git_commit": version_info["git_commit"],
            "evaluator_version": version_info["evaluator_version"],
            "seed": args.seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, model_path)
        logger.info("  Fold %d: best checkpoint (epoch %d, weights %s) -> %s",
                    fold, history.get("best_epoch_idx", 0) + 1,
                    weight_hash, model_path)

        # Evaluate the exact deployed checkpoint ONCE on the held-out outer
        # test — the honest out-of-sample number, never used for selection.
        # Delist-day index in the fold's simulation column space, via the
        # shared helper (§P0-5) so the baselines force-sell delisted positions
        # exactly as the deep model does.
        Wp = outer_test_data["close_price"].shape[1] - config.seq_len
        delist_day = _fold_delist_day(
            delist_global, fold_eligible, val_start, Wp)
        outer_m = evaluate_portfolio(
            model, outer_test_data, config, device,
            horizon=config.horizon,
            top_fraction=0.1,
            raw_returns=outer_test_data["realized_return"],
            # Formal training must use the chronological
            # sleeve account — a prebuilt panel without price paths is a data
            # bug, not a reason to silently downgrade to the legacy estimator.
            require_price_path=True,
            # Emit the per-position ledger so the OOS tape
            # records every fill the account actually made, offline-replayable.
            return_ledger=True,
            # Known-delisted stocks are force-sold at the delisting close
            # instead of dangling as UNRESOLVED (§七-1).
            delist_day=delist_day,
            # §十五-1: project-wide trial count for the DSR multiplicity.
            n_trials=n_trials,
        )
        best_epoch = history.get("best_epoch_idx", 0) + 1

        # Daily OOS predictions: one return forecast per
        # (stock, entry day).  A window's entry is global column val_start+d,
        # so entry dates run global_dates[val_start .. val_start+val_len-1].
        oos_preds = _predict_outer(model, outer_test_data, config, device)
        if oos_preds is not None:
            n_w = oos_preds.shape[1]
            p0 = config.seq_len
            entry_dates = [_fmt_date(global_dates, val_start + d) for d in range(n_w)]
            # Window-day grid arrays (column d ↔ panel column seq_len+d), all
            # aligned exactly as evaluate_portfolio slices them, so a tape
            # consumer can reconstruct the sleeve account offline: the
            # selection pool (decision & history), the entry/open-validity
            # fill gate, the clean open->open return target (saved before
            # z-score) and its mask.
            dec = outer_test_data["decision_eligible_mask"][:, p0:p0 + n_w]
            hist = outer_test_data["history_eligible_mask"][:, p0:p0 + n_w]
            pool = dec & hist
            elig = outer_test_data["entry_eligible_mask"][:, p0:p0 + n_w]
            rt_mask = outer_test_data["return_target_mask"][:, p0:p0 + n_w]
            rt = outer_test_data["y_return_raw"][:, p0:p0 + n_w]
            # Price paths on the same grid, with `horizon` EXTRA columns so the
            # sleeve entered on the last signal day W-1 can still liquidate at
            # open[W-1+horizon] — identical to the grid
            # evaluate_portfolio passes to the sleeve simulator.
            price_grid = outer_test_data["close_price"][:, p0:p0 + n_w + config.horizon]
            open_grid = outer_test_data["open_price"][:, p0:p0 + n_w + config.horizon]
            price_dates = [_fmt_date(global_dates, val_start + d)
                           for d in range(n_w + config.horizon)]
            np.savez(
                os.path.join(oos_dir, f"fold_{fold:03d}.npz"),
                preds=oos_preds,
                dates=np.array(entry_dates),
                stocks=np.array(fold_stocks),
                decision_eligible=dec,
                history_eligible=hist,
                pool=pool,
                entry_eligible=elig,
                return_target_mask=rt_mask,
                return_target=rt,
                close_price=price_grid,
                open_price=open_grid,
                price_dates=np.array(price_dates),
                horizon=config.horizon,
                seq_len=config.seq_len,
                top_fraction=0.1,
                cost=config.txn_cost,
                # §P0-6: the force-sell delist-day grid (in this fold's sim
                # column space) so an offline replay force-sells delisted
                # positions exactly as the live sleeve did, and the content
                # hashes of the universe records the gates consumed.
                delist_day=delist_day,
                universe_status_hash=universe_hashes["universe_status_hash"],
                membership_hash=universe_hashes["membership_hash"],
                # §十二.2: calendar content hash — the tape must not blend folds
                # trained under a different holiday set / verified_until.
                calendar_hash=version_info["calendar_artifact_hash"],
                # A tape row must identify the data + model
                # that produced it, so every return number is traceable.
                data_version=version_info["data_manifest_hash"],
                model_hash=version_info["model_hash"],
                # §十六: the split model-identity hashes the formal continuous
                # replay REQUIRES to be identical across folds (architecture /
                # config / feature-schema).  weight_hash below — the actual
                # trained parameters — is allowed to differ per fold.
                model_source_hash=version_info["model_source_hash"],
                model_config_hash=version_info["model_config_hash"],
                feature_schema_hash=version_info["feature_schema_hash"],
                # Trained-parameter hash — the fold's tape maps
                # to the exact weights in fold_XXX_model.pt (config+source
                # model_hash is shared by all folds, this one is not).
                weight_hash=weight_hash,
                # §十五-1: the strategy policy + evaluator identity that produced
                # this tape.  The continuous replay REJECTS a directory whose
                # folds disagree on any of these — otherwise a horizon=5 fold
                # mixed with a horizon=20 fold would be replayed with the first
                # tape's policy explaining the whole account.
                evaluator_version=version_info["evaluator_version"],
                price_convention="open_to_open",
                exit_policy="scheduled_horizon_delayed_delist_force_sell",
                strategy_mode="long_top_fraction",
            )
            # Per-position ledger: the exact fills the long
            # sleeve account made — entry/exit price, exit status, gross/net
            # PnL and attributed costs — mapped to dates and stock codes so the
            # tape is self-contained.  Sum over net_pnl (resolved) +
            # unrealized_pnl (unresolved) == final_nav - 1 holds per fold by
            # construction (enforced inside _run_sleeve_sim, §十三-3).
            ledger_rows = outer_m.get("long_ledger")
            if ledger_rows:
                ldf = pd.DataFrame(ledger_rows)
                si = ldf["stock"].to_numpy(dtype=int)
                di = ldf["entry_day"].to_numpy(dtype=int)
                ldf["entry_date"] = [entry_dates[c] for c in di]
                ldf["stock_code"] = [fold_stocks[i] for i in si]
                ldf["prediction"] = oos_preds[si, di]
                ldf["candidate_eligible"] = pool[si, di]
                ldf["entry_eligible"] = elig[si, di]
                ldf["fold"] = fold
                # Data/model provenance columns on every tape
                # row — realized return + executed weight make the P&L fully
                # recomputable from prices alone.
                ldf["data_version"] = version_info["data_manifest_hash"]
                ldf["model_hash"] = version_info["model_hash"]
                # §十四-2: `entry_value`/`executed_weight` (both the nominal) are
                # split into entry_notional + target_weight + executed_weight
                # (notional/entry_nav) + entry_nav, so an offline consumer can
                # distinguish the intended weight from the cash-cap-reduced one.
                ldf = ldf[["fold", "entry_day", "entry_date", "stock",
                           "stock_code", "mode", "prediction",
                           "candidate_eligible", "entry_eligible",
                           "entry_price", "entry_notional", "target_weight",
                           "executed_weight", "entry_nav",
                           "shares", "scheduled_exit_day", "actual_exit_day",
                           "exit_status", "exit_price", "realized_return",
                           "mark_day", "mark_price", "gross_pnl",
                           "entry_cost", "exit_cost", "net_pnl",
                           "unrealized_pnl"]]
                ledger_path = os.path.join(oos_dir, f"fold_{fold:03d}_ledger.parquet")
                ldf.to_parquet(ledger_path)
                oos_ledgers.append(ldf)
                logger.info("  Fold %d: ledger %d filled positions -> %s",
                            fold, len(ldf), ledger_path)
            for d in range(n_w):
                oos_dates_all.extend([entry_dates[d]] * len(fold_stocks))
                oos_stocks_all.extend(fold_stocks)
                oos_preds_all.append(oos_preds[:, d])
                oos_pool_all.append(pool[:, d])
                oos_fold_all.extend([fold] * len(fold_stocks))
                oos_weight_hash_all.extend([weight_hash] * len(fold_stocks))

        if outer_m["n_periods"] >= 2:
            best_ls = outer_m["ls_sharpe"]
            all_sharpes.append(best_ls)
            # Inner-val eval nearest the deployed checkpoint — what selection
            # actually saw, reported honestly alongside the held-out outer
            # metrics (never report a post-hoc max).
            inner_eval_m, inner_eval_epoch = _best_eval_metrics(history)
            # Input-context date bounds of each segment — column t of the panel
            # is global_dates[t], so a slice [a,b) covers dates [a, b-1].
            # Semantic dates: entry day e buys at open[e],
            # the signal is produced after close[e-1], and the input context is
            # the seq_len days [e-seq_len, e).
            fold_histories.append({
                "history": history,
                "outer_metrics": outer_m,
                "best_epoch": best_epoch,
                "inner_eval_epoch": inner_eval_epoch,
                "inner_eval_ls_sharpe": inner_eval_m.get("ls_sharpe"),
                "inner_eval_ic": inner_eval_m.get("ic_mean"),
                "weight_hash": weight_hash,
                # §P0-6: the universe records this fold's gates consumed, so the
                # fold result is provably tied to those delist/membership files.
                "universe_status_hash": universe_hashes["universe_status_hash"],
                "membership_hash": universe_hashes["membership_hash"],
                "model_path": f"oos_preds/fold_{fold:03d}_model.pt",
                "train_start": _fmt_date(global_dates, 0),
                "train_end": _fmt_date(global_dates, inner_train_end - 1),
                "inner_val_start": _fmt_date(global_dates, inner_val_context_start),
                "inner_val_end": _fmt_date(global_dates, train_end - 1),
                "context_start": _fmt_date(global_dates, val_context_start),
                "signal_start": _fmt_date(global_dates, val_start - 1),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "exit_end": _fmt_date(
                    global_dates,
                    min(val_start + val_len - 1 + config.horizon, n_timesteps - 1)),
                "test_start": _fmt_date(global_dates, val_context_start),
                "test_end": _fmt_date(global_dates, val_end - 1),
            })
            logger.info(
                "  Fold %d: best@epoch%d OUTER-TEST LS_Sharpe=%.2f IC=%.4f(IR=%.2f) "
                "Long_Sharpe=%.2f Q5-Q1=%.1fbp ElgEW_Sharpe=%.2f SelUniEW_Sharpe=%.2f (%.1fs)",
                fold, best_epoch, best_ls,
                outer_m.get("ic_mean", 0), outer_m.get("ic_ir", 0),
                outer_m.get("long_sharpe", 0),
                outer_m.get("q5mq1_ret", 0) * 10000,
                outer_m.get("eligible_ew_sharpe", 0),
                outer_m.get("selected_universe_ew_sharpe", 0),
                elapsed,
            )
        else:
            logger.warning(
                "  Fold %d: outer-test too short for metrics (%.1fs)", fold, elapsed,
            )

        val_start -= step
    return _FoldLoopResult(
        oos_preds_all=oos_preds_all,
        oos_dates_all=oos_dates_all,
        oos_stocks_all=oos_stocks_all,
        oos_pool_all=oos_pool_all,
        oos_fold_all=oos_fold_all,
        oos_weight_hash_all=oos_weight_hash_all,
        oos_ledgers=oos_ledgers,
        all_sharpes=all_sharpes,
        fold_histories=fold_histories,
    )
