"""Per-fold panel slicing and universe gates for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — per-fold PIT stock
eligibility, stock masking, the whole-run universe gates (delist / index
membership), candidate-pool gate application, cross-sectional normalization,
panel slicing, date formatting, sequence augmentation, and the experiment
artifacts writer.  ``train_panel`` re-exports these names for backward
compatibility.
"""
import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from stoke_ml.data.universe import (
    delist_global_index,
    index_membership_mask,
    load_index_membership,
    load_universe_status,
    not_delisted_mask,
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
