"""Train VSN+xLSTM panel model on A-share stocks.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --stocks 500 --epochs 30 --max-folds 1
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --universe csi300 --stocks 300 --outdir reports/exp/csi300
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --stock-list 600519,000001,000858
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/train_panel.py --no-aux  # skip auxiliary data for quick test

Universe modes (--universe): first / random / stratified / all / csi300 / csi500 / csi800.
Artifacts (args.json, universe_resolved.txt, universe_used.txt, summary.json)
are saved to --outdir (default reports/experiments/<timestamp>).
"""
import argparse
import dataclasses
import hashlib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stoke_ml.config import load_config
from stoke_ml.data.calendar import (
    TradingCalendar,
    build_calendar_frame,
    load_calendar,
)
from stoke_ml.data.universe import (
    delist_global_index,
    index_membership_mask,
    load_index_membership,
    load_universe_status,
    not_delisted_mask,
)
from stoke_ml.features import cache_manifest
from stoke_ml.features.pipeline import (
    FeaturePipeline, _PIT_STATIC_COLS, fold_dead_feature_columns,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset, panel_collate
from stoke_ml.models.panel.train import train_panel
from stoke_ml.models.panel.evaluate import (
    EVALUATOR_VERSION,
    _run_sleeve_sim,
    compute_equity_curve,
    compute_max_drawdown,
    compute_sharpe,
    evaluate_portfolio,
)
from stoke_ml.models.panel.inference import (
    compute_deflated_sharpe,
    compute_psr,
)
from scripts.production.data_quality_gate import (
    QUALITY_GATE_VERSION, contract_version, dataset_fingerprint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _discover_stocks(data_dir: str, limit: int | None = None) -> list[str]:
    from stoke_ml.data.storage import DataStorage
    stocks = DataStorage(data_dir).list_stocks()
    return stocks[:limit] if limit else stocks


def _exchange_group(stock_code: str) -> str:
    """Exchange bucket by code prefix: SH=6, SZ=0/3, BJ=4/8."""
    if stock_code.startswith("6"):
        return "SH"
    if stock_code.startswith(("0", "3")):
        return "SZ"
    return "BJ"


def _load_index_universe(data_dir: str, index_codes: set[str]) -> list[str]:
    """Stocks ever in the given indices (PIT in_date/out_date union).

    Any stock that was a member at any point in history is included — the
    training span runs 2000-2026, so a fixed-date snapshot would silently
    drop stocks that left the index mid-period.
    """
    path = os.path.join(
        data_dir, "a_shares", "index_constituents_hist", "membership.parquet",
    )
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    members = df[df["index_code"].isin(index_codes)]["stock_code"].unique()
    return sorted(members)


def _require_quality_gate(
    data_dir: str,
    prebuilt_dir: str | None,
    report_path: str,
) -> dict:
    """Verify a matching quality-gate report covers the data this run consumes.

    §六-2: training must not read data the gate has not validated, or that
    changed since the gate PASS.  A missing / stale / mismatched report exits
    the run; --no-require-quality-gate disables the whole gate (dev smoke).
    """
    if not os.path.isfile(report_path):
        raise SystemExit(
            f"quality gate report not found at {report_path} — run "
            f"scripts/production/build_features.py --quality-gate (or the gate directly) "
            f"before training, or pass --no-require-quality-gate to bypass."
        )
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    problems: list[str] = []
    if not report.get("passed"):
        problems.append("gate did not PASS")
    if report.get("quality_gate_version") != QUALITY_GATE_VERSION:
        problems.append(
            f"gate version {report.get('quality_gate_version')!r} "
            f"!= {QUALITY_GATE_VERSION!r}"
        )
    if os.path.realpath(str(report.get("data_root") or "")) != os.path.realpath(data_dir):
        problems.append(f"data_root {report.get('data_root')!r} != {data_dir!r}")
    if report.get("calendar_version") != TradingCalendar.CALENDAR_VERSION:
        problems.append(
            f"calendar {report.get('calendar_version')!r} "
            f"!= {TradingCalendar.CALENDAR_VERSION!r}"
        )
    if report.get("contract_version") != contract_version():
        problems.append("daily contract changed since the gate ran")
    consumed = {"daily"}
    if prebuilt_dir:
        name = os.path.basename(os.path.normpath(prebuilt_dir))
        if name in ("features", "features_panel"):
            consumed.add(name)
    gate_required = set(report.get("required_datasets") or [])
    if not consumed <= gate_required:
        problems.append(
            f"gate required datasets {sorted(gate_required)} do not cover "
            f"consumed {sorted(consumed)}"
        )
    live_hash = dataset_fingerprint(data_dir, sorted(gate_required))
    if report.get("data_manifest_hash") != live_hash:
        problems.append("dataset fingerprint changed since the gate PASS")
    if problems:
        raise SystemExit(
            "quality-gate report mismatch — refusing to train on unvalidated "
            "data: " + "; ".join(problems)
        )
    return report


def _check_verified_until_scope(global_dates, enforce: bool) -> str | None:
    """§九-3: refuse a formal run whose panel axis reaches past verified_until.

    Forward-estimate trading days (2027+ A-share closures) are not verified
    exchange fact, so a formal run (quality gate enforced) that spans them is
    refused instead of silently mixing guessed holidays into the panel axis.
    Uses the strict calendar's own bound so the refusal is exactly the
    strict-mode contract.  ``enforce`` is False for exploratory runs that opt
    out via --no-require-quality-gate.  Returns the refusal message (caller
    raises SystemExit) or None when in scope.
    """
    if not enforce or global_dates is None or not len(global_dates):
        return None
    strict_cal = TradingCalendar("a_shares", strict=True)
    lo = pd.Timestamp(global_dates[0]).date()
    hi = pd.Timestamp(global_dates[-1]).date()
    try:
        strict_cal.get_trading_days(lo, hi)
    except ValueError as exc:
        return f"formal run refused: {exc}"
    return None


def _resolve_universe(
    all_stocks: list[str],
    universe: str,
    limit: int | None,
    seed: int,
    data_dir: str,
) -> tuple[list[str], str]:
    """Select the training universe by name.

    Returns (resolved_stocks, description).  The description records exactly
    what was selected (mode / seed / count) for the experiment artifacts.

    Modes:
      first      — first N by code (legacy behaviour, alphabetical bias)
      random     — seeded sample of N from all stocks (no code-order bias)
      stratified — seeded sample targeting N, balanced across SH/SZ/BJ
      all        — every stock (ignores --stocks)
      csi300/500/800 — index constituents (PIT union), --stocks caps count
    """
    if limit is None:
        limit = len(all_stocks)
    all_sorted = sorted(all_stocks)
    if universe == "all":
        return all_sorted, f"all ({len(all_sorted)} stocks)"
    if universe == "first":
        chosen = all_sorted[:limit]
        return chosen, f"first {len(chosen)} (sorted by code)"
    if universe == "random":
        rng = np.random.RandomState(seed)
        k = min(limit, len(all_stocks))
        chosen = rng.choice(all_sorted, size=k, replace=False).tolist()
        return chosen, f"random {len(chosen)} (seed={seed})"
    if universe == "stratified":
        rng = np.random.RandomState(seed)
        by_group: dict[str, list[str]] = {"SH": [], "SZ": [], "BJ": []}
        for code in all_sorted:
            by_group[_exchange_group(code)].append(code)
        chosen: list[str] = []
        remaining = limit
        for group, codes in by_group.items():
            share = min(remaining // 3, len(codes))
            chosen.extend(rng.choice(codes, size=share, replace=False).tolist())
            remaining -= share
        if remaining > 0:
            used = set(chosen)
            leftover = [c for codes in by_group.values() for c in codes if c not in used]
            chosen.extend(
                rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist()
            )
        rng.shuffle(chosen)
        return chosen, f"stratified {len(chosen)} (seed={seed}, SH/SZ/BJ balanced)"
    if universe in ("csi300", "csi500", "csi800"):
        idx_map = {"csi300": {"000300"}, "csi500": {"000905"}, "csi800": {"000300", "000905"}}
        members = _load_index_universe(data_dir, idx_map[universe])
        if not members:
            return [], f"{universe}: no membership data"
        # Keep only members we actually have daily K-line for — the membership
        # file includes codes with no data (delisted / not downloaded).
        have = set(all_stocks)
        members = [c for c in members if c in have]
        chosen = members[:limit] if limit else members
        return chosen, f"{universe} {len(chosen)} (PIT union, cap={limit})"
    raise ValueError(f"unknown --universe: {universe}")


# Channel coverage manifest: every aux channel is loaded
# per-stock with per-stock error counting, so an experiment that silently lost a
# whole channel (storage schema update, missing dir) is caught instead of
# finishing quietly.  `status` distinguishes three empty states:
#   MISSING — channel absent from disk (loaded 0, errors 0)
#   FAILED  — storage construction/read broke (errors == n_stocks)
#   PARTIAL — some stocks loaded, some errored
_HAS_FLAG_CHANNELS = {
    "has_news": "sentiment",
    "has_guba_post": "guba",
    "has_comment": "comment",
    "has_announce": "announcement",
    "has_forecast": "earnings",
    "has_pledge": "pledge",
    "has_hot_board": "concept",
}


def _new_channel_entry(requested: bool, required: bool) -> dict:
    return {
        "requested": requested,
        "required": required,
        "loaded_stocks": 0,
        "coverage": 0.0,
        "errors": 0,
        "status": "MISSING",
    }


def _finalize_channel(entry: dict, name: str, loaded: int, errors: int, n: int) -> None:
    entry["loaded_stocks"] = loaded
    entry["errors"] = errors
    entry["coverage"] = round(loaded / n, 4) if n else 0.0
    entry["status"] = (
        "FAILED" if loaded == 0 and errors > 0 else
        "MISSING" if loaded == 0 else
        "PARTIAL" if loaded < n else "OK"
    )
    logger.info("[%s] loaded %d/%d stocks (errors=%d) %s",
                name, loaded, n, errors, entry["status"])


def _load_channel_aux(
    name: str,
    stock_list: list[str],
    result: dict[str, dict[str, pd.DataFrame]],
    manifest: dict[str, dict],
    make_storage,      # Callable[[], object] — storage construction (raises → channel FAILED)
    load_one,          # Callable[[object, str], pd.DataFrame | None]
    required: bool = False,
) -> None:
    """Per-stock aux load with per-stock error counting."""
    entry = _new_channel_entry(True, required)
    manifest[name] = entry
    n = len(stock_list)
    try:
        storage = make_storage()
    except Exception as exc:
        entry["errors"] = n
        entry["status"] = "FAILED"
        entry["note"] = f"storage construction failed: {exc}"
        logger.warning("[%s] storage unavailable — %s", name, exc)
        return
    loaded = 0
    errors = 0
    for code in stock_list:
        try:
            df = load_one(storage, code)
            if df is not None and not df.empty:
                result[code][name] = df
                loaded += 1
        except Exception:
            errors += 1
    _finalize_channel(entry, name, loaded, errors, n)


def load_aux_data(
    stock_list: list[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    required_channels: set[str] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict]:
    """Load auxiliary data (sentiment, guba, margin, etc.) per stock.

    Returns (result, manifest):
      result   — {stock_code: {"sentiment": df, "guba": df, ...}}
      manifest — per-channel coverage (requested/required/loaded_stocks/
                 coverage/errors/status).
    """
    from stoke_ml.data.news_storage import NewsStorage
    from stoke_ml.data.guba_storage import GubaStorage
    from stoke_ml.data.market_wide_storage import MarketWideStorage
    from stoke_ml.data.fundamental_storage import FundamentalStorage
    from stoke_ml.data.comment_storage import CommentStorage
    from stoke_ml.data.announcement_storage import AnnouncementStorage

    result: dict[str, dict[str, pd.DataFrame]] = {c: {} for c in stock_list}
    manifest: dict[str, dict] = {}
    required_set = set(required_channels or ())

    # Sentiment (news)
    _load_channel_aux(
        "sentiment", stock_list, result, manifest,
        make_storage=lambda: NewsStorage(data_dir),
        load_one=lambda ns, code: ns.load_daily_sentiment(code, start_date, end_date),
        required=("sentiment" in required_set),
    )

    # Announcements (CNINFO PDF body sentiment preferred, EastMoney fallback)
    def _make_ann():
        cninfo_dir = os.path.join(data_dir, "a_shares", "cninfo_announcements", "sentiment")
        return (cninfo_dir, AnnouncementStorage(data_dir))

    def _load_ann(storage_tuple, code):
        cninfo_dir, a_store = storage_tuple
        path = os.path.join(cninfo_dir, f"{code}.parquet")
        if os.path.isfile(path):
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            if not df.empty:
                return df.sort_values("date").reset_index(drop=True)
            return None
        return a_store.load_daily_sentiment(code, start_date, end_date)

    _load_channel_aux(
        "announcement", stock_list, result, manifest,
        make_storage=_make_ann,
        load_one=_load_ann,
        required=("announcement" in required_set),
    )

    # Guba
    _load_channel_aux(
        "guba", stock_list, result, manifest,
        make_storage=lambda: GubaStorage(data_dir),
        load_one=lambda gs, code: gs.load_daily_sentiment(code, start_date, end_date),
        required=("guba" in required_set),
    )

    # Comment
    _load_channel_aux(
        "comment", stock_list, result, manifest,
        make_storage=lambda: CommentStorage(data_dir),
        load_one=lambda cs, code: cs.build_features(code, start_date, end_date),
        required=("comment" in required_set),
    )

    # Fundamental (quarterly, backfilled from 2010 so the forward-fill spans)
    _load_channel_aux(
        "fundamental", stock_list, result, manifest,
        make_storage=lambda: FundamentalStorage(data_dir),
        load_one=lambda fs, code: fs.load(code, "2010-01-01", end_date),
        required=("fundamental" in required_set),
    )

    # MarketWideStorage channels (margin/northbound/dragon_tiger/capital_flow/
    # block_trade/shareholder/lockup/dividend/valuation) — identical pattern.
    for ch in (
        "margin", "northbound", "dragon_tiger", "capital_flow", "block_trade",
        "shareholder", "lockup", "dividend", "valuation",
    ):
        _load_channel_aux(
            ch, stock_list, result, manifest,
            make_storage=lambda ch=ch: MarketWideStorage(data_dir, ch),
            load_one=lambda st, code, ch=ch: st.load(code, start_date, end_date),
            required=(ch in required_set),
        )

    # ETF Flow (sector-level, aggregated to market-wide per date, broadcast to
    # every stock — not a per-stock channel).
    entry = _new_channel_entry(True, "etf_flow" in required_set)
    manifest["etf_flow"] = entry
    try:
        etf_base = os.path.join(data_dir, "a_shares", "etf_flow")
        etf_frames = []
        if os.path.isdir(etf_base):
            for f in os.listdir(etf_base):
                if f.startswith("sector_") and f.endswith(".parquet"):
                    etf_frames.append(pd.read_parquet(os.path.join(etf_base, f)))
        if etf_frames:
            etf_all = pd.concat(etf_frames, ignore_index=True)
            etf_all["date"] = pd.to_datetime(etf_all["date"])
            etf_agg = etf_all.groupby("date").agg(
                etf_flow_sum=("etf_flow_sum", "sum"),
                etf_amount_sum=("etf_amount_sum", "sum"),
            ).reset_index()
            for code in stock_list:
                result[code]["etf_flow"] = etf_agg
            entry["loaded_stocks"] = len(stock_list)
            entry["coverage"] = 1.0 if stock_list else 0.0
            entry["status"] = "OK"
            logger.info("[etf_flow] aggregated from %d sector files "
                        "(broadcast to %d stocks)", len(etf_frames), len(stock_list))
        else:
            logger.info("[etf_flow] no sector files found — MISSING")
    except Exception as exc:
        entry["errors"] = len(stock_list)
        entry["status"] = "FAILED"
        entry["note"] = str(exc)
        logger.warning("[etf_flow] aggregation failed — %s", exc)

    loaded = sum(1 for v in result.values() if v)
    logger.info("Aux data loaded for %d/%d stocks", loaded, len(stock_list))
    return result, manifest


def _prebuilt_channel_coverage(panel_data: dict) -> dict:
    """Channel coverage probed from a prebuilt panel's has_* flags.

    past_observed is (N, T, D); each has_* flag is True on exactly the
    (stock, day) cells where that aux channel delivered data (the pipeline
    ZI-fills absent cells, so False == no data).  Coverage = fraction of grid
    cells with the flag set, across the whole loaded panel.  Channels without
    a has_* flag carry no presence marker in the arrays, so their coverage is
    left null (not decodable).
    """
    po = panel_data.get("past_observed")
    col_names = panel_data.get("past_observed_cols") or []
    index = {name: i for i, name in enumerate(col_names)}
    channels: dict[str, dict] = {}
    if po is None or po.ndim != 3:
        channels["_note"] = {
            "status": "UNKNOWN",
            "message": "panel lacks a past_observed grid",
        }
        return channels
    for flag, channel in _HAS_FLAG_CHANNELS.items():
        if flag not in index:
            continue
        mask = po[:, :, index[flag]] > 0
        present = int(np.count_nonzero(mask))
        channels[channel] = {
            "requested": True,
            "required": False,
            "loaded_stocks": None,  # cell-level probe, not per-stock
            "coverage": round(float(mask.mean()), 4),
            "errors": 0,
            "status": "OK" if present else "MISSING",
            "cells": int(mask.size),
            "flag": flag,
        }
    return channels


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
        "date_indices": panel_data["date_indices"][:, tslice].copy(),
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


def _weight_hash(model) -> str:
    """Content hash of a model's TRAINED parameters (float32, CPU).

    The version dict's `model_hash` only fingerprints config + architecture
    source — every fold shares it.  This one hashes the actual state_dict so
    an OOS tape row / checkpoint can be tied to the exact weights that
    produced it, and two differently-trained folds get different digests.
    """
    h = hashlib.sha1()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(b"=")
        arr = param.detach().to("cpu", dtype=torch.float32).contiguous().view(-1)
        h.update(arr.numpy().tobytes())
        h.update(b";")
    return h.hexdigest()[:16]


def _augment_sequence(
    pk: np.ndarray,
    po: np.ndarray,
    obs_mask: np.ndarray | None = None,
    noise_std: float = 0.01,
    mask_prob: float = 0.05,
    feat_dropout: float = 0.02,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight time-series augmentation for financial data.

    Three independent augmentations:
    1. Gaussian noise ~ N(0, noise_std) — improves robustness
    2. Time masking — zero out random contiguous segments (simulates missing data)
    3. Feature dropout — zero out random feature dimensions

    All augmentations are conservative (small magnitudes) to avoid
    distorting the financial signal.  Gaussian noise is gated by `obs_mask`
    (True = real observation) so zero-padded history of new listings stays
    exactly zero instead of gaining fake noise that the model would read as
    real data.
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


def _calendar_freeze(data_dir: str) -> dict:
    """§九-4: freeze the calendar CONTENT an experiment consumed.

    A bare manual version string (``CALENDAR_VERSION``) does not pin the
    actual trading days — a holiday-set edit with a forgotten version bump
    would go unnoticed.  Hash the materialized calendar (dates + is_open +
    source + verified_until + status), which is invariant to the
    non-deterministic ``generated_at`` stamp, and record ``verified_until`` +
    ``source``.  The on-disk artifact is the authoritative frame when present
    (data-driven corrections win over code); a malformed/absent artifact falls
    back to the code-derived frame the panel pipeline actually uses.
    """
    try:
        frame = load_calendar(data_dir, "a_shares")
    except Exception:
        frame = None
    if frame is None:
        frame = build_calendar_frame("a_shares")
    verified_until = str(pd.Timestamp(frame["verified_until"].iloc[0]).date())
    source = str(frame["source"].iloc[0])
    canonical = (frame.drop(columns=["generated_at"])
                 if "generated_at" in frame.columns else frame)
    # Deterministic text encoding (not parquet bytes): pandas str() normalizes
    # datetime values regardless of dtype unit (datetime64[s] vs [ms] after a
    # parquet round-trip), so a disk artifact and the code formula hash equal.
    cols = sorted(canonical.columns)
    key = "date"
    canon_sorted = canonical.sort_values(key).reset_index(drop=True)
    lines = []
    for _, row in canon_sorted.iterrows():
        lines.append("|".join(str(row[c]) for c in cols))
    digest = "\n".join(lines).encode("utf-8")
    checksum = hashlib.sha1(digest).hexdigest()[:16]
    return {
        "calendar_artifact_hash": checksum,
        "verified_until": verified_until,
        "calendar_source": source,
    }


def _experiment_version(
    data_dir: str,
    universe_used: list[str],
    prebuilt_dir: str | None,
    static_dim: int,
    past_known_dim: int,
    past_observed_dim: int,
    config: PanelConfig,
    start: str,
    end: str,
    seed: int,
) -> dict:
    """Freeze the data/code/feature versions an experiment consumed.

    Every hash is content-addressed and deterministic — the same commit +
    source files + feature schemas + universe must yield the same digest, so a
    days-old run stays explainable.  `data_manifest_hash` covers the raw source
    files actually fed to feature engineering; `feature_schema_hash` covers the
    feature column set (prebuilt sidecar manifests when available, else the
    panel dims).
    """
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    src = hashlib.sha1()
    feat = hashlib.sha1()
    feat.update(f"S={static_dim}|PK={past_known_dim}|PO={past_observed_dim}|".encode())

    manifest_dir = os.path.join(prebuilt_dir, ".manifests") if prebuilt_dir else None
    for code in sorted(universe_used):
        src.update(code.encode())
        src.update(b":")
        if manifest_dir is not None:
            mp = os.path.join(manifest_dir, f"{code}.json")
            m = None
            if os.path.isfile(mp):
                try:
                    with open(mp, encoding="utf-8") as f:
                        m = json.load(f)
                except Exception:
                    m = None
            if m:
                for name in sorted(m.get("source_files") or {}):
                    src.update(name.encode())
                    src.update(b"=")
                    src.update(str(m["source_files"][name].get("hash")).encode())
                    src.update(b";")
                feat.update(str(m.get("feature_schema_hash", "")).encode())
                feat.update(b";")
                continue
            # No readable manifest → fingerprint the prebuilt feature file.
            p = os.path.join(prebuilt_dir, f"{code}.parquet")
            src.update(b"prebuilt=")
            src.update(str(cache_manifest.file_fingerprint(p)).encode())
            src.update(b";")
            feat.update(str(cache_manifest.schema_hash(p)).encode())
            feat.update(b";")
        else:
            _, source_files = cache_manifest.channels_and_source_files(
                data_dir, code, start, end,
            )
            for name in sorted(source_files):
                src.update(name.encode())
                src.update(b"=")
                src.update(str(source_files[name].get("hash")).encode())
                src.update(b";")

    # Model hash: content-addressed over the architecture
    # source + the PanelConfig hyper-parameters, so an OOS tape row records
    # exactly which model produced it — config change or arch edit flips it.
    model_h = hashlib.sha1()
    model_h.update(repr(config).encode("utf-8"))
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("stoke_ml/models/panel/model.py",
                "stoke_ml/models/panel/vsn.py",
                "stoke_ml/models/panel/xlstm.py",
                "stoke_ml/models/panel/heads.py",
                "stoke_ml/models/panel/config.py",
                "stoke_ml/models/panel/loss.py"):
        fp = cache_manifest.file_fingerprint(os.path.join(_root, rel)) or "absent"
        model_h.update(rel.encode("utf-8"))
        model_h.update(b"=")
        model_h.update(fp.encode("utf-8"))
        model_h.update(b";")

    # §九-4: calendar content freeze — artifact hash + verified_until + source
    # (see _calendar_freeze), alongside the manual version string.
    calendar_freeze = _calendar_freeze(data_dir)
    return {
        "git_commit": cache_manifest.git_head(),
        "data_manifest_hash": src.hexdigest()[:16],
        "calendar_version": TradingCalendar.CALENDAR_VERSION,
        **calendar_freeze,
        "feature_schema_hash": feat.hexdigest()[:16],
        "universe_hash": _sha1("\n".join(sorted(universe_used))),
        "model_hash": model_h.hexdigest()[:16],
        "evaluator_version": EVALUATOR_VERSION,
        "cost_model": f"sleeve per-side txn_cost={config.txn_cost}, top_fraction=0.1",
        "random_seed": seed,
    }


# §十五-1 experiment registry — a durable count of every research run, so the
# DSR deflation has a defensible N (the number of trials iterated across the
# project, not just the strategies inside one report).  Lives at a STABLE path
# (independent of the timestamped outdir) so it accumulates across runs.
_EXPERIMENT_REGISTRY_PATH = os.path.join(
    "reports", "experiments", "experiment_registry.json")


def _load_experiment_registry(path: str) -> list[dict]:
    """Prior experiment entries; [] when missing or corrupt (never aborts a run)."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _append_experiment_registry(path: str, entry: dict) -> None:
    """Atomically append an entry; an existing row for the same outdir is replaced
    (a re-run of the same directory must not double-count the trial)."""
    entries = _load_experiment_registry(path)
    entries = [e for e in entries if e.get("outdir") != entry.get("outdir")]
    entries.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


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


def _predict_outer(model, outer_data, config, device) -> np.ndarray | None:
    """Run the deployed checkpoint over the outer-test panel.

    Uses the same no-sampler DataLoader as evaluate_portfolio, so every
    (stock, window) pair is emitted in index order and the flattened return
    predictions reshape back to (n_stocks, n_windows).  Window d enters at
    panel column seq_len + d — i.e. global column val_start + d of the full
    panel.  Returns None only when the outer panel has no windows.
    """
    val_ds = PanelDataset(outer_data, seq_len=config.seq_len,
                          min_history=config.min_history)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=panel_collate,
        num_workers=0, pin_memory=False,
    )
    model.eval()
    preds_parts = []
    with torch.no_grad():
        for batch in val_loader:
            static, pk, po, *_ = batch
            static = static.to(device)
            pk = pk.to(device)
            po = po.to(device)
            _, pred_ret, _ = model(static, pk, po)
            preds_parts.append(pred_ret.cpu().squeeze(-1))
    if not preds_parts:
        return None
    preds = torch.cat(preds_parts)
    n_stocks = outer_data["static_features"].shape[0]
    return preds.reshape(n_stocks, val_ds.n_windows).numpy()


def _replay_continuous_oos(oos_dir: str, n_trials: int | None = None) -> dict | None:
    """§十四-4: replay ONE continuous long sleeve account across ALL fold tapes.

    Each fold tape is a self-contained grid: (fold stocks, fold signal days,
    fold price days).  A single continuous account needs a UNION grid — one
    stock axis, one price axis — where every cell comes from the fold that
    owned that day.  The folds' signal windows tile contiguously (step ==
    val_len) and each price path extends `horizon` columns past its last signal
    day, so a sleeve entered in fold A liquidates inside fold B's window with
    the SAME real prices (the overlapping price columns carry identical values,
    so overwrite order is irrelevant).  The account's NAV carries across fold
    boundaries: a sleeve entered in fold A stays open and marks to market
    through fold B while fold B's model starts producing its own signals —
    exactly what the old per-fold "NAV restarts at 1 then average Sharpe"
    aggregation could not show.

    Grid layout: the PRICE axis is the sorted union of every fold's price
    dates, and the preds/pool grids are built on the SAME axis — a signal fires
    only on days a fold actually predicted (NaN preds / False pool elsewhere),
    so `_run_sleeve_sim`'s entry gate simply never fires on non-signal days.
    This keeps column d ↔ date monotonic without assuming the signal days
    occupy the first columns, so a fold whose window came out shorter than
    `val_len` (gap days) is handled honestly too.

    `delist_day` is intentionally None: the tape stores prices but not the
    delisting record, so a known-delisted stock degrades to a carried-close
    UNRESOLVED hold instead of the force-sold DELISTED exit the original
    per-fold run booked.  The tape is the persisted source of truth for this
    re-run; the delist force-sell needs the universe record only the live run
    had.

    Returns the account dict (see `_run_sleeve_sim`), the global price dates,
    union stock codes, ledger, and headline metrics; None when no fold tapes
    exist.
    """
    if not os.path.isdir(oos_dir):
        return None
    tapes = sorted(
        os.path.join(oos_dir, f)
        for f in os.listdir(oos_dir)
        if f.startswith("fold_") and f.endswith(".npz"))
    if not tapes:
        return None
    recs = []
    for p in tapes:
        z = np.load(p, allow_pickle=False)

        def _strs(a: np.ndarray) -> list[str]:
            return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

        recs.append({
            "stocks": _strs(z["stocks"]),
            "dates": _strs(z["dates"]),
            "price_dates": _strs(z["price_dates"]),
            "preds": z["preds"],
            "pool": z["pool"],
            "close": z["close_price"],
            "open": z["open_price"],
            "horizon": int(z["horizon"]),
            "cost": float(z["cost"]),
            "top_fraction": float(z["top_fraction"]),
        })
    # fold index grows as val_start walks BACKWARD, so chronological order is
    # the reverse of the lexicographic tape order.
    recs = list(reversed(recs))

    union_stocks: list[str] = []
    for r in recs:
        for s in r["stocks"]:
            if s not in union_stocks:
                union_stocks.append(s)
    row_of = {s: i for i, s in enumerate(union_stocks)}

    global_price_dates = sorted({
        d for r in recs for d in r["price_dates"]
    })  # ISO strings sort chronologically
    pcol_of = {d: i for i, d in enumerate(global_price_dates)}

    n_glob = len(union_stocks)
    wp_glob = len(global_price_dates)
    close_glob = np.full((n_glob, wp_glob), np.nan, dtype=np.float64)
    open_glob = np.full((n_glob, wp_glob), np.nan, dtype=np.float64)
    preds_glob = np.full((n_glob, wp_glob), np.nan, dtype=np.float64)
    pool_glob = np.zeros((n_glob, wp_glob), dtype=bool)
    for r in recs:
        rows = np.array([row_of[s] for s in r["stocks"]], dtype=int)
        pcols = np.array([pcol_of[d] for d in r["price_dates"]], dtype=int)
        close_glob[np.ix_(rows, pcols)] = r["close"]
        open_glob[np.ix_(rows, pcols)] = r["open"]
        # Signal columns sit at their true price positions; other columns keep
        # NaN preds so no entry fires there.
        scols = np.array([pcol_of[d] for d in r["dates"]], dtype=int)
        preds_glob[np.ix_(rows, scols)] = r["preds"]
        pool_glob[np.ix_(rows, scols)] = r["pool"]

    acc = _run_sleeve_sim(
        preds_glob, close_glob, open_glob, pool_glob,
        horizon=recs[0]["horizon"],
        top_fraction=recs[0]["top_fraction"],
        cost=recs[0]["cost"],
        mode="long",
        return_ledger=True,
    )

    daily = np.asarray(acc["daily"], dtype=np.float64)
    t = torch.tensor(daily, dtype=torch.float32)
    final_nav = acc["final_nav"]
    n_days = int(daily.size)
    cagr = (final_nav ** (252.0 / n_days) - 1.0
            if final_nav is not None and final_nav > 0 and n_days > 0
            else None)
    metrics = {
        "sharpe": compute_sharpe(t, horizon=1),
        "maxdd": compute_max_drawdown(compute_equity_curve(t)),
        "cagr": cagr,
        "final_nav": final_nav,
        "n_days": n_days,
        "n_stocks": n_glob,
    }
    # §十五-1: the continuous account's Sharpe read against data-snooping.  The
    # trial-Sharpe dispersion comes from the block-bootstrap proxy (the replay
    # only runs the long sleeve, so there are no sibling-trial Sharpes here);
    # `n_trials` is the project-wide registry count from the training script.
    metrics["psr"] = compute_psr(daily, 0.0, 1)
    metrics["dsr"] = compute_deflated_sharpe(
        daily, n_trials, None, 1) if (n_trials is not None and n_trials >= 2) else float("nan")
    metrics["dsr_n_trials"] = int(n_trials) if n_trials is not None else None

    ledger = acc.get("ledger") or []
    ldf = None
    if ledger:
        ldf = pd.DataFrame(ledger)
        si = ldf["stock"].to_numpy(dtype=int)
        di = ldf["entry_day"].to_numpy(dtype=int)
        ldf["entry_date"] = [global_price_dates[c] for c in di]
        ldf["stock_code"] = [union_stocks[i] for i in si]
        ldf = ldf.sort_values(["entry_date", "stock", "mode"]).reset_index(drop=True)

    return {
        "price_dates": global_price_dates,
        "stocks": union_stocks,
        "account": acc,
        "metrics": metrics,
        "ledger": ldf,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Train VSN+xLSTM panel model")
    parser.add_argument("--stocks", type=int, default=500,
                        help="Universe size / cap (default: 500; with "
                             "--universe first: first N sorted; random/stratified: "
                             "N sampled; csi*: N cap)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"],
                        help="Stock universe selection (default: random; "
                             "csi* = index constituents, PIT ever-held union)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for universe sampling and data "
                             "augmentation (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Experiment artifact dir (default: "
                             "reports/experiments/<timestamp>)")
    parser.add_argument("--stock-list", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=3,
                        help="Limit number of walk-forward folds (default: 3)")
    parser.add_argument("--lockbox-months", type=int, default=12,
                        help="Reserve the last N months as an untouched lockbox "
                             "— no fold trains on or "
                             "evaluates it; kept for a single final run once "
                             "the design freezes (default: 12)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in days (1/5/20)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Model hidden dimension (default: 128)")
    parser.add_argument("--xlstm-blocks", type=int, default=2,
                        help="Number of xLSTM blocks (default: 2)")
    parser.add_argument("--rank-weight", type=float, default=0.1,
                        help="Ranking loss weight (0=disable, default: 0.1)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable time-series data augmentation")
    parser.add_argument("--log-gradient-flow", action="store_true",
                        help="Log per-parameter-group gradient norms each epoch "
                             "(after optimizer.step, before zero_grad)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip auxiliary data loading (faster startup)")
    parser.add_argument("--require-aux-channels", type=str, default="",
                        help="Comma-separated aux channels that must have "
                             "loaded_stocks>0; experiment "
                             "FAILS otherwise. Default: none required")
    parser.add_argument("--prebuilt", type=str, default=None,
                        help="Load panel-mode prebuilt features from this dir "
                             "(built via build_features.py --panel-mode). "
                             "Skips aux data loading and live feature "
                             "engineering — the panel is built from the "
                             "prebuilt parquets")
    parser.add_argument("--require-feature-manifest",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Require every prebuilt feature parquet to carry a "
                             "matching sidecar manifest (missing / stale / "
                             "schema-drift / different-git-commit FAILS the run "
                             "instead of warning). Default: on. Use "
                             "--no-require-feature-manifest for legacy prebuilt "
                             "dirs built without manifests (§十一-1)")
    parser.add_argument("--no-require-quality-gate", action="store_true",
                        help="Skip the required quality-gate report check "
                             "(dev smoke only; §六-2 wants a matching report "
                             "before any real training run)")
    parser.add_argument("--quality-gate-report", type=str, default=None,
                        help="Path to the quality-gate report to verify "
                             "(default: <repo>/reports/data_quality_gate.json)")
    parser.add_argument("--minute", action="store_true",
                        help="Use minute-frequency K-line data instead of daily")
    parser.add_argument("--minute-frequency", type=str, default="60",
                        choices=["5", "15", "30", "60"],
                        help="Bar frequency for minute mode (default: 60)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60 daily, 64 minute)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    if not args.no_require_quality_gate:
        _report_path = args.quality_gate_report or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports", "data_quality_gate.json",
        )
        _require_quality_gate(data_dir, args.prebuilt, _report_path)
        logger.info("Quality-gate report verified: %s", _report_path)

    if args.stock_list:
        stock_list = [c.strip() for c in args.stock_list.split(",")]
        universe_desc = f"stock-list (explicit, n={len(stock_list)})"
    elif args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        stock_list = MinuteStorage(data_dir).list_stocks(args.minute_frequency)
        if args.stocks:
            stock_list = stock_list[:args.stocks]
        universe_desc = f"minute-mode (n={len(stock_list)})"
    else:
        all_stocks = _discover_stocks(data_dir, None)
        stock_list, universe_desc = _resolve_universe(
            all_stocks, args.universe, args.stocks, args.seed, data_dir,
        )

    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)

    universe_resolved = list(stock_list)

    # Stock-level quality is judged per-fold, point-in-time, inside the fold
    # loop (_fold_eligible_stocks uses only columns before train_end) — no
    # full-history ejection up front.  Row-level badness is
    # handled as masks in the pipeline, not stock ejection.
    universe_used = list(stock_list)

    logger.info("Universe: %s", universe_desc)
    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    if args.minute:
        from stoke_ml.data.minute_storage import MinuteStorage
        ms = MinuteStorage(data_dir)
        frames = []
        for code in stock_list:
            df = ms.load(code, args.start, args.end, args.minute_frequency)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date
                df["stock_code"] = code
                frames.append(df)
        if not frames:
            logger.error("No minute data loaded for any stock — run download_minute.py first")
            sys.exit(1)
        logger.info("Minute mode: %d stocks @ %s-min, %d available in storage",
                    len(frames), args.minute_frequency,
                    len(ms.list_stocks(args.minute_frequency)))
    else:
        from stoke_ml.data.storage import DataStorage
        ds = DataStorage(data_dir)
        frames = []
        for code in stock_list:
            df = ds.load_daily(code, args.start, args.end,
                               require_valid_manifest=True)
            if df is not None and not df.empty:
                df["stock_code"] = code
                frames.append(df)
        if not frames:
            logger.error("No data loaded for any stock")
            sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    logger.info("Panel shape: %s", panel.shape)
    # Panel stock order is sorted-unique (build_panel_features sorts codes);
    # derive the code list from the panel itself so OOS artifacts stay aligned
    # with the panel arrays even if a stock's data was dropped as empty.
    panel_stocks = sorted(panel["stock_code"].unique())

    # Load auxiliary data (unless --no-aux or --prebuilt)
    required_set = {c.strip() for c in (args.require_aux_channels or "").split(",") if c.strip()}
    aux_data = None
    channel_manifest: dict = {}
    if not args.no_aux and not args.prebuilt:
        logger.info("Loading auxiliary data...")
        t_aux = time.time()
        aux_data, channel_manifest = load_aux_data(
            stock_list, data_dir, args.start, args.end,
            required_channels=required_set,
        )
        logger.info("Aux data loaded in %.1fs", time.time() - t_aux)

    # Build features
    seq_len = args.seq_len or (64 if args.minute else 60)
    fp = FeaturePipeline(
        seq_len=seq_len,
        minute_mode=args.minute,
        use_sentiment=True, use_announcements=True,
        use_guba=True, use_comment=True, use_margin=True,
        use_northbound=True, use_dragon_tiger=True,
        use_fundamental=True, use_etf_flow=True,
        use_capital_flow=True, use_block_trade=True,
        use_shareholder=True, use_lockup=True, use_dividend=True,
        use_valuation=True,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(
        panel, aux_data=aux_data, horizon=args.horizon, prebuilt_dir=args.prebuilt,
        require_feature_manifest=args.require_feature_manifest,
    )

    if args.prebuilt:
        # Live per-channel loading is skipped in prebuilt mode; probe the panel's
        # has_* flags instead so the experiment still records what actually got
        # in.
        channel_manifest = _prebuilt_channel_coverage(panel_data)

    # Required-channel gate: a required channel with ZERO
    # coverage aborts the experiment instead of silently training on air.
    missing_required = sorted(
        ch for ch in required_set
        if channel_manifest.get(ch, {}).get("loaded_stocks", 0) == 0
        and channel_manifest.get(ch, {}).get("coverage", 0.0) == 0
    )
    if missing_required:
        logger.error("Required aux channels have ZERO coverage: %s — aborting",
                     ", ".join(missing_required))
        sys.exit(1)
    for ch in sorted(required_set):
        if ch not in channel_manifest:
            logger.warning("Required aux channel '%s' has no coverage probe in "
                           "this mode (prebuilt panel without has_* flag) — "
                           "coverage cannot be verified", ch)
    if channel_manifest:
        summary_bits = ", ".join(
            f"{k}={v.get('status')}({v.get('coverage')})"
            for k, v in sorted(channel_manifest.items()) if not k.startswith("_")
        )
        logger.info("Channel coverage manifest: %s", summary_bits)

    # Union trading calendar (datetime64[ns]) — fold boundaries in index space
    # map back to real dates for the summary.
    global_dates = panel_data.get("global_dates")

    # §九-3: strict formal run — the panel must stay within the VERIFIED
    # calendar window (see _check_verified_until_scope).  Exploratory runs can
    # pass --no-require-quality-gate to opt out.
    refusal = _check_verified_until_scope(
        global_dates, enforce=not args.no_require_quality_gate)
    if refusal:
        raise SystemExit(refusal)

    # Per-stock delisting day in global panel-column space, from the universe
    # delisting records (§七-1).  The sleeve simulator force-sells a
    # known-delisted position at the delisting close; without this, a stock
    # whose data simply ends would stay UNRESOLVED even when the universe says
    # it was really delisted.  Missing universe parquets → empty status → all
    # -1 (no force-sell), so this never crashes on a data-dir without records.
    universe_status = load_universe_status(data_dir)
    delist_global = delist_global_index(
        global_dates, universe_status, panel_stocks,
    )

    # §七-3 universe gates, both in the panel's (N_stocks, T) grid space:
    #   nd_mask  — 未退市: blocks ENTRY from a known delisting column on.
    #   mem_mask — per-day index membership (in_date <= date < out_date) for
    #              csi300/csi500/csi800; None for other universes so the fold
    #              gates become a no-op for the "all A" / stratified studies.
    nd_mask = not_delisted_mask(global_dates, panel_stocks, universe_status)
    universe_index_codes = {
        "csi300": {"000300"},
        "csi500": {"000905"},
        "csi800": {"000300", "000905"},
    }.get(args.universe, set())
    mem_mask = None
    if universe_index_codes:
        membership_df = load_index_membership(data_dir, sorted(universe_index_codes))
        if membership_df.empty:
            logger.warning(
                "universe=%s: no membership.parquet intervals — per-day "
                "index-membership gate is a no-op (candidate pool keeps the "
                "full historical-member union)", args.universe,
            )
        else:
            mem_mask = index_membership_mask(global_dates, panel_stocks, membership_df)

    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    # Static features are (N, T, D) PIT — feature dim is axis 2.
    static_dim = panel_data["static_features"].shape[2]
    dims = f"S={static_dim} " \
           f"PK={panel_data['past_known'].shape[2]} " \
           f"PO={panel_data['past_observed'].shape[2]}"
    logger.info("Panel data: %d stocks × %d timesteps  dims: %s  horizon=%d",
                n_stocks, n_timesteps, dims, args.horizon)

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=static_dim,
        past_known_dim=panel_data["past_known"].shape[2],
        past_observed_dim=panel_data["past_observed"].shape[2],
        hidden_dim=args.hidden_dim,
        xlstm_num_blocks=args.xlstm_blocks,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        compile_model=not args.no_compile,
        num_workers=0,
        horizon=args.horizon,
        rank_loss_weight=args.rank_weight,
        seed=args.seed,
        log_gradient_flow=args.log_gradient_flow,
    )
    logger.info("VSN+xLSTM config: hidden=%d blocks=%d heads=%d batch=%d lr=%.1e rank_w=%.2f",
                config.hidden_dim, config.xlstm_num_blocks, config.xlstm_num_heads,
                config.batch_size, config.learning_rate, config.rank_loss_weight)

    # Freeze the data/code/feature versions up front so the run
    # stays explainable even if every fold fails.  Written to version.json
    # unconditionally; also embedded in summary.json when folds complete.
    version_info = _experiment_version(
        data_dir, universe_used, args.prebuilt,
        static_dim,
        panel_data["past_known"].shape[2],
        panel_data["past_observed"].shape[2],
        config, args.start, args.end, args.seed,
    )
    logger.info(
        "Version freeze: commit=%s data=%s feat=%s uni=%s cal=%s eval=%s",
        version_info["git_commit"][:10], version_info["data_manifest_hash"],
        version_info["feature_schema_hash"], version_info["universe_hash"],
        version_info["calendar_version"], version_info["evaluator_version"],
    )

    # Purged walk-forward splits
    if args.minute:
        val_len = 250      # ~62 trading days
    else:
        val_len = 126      # ~6 months daily
    # OOS folds are NON-OVERLAPPING — step == val_len, so
    # adjacent folds evaluate disjoint test windows.  The old step < val_len
    # made every fold share test days with its neighbours, inflating fold
    # count and letting mean±std masquerade as independent dispersion.
    step = val_len
    purge = config.seq_len
    all_sharpes = []
    fold_histories = []

    # Reserve the last N months as an untouched lockbox.
    # No fold trains on or evaluates it; it is kept for a single final run
    # once the design freezes.  Daily ≈ 21 bars/month; minute mode scales by
    # bars per day so lockbox_months spans the same wall-clock time.
    bars_per_day = {"5": 48, "15": 16, "30": 8, "60": 4}[args.minute_frequency]
    lockbox_len = int(args.lockbox_months * 21 * (bars_per_day if args.minute else 1))
    lockbox_start = max(0, n_timesteps - lockbox_len)
    if lockbox_start <= 0:
        logger.error("Lockbox (%d steps) leaves no trainable panel "
                     "(n_timesteps=%d) — reduce --lockbox-months",
                     lockbox_len, n_timesteps)
        sys.exit(1)
    logger.info("Lockbox [%d:%d] %d steps (%.1f months) — %s .. %s",
                lockbox_start, n_timesteps, lockbox_len, args.lockbox_months,
                _fmt_date(global_dates, lockbox_start),
                _fmt_date(global_dates, n_timesteps - 1))

    outdir = args.outdir or os.path.join(
        "reports", "experiments", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    oos_dir = os.path.join(outdir, "oos_preds")
    os.makedirs(oos_dir, exist_ok=True)
    # §十五-1: this run is one more trial in the project-wide experiment
    # registry — the DSR deflation N counts ALL research trials iterated so
    # far, not just the strategies inside a single report.
    experiment_registry = _load_experiment_registry(_EXPERIMENT_REGISTRY_PATH)
    n_trials = len(experiment_registry) + 1
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
            # PanelDataset needs at least seq_len+1 rows for one window
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
        # membership) for csi300/csi500/csi800.  inner_train is left ungated —
        # the model still learns from all data; only what gets RANKED as a
        # tradable candidate is restricted.  Applied in this fold's row/column
        # space (rows = surviving original stock rows, cols = the slice's
        # global columns) so _candidate_pool picks the gates up automatically.
        rows = np.where(fold_eligible)[0]
        for name, tslice, dd in (
            ("inner_val", inner_val_slice, inner_val_data),
            ("outer_test", outer_test_slice, outer_test_data),
        ):
            cols = np.arange(tslice.start, tslice.stop)
            gate = nd_mask[np.ix_(rows, cols)]
            if mem_mask is not None:
                gate = gate & mem_mask[np.ix_(rows, cols)]
            dd["decision_eligible_mask"] &= gate

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

        # Time-series data augmentation on the inner-training data.
        # Each stock's sequence gets independent noise/masking/dropout —
        # Gaussian noise gated by observation_mask so zero-padded history
        # (new listings) stays exactly zero instead of gaining fake noise.
        if not args.no_augment:
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
            "model_source_hash": version_info["model_hash"],
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
        # Delist-day index in the fold's simulation column space: sim column d
        # ↔ global column val_start+d (evaluate_portfolio slices prices at
        # seq_len within the outer_test window that starts at val_context_start
        # = val_start - seq_len), so subtract val_start.  Clamp values outside
        # [0, Wp) to -1 (never fires within this window).
        dd_global = delist_global[fold_eligible]
        Wp = outer_test_data["close_price"].shape[1] - config.seq_len
        delist_day = np.where(
            (dd_global >= val_start) & (dd_global < val_start + Wp),
            dd_global - val_start, -1,
        )
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
                # A tape row must identify the data + model
                # that produced it, so every return number is traceable.
                data_version=version_info["data_manifest_hash"],
                model_hash=version_info["model_hash"],
                # Trained-parameter hash — the fold's tape maps
                # to the exact weights in fold_XXX_model.pt (config+source
                # model_hash is shared by all folds, this one is not).
                weight_hash=weight_hash,
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

    # Combined daily OOS series: one row per (stock, entry
    # day) across all non-overlapping folds — the input to the sleeve-account
    # backtest, kept separate from fold-level aggregates.
    if oos_preds_all:
        oos_series = pd.DataFrame({
            "entry_date": oos_dates_all,
            "stock_code": oos_stocks_all,
            "pred": np.concatenate(oos_preds_all),
            # The exact select pool the sleeve account ranked over (decision &
            # history) — a tape must expose the candidate set it was built
            # from, not only the selected fills.
            "candidate_eligible": np.concatenate(oos_pool_all),
            # Provenance: data + model versions so every tape
            # row is traceable to the exact experiment it was produced by.
            "data_version": version_info["data_manifest_hash"],
            "model_hash": version_info["model_hash"],
            # fold + trained-parameter hash per row so the tape
            # maps to the exact weights (fold_XXX_model.pt) that produced it.
            "fold": oos_fold_all,
            "weight_hash": oos_weight_hash_all,
        })
        # §十四-3: folds are walked most-recent-first, so the concatenated rows
        # are reverse-chronological chunk by chunk.  Sort before persisting so
        # the tape is date-ordered regardless of fold iteration order — a naive
        # consumer must not need to re-sort to feed the series chronologically.
        oos_series = oos_series.sort_values(
            ["entry_date", "stock_code"]).reset_index(drop=True)
        oos_series_path = os.path.join(outdir, "oos_series.parquet")
        oos_series.to_parquet(oos_series_path)
        logger.info("OOS series: %d rows -> %s", len(oos_series), oos_series_path)

    # Combined per-position ledger across all folds — the
    # single file a consumer reads to reproduce every fill of the backtest.
    # Same reverse-chunk problem as the series: sort by entry date then fold /
    # entry_day so the combined tape is chronological.
    if oos_ledgers:
        combined_ledger = pd.concat(oos_ledgers, ignore_index=True)
        combined_ledger = combined_ledger.sort_values(
            ["entry_date", "fold", "entry_day", "stock", "mode"]
        ).reset_index(drop=True)
        oos_ledger_path = os.path.join(outdir, "oos_ledger.parquet")
        combined_ledger.to_parquet(oos_ledger_path)
        logger.info("Combined OOS ledger: %d rows -> %s",
                    len(combined_ledger), oos_ledger_path)

    # §十四-4: ONE continuous long sleeve account replayed across ALL fold
    # tapes.  Each fold restarts NAV at 1 and is aggregated by mean Sharpe —
    # that is a set of disjoint OOS signal windows, not a continuous strategy.
    # This replay keeps a single account whose NAV carries over fold
    # boundaries (the previous fold's sleeves keep settling while the next
    # fold's model signals), and the FINAL Sharpe/MDD/CAGR come from THIS
    # account only.
    cont = _replay_continuous_oos(oos_dir, n_trials=n_trials) if oos_preds_all else None
    if cont is not None:
        daily = np.asarray(cont["account"]["daily"], dtype=np.float64)
        # The account starts at NAV 1.0 on the close BEFORE day 0, so the NAV
        # after day c's close is the cumulative product through c (one row per
        # price date — final entry == final_nav by the simulator's identity).
        nav = (1.0 + daily).cumprod()
        pd.DataFrame({
            "price_date": cont["price_dates"],
            "nav": nav,
            "daily_return": daily,
        }).to_parquet(os.path.join(outdir, "oos_continuous.parquet"))
        if cont["ledger"] is not None:
            cont["ledger"].to_parquet(
                os.path.join(outdir, "oos_continuous_ledger.parquet"))
        logger.info(
            "Continuous OOS account: %d days across %d stocks | "
            "Sharpe=%.2f MaxDD=%.2f CAGR=%.2f final_nav=%.3f",
            len(daily), len(cont["stocks"]),
            cont["metrics"]["sharpe"], cont["metrics"]["maxdd"],
            cont["metrics"]["cagr"] if cont["metrics"]["cagr"] is not None
            else float("nan"),
            cont["metrics"]["final_nav"] if cont["metrics"]["final_nav"]
            is not None else float("nan"),
        )

    summary_data = None
    if all_sharpes:
        logger.info("=== %d-Fold Summary ===", len(all_sharpes))
        logger.info("LS_Sharpe mean: %.2f ± %.2f", np.mean(all_sharpes), np.std(all_sharpes))
        # IC comes from the outer-test evaluation of the exact deployed
        # checkpoint (outer_metrics) — never an in-loop proxy.
        all_ics = [
            f["outer_metrics"].get("ic_mean", float("nan"))
            for f in fold_histories if f.get("outer_metrics")
        ]
        all_ics = [x for x in all_ics if not np.isnan(x)]
        if all_ics:
            logger.info("IC mean: %.4f ± %.4f", np.mean(all_ics), np.std(all_ics))
        summary_data = {
            # Freeze the data/code/feature versions so the run
            # stays explainable days later (same info also in version.json).
            "version": version_info,
            # §十五-1: how many research trials (incl. this one) the DSR
            # multiplicity was computed against.
            "n_trials": n_trials,
            "n_folds": len(all_sharpes),
            "ls_sharpe_mean": float(np.mean(all_sharpes)),
            "ls_sharpe_std": float(np.std(all_sharpes)),
            "ic_mean": float(np.mean(all_ics)) if all_ics else None,
            "ic_std": float(np.std(all_ics)) if all_ics else None,
            "universe": universe_desc,
            # Non-overlapping folds (step == val_len) — each
            # fold's test days are disjoint, so mean±std is the dispersion of
            # disjoint OOS return windows.
            "folds_overlap": False,
            "fold_note": (
                "disjoint OOS return windows (step == val_len); per-fold metrics "
                "come from separate trainings, not repeated experiments on one model"
            ),
            "lockbox": {
                "months": args.lockbox_months,
                "start": _fmt_date(global_dates, lockbox_start),
                "end": _fmt_date(global_dates, n_timesteps - 1),
                "n_steps": lockbox_len,
                "note": "Reserved for a single final run once the design "
                        "freezes — no fold trains on or evaluates it.  The "
                        "horizon steps before it are a settlement buffer so "
                        "the last fold's exits stop before the lockbox opens.",
            },
            "oos_series": "oos_series.parquet",
            # The per-position fill ledger written above.
            "oos_ledger": "oos_ledger.parquet" if oos_ledgers else None,
            # §十四-4: headline comes from ONE continuous long sleeve account
            # replayed across all fold tapes — not the mean of fold-restart
            # NAVs.  Sharpe/MDD/CAGR are that account's, annualized at 252.
            "oos_continuous": (
                {
                    "file": "oos_continuous.parquet",
                    "ledger": ("oos_continuous_ledger.parquet"
                               if cont["ledger"] is not None else None),
                    "sharpe": cont["metrics"]["sharpe"],
                    "maxdd": cont["metrics"]["maxdd"],
                    "cagr": cont["metrics"]["cagr"],
                    "final_nav": cont["metrics"]["final_nav"],
                    "n_days": cont["metrics"]["n_days"],
                    "n_stocks": cont["metrics"]["n_stocks"],
                    # §十五-1: the continuous Sharpe read against data-snooping
                    # (PSR vs zero; DSR vs the expected max of n_trials).
                    "psr": cont["metrics"]["psr"],
                    "dsr": cont["metrics"]["dsr"],
                    "dsr_n_trials": cont["metrics"]["dsr_n_trials"],
                    "note": "One continuous long sleeve account replayed across "
                            "all fold tapes; NAV carries over fold boundaries.  "
                            "Final Sharpe/MDD/CAGR come from this account only, "
                            "not a mean of fold-restart NAVs.",
                }
                if cont is not None else None),
            "folds": [],
        }
        for i, f in enumerate(fold_histories):
            m = f["outer_metrics"]
            summary_data["folds"].append({
                "fold": i + 1,
                "best_epoch": f["best_epoch"],
                "eval_epoch": f["best_epoch"],
                "inner_eval_epoch": f.get("inner_eval_epoch"),
                "inner_eval_ls_sharpe": f.get("inner_eval_ls_sharpe"),
                "inner_eval_ic": f.get("inner_eval_ic"),
                "weight_hash": f.get("weight_hash"),
                "model_path": f.get("model_path"),
                "train_start": f.get("train_start"),
                "train_end": f.get("train_end"),
                "inner_val_start": f.get("inner_val_start"),
                "inner_val_end": f.get("inner_val_end"),
                "context_start": f.get("context_start"),
                "signal_start": f.get("signal_start"),
                "entry_start": f.get("entry_start"),
                "entry_end": f.get("entry_end"),
                "exit_end": f.get("exit_end"),
                "test_start": f.get("test_start"),
                "test_end": f.get("test_end"),
                "ls_sharpe": m.get("ls_sharpe"),
                "ic_mean": m.get("ic_mean"),
                "ic_ir": m.get("ic_ir"),
                "long_sharpe": m.get("long_sharpe"),
                "q5mq1_ret": m.get("q5mq1_ret"),
                "eligible_ew_sharpe": m.get("eligible_ew_sharpe"),
                "selected_universe_ew_sharpe": m.get("selected_universe_ew_sharpe"),
            })
    else:
        logger.warning("No valid folds completed")

    _save_artifacts(
        outdir, args, universe_resolved, universe_used, universe_desc, summary_data,
        channel_manifest=channel_manifest,
        version=version_info,
    )

    # §十五-1: register this run in the project-wide experiment ledger so the
    # NEXT run's DSR multiplicity counts it.  Written even when no fold
    # completed — an aborted / short run is still a research trial.
    registry_entry = {
        "outdir": outdir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": version_info.get("git_commit"),
        "data_manifest_hash": version_info.get("data_manifest_hash"),
        "feature_schema_hash": version_info.get("feature_schema_hash"),
        "model_hash": version_info.get("model_hash"),
        "universe_hash": version_info.get("universe_hash"),
        "lockbox": {
            "months": args.lockbox_months,
            "start": _fmt_date(global_dates, lockbox_start),
            "end": _fmt_date(global_dates, n_timesteps - 1),
        },
        "n_folds": len(all_sharpes) if all_sharpes else 0,
        "ls_sharpe_mean": float(np.mean(all_sharpes)) if all_sharpes else None,
        "oos_continuous_sharpe": (
            cont["metrics"]["sharpe"] if cont is not None else None),
        "psr": cont["metrics"]["psr"] if cont is not None else None,
        "dsr": cont["metrics"]["dsr"] if cont is not None else None,
        "dsr_n_trials": n_trials,
    }
    _append_experiment_registry(_EXPERIMENT_REGISTRY_PATH, registry_entry)
    logger.info(
        "Experiment registry: %d prior trials -> this is trial #%d (%s)",
        len(experiment_registry), n_trials, outdir,
    )


if __name__ == "__main__":
    main()
