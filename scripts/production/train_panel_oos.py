"""Continuous-OOS replay of fold tapes (§二十一).

Extracted from ``scripts.production.train_panel`` — ``_replay_continuous_oos``
rebuilds ONE continuous long sleeve account across ALL fold tapes, with the
tape-to-checkpoint weight verification (``_verify_tape_weight_hash``) that
guards it and the content hashes it re-derives.  ``train_panel`` re-exports
these names for backward compatibility.
"""
import hashlib
import logging
import os
import re

import numpy as np
import pandas as pd
import torch

from stoke_ml.models.panel.evaluate import (
    _run_sleeve_sim,
    compute_sharpe,
    compute_max_drawdown,
    compute_equity_curve,
)
from stoke_ml.models.panel.inference import (
    compute_deflated_sharpe,
    compute_psr,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


def _file_sha256(path: str) -> str:
    """Content SHA-256 of a file's bytes (full-length digest).

    The baseline tapes' ``weight_hash`` is the SHA-256 of the retained .pkl
    (train_baselines_panel._file_sha256); the replay re-derives it from the same
    file to verify a tape's weight fingerprint (§十八-C1).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_dict_hash(state_dict) -> str:
    """Content hash of a state_dict's tensors (float32, CPU, sorted by name).

    Independent of a live model instance — a persisted checkpoint's state_dict
    (or a freshly-trained model's) hashes to the same value, so the offline
    replay (§十八-C1) can re-derive the trained-parameter hash straight from a
    saved ``fold_XXX_model.pt`` without rebuilding the architecture.
    """
    h = hashlib.sha1()
    for name in sorted(state_dict.keys()):
        h.update(name.encode("utf-8"))
        h.update(b"=")
        param = state_dict[name]
        arr = param.detach().to("cpu", dtype=torch.float32).contiguous().view(-1)
        h.update(arr.numpy().tobytes())
        h.update(b";")
    return h.hexdigest()[:16]


def _verify_tape_weight_hash(rec: dict, oos_dir: str,
                             model_name: str | None) -> None:
    """§十八-C1: tie a tape's recorded weight_hash to its retained checkpoint.

    ``rec["path"]`` is the tape file (fold_NNN.npz for the deep model,
    fold_NNN_<model>.npz for a baseline).  For the deep model the counterpart is
    fold_NNN_model.pt: the recorded weight_hash is re-derived from the saved
    state_dict (via ``_state_dict_hash``) and compared to the tape's value, and
    the checkpoint's own stored weight_hash is checked too.  For a baseline the
    counterpart is fold_NNN_<model>.pkl whose byte SHA-256 is the tape's
    weight_hash (train_baselines_panel._file_sha256).  Any mismatch fails the
    replay; a tape that predates weight_hash, or whose checkpoint is absent, is
    flagged with a warning (legacy tolerance) rather than silently trusted.
    """
    wh = rec.get("weight_hash")
    if wh is None:
        return  # legacy tape predates the weight hash — tolerated
    stem = os.path.basename(rec["path"])
    m = re.match(r"fold_(\d+)", stem)
    if not m:
        return  # unexpected tape name — nothing to bind
    fold_n = int(m.group(1))
    if model_name is not None:
        ckpt = os.path.join(oos_dir, f"fold_{fold_n:03d}_{model_name}.pkl")
        if not os.path.isfile(ckpt):
            logger.warning(
                "§十八-C1: tape %s records weight_hash but its retained pickle "
                "%s is absent — weight-verification skipped", rec["path"], ckpt)
            return
        got = _file_sha256(ckpt)
        if got != wh:
            raise ValueError(
                "§十八-C1: fold tape weight_hash does not match its retained "
                f"pickle ({ckpt}): tape={wh!r} file={got[:16]!r} — the tape "
                "does not describe the fitted weights on disk")
        return
    ckpt = os.path.join(oos_dir, f"fold_{fold_n:03d}_model.pt")
    if not os.path.isfile(ckpt):
        logger.warning(
            "§十八-C1: tape %s records weight_hash but its checkpoint %s is "
            "absent — weight-verification skipped", rec["path"], ckpt)
        return
    try:
        ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "§十八-C1: could not load checkpoint "
            f"{ckpt} to verify tape weight_hash: {exc}") from exc
    recorded = str(ckpt_obj.get("weight_hash", ""))
    if recorded and recorded != wh:
        raise ValueError(
            "§十八-C1: fold tape weight_hash does not match its checkpoint's "
            f"recorded weight_hash ({ckpt}): tape={wh!r} ckpt={recorded!r}")
    sd = ckpt_obj.get("state_dict")
    if sd is not None:
        recomputed = _state_dict_hash(sd)
        if recomputed != wh:
            raise ValueError(
                "§十八-C1: fold tape weight_hash does not match the state_dict "
                f"of its checkpoint ({ckpt}): tape={wh!r} "
                f"recomputed={recomputed!r}")


def _replay_continuous_oos(
    oos_dir: str,
    n_trials: int | None = None,
    trial_sharpes: list[float] | None = None,
    formal: bool = False,
    model_name: str | None = None,
) -> dict | None:
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

    `delist_day` (per-stock, in this fold's sim column space) is persisted in
    the tape (§P0-6), so this replay maps it onto the union price axis and
    passes it to `_run_sleeve_sim`: a known-delisted stock is force-sold at the
    delisting close exactly as the live per-fold run booked — no silent
    degradation to a carried-close UNRESOLVED hold.

    §十二.2: building the union grid asserts every overlapping price cell is
    equal within tolerance across the folds that own it (identical real prices,
    same data version) and that all tapes share the same data_version /
    universe_status_hash / membership_hash / calendar_hash — a silent
    overwrite, a delist-file/membership edit, or a calendar-content change
    between runs fails the replay instead of passing.  §十五-1 additionally
    requires the strategy POLICY (horizon / cost / top_fraction /
    evaluator_version / price_convention / exit_policy / strategy_mode) to be
    identical across folds, so a mixed-policy directory is never explained by
    the first tape's policy.  §十六 further requires the MODEL IDENTITY
    (model_source_hash / model_config_hash / feature_schema_hash) to be
    identical across folds in formal mode, so a mid-run architecture switch
    (e.g. "first two folds VSN+xLSTM, last three plain LSTM") or a config /
    feature-schema edit is refused instead of silently blended; only
    `weight_hash` (the trained parameters) is allowed to differ per fold.

    `formal=True` (§十五-2) is the production headline mode: it refuses ANY
    tape that lacks required metadata (universe/delist/calendar/version/policy
    keys) instead of replaying it under legacy semantics — the final continuous
    account must never silently blend a legacy tape.  Non-formal replay keeps
    the legacy tolerance for unit tests and migrations.

    `model_name` (§十五-3) selects ONE baseline model's tapes
    (`fold_000_lgbm.npz`, ...) for its own continuous account.  The default
    scan matches only digit-stem tapes (`fold_000.npz`, the deep model), so a
    baseline tape sharing an oos_dir is never blended into the deep replay.

    Returns the account dict (see `_run_sleeve_sim`), the global price dates,
    union stock codes, ledger, and headline metrics; None when no fold tapes
    match.
    """
    if not os.path.isdir(oos_dir):
        return None
    if model_name is not None:
        pat = re.compile(rf"fold_\d+_{re.escape(model_name)}\.npz$")
    else:
        # Deep-model tapes are the bare `fold_000.npz` form; baseline tapes
        # carry a `_modelname` suffix and must not join the deep replay.
        pat = re.compile(r"fold_\d+\.npz$")
    tapes = sorted(
        os.path.join(oos_dir, f)
        for f in os.listdir(oos_dir)
        if pat.match(f))
    if not tapes:
        return None
    recs = []
    for p in tapes:
        z = np.load(p, allow_pickle=False)

        def _strs(a: np.ndarray) -> list[str]:
            return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

        recs.append({
            "path": p,
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
            # §P0-6 tape keys: optional so legacy/test tapes (written without
            # the delist record) replay with their historical no-force-sell
            # semantics instead of crashing on a missing key.
            "delist_day": z["delist_day"] if "delist_day" in z.files else None,
            "data_version": (str(z["data_version"])
                             if "data_version" in z.files else None),
            "universe_status_hash": (str(z["universe_status_hash"])
                                     if "universe_status_hash" in z.files else None),
            "membership_hash": (str(z["membership_hash"])
                                if "membership_hash" in z.files else None),
            "calendar_hash": (str(z["calendar_hash"])
                              if "calendar_hash" in z.files else None),
            # §十五-1: policy + evaluator identity — `None` marks a legacy tape
            # that predates these keys (tolerated by non-formal replay).
            "evaluator_version": (str(z["evaluator_version"])
                                  if "evaluator_version" in z.files else None),
            "price_convention": (str(z["price_convention"])
                                 if "price_convention" in z.files else None),
            "exit_policy": (str(z["exit_policy"])
                            if "exit_policy" in z.files else None),
            "strategy_mode": (str(z["strategy_mode"])
                              if "strategy_mode" in z.files else None),
            # §十六: split model-identity hashes — `None` marks a legacy tape
            # that predates these keys (tolerated by non-formal replay).
            "model_source_hash": (str(z["model_source_hash"])
                                  if "model_source_hash" in z.files else None),
            "model_config_hash": (str(z["model_config_hash"])
                                  if "model_config_hash" in z.files else None),
            "feature_schema_hash": (str(z["feature_schema_hash"])
                                    if "feature_schema_hash" in z.files else None),
            # §十八-C1: the trained-parameter hash recorded on the tape — tied
            # to the retained checkpoint (fold_NNN_model.pt / fold_NNN_<m>.pkl)
            # below, so a tape's preds are verifiably the product of its weights.
            "weight_hash": (str(z["weight_hash"])
                            if "weight_hash" in z.files else None),
        })
    # fold index grows as val_start walks BACKWARD, so chronological order is
    # the reverse of the lexicographic tape order.
    recs = list(reversed(recs))

    # §十二.2: every fold tape must describe the SAME data + universe records —
    # a fold re-downloaded mid-run, or a delist-file/membership edit between
    # runs, would otherwise be silently blended into one continuous account.
    #
    # §十五-1: the STRATEGY POLICY must be identical across folds too.  The
    # replay explains the whole account with ONE policy (horizon, cost,
    # top_fraction, ...); a horizon=5 fold mixed with a horizon=20 fold, or a
    # 10bp fold with a 30bp fold, would otherwise be silently replayed with the
    # first tape's policy.  A legacy tape that predates a key is `None` and
    # only tolerated by NON-formal replay.
    #
    # §十五-2: FORMAL replay (the production headline account) refuses ANY
    # missing required metadata — universe/delist/calendar/version keys that a
    # legacy tape lacks.  The final continuous account must never silently
    # blend a legacy tape, so `formal=True` upgrades every consistency check to
    # require the key to be present.
    def _consistent(name: str, values: list, *, required: bool = False) -> None:
        if required and any(v is None for v in values):
            raise ValueError(
                f"formal continuous-OOS replay: tape missing required "
                f"{name} — refusing to blend a legacy tape (§十五-2/§十六)")
        known = [v for v in values if v is not None]
        if len(set(known)) > 1:
            # Name the exact folds carrying each value so a model switch is
            # traceable (e.g. "folds [0,1] -> arch A; folds [2,3,4] -> arch B").
            by_val: dict[str, list[int]] = {}
            for i, v in enumerate(values):
                if v is not None:
                    by_val.setdefault(v, []).append(i)
            detail = "; ".join(
                f"folds {sorted(idxs)} -> {v!r}"
                for v, idxs in sorted(by_val.items()))
            raise ValueError(
                f"§十二.2/§十六: fold tapes disagree on {name} — {detail}")

    _consistent("data_version", [r["data_version"] for r in recs],
                required=formal)
    _consistent("universe_status_hash",
                [r["universe_status_hash"] for r in recs], required=formal)
    _consistent("membership_hash", [r["membership_hash"] for r in recs],
                required=formal)
    _consistent("calendar_hash", [r["calendar_hash"] for r in recs],
                required=formal)
    _consistent("horizon", [r["horizon"] for r in recs], required=formal)
    _consistent("cost", [r["cost"] for r in recs], required=formal)
    _consistent("top_fraction", [r["top_fraction"] for r in recs],
                required=formal)
    _consistent("evaluator_version", [r["evaluator_version"] for r in recs],
                required=formal)
    _consistent("price_convention", [r["price_convention"] for r in recs],
                required=formal)
    _consistent("exit_policy", [r["exit_policy"] for r in recs],
                required=formal)
    _consistent("strategy_mode", [r["strategy_mode"] for r in recs],
                required=formal)
    # §十六: the MODEL identity must be identical across folds too — a "first
    # two folds VSN+xLSTM, last three plain LSTM" switch (or a config / feature
    # schema edit between runs) is a model switch the continuous account must
    # never silently blend.  Architecture / config / schema hashes are required
    # to be present AND equal in formal mode.  `weight_hash` (the actual trained
    # parameters) is deliberately allowed to differ per fold — it is NOT checked
    # for cross-fold equality; instead §十八-C1 ties EACH tape's weight_hash to
    # the retained checkpoint that produced it, so a tape's preds are verifiably
    # the product of its own weights.
    _consistent("model_source_hash", [r["model_source_hash"] for r in recs],
                required=formal)
    _consistent("model_config_hash", [r["model_config_hash"] for r in recs],
                required=formal)
    _consistent("feature_schema_hash", [r["feature_schema_hash"] for r in recs],
                required=formal)
    if formal and any(r["delist_day"] is None for r in recs):
        raise ValueError(
            "formal continuous-OOS replay: tape missing required delist_day — "
            "refusing to blend a legacy tape (§十五-2)")
    # §十八-1: FORMAL replay requires weight_hash in EVERY fold.  `rec["weight_hash"]`
    # is None exactly when the tape's npz LACKS the key (the reader stringifies a
    # stored value), so a tape whose predictions cannot be tied to a retained
    # checkpoint — even if the checkpoint itself is absent — is refused instead of
    # being replayed under the legacy skip in `_verify_tape_weight_hash`.
    if formal and any(r["weight_hash"] is None for r in recs):
        raise ValueError(
            "formal continuous-OOS replay: tape missing required weight_hash — "
            "refusing to replay a tape whose predictions cannot be tied to a "
            "retained checkpoint (§十八-1)")

    # §十八-C1: verify each tape's recorded weight_hash against the retained
    # checkpoint.  The deep folds write weight_hash = _state_dict_hash(state_dict)
    # into both fold_NNN.npz and fold_NNN_model.pt; a checkpoint regenerated with
    # different weights, or a tape whose weight points at another checkpoint,
    # fails the replay instead of blending preds of unverifiable provenance.
    # Baseline tapes hash the retained .pkl bytes and are re-derived the same way.
    # Tapes that predate weight_hash, or whose checkpoint is absent, are flagged
    # (legacy tolerance) rather than silently trusted.
    for _rec in recs:
        _verify_tape_weight_hash(_rec, oos_dir, model_name)

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
        # §十二.2: a (stock, day) cell owned by two folds MUST carry the same
        # real price — a silent later-write-overwrites-earlier would hide a
        # data-version split mid-run.  Fail loudly instead.
        old_c = close_glob[np.ix_(rows, pcols)]
        new_c = r["close"]
        both_c = np.isfinite(old_c) & np.isfinite(new_c)
        if np.any(both_c & ~np.isclose(old_c, new_c, rtol=1e-6, atol=1e-9)):
            ri, ci = np.nonzero(
                both_c & ~np.isclose(old_c, new_c, rtol=1e-6, atol=1e-9))
            raise ValueError(
                "§十二.2: overlapping fold close prices disagree for "
                f"{r['stocks'][int(ri[0])]} on {r['price_dates'][int(ci[0])]}: "
                f"{old_c[ri[0], ci[0]]!r} vs {new_c[ri[0], ci[0]]!r}")
        old_o = open_glob[np.ix_(rows, pcols)]
        new_o = r["open"]
        both_o = np.isfinite(old_o) & np.isfinite(new_o)
        if np.any(both_o & ~np.isclose(old_o, new_o, rtol=1e-6, atol=1e-9)):
            ri, ci = np.nonzero(
                both_o & ~np.isclose(old_o, new_o, rtol=1e-6, atol=1e-9))
            raise ValueError(
                "§十二.2: overlapping fold open prices disagree for "
                f"{r['stocks'][int(ri[0])]} on {r['price_dates'][int(ci[0])]}: "
                f"{old_o[ri[0], ci[0]]!r} vs {new_o[ri[0], ci[0]]!r}")
        close_glob[np.ix_(rows, pcols)] = r["close"]
        open_glob[np.ix_(rows, pcols)] = r["open"]
        # Signal columns sit at their true price positions; other columns keep
        # NaN preds so no entry fires there.  §十八-C2: a (stock, signal-day)
        # cell must be owned by EXACTLY ONE fold — the folds' signal windows are
        # strictly non-overlapping (step == val_len), so a second tape predicting
        # the same cell is a data/overlap bug, not a case to silently overwrite.
        scols = np.array([pcol_of[d] for d in r["dates"]], dtype=int)
        old_preds = preds_glob[np.ix_(rows, scols)]
        new_preds = r["preds"]
        both_preds = np.isfinite(old_preds) & np.isfinite(new_preds)
        if np.any(both_preds):
            ri, ci = np.nonzero(both_preds)
            raise ValueError(
                "§十八-C2: same stock + signal day predicted by two fold tapes "
                f"— {r['stocks'][int(ri[0])]} on {r['dates'][int(ci[0])]}.  "
                "Fold signal windows must be disjoint; an overlapping cell means "
                "the folds' signal windows are not strictly non-overlapping.")
        preds_glob[np.ix_(rows, scols)] = r["preds"]
        pool_glob[np.ix_(rows, scols)] = r["pool"]

    # §P0-6: map each tape's per-stock delist index (its own sim column space)
    # onto the union price axis so the replayed account force-sells delisted
    # positions at the same global day the live per-fold runs did.
    delist_glob = np.full(n_glob, -1, dtype=int)
    for r in recs:
        if r["delist_day"] is None:
            continue
        dd = np.asarray(r["delist_day"], dtype=int)
        pdates = r["price_dates"]
        for i, s in enumerate(r["stocks"]):
            d = int(dd[i])
            if d < 0 or d >= len(pdates):
                continue  # force-sell outside this fold's window — never fired
            gcol = pcol_of[pdates[d]]
            row = row_of[s]
            if delist_glob[row] >= 0 and delist_glob[row] != gcol:
                raise ValueError(
                    "§十二.1: fold tapes disagree on the delist day of "
                    f"{s}: union column {delist_glob[row]} vs {gcol}")
            delist_glob[row] = gcol

    acc = _run_sleeve_sim(
        preds_glob, close_glob, open_glob, pool_glob,
        horizon=recs[0]["horizon"],
        top_fraction=recs[0]["top_fraction"],
        cost=recs[0]["cost"],
        mode="long",
        return_ledger=True,
        delist_day=delist_glob,
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
    # §十五-1 / §十二.3: the continuous account's Sharpe read against
    # data-snooping.  The trial-Sharpe dispersion is the HISTORICAL research-
    # trial OOS Sharpe distribution (prior registry rows), NOT the same-run
    # leverage/benchmark variants the review rejected; when fewer than two
    # historical Sharpes exist, the block-bootstrap proxy of this account's
    # returns is the documented fallback.  `n_trials` is the project-wide
    # DISTINCT-experiment count from the training script.
    # §十二.5: the sleeve's daily returns share overlapping holdings, so PSR/DSR
    # use the autocorrelation-adjusted effective sample size, never raw n.
    n_eff = effective_sample_size(daily, horizon=1)
    metrics["n_eff"] = int(n_eff)
    metrics["psr"] = compute_psr(daily, 0.0, 1, n_obs=n_eff)
    metrics["dsr"] = compute_deflated_sharpe(
        daily, n_trials, trial_sharpes, 1, n_obs=n_eff) if (n_trials is not None and n_trials >= 2) else float("nan")
    metrics["dsr_n_trials"] = int(n_trials) if n_trials is not None else None
    hist = [float(x) for x in (trial_sharpes or []) if np.isfinite(x)]
    metrics["dsr_trial_sharpes_n"] = len(hist)
    metrics["dsr_trial_variance_source"] = (
        "historical_registry" if len(hist) >= 2 else "bootstrap_proxy")

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
