"""Chronological sleeve-account backtest simulation (§二十一).

Extracted from ``stoke_ml.models.panel.evaluate`` — the ``_run_sleeve_sim`` /
``_simulate_sleeve_account`` core simulator, the long / short / equal-weight
metric assembly (``_sleeve_account_metrics``), the LS exposure ledger and the
no-rebalance book blend.  ``evaluate`` re-exports these for backward
compatibility.
"""
import numpy as np
import torch

from stoke_ml.models.panel.evaluate_metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_daily_return_profit_factor,
    compute_equity_curve,
    compute_bootstrap_sharpe_ci,
)
from stoke_ml.models.panel.inference import (
    compute_deflated_sharpe,
    compute_psr,
    effective_sample_size,
    block_bootstrap_max_mean,
)


def _build_portfolio_returns(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    n_windows: int,
    horizon: int,
    top_fraction: float,
    mask: torch.Tensor | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Build long-only top-K, short bottom-K, and long-short spread returns.

    All `horizon` entry phases (offset 0..horizon-1) are evaluated and
    concatenated into ONE return series, so no in-sample entry days are wasted
    (the old single-phase loop discarded 4/5 of the data at horizon=5).

    mask: (n_stocks, n_windows) bool — the candidate pool on day t.  Only
    stocks in the pool are selectable; a day with <2 pool members is skipped.
    """
    long_rets, short_rets, spread_rets = [], [], []
    for offset in range(horizon):
        for t in range(offset, n_windows, horizon):
            if mask is not None:
                day_mask = mask[:, t]
                valid_idx = day_mask.nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() < 2:
                    continue
                p_day = preds[valid_idx, t]
                a_day = actuals[valid_idx, t]
            else:
                p_day = preds[:, t]
                a_day = actuals[:, t]

            # Drop non-finite predictions.  Head clamps bound magnitudes but pass
            # NaN through, and descending argsort sorts NaN to the head — that
            # would inject garbage rows into the top-K long / bottom-K short books.
            keep = torch.isfinite(p_day) & torch.isfinite(a_day)
            p_day = p_day[keep]
            a_day = a_day[keep]
            n_candidates = p_day.numel()
            if n_candidates < 2:
                continue

            k = max(1, min(n_candidates // 2, int(round(top_fraction * n_candidates))))
            sorted_idx = torch.argsort(p_day, descending=True)
            top_idx = sorted_idx[:k]
            bot_idx = sorted_idx[-k:]

            long_r = a_day[top_idx].mean().item()
            short_r = a_day[bot_idx].mean().item()
            if np.isfinite(long_r) and np.isfinite(short_r):
                long_rets.append(long_r)
                short_rets.append(short_r)
                spread_rets.append(long_r - short_r)

    return long_rets, short_rets, spread_rets


def _ffill_last_np(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Forward-fill the last known value along axis 1 (per stock).

    NaN before a stock's first real price becomes `fill` (0) so a position in
    a never-traded stock marks to zero instead of poisoning NAV with NaN.
    """
    a = np.array(arr, dtype=np.float64)
    n, m = a.shape
    mask = np.isfinite(a)
    idx = np.where(mask, np.arange(m)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1)
    row_has_valid = mask.any(axis=1)
    first_valid = np.where(row_has_valid, np.argmax(mask, axis=1), m)
    col = np.arange(m)[None, :]
    out = np.where(col < first_valid[:, None], fill, out)
    return np.nan_to_num(out, nan=fill)


def _run_sleeve_sim(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    mode: str = "long",
    return_ledger: bool = False,
    delist_day: np.ndarray | None = None,
) -> dict:
    """Chronological sleeve-account backtest, ONE run at a fixed cost.

    Core simulator shared by `_simulate_sleeve_account`, which runs it twice —
    once at the real cost (net NAV series) and once at cost=0 (the TRUE
    cost-free counterfactual used for gross returns).  This function returns
    the net daily series plus the realized P&L / exit ledger for THIS run; it
    does NOT compute a gross series (the old `gross_daily = net + cost_paid`
    approximation was not a real counterfactual — a zero-cost account would
    reallocate different sleeve sizes, so its daily returns differ day-by-day,
    not just by an additive cost term).

    Every signal day d (< W) enters a new sleeve that buys the top-K (long),
    shorts the bottom-K (short), or buys every pool member (ew), at open[d],
    and is scheduled to liquidate at open[d+horizon].  Each sleeve carries
    weight 1/h of the account's NAV at entry, so h sleeves are concurrently
    invested; the NAV is marked to close each day, giving a TRUE daily return
    series annualized with sqrt(252) — fixing the phase-concatenation bug.

    The simulation runs to the END of the price path (Wp columns), NOT just to
    the last signal day: the sleeve entered on the last signal
    day W-1 must still be liquidated at open[W-1+horizon], with its exit cost
    booked and the book reaching zero active sleeves.  Entry only happens on
    days d < W (no signal beyond W).

    Exits are TRUE delayed exits: a position whose scheduled
    exit open is missing (suspension) is NOT sold at its stale close — it keeps
    holding, marks to its last known close, and retries the exit on every later
    day until a real open appears (DELAYED) or the price path ends with the
    position still open (UNRESOLVED_AT_END, booked at the last carried close,
    capital left locked — no fake exit, no fictitious sell cost).  A carried
    close <= 0 is treated as DATA-MISSING, NOT an executable delist: after
    ffill a zero price cannot
    distinguish a real delisting from a dead data row, so the position is never
    force-sold at 0.  exit_status covers every successfully entered position.

    A stock with a KNOWN delisting day (§七-1) is the exception: `delist_day`
    carries the per-stock simulation-column index where the stock is delisted
    (-1 = never).  A held position on a stock at/after its delist day is
    FORCE-sold at the last carried close (the delisting close) and classified
    DELISTED — even before the sleeve's scheduled exit day, because a delisted
    stock cannot be held to horizon.  This is how a real delisting (from the
    universe delisting record) is told apart from a dead data row: only stocks
    with a delist record reach the force-sell path; everything else stays
    unresolved.

    Fills respect entry eligibility: a selected stock with no real open[d]
    stays unfilled (its weight remains cash) and is NEVER backfilled.  Long/EW
    entries are additionally capped so total spend (notional
    + entry cost) never exceeds available cash — the old cap
    booked the cost AFTER sizing, so a full-buy at 10bp went cash-negative.

    The account starts at nav 1.0 on the close BEFORE the first price day, so
    the returned daily series INCLUDES day 0.  An internal
    assertion enforces the account identity np.prod(1+daily) == final_nav.

    preds_np: (N, W) predictions for entry at window column d.  close_np /
    open_np: (N, Wp) price paths aligned to the SAME window-day grid (column d
    = entry day d = panel column seq_len+d), Wp >= W.  The exit at open[d+h]
    is the sleeve popped on day d+h reading open_np[:, d+h].  If Wp < W+horizon
    the trailing sleeves end up UNRESOLVED.
    """
    N, W = preds_np.shape
    Wp = close_np.shape[1]
    if Wp < W:
        raise ValueError(f"close path {Wp} columns < preds {W} columns")
    if delist_day is None:
        delist_day = np.full(N, -1, dtype=int)
    else:
        dd = np.asarray(delist_day, dtype=int)
        if dd.shape != (N,):
            raise ValueError(f"delist_day shape {dd.shape} != preds {preds_np.shape}")
        delist_day = dd
    side = {"long": 1.0, "ew": 1.0, "short": -1.0}[mode]
    carried = _ffill_last_np(close_np)

    nav_prev = 1.0
    cash = 1.0
    # §十三-2: the day the account's NAV first goes <= 0 (blow-up).  From that
    # day on the account is frozen — no entries, no marks, no further returns —
    # so post-bankruptcy days cannot manufacture economically meaningless
    # returns/exposures.
    bankrupt_day: int | None = None
    sleeves: dict[int, dict] = {}
    net_daily = np.zeros(Wp)
    cost_paid = np.zeros(Wp)
    buy_notional = np.zeros(Wp)
    sell_notional = np.zeros(Wp)
    nav_before_day = np.zeros(Wp)
    # Daily exposure ledger: held notional / held value / cash /
    # active sleeves / delayed capital, each normalized by start-of-day NAV so
    # the ratios are scale-free and comparable across sub-accounts.
    gross_exposure = np.zeros(Wp)
    net_exposure = np.zeros(Wp)
    cash_ratio = np.zeros(Wp)
    active_sleeves = np.zeros(Wp, dtype=int)
    delayed_capital = np.zeros(Wp)
    exit_counts = {"clean": 0, "delayed": 0, "delisted": 0,
                   "unresolved": 0, "unfilled": 0}
    exit_pnl = {k: 0.0 for k in exit_counts}
    exit_days = {k: 0.0 for k in exit_counts}
    # Per-position ledger: one record per FILLED position,
    # with entry/exit price, scheduled/actual exit day, exit status, gross PnL
    # and attributed costs.  Sum(net_pnl) == final_nav - 1 by construction for
    # resolved positions; unresolved ones carry unrealized_pnl instead (§十三-3)
    # and sum over both fields == final_nav - 1.  Each position's gross PnL
    # matches exit_pnl aggregation and entry/exit costs match cost_paid
    # day-by-day.  Built only when requested.
    ledger: list[dict] = []

    for d in range(Wp):
        nav_before_day[d] = nav_prev
        if bankrupt_day is not None:
            # Frozen post-bankruptcy: NAV pinned at the blow-up level so
            # prod(1+daily) == final_nav still holds; no new entries, no marks.
            continue
        # ── Liquidate positions whose scheduled exit is due (or overdue) ──
        for c, sl in list(sleeves.items()):
            held = sl["mask"]
            if not held.any():
                del sleeves[c]
                continue
            # A stock DELISTED on/before today (KNOWN from the universe delisting
            # record) is FORCE-sold at its last carried close — the delisting
            # event forces the exit, even before the sleeve's scheduled exit
            # day, because a delisted stock cannot be held to horizon.
            delisted_here = held & (delist_day >= 0) & (d >= delist_day)
            # Valid (all-False) in the early-branch path so `is_clean` below is
            # always defined; the real value is assigned in the else branch.
            ex_open_ok = np.zeros(N, dtype=bool)
            if d < sl["scheduled_exit_day"]:
                if not delisted_here.any():
                    continue
                liquidate = delisted_here
            else:
                ex_open = open_np[:, d]
                ex_open_ok = np.isfinite(ex_open) & (ex_open > 0)
                exitable = held & ex_open_ok
                if d == sl["scheduled_exit_day"]:
                    # First attempt: positions that cannot exit become PENDING and
                    # are held (marked to last close) until a real open appears.
                    # A delisted position is NOT pending — it is force-sold below.
                    sl["pending"] |= held & (~ex_open_ok) & (carried[:, d] > 0) & (~delisted_here)
                # A held position whose carried close is <= 0 is DATA-MISSING, not
                # an executable delist: price<=0 after ffill
                # cannot tell a real delisting from a dead data row, so it is never
                # force-sold at 0.  It stays held at its (zero) carried mark and
                # resolves as UNRESOLVED_AT_END at the path end.
                liquidate = exitable | delisted_here
            if liquidate.any():
                # Delisted positions sell at the last carried close (the
                # delisting close); normal positions sell at the day's real open.
                ex_price = carried[:, d].copy()
                normal = liquidate & ~delisted_here
                if normal.any():
                    ex_price[normal] = np.where(
                        ex_open_ok[normal], ex_open[normal], carried[normal, d])
                ex_value = sl["shares"] * ex_price
                proceeds = side * ex_value[liquidate].sum()
                cost_here = cost * np.abs(ex_value[liquidate]).sum()
                cash += proceeds - cost_here
                cost_paid[d] += cost_here
                sell_notional[d] += np.abs(ex_value[liquidate]).sum()
                is_clean = ((d == sl["scheduled_exit_day"]) & ex_open_ok
                            & (~sl["pending"]) & (~delisted_here))
                for cls, cm in (("clean", liquidate & is_clean),
                                ("delayed", liquidate & ~is_clean & ~delisted_here),
                                ("delisted", liquidate & delisted_here)):
                    if cm.any():
                        cls_pnl = side * (ex_value[cm].sum() - sl["entry_val"][cm].sum())
                        exit_pnl[cls] += float(cls_pnl)
                        exit_counts[cls] += int(cm.sum())
                        exit_days[cls] += float((d - c) * int(cm.sum()))
                if return_ledger:
                    for j in np.nonzero(liquidate)[0]:
                        ej = float(sl["entry_val"][j])
                        xj = float(ex_value[j])
                        ledger.append({
                            "entry_day": c,
                            "stock": int(j),
                            "mode": mode,
                            "entry_price": float(open_np[j, c]),
                            "entry_notional": ej,
                            "target_weight": (float(sl["target_w"] / sl["entry_nav"])
                                              if sl["entry_nav"] > 0 else 0.0),
                            "executed_weight": (float(ej / sl["entry_nav"])
                                                if sl["entry_nav"] > 0 else 0.0),
                            "entry_nav": float(sl["entry_nav"]),
                            "shares": float(sl["shares"][j]),
                            "scheduled_exit_day": int(sl["scheduled_exit_day"]),
                            "actual_exit_day": d,
                            "exit_status": ("delisted" if delisted_here[j]
                                            else ("clean" if is_clean[j] else "delayed")),
                            "exit_price": float(ex_price[j]),
                            "realized_return": float((xj - ej) / ej) if ej > 0 else 0.0,
                            "mark_day": None,
                            "mark_price": None,
                            "gross_pnl": float(side * (xj - ej)),
                            "entry_cost": cost * ej,
                            "exit_cost": cost * abs(xj),
                            "net_pnl": float(side * (xj - ej) - cost * ej - cost * abs(xj)),
                            "unrealized_pnl": None,
                        })
                sl["mask"] = held & ~liquidate
            if not sl["mask"].any():
                del sleeves[c]

        # ── Enter the sleeve at signal day d ──
        if d < W:
            w = nav_prev / horizon
            pool_d = select_pool[:, d] & np.isfinite(preds_np[:, d])
            pool_idx = np.nonzero(pool_d)[0]
            shares = np.zeros(N)
            entry_val = np.zeros(N)
            fillable = np.zeros(N, dtype=bool)  # empty pool → empty (cash) sleeve
            # §十四-2: `target_w` is the uncapped per-stock nominal the sleeve
            # intended (w/|chosen|, NAV units) and `entry_nav` is the account NAV
            # at this sleeve's entry — together they let the ledger report an
            # honest `target_weight` (1/(horizon*|chosen|)) vs `executed_weight`
            # (entry_notional/entry_nav): when the cash cap bites, executed
            # drops below target instead of silently passing the capped amount
            # off as a "weight".
            target_w = 0.0
            entry_nav = nav_prev
            if pool_idx.size > 0:
                if mode == "ew":
                    chosen = pool_idx
                else:
                    k = max(1, min(pool_idx.size, int(round(top_fraction * pool_idx.size))))
                    order = pool_idx[np.argsort(preds_np[pool_idx, d])]
                    chosen = order[-k:] if mode == "long" else order[:k]
                stock_mask = np.zeros(N, dtype=bool)
                stock_mask[chosen] = True
                fillable = stock_mask & np.isfinite(open_np[:, d]) & (open_np[:, d] > 0)
                exit_counts["unfilled"] += int((stock_mask & ~fillable).sum())
                n_fill = int(fillable.sum())
                if n_fill > 0:
                    # Fixed per-stock weight w/|chosen| — the unfilled fraction
                    # stays cash (NO backfill / 递补).  Long/EW
                    # cap the whole sleeve at available cash so leverage cannot
                    # drift past what the account holds; the
                    # short leg is a theoretical factor book and needs no cap.
                    target_w = w / chosen.size
                    per_stock = target_w
                    if side > 0:
                        # Cap total spend (notional + entry cost) at available
                        # cash: per_stock*n_fill*(1+cost) <= cash
                        # so the entry cost can never push cash negative.
                        per_stock = min(per_stock, cash / ((1.0 + cost) * n_fill))
                    shares[fillable] = per_stock / open_np[fillable, d]
                    entry_val[fillable] = per_stock
                    spent = per_stock * n_fill
                    # The entry cost is deducted from cash along with the
                    # notional — previously only the notional
                    # left cash, so net returns excluded the entry cost.
                    cash -= side * spent + cost * spent
                    cost_paid[d] += cost * spent
                    buy_notional[d] = spent
            sleeves[d] = {"shares": shares, "mask": fillable, "entry_val": entry_val,
                          "target_w": target_w, "entry_nav": entry_nav,
                          "scheduled_exit_day": d + horizon,
                          "pending": np.zeros(N, dtype=bool)}

        # ── Mark to market at close d ──
        pos_value = 0.0
        gross_value = 0.0
        active_count = 0
        delayed_value = 0.0
        for c, sl in sleeves.items():
            if sl["mask"].any():
                mv = float((sl["shares"] * carried[:, d]).sum())
                pos_value += side * mv
                gross_value += abs(mv)
                active_count += 1
                if d > sl["scheduled_exit_day"]:
                    delayed_value += abs(mv)
        nav_close = cash + pos_value
        # The account starts at nav 1.0 on the close BEFORE the
        # first price day, so day-0's open->close P&L IS part of the return
        # series — dropping it broke the identity prod(1+daily) == final_nav.
        if nav_prev > 0:
            net_daily[d] = nav_close / nav_prev - 1.0
        active_sleeves[d] = active_count
        if nav_close > 0:
            # Exposure ratios are close-market-value / close-NAV (§十三-1) —
            # exactly the position's value share at that close.  The old
            # start-of-day base produced an "open-NAV multiple of close value"
            # hybrid that, once the LS ledger rescaled it by the end-of-day
            # NAV, re-applied the day's price change.  The nominal
            # 1/h-per-sleeve book is only an approximation — unfilled slots
            # stay cash (gross < target) while delayed/unresolved sleeves push
            # gross above it.
            gross_exposure[d] = gross_value / nav_close
            net_exposure[d] = pos_value / nav_close
            cash_ratio[d] = cash / nav_close
            delayed_capital[d] = delayed_value / nav_close
        else:
            # NAV <= 0 → bankruptcy (§十三-2): the account is dead at this
            # mark.  The blow-up day's return above already records the
            # collapse; from here the account is frozen (see loop top).
            bankrupt_day = d
        nav_prev = nav_close

    # ── End of price path: any holding that never found a real exit is
    # UNRESOLVED_AT_END, booked at its last carried close: the
    # capital stays locked in the position — no fake exit, no fictitious sell
    # cost. ──
    for c, sl in sleeves.items():
        held = sl["mask"]
        if held.any():
            if bankrupt_day is not None:
                # §十三-2: the account is already dead — the frozen positions'
                # value is embedded in the blow-up day's return.  Booking them
                # again at the final mark would double-count later price moves,
                # so a bankrupt account books no unresolved P&L.
                continue
            # Mark the unresolved hold at the FINAL carried close:
            # carried[:, Wp-1] is what final_nav's mark-to-market used on the
            # last day, so the ledger reconciles with the NAV series.  The
            # old min(c+horizon, Wp-1) booked at the scheduled exit day, which
            # drifts from final_nav whenever price moved after that day.
            last_close = carried[:, Wp - 1]
            ex_value = sl["shares"] * last_close
            pnl = side * (ex_value[held].sum() - sl["entry_val"][held].sum())
            exit_pnl["unresolved"] += float(pnl)
            exit_counts["unresolved"] += int(held.sum())
            exit_days["unresolved"] += float((Wp - 1 - c) * int(held.sum()))
            if return_ledger:
                for j in np.nonzero(held)[0]:
                    ej = float(sl["entry_val"][j])
                    xj = float(ex_value[j])
                    ledger.append({
                        "entry_day": c,
                        "stock": int(j),
                        "mode": mode,
                        "entry_price": float(open_np[j, c]),
                        "entry_notional": ej,
                        "target_weight": (float(sl["target_w"] / sl["entry_nav"])
                                          if sl["entry_nav"] > 0 else 0.0),
                        "executed_weight": (float(ej / sl["entry_nav"])
                                            if sl["entry_nav"] > 0 else 0.0),
                        "entry_nav": float(sl["entry_nav"]),
                        "shares": float(sl["shares"][j]),
                        "scheduled_exit_day": int(sl["scheduled_exit_day"]),
                        # §十三-3: an unresolved hold is NOT a realized exit.
                        # actual_exit_day stays null and the end-of-path close is
                        # a mark, so no exit_price/realized_return/net_pnl are
                        # fabricated — the open P&L is unrealized_pnl at
                        # mark_day/mark_price instead.
                        "actual_exit_day": None,
                        "exit_status": "unresolved",
                        "exit_price": None,
                        "realized_return": None,
                        "mark_day": int(Wp - 1),
                        "mark_price": float(last_close[j]),
                        "gross_pnl": float(side * (xj - ej)),
                        "entry_cost": cost * ej,
                        "exit_cost": 0.0,
                        "net_pnl": None,
                        "unrealized_pnl": float(side * (xj - ej) - cost * ej),
                    })

    total_pnl = sum(exit_pnl.values())
    abs_total = sum(abs(v) for v in exit_pnl.values())

    # P&L reconciliation: every sleeve's realized P&L net of
    # ALL entry/exit costs must equal the account's total NAV change.  Booked
    # at a mark different from the final one (or costs missing from cost_paid)
    # would silently break this identity, so assert it explicitly.
    if (bankrupt_day is None and np.isfinite(nav_close)
            and np.isfinite(total_pnl)):
        reconciled = float(total_pnl) - float(cost_paid.sum())
        target = float(nav_close) - 1.0
        if not np.isclose(reconciled, target, rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"sleeve P&L reconciliation violated: sum(pnl) - sum(cost) = "
                f"{reconciled:.8f} != final_nav - 1 = {target:.8f}")
    # Turnover is the traded notional scaled by the NAV the day opened on —
    # a ratio, invariant to the initial capital.
    turnover_daily = (buy_notional + sell_notional) / np.maximum(nav_before_day, 1e-8)

    # The signed PnL share explodes when the book's total
    # PnL is near zero, so alongside the signed share (kept for compat) report
    # the signed PnL, an ABSOLUTE-PnL share, capital-occupancy days, and the
    # mean delay of delayed exits.
    avg_delayed_days = (exit_days["delayed"] / exit_counts["delayed"]
                        if exit_counts["delayed"] else 0.0)

    # Enforce the account identity without relying on manual inspection:
    # np.prod(1+daily) must equal final_nav / initial_nav.
    if np.isfinite(nav_close) and nav_close > 0 and np.all(nav_before_day > 0):
        cum = float(np.prod(1.0 + net_daily))
        if not np.isclose(cum, nav_close, rtol=1e-5, atol=1e-8):
            raise AssertionError(
                f"sleeve account identity violated: prod(1+daily)={cum:.8f} != "
                f"final_nav={nav_close:.8f}")

    return {
        "daily": net_daily,
        # The account's final mark, exposed so a consumer can verify the
        # identity prod(1+daily) == final_nav WITHOUT recomputing nav_close —
        # the strongest cross-check a backtest can carry.
        "final_nav": float(nav_close) if np.isfinite(nav_close) else None,
        "exit_stats": {
            "counts": exit_counts,
            "pnl": {k: float(v) for k, v in exit_pnl.items()},
            "pnl_share": {k: (float(v) / total_pnl if total_pnl != 0.0 else 0.0)
                          for k, v in exit_pnl.items()},
            "abs_pnl_share": {k: (float(abs(v)) / abs_total if abs_total != 0.0 else 0.0)
                              for k, v in exit_pnl.items()},
            "capital_days": {k: float(v) for k, v in exit_days.items()},
            "avg_delayed_days": float(avg_delayed_days),
        },
        "turnover": {
            "daily_avg": float(turnover_daily.mean()) if Wp > 0 else 0.0,
        },
        # Daily exposure ledger: ratios normalized by NAV.
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "cash_ratio": cash_ratio,
        "active_sleeves": active_sleeves,
        "delayed_capital": delayed_capital,
        # §十三-2: first day NAV <= 0 (None = account survived the full path).
        "bankrupt_day": bankrupt_day,
        # Per-position ledger; empty unless return_ledger.
        "ledger": ledger,
    }


def _simulate_sleeve_account(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    mode: str = "long",
    return_ledger: bool = False,
    delist_day: np.ndarray | None = None,
) -> dict:
    """Sleeve-account backtest with a TRUE cost-free gross counterfactual.

    Runs `_run_sleeve_sim` twice — once at the real `cost` (net NAV series,
    realized exit ledger, turnover) and once at cost=0 (the independent
    cost-free account).  `daily` is the NET series; `gross_daily` is the
    cost=0 series.  Because the zero-cost run re-sizes sleeve positions, its
    daily returns differ from the net series day-by-day in a way the old
    `net_daily + cost_paid/nav` additive approximation could not capture.  Both
    runs trigger the internal account-identity and P&L-reconciliation asserts,
    so each returned series is internally consistent.  Return keys are
    backward-compatible with the pre-refactor shape.

    `return_ledger` surfaces the NET run's per-position ledger so a caller can
    persist every filled position's entry/exit prices,
    status and attributed costs — the audit trail that lets the OOS account be
    reconstructed offline from a tape.  `delist_day` is forwarded to
    `_run_sleeve_sim` so known-delisted stocks are force-sold (§七-1).
    """
    net = _run_sleeve_sim(preds_np, close_np, open_np, select_pool,
                          horizon, top_fraction, cost, mode=mode,
                          return_ledger=return_ledger, delist_day=delist_day)
    gross = _run_sleeve_sim(preds_np, close_np, open_np, select_pool,
                            horizon, top_fraction, 0.0, mode=mode,
                            delist_day=delist_day)
    out = {
        "daily": net["daily"],
        "gross_daily": gross["daily"],
        "exit_stats": net["exit_stats"],
        "turnover": net["turnover"],
        # Exposure ledger is measured on the REAL-cost run —
        # it describes the account as actually traded, not the counterfactual.
        "gross_exposure": net["gross_exposure"],
        "net_exposure": net["net_exposure"],
        "cash_ratio": net["cash_ratio"],
        "active_sleeves": net["active_sleeves"],
        "delayed_capital": net["delayed_capital"],
        "bankrupt_day": net["bankrupt_day"],
    }
    if return_ledger:
        out["ledger"] = net["ledger"]
    return out


def _ls_exposure_ledger(long_a: dict, short_a: dict) -> dict:
    """Combined LS-book daily exposure ledger + summary.

    The LS book holds the long and short sub-accounts at 0.5 weight each.  Each
    sub-account reports close-NAV-relative ratios from its own simulation
    (close market value / close NAV, §十三-1); rescaling that ratio by the same
    end-of-day NAV recovers the leg's currency value, so the day's price change
    is applied exactly once — the old start-of-day base double-counted it.

    Sign convention: `long_market_value` / `short_market_value` are the SIGNED
    position values of each leg as fractions of the combined NAV — the short
    leg is negative because a short position has negative value (a liability).
    This makes the accounting identity

        long_market_value + short_market_value + cash == 1

    hold exactly on every solvent day (the short leg's cash collateral backs
    its liability), with

        gross == abs(long_market_value) + abs(short_market_value)
        net   == long_market_value + short_market_value

    (`net` is the signed SUM of the legs — the spec's "net == long - short"
    used short as a positive magnitude, which would double-count the liability
    and break the identity above; the signed convention is the consistent one).

    On a day where either leg or the combined book is bankrupt (NAV <= 0,
    §十三-2) the value-share identity is meaningless, so those days are NaN in
    the daily ledger and excluded from the summary statistics.

    `target` is the nominal book construction (100% gross / 0% net); `realized`
    is what actually traded — unfilled cash slots push gross below target while
    delayed/unresolved sleeves push it above, so the two must not be conflated.
    """
    long_d = np.asarray(long_a["daily"], dtype=np.float64)
    short_d = np.asarray(short_a["daily"], dtype=np.float64)
    long_nav = np.cumprod(1.0 + long_d)
    short_nav = np.cumprod(1.0 + short_d)
    comb_nav = 0.5 * long_nav + 0.5 * short_nav
    # §十三-2: a value share is only defined for a solvent book; a bankrupt leg
    # or combined NAV has no defined share, so those days are NaN (excluded
    # from stats) rather than silently divided by a fake 1.0 denominator.
    healthy = (np.isfinite(long_nav) & (long_nav > 0)
               & np.isfinite(short_nav) & (short_nav > 0)
               & np.isfinite(comb_nav) & (comb_nav > 0))
    safe = np.where(healthy, comb_nav, np.nan)

    ln = np.asarray(long_a["net_exposure"], dtype=np.float64)
    sn = np.asarray(short_a["net_exposure"], dtype=np.float64)
    lc = np.asarray(long_a["cash_ratio"], dtype=np.float64)
    sc = np.asarray(short_a["cash_ratio"], dtype=np.float64)
    ld = np.asarray(long_a["delayed_capital"], dtype=np.float64)
    sd = np.asarray(short_a["delayed_capital"], dtype=np.float64)
    la = np.asarray(long_a["active_sleeves"], dtype=np.int64)
    sa = np.asarray(short_a["active_sleeves"], dtype=np.int64)

    # Long leg is long-only → net ratio == gross; the short leg's net ratio is
    # NEGATIVE (short positions have negative value).  Rescaling each leg's
    # close-NAV ratio by that same close NAV (long_nav) gives the currency
    # value; dividing by the combined NAV gives the signed value share.
    long_mv = (0.5 * ln * long_nav) / safe
    short_mv = (0.5 * sn * short_nav) / safe
    ls_cash = (0.5 * lc * long_nav + 0.5 * sc * short_nav) / safe
    ls_gross = np.abs(long_mv) + np.abs(short_mv)
    ls_net = long_mv + short_mv
    ls_delayed = (0.5 * ld * long_nav + 0.5 * sd * short_nav) / safe

    # §十三-1 invariants, enforced on every solvent day.  The identities are
    # only consistent under the signed-short convention above.
    if healthy.any():
        tol = 1e-6
        ident = long_mv[healthy] + short_mv[healthy] + ls_cash[healthy]
        g_ident = (ls_gross[healthy] - np.abs(long_mv[healthy])
                   - np.abs(short_mv[healthy]))
        n_ident = ls_net[healthy] - long_mv[healthy] - short_mv[healthy]
        if (np.max(np.abs(ident - 1.0)) > tol or np.max(np.abs(g_ident)) > tol
                or np.max(np.abs(n_ident)) > tol):
            raise AssertionError(
                "LS exposure identities violated on a solvent day: "
                f"max|long+short+cash-1|={np.max(np.abs(ident - 1.0)):.3e} "
                f"max|gross-abs-sum|={np.max(np.abs(g_ident)):.3e} "
                f"max|net-(long+short)|={np.max(np.abs(n_ident)):.3e}")

    g = ls_gross[np.isfinite(ls_gross)]
    n = ls_net[np.isfinite(ls_net)]
    dly = ls_delayed[np.isfinite(ls_delayed)]
    # Target gross is 1.0; count days deviating >5% either way and the mean
    # over-leverage (gross above 1.0) — the excess delayed exits can create.
    off_target = int(np.sum(np.abs(g - 1.0) > 0.05)) if g.size else 0
    excess = float(np.mean(np.maximum(g - 1.0, 0.0))) if g.size else 0.0

    return {
        "target": {
            "gross_exposure": 1.0,
            "net_exposure": 0.0,
            "long_exposure": 0.5,
            "short_exposure": 0.5,
        },
        "realized": {
            "mean_gross_exposure": float(g.mean()) if g.size else 0.0,
            "max_gross_exposure": float(g.max()) if g.size else 0.0,
            "p95_gross_exposure": float(np.percentile(g, 95)) if g.size else 0.0,
            "mean_net_exposure": float(n.mean()) if n.size else 0.0,
            "days_off_target_gross": off_target,
            "mean_excess_over_target_gross": excess,
            "mean_delayed_capital": float(dly.mean()) if dly.size else 0.0,
            "max_delayed_capital": float(dly.max()) if dly.size else 0.0,
            "daily": {
                "long_market_value": long_mv.tolist(),
                "short_market_value": short_mv.tolist(),
                "cash": ls_cash.tolist(),
                "gross_exposure": ls_gross.tolist(),
                "net_exposure": ls_net.tolist(),
                "active_long_sleeves": la.tolist(),
                "active_short_sleeves": sa.tolist(),
                "delayed_capital": ls_delayed.tolist(),
            },
        },
        # §十三-2: the first day each leg's NAV went <= 0 (None = survived).
        "bankrupt": {
            "long_day": long_a.get("bankrupt_day"),
            "short_day": short_a.get("bankrupt_day"),
        },
        "note": ("target = nominal book (ls_sharpe is 100% gross, 50% long / "
                 "50% short, net 0; ls2x_* is 200% gross); realized = daily "
                 "measured exposure from the sleeve simulation.  Value shares "
                 "are signed (short leg negative); a bankrupt day (any leg or "
                 "combined NAV <= 0) is NaN in the daily ledger and excluded "
                 "from the summary."),
    }


def _combine_book_daily(
    a_daily: np.ndarray,
    b_daily: np.ndarray,
    w_a: float,
    w_b: float,
    subtract: float = 0.0,
) -> np.ndarray:
    """Daily returns of a NO-rebalance blend of two sub-account NAVs.

    Each sub-account starts at ``w_a`` / ``w_b`` of unit capital and is never
    rebalanced, so the combined daily return is the NAV ratio of the weighted
    NAV blend — the same NAV ``_ls_exposure_ledger`` reports, keeping the
    metric and the ledger on one policy (方案 A).  ``subtract`` folds in
    borrowed capital for leverage >100% gross: the 2x book holds a full unit
    long AND a full unit short, so its NAV = long_nav + short_nav - 1.0 (the
    short leg's margin loan).  The first element is ``nav[0] - 1`` because the
    book starts at unit capital.

    §十三-2: if the blended NAV goes <= 0 the book is bankrupt.  The blow-up
    day's return is still the true NAV ratio (<= -1); from there the account
    is closed and produces zero further returns.  The old
    ``np.where(nav[:-1] > 0, nav[:-1], 1.0)`` substituted a fake 1.0
    denominator, manufacturing economically meaningless post-bankruptcy
    returns.
    """
    a = np.cumprod(1.0 + np.asarray(a_daily, dtype=np.float64))
    b = np.cumprod(1.0 + np.asarray(b_daily, dtype=np.float64))
    nav = w_a * a + w_b * b - subtract
    out = np.empty_like(nav)
    out[0] = nav[0] - 1.0
    blowup = nav <= 0
    if np.any(blowup):
        first = int(np.argmax(blowup))
        if first > 0:
            out[1:first + 1] = nav[1:first + 1] / nav[:first] - 1.0
        out[first + 1:] = 0.0
    else:
        out[1:] = nav[1:] / nav[:-1] - 1.0
    return out


def _sleeve_account_metrics(
    preds_np: np.ndarray,
    close_np: np.ndarray,
    open_np: np.ndarray,
    select_pool: np.ndarray,
    horizon: int,
    top_fraction: float,
    cost: float,
    n_boot: int,
    return_ledger: bool = False,
    delist_day: np.ndarray | None = None,
    n_trials: int | None = None,
) -> dict:
    """Run long / short / equal-weight sleeve accounts and assemble metrics.

    When `n_trials` is given (>1), the report also carries the §十五-1
    multiple-testing corrections — PSR, DSR and the block-bootstrap max-mean
    reality-check p-value vs the selected-universe equal-weight benchmark — so
    the headline Sharpe is read together with how much of it survives
    data-snooping.  The reality check is an un-studentized max-mean screen, not
    a full Hansen SPA (§十二.4).
    """
    long_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                      horizon, top_fraction, cost, "long",
                                      return_ledger=return_ledger,
                                      delist_day=delist_day)
    short_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                       horizon, top_fraction, cost, "short",
                                       return_ledger=return_ledger,
                                       delist_day=delist_day)
    # Eligible candidate-pool equal-weight: every pool member
    # gets an equal slice — NOT a full-market/index benchmark.  A SEPARATE
    # selected-universe equal-weight proxy (all fold stocks, no eligibility
    # gate) is run below as the naive "buy everything" reference.
    ew_a = _simulate_sleeve_account(preds_np, close_np, open_np, select_pool,
                                    horizon, top_fraction, cost, "ew",
                                    return_ledger=return_ledger,
                                    delist_day=delist_day)
    sel_uni_a = _simulate_sleeve_account(
        preds_np, close_np, open_np, np.ones_like(select_pool),
        horizon, top_fraction, cost, "ew", return_ledger=return_ledger,
        delist_day=delist_day)

    long_d = np.asarray(long_a["daily"], dtype=np.float64)
    short_d = np.asarray(short_a["daily"], dtype=np.float64)
    long_g = np.asarray(long_a["gross_daily"], dtype=np.float64)
    short_g = np.asarray(short_a["gross_daily"], dtype=np.float64)
    sel_uni_d = np.asarray(sel_uni_a["daily"], dtype=np.float64)

    # `short_d` is already the short account's REAL daily return
    # (side=-1 applied inside), so the long-short book ADDS the legs — NOT
    # long_d - short_d, which cancels them when both sides profit.
    # The LS book does NOT rebalance daily.  The two legs each
    # start at their target weight of unit capital and are left alone, so the
    # combined daily return is the NAV ratio of the weighted NAV blend (方案 A)
    # — NOT the arithmetic mean of the legs' daily returns, which silently
    # assumes a daily 50/50 reallocation and drifts from the ledger NAV.  All
    # Sharpe/MDD/CAGR now derive from the same combined NAV the exposure
    # ledger uses (_ls_exposure_ledger).  Primary metric = 100% gross (50% long
    # + 50% short, net 0); 200% gross (100% long + 100% short) is reported
    # alongside as ls2x (NAV = long_nav + short_nav - 1).  Both are explicit so
    # two different-leverage books are never compared as one.
    ls_d = _combine_book_daily(long_d, short_d, 0.5, 0.5)
    ls2x_d = _combine_book_daily(long_d, short_d, 1.0, 1.0, subtract=1.0)
    ls_g = _combine_book_daily(long_g, short_g, 0.5, 0.5)
    ls2x_g = _combine_book_daily(long_g, short_g, 1.0, 1.0, subtract=1.0)

    def _met(daily, gross):
        t = torch.tensor(daily, dtype=torch.float32)
        eq = compute_equity_curve(t)
        return {
            "sharpe": compute_sharpe(t, horizon=1),
            "gross_sharpe": compute_sharpe(
                torch.tensor(gross, dtype=torch.float32), horizon=1),
            "sortino": compute_sortino(t, horizon=1),
            "calmar": compute_calmar(t, horizon=1),
            "maxdd": compute_max_drawdown(eq),
            "daily_return_pf": compute_daily_return_profit_factor(t),
        }

    long_m = _met(long_d, long_g)
    ls_m = _met(ls_d, ls_g)
    ls2x_m = _met(ls2x_d, ls2x_g)
    eligible_ew_sharpe = compute_sharpe(
        torch.tensor(ew_a["daily"], dtype=torch.float32), horizon=1)
    selected_universe_ew_sharpe = compute_sharpe(
        torch.tensor(sel_uni_a["daily"], dtype=torch.float32), horizon=1)

    # §十五-3: the daily sleeve returns share overlapping holdings when
    # horizon>1, so the resampled block is floored at the strategy horizon.
    # horizon=1 → floor is ceil(n^(1/3)) as before (no iid degeneration); the
    # `horizon=1` arg here only sets the ANNUALIZATION (√252, per-day returns).
    ci_block = max(2, int(np.ceil(len(long_d) ** (1 / 3))), horizon)
    long_lo, long_hi = compute_bootstrap_sharpe_ci(
        long_d, horizon=1, n_boot=n_boot, block_len=ci_block)
    ls_lo, ls_hi = compute_bootstrap_sharpe_ci(
        ls_d, horizon=1, n_boot=n_boot, block_len=ci_block)

    # §十五-1 multiple-testing corrections.  The trial-Sharpe dispersion comes
    # from the strategies evaluated HERE (long / ls / ls2x / the two ew
    # proxies) — the within-report multiplicity — while `n_trials` counts the
    # research trials iterated across the project (experiment registry); the
    # registry count is passed in by the training script, else it defaults to
    # the number of strategies in this report.
    # §十二.5: sleeve daily returns share overlapping holdings and volatility
    # clustering, so PSR/DSR must read the autocorrelation-adjusted effective
    # sample size — the raw n overstates precision and inflates the probabilities.
    long_n_eff = effective_sample_size(long_d, horizon=1)
    ls_n_eff = effective_sample_size(ls_d, horizon=1)
    long_psr = compute_psr(long_d, 0.0, 1, n_obs=long_n_eff)
    ls_psr = compute_psr(ls_d, 0.0, 1, n_obs=ls_n_eff)
    trial_sharpes = [long_m["sharpe"], ls_m["sharpe"], ls2x_m["sharpe"],
                     eligible_ew_sharpe, selected_universe_ew_sharpe]
    nt = n_trials if (n_trials is not None and n_trials >= 2) else len(trial_sharpes)
    long_dsr = compute_deflated_sharpe(long_d, nt, trial_sharpes, 1, n_obs=long_n_eff)
    ls_dsr = compute_deflated_sharpe(ls_d, nt, trial_sharpes, 1, n_obs=ls_n_eff)
    bbmm = block_bootstrap_max_mean(np.stack([long_d, ls_d, ls2x_d]),
                                    sel_uni_d, horizon=horizon)

    out = {
        "n_periods": int(long_d.size),
        "long_sharpe": long_m["sharpe"],
        "long_gross_sharpe": long_m["gross_sharpe"],
        "long_sharpe_lo": long_lo,
        "long_sharpe_hi": long_hi,
        "long_sortino": long_m["sortino"],
        "long_calmar": long_m["calmar"],
        "long_maxdd": long_m["maxdd"],
        "long_daily_return_pf": long_m["daily_return_pf"],
        "ls_sharpe": ls_m["sharpe"],
        "ls_gross_sharpe": ls_m["gross_sharpe"],
        "ls_sharpe_lo": ls_lo,
        "ls_sharpe_hi": ls_hi,
        "ls_sortino": ls_m["sortino"],
        "ls_calmar": ls_m["calmar"],
        "ls_maxdd": ls_m["maxdd"],
        "ls2x_sharpe": ls2x_m["sharpe"],
        "ls2x_gross_sharpe": ls2x_m["gross_sharpe"],
        "ls2x_maxdd": ls2x_m["maxdd"],
        # §十五-1 multiple-testing corrections (NaN when the series is too
        # short or the multiplicity undefined).
        "long_psr": long_psr,
        "long_dsr": long_dsr,
        "ls_psr": ls_psr,
        "ls_dsr": ls_dsr,
        # §十二.5: effective (autocorrelation-adjusted) sample sizes behind the
        # PSR/DSR above — the reader can see how much the overlap discount was.
        "long_n_eff": int(long_n_eff),
        "ls_n_eff": int(ls_n_eff),
        "dsr_n_trials": nt,
        "bbmm_stat": bbmm["stat"],
        "bbmm_p_value": bbmm["p_value"],
        "bbmm_n_strategies": bbmm["n_strategies"],
        # The short leg is a theoretical bottom-quantile factor book (A-share
        # stocks cannot be shorted directly) — the exposure metadata keeps the
        # leverage assumption explicit.  The
        # nominal target is only a book construction — the REALIZED daily
        # exposure is measured from the sleeve simulation (unfilled slots stay
        # cash → gross below target; delayed/unresolved sleeves → gross above
        # target), so the metadata reports the measured distribution + full
        # daily ledger instead of a constant.
        "exposure": _ls_exposure_ledger(long_a, short_a),
        # Equal-weight of the ELIGIBLE candidate pool — NOT a
        # full-market/index benchmark.  `selected_universe_ew_sharpe` is the separate
        # selected-universe equal-weight (all fold stocks) proxy.
        "eligible_ew_sharpe": eligible_ew_sharpe,
        "selected_universe_ew_sharpe": selected_universe_ew_sharpe,
        "long_turnover": long_a["turnover"]["daily_avg"],
        "ls_turnover": (long_a["turnover"]["daily_avg"]
                        + short_a["turnover"]["daily_avg"]) / 2.0,
        "ew_turnover": ew_a["turnover"]["daily_avg"],
        "exit_status": long_a["exit_stats"],
        # Per-leg exit status: the long leg alone cannot
        # reveal whether the short book's abnormal returns come from more
        # unfillable or trailing positions — each leg must be auditable on its
        # own.  `exit_status` stays as the long-leg alias for backward compat.
        "long_exit_status": long_a["exit_stats"],
        "short_exit_status": short_a["exit_stats"],
        "eligible_ew_exit_status": ew_a["exit_stats"],
    }
    if return_ledger:
        # §十四-1: the long-only ledger was the only one saved, which hid the
        # short / equal-weight legs' own entry/exit tapes.  Expose each leg's
        # ledger, plus a combined long+short LS tape sorted by (entry_day,
        # stock, mode) — every row already carries its own `mode`, so the
        # concatenation is directly replayable.
        out["long_ledger"] = long_a.get("ledger")
        out["short_ledger"] = short_a.get("ledger")
        out["ew_ledger"] = ew_a.get("ledger")
        out["selected_universe_ew_ledger"] = sel_uni_a.get("ledger")
        ls_ledger = []
        if long_a.get("ledger") is not None:
            ls_ledger.extend(long_a["ledger"])
        if short_a.get("ledger") is not None:
            ls_ledger.extend(short_a["ledger"])
        out["ls_ledger"] = sorted(
            ls_ledger,
            key=lambda r: (r["entry_day"], r["stock"], r["mode"]),
        )
    return out
