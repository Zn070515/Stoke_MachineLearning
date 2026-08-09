"""Universe resolution for panel training (§二十一).

Extracted from ``scripts.production.train_panel`` — stock discovery, the
market-authority exchange bucketing, the strict-CSI universe predicates, index
universe loading/resolution, and the ``--universe all`` prebuilt requirement.
``train_panel`` re-exports these names for backward compatibility.
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _discover_stocks(data_dir: str, limit: int | None = None) -> list[str]:
    from stoke_ml.data.storage import DataStorage
    stocks = DataStorage(data_dir).list_stocks()
    return stocks[:limit] if limit else stocks


def _exchange_group(stock_code: str) -> str:
    """Exchange bucket via the single market authority (§六): SH/SZ/BJ.

    The old ``6→SH; 0/3→SZ; else BJ`` heuristic already bucketed the real BSE
    ranges (43/83/87/88/920) into BJ, but it re-derived the market locally.
    Route through market_of_code so every caller shares one authority.
    """
    from stoke_ml.data.codes import market_of_code

    market = market_of_code(stock_code)
    # The panel daily store only holds A-share common equity, so market is
    # always SH/SZ/BJ.  market_of_code returns None for anything that is not a
    # known equity prefix (incl. legacy 老三板 4/8 codes and garbage input); the
    # fallback keeps that in the BJ bucket rather than KeyError-ing into
    # by_group or silently mis-bucketing it as SH.
    if market is not None:
        return market
    return "BJ"


def _is_csi_universe(universe: str) -> bool:
    """§T6 decision 2: the universe is one of the strict-CSI index universes."""
    return universe in ("csi300", "csi500", "csi800")


def _strict_index_training_effective(cli_value: bool | None, universe: str) -> bool:
    """§T6 decision 2: resolve the strict inner-train membership gate.

    ``None`` (the new default) means "decide from the universe": ON for the
    strict-CSI universes (csi300/csi500/csi800), OFF otherwise.  An explicit
    ``--strict-index-training`` / ``--no-strict-index-training`` forces the
    value regardless of universe.
    """
    if cli_value is not None:
        return bool(cli_value)
    return _is_csi_universe(universe)


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

def _resolve_universe(
    all_stocks: list[str],
    universe: str,
    limit: int | None,
    seed: int,
    data_dir: str,
    formal: bool = False,
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

    §八-2: index membership is never silently truncated by missing data.
    csi* members with no daily K-line on disk are counted and surfaced.  When
    universe reconciliation is ENFORCED (the quality gate is required — a run
    whose gate report must reconcile the requested universe, §八-2) the run
    refuses to start: the dropped members are listed, so the gap is visible
    instead of silently shrinking the index to what happened to be downloaded.
    When reconciliation is not enforced (exploratory / --no-formal / dev smoke
    with --no-require-quality-gate), the run degrades to a prominent warning
    and records the drop count in the description so the artifact still
    captures what was dropped.

    ``formal`` is the gate-enforcement predicate (:func:`_gate_enforced`), not
    merely ``--no-formal`` — the refusal must key on whether the run is
    actually required to reconcile the requested universe.
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
        # §八-2: keep only members we have daily K-line for, but NEVER silently.
        # The membership file includes codes with no data (delisted / not
        # downloaded); a formal run refuses so the index is not silently shrunk
        # to whatever was downloaded, and exploratory runs record the drop so
        # the artifact still exposes it.
        have = set(all_stocks)
        dropped = sorted(c for c in members if c not in have)
        members = [c for c in members if c in have]
        if dropped:
            listing = ", ".join(dropped[:20]) + (" ..." if len(dropped) > 20 else "")
            if formal:
                raise SystemExit(
                    f"universe={universe}: {len(dropped)} index members have no "
                    f"daily K-line on disk and would be silently dropped — "
                    f"refusing to run because universe reconciliation is "
                    f"enforced: the run must NOT silently shrink the requested "
                    f"index (§八-2).  Missing: {listing}.  Download the missing "
                    f"members or re-run with --no-require-quality-gate to "
                    f"degrade explicitly and record the drop."
                )
            logger.warning(
                "universe=%s: dropped %d/%d index members with no daily K-line "
                "on disk (recorded in description): %s",
                universe, len(dropped), len(dropped) + len(members), listing,
            )
        chosen = members[:limit] if limit else members
        note = f" (PIT union, cap={limit})"
        if dropped and not formal:
            note += f", dropped {len(dropped)} no-data members"
        return chosen, f"{universe} {len(chosen)}{note}"
    raise ValueError(f"unknown --universe: {universe}")

_PREBUILT_MAINLINE_THRESHOLD = 1000


def _require_prebuilt_mainline(
    universe: str,
    prebuilt: str | None,
    store_complete: bool = False,
    n_resolved: int = 0,
    formal_research: bool = False,
    threshold: int = _PREBUILT_MAINLINE_THRESHOLD,
) -> None:
    """§七-P0 / §v19 P0#3: refuse a run without the prebuilt feature mainline.

    Two refusal branches:
      * ``universe == "all"`` without ``--prebuilt`` / a complete
        ``--panel-store`` is refused outright — the full market cannot be
        feature-engineered in RAM (~225GB of feature arrays on a ~96GB host),
        regardless of mode (§七-P0).
      * ANY FORMAL-RESEARCH run without ``--prebuilt`` / a complete store is
        refused — live feature engineering is reserved for debug / smoke /
        exploratory runs; formal named-profile research uses the prebuilt
        mainline at any universe size (§v19 P0#3).
    ``threshold`` / ``n_resolved`` are retained only for signature
    compatibility with the call site in ``train_panel.py`` (which passes
    ``n_resolved``); the formal branch no longer consults them — the
    count-based escape hatch is gone.
    A complete ``--panel-store`` is an equivalent source in both cases (the
    panel was already built and persisted as mmap'd arrays).
    """
    if universe == "all" and not prebuilt and not store_complete:
        raise SystemExit(
            "--universe all requires --prebuilt (or a complete --panel-store): "
            "the full market cannot be feature-engineered in memory (5530 "
            "stocks x the full feature grid ~= 225GB on a ~96GB host, §七-P0).  "
            "Run scripts/production/build_features.py --panel-mode first, then "
            "re-run with --prebuilt data/features_panel, or point --panel-store "
            "at a previously-built store."
        )
    if formal_research and not prebuilt and not store_complete:
        raise SystemExit(
            f"formal research requires the prebuilt feature mainline: live "
            f"feature engineering is reserved for debug / smoke / exploratory "
            f"runs (§v19 P0#3).  Run "
            f"scripts/production/build_features.py --panel-mode to build "
            f"data/features_panel once, then re-run with --prebuilt "
            f"data/features_panel (or point --panel-store at a previously-built "
            f"complete store).  To run live anyway, pass --no-formal "
            f"(exploratory) or --no-require-quality-gate (dev smoke) explicitly."
        )
