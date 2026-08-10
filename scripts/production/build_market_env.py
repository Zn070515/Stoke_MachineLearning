#!/usr/bin/env python
"""Formal production builder for the market_env (市场环境) broadcast channel.

SUPERSEDES the archived ``scripts/maintenance/legacy/_preprocess_market_env.py``
(§八/T5): the legacy script self-labels ARCHIVED / NOT part of the canonical
pipeline yet still produced the headline ``a_shares/market_breadth/
market_env_daily.parquet`` file, and it mapped the monthly account stats'
data-MONTH label ("2015-04") straight to a proxy day ("2015-04-28") with no
honest PIT declaration.  This module is the canonical replacement.

Consumer-facing shape is UNCHANGED (backward compat): the output is the same
single ``market_env_daily.parquet`` on the same path with the same 7 columns
over a DatetimeIndex named ``date`` — exactly what ``aux_aligner._merge_market_env``,
``train_panel_panel``'s broadcast probe + formal manifest gate, and
``cache_manifest.SOURCE_SUBDIRS`` already read.  What is NEW is the honest
price/account split:

  - **price part** (``high_low_ratio`` / ``market_adv_ratio`` /
    ``market_turnover_z``): a COMPOSITE — ``high_low_ratio`` and
    ``market_turnover_z`` are same-day trade data (verified by nature), but
    ``market_adv_ratio`` inherits its PIT from ``industry_ranking``.  The part's
    ``pit_alignment`` is therefore the WEAKEST constituent: ``"verified"`` ONLY
    when ``industry_advance_pit == "verified"``, otherwise ``"proxy"`` (§二十: a
    proxy constituent must never be smuggled inside a "verified" part).
    ``market_adv_ratio`` is the CSRC broad-sector advance ratio (fraction of
    证监会 门类 sectors A–S with positive equal-weighted return) derived from the
    PIT ``industry_ranking``; its per-source PIT is recorded in ``parts`` as
    ``industry_advance_pit`` (``"verified"`` for the PIT membership path,
    ``"proxy"`` only when the legacy snapshot fallback was forced).
  - **account part** (``mkt_cap_total_z`` / ``avg_account_cap_z`` /
    ``investor_new_num`` / ``investor_new_z``): monthly account statistics
    whose real publish date is NOT determinable from the shipped
    ``account_stats.parquet`` (only the data-month label) → ``"proxy"``; if the
    raw source ever records a real publish-date column, it is used and the part
    is ``"verified"``.

Because one file carries both parts, the channel-level manifest label is the
STRICTER of the two (``vintage_pit="proxy"``, drawn from channel_vintage's
``market_env`` entry); the finer per-part labels are recorded in the manifest's
``parts`` field and in ``stoke_ml/config/feature_profile.py``'s
``MARKET_ENV_PRICE_COLS`` / ``MARKET_ENV_ACCOUNT_COLS`` split.  The critical
invariant (§八): revision-safe formal training must never SILENTLY train on
account data whose PIT is unverified — the proxy-ness is declared in three
places (manifest, channel_vintage, feature_profile), never implied-verified.

The manifest is written via ``broadcast_assets.MARKET_ENV_ASSET`` with the same
``write_asset_manifest``/``check_asset_read`` mechanism broadcast_assets'
industry asset uses, so the §T4 formal gate
(``train_panel_panel._enforce_formal_manifests``) validates it unchanged.

Run:
    PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_market_env.py
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

from scripts.production.data_quality_gate import dataset_fingerprint
from scripts.production.download_industry_ranking import INDUSTRY_RANKING_ASSET
from scripts.production.download_market_breadth import HIGH_LOWS_ASSET
from stoke_ml.config import load_config
from stoke_ml.config.feature_profile import (
    MARKET_ENV_ACCOUNT_COLS,
    MARKET_ENV_PRICE_COLS,
)
from stoke_ml.data.asset_contract import (
    AtomicCommit,
    check_asset_read,
    validate_asset_manifest,
    write_asset_manifest,
)
from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET
from stoke_ml.features.aux_cols import MARKET_ENV_COLS
from stoke_ml.models.panel.code_tree_hash import hash_json

# The exact consumer-facing column order (matches aux_cols.MARKET_ENV_COLS).
OUTPUT_COLUMNS: tuple[str, ...] = tuple(MARKET_ENV_COLS)

# Real publish-date column candidates for the monthly account part.  The shipped
# account_stats.parquet has NONE (only the data-month label), so the account part
# is PROXY until a future source records the actual publication day.
_PUBLISH_DATE_COLS: tuple[str, ...] = (
    "发布日期", "公布日期", "披露日期", "publish_date",
)


def _z(s: pd.Series, win: int = 20) -> pd.Series:
    """20-window rolling z-score, NaN→0 (identical to the legacy formula)."""
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0)


def build_turnover_daily(base: str) -> pd.Series:
    """Sum 'amount' across all daily flat files per date -> z-scored turnover.

    ``base`` is the ``a_shares`` dir (``data_dir/a_shares``).  Same-day trade
    data, so this is the VERIFIED price part.  §v19 P0#4: a formal builder must
    NEVER silently skip an unreadable / un-manifested daily file — every daily
    parquet is read through ``DataStorage``'s canonical validated path
    (``require_valid_manifest=True``), and ANY file that fails validation aborts
    the whole build (a missing stock would silently change market turnover).
    """
    from stoke_ml.data.storage import DataStorage
    storage = DataStorage(os.path.dirname(base))
    codes = storage.list_stocks("a_shares")
    problems: list[str] = []
    amounts: list[pd.Series] = []
    for code in codes:
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
        except (ValueError, OSError) as exc:
            problems.append(f"{code}: {exc}")
            continue
        if d.empty:
            continue
        amounts.append(d.groupby(d["date"])["amount"].sum())
    if problems:
        raise SystemExit(
            f"build_market_env: {len(problems)}/{len(codes)} daily files "
            "FAILED canonical validation — refusing to build a turnover "
            "series over incomplete inputs (§v19 P0#4):\n  "
            + "\n  ".join(problems[:20]))
    if not amounts:
        return pd.Series(dtype="float64")
    tot = pd.concat(amounts).groupby(level=0).sum()
    return _z(tot)


def build_industry_advance(base: str) -> tuple[pd.Series, str]:
    """(Fraction of sectors with positive change_pct per date, pit_alignment).

    ``base`` is the ``a_shares`` dir.  The real upstream is
    ``download_industry_ranking.py``'s ``a_shares/industry_ranking.parquet``
    (date/sector_code/change_pct — sector equal-weighted return), NOT the
    legacy ``industry/industry_ranking_computed.parquet`` (date/ind_return)
    that no production pipeline writes.  §二十: a PRESENT ranking must first
    pass its own INDUSTRY_RANKING_ASSET manifest check
    (``require_valid_manifest=True``, fail-closed) — a bare or tampered
    parquet is never silently trusted as market breadth.  The returned
    ``pit_alignment`` is read from the manifest's ``pit_alignment`` (``"verified"``
    for the PIT sector-membership path, ``"proxy"`` when the legacy snapshot
    fallback was forced); a present manifest that carries NO explicit
    ``pit_alignment`` defaults conservatively to ``"proxy"`` — a proxy pit must
    never silently upgrade to the strict headline.  The absent-file case
    returns ``(empty, "verified")`` because a missing ranking is not a manifest
    problem — the required-price-column assertion downstream handles absence.
    """
    path = os.path.join(base, "industry_ranking.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype="float64"), "verified"
    raw = pd.read_parquet(path)
    check_asset_read(path, INDUSTRY_RANKING_ASSET, raw, require_valid_manifest=True)
    if "date" not in raw.columns or "change_pct" not in raw.columns:
        return pd.Series(dtype="float64"), "verified"
    d = raw[["date", "change_pct"]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.dropna(subset=["change_pct"])
    if d.empty:
        return pd.Series(dtype="float64"), "verified"
    adv = d.groupby("date")["change_pct"].apply(
        lambda x: float((x > 0).mean()))
    report = validate_asset_manifest(path, INDUSTRY_RANKING_ASSET, df=raw)
    # conservative: a manifest with no explicit pit_alignment is treated as
    # proxy — a proxy pit must never silently upgrade to the strict headline
    # (§v19 §二十).
    pit = (report["manifest"] or {}).get("pit_alignment") or "proxy"
    return adv.rename("market_adv_ratio"), pit


def _resolve_account_dates(acc: pd.DataFrame) -> tuple[pd.Series, str]:
    """(effective-date Series, pit_alignment) for the monthly account part.

    Uses a REAL publish-date column when the source records one (→ ``verified``);
    otherwise falls back to the legacy month-end proxy "2015-04"→"2015-04-28"
    and declares ``proxy`` — the honest label for an undeterminable publish day.
    """
    for col in _PUBLISH_DATE_COLS:
        if col in acc.columns:
            return pd.to_datetime(acc[col], errors="coerce"), "verified"
    if "数据日期" not in acc.columns:
        # No publish-date column AND no month-label column: the source schema is
        # not what this builder expects.  Degrade to all-NaT dates → the caller's
        # dropna produces an empty account part (proxy) — NEVER raise KeyError.
        return pd.Series(pd.NaT, index=acc.index), "proxy"
    return pd.to_datetime(acc["数据日期"].astype(str) + "-28", errors="coerce"), "proxy"


def build_account_part(data_dir: str) -> tuple[pd.DataFrame, str]:
    """The ACCOUNT part (monthly investor/cap stats) as a DatetimeIndexed frame.

    Returns ``(df, pit_alignment)`` — ``df`` empty when account_stats is absent
    or lacks the expected columns (the caller then contributes no account
    columns; the consumer's ``[c for c in MARKET_ENV_COLS if c in cols]`` merge
    degrades gracefully).
    """
    br = os.path.join(data_dir, "a_shares", "market_breadth")
    path = os.path.join(br, "account_stats.parquet")
    if not os.path.isfile(path):
        return pd.DataFrame(), "proxy"
    acc = pd.read_parquet(path)
    dates, pit = _resolve_account_dates(acc)
    acc = acc.assign(_date=dates).dropna(subset=["_date"])
    if acc.empty:
        return pd.DataFrame(), pit
    acc = acc.set_index("_date").sort_index()
    acc = acc.rename(columns={
        "新增投资者-数量": "investor_new_num",
        "沪深总市值": "mkt_cap_total",
        "沪深户均市值": "avg_account_cap",
    })
    raw_cols = ["investor_new_num", "mkt_cap_total", "avg_account_cap"]
    if not all(c in acc.columns for c in raw_cols):
        return pd.DataFrame(), pit
    acc_raw = acc[raw_cols].resample("D").ffill()
    acc_z = acc[raw_cols].apply(_z).resample("D").ffill()
    return pd.DataFrame({
        "mkt_cap_total_z": acc_z["mkt_cap_total"],
        "avg_account_cap_z": acc_z["avg_account_cap"],
        "investor_new_num": acc_raw["investor_new_num"],
        "investor_new_z": acc_z["investor_new_num"],
    }), pit


def build_market_env(data_dir: str) -> tuple[pd.DataFrame, dict]:
    """Assemble the market_env daily panel WITHOUT writing.

    Args:
        data_dir: the project data root (contains ``a_shares/``).

    Returns:
        (df, parts) — ``df`` the 7-column DatetimeIndexed panel; ``parts`` the
        honest price/account split declaration consumed by ``write_market_env``.
    """
    base = os.path.join(data_dir, "a_shares")
    series: dict[str, pd.Series] = {}

    # price part — COMPOSITE: high_low_ratio + market_turnover_z are same-day
    # trade data (verified by nature), but market_adv_ratio inherits its PIT
    # from industry_ranking; the part's pit_alignment is the WEAKEST constituent
    hl = _build_high_low_ratio(base)
    if not hl.empty:
        series["high_low_ratio"] = hl
    adv, adv_pit = build_industry_advance(base)
    if not adv.empty:
        series["market_adv_ratio"] = adv
    turn = build_turnover_daily(base)
    if not turn.empty:
        series["market_turnover_z"] = turn

    # account part — PROXY unless a real publish date is recorded
    acc_part, account_pit = build_account_part(data_dir)
    for c in MARKET_ENV_ACCOUNT_COLS:
        if c in acc_part.columns:
            series[c] = acc_part[c]

    out = pd.DataFrame(series).sort_index()
    # §v18-5: the required PRICE part is atomic — a build that cannot produce
    # every MARKET_ENV_PRICE_COLS column must FAIL, never write a partial asset
    # that still earns a valid manifest (the §二十-5 failure mode).  The account
    # part stays optional (ablation-only).
    missing_price = sorted(MARKET_ENV_PRICE_COLS - set(out.columns))
    if missing_price:
        raise ValueError(
            "build_market_env: market_env PRICE columns missing: "
            f"{missing_price} — the market_env required channel must carry the "
            f"full {sorted(MARKET_ENV_PRICE_COLS)} set.  Fix the upstream "
            "sources (highs_lows.parquet / industry_ranking.parquet / daily).")
    # State-channel honesty (§九-4): NEVER zero-fill a missing observation.
    #   * ACCOUNT part — a missing monthly value means "unchanged", not zero, so
    #     forward-fill to the end of the panel (a stale carry is then flagged by
    #     the consumer's {prefix}_staleness_days rather than read as a real 0).
    #   * PRICE part — same-day trade data; a date one source lacks but another
    #     covers stays NaN so the consumer's _batch_fill_shift (policy="ffill")
    #     carries the last breadth.  A hard-coded 0 here would bypass that ffill
    #     and inject a fake "zero breadth / zero new-investor" day.
    for c in MARKET_ENV_ACCOUNT_COLS:
        if c in out.columns:
            out[c] = out[c].ffill()
    out.index.name = "date"
    out.attrs["source"] = ("derived market-breadth panel "
                           "(build_market_env.py: account_stats/highs_lows/"
                           "industry advance/turnover)")
    # The price part is a COMPOSITE: high_low_ratio + market_turnover_z are
    # same-day trade data (verified by nature), but market_adv_ratio inherits
    # its PIT from industry_ranking — so the part's pit_alignment is the
    # WEAKEST constituent: verified ONLY when adv_pit == "verified", else proxy
    # (§二十: a proxy constituent must never be smuggled inside a "verified"
    # part).
    price_pit = "verified" if adv_pit == "verified" else "proxy"
    parts = {
        "price": {
            "columns": sorted(MARKET_ENV_PRICE_COLS),
            "pit_alignment": price_pit,
            "industry_advance_pit": adv_pit,
            "note": "same-day trade data (high/low breadth, market turnover) "
                    "plus market_adv_ratio — the CSRC broad-sector advance "
                    "ratio (fraction of 证监会 门类 sectors A–S with positive "
                    "equal-weighted return) derived from industry_ranking. "
                    "This part is a COMPOSITE: pit_alignment inherits the "
                    "WEAKEST constituent, so it is 'verified' ONLY when "
                    "industry_advance_pit == 'verified', otherwise 'proxy' "
                    "(§二十)",
        },
        "account": {
            "columns": sorted(MARKET_ENV_ACCOUNT_COLS),
            "pit_alignment": account_pit,
            "note": ("monthly account statistics — real publish date NOT "
                     "determinable from account_stats.parquet (only the "
                     "data-month label); month-end proxy date used"
                     if account_pit == "proxy"
                     else "monthly account statistics — real publish date "
                          "recorded in the raw source"),
        },
    }
    return out, parts


def _build_high_low_ratio(base: str) -> pd.Series:
    """high20/(high20+low20) breadth from highs_lows.parquet (VERIFIED part).

    §v19: a PRESENT highs_lows must first pass its own HIGH_LOWS_ASSET manifest
    check (``require_valid_manifest=True``, fail-closed) — a bare or tampered
    parquet is never silently trusted as market breadth.  The absent-file case
    returns empty (a missing file is not a manifest problem; the
    required-price-column assertion downstream handles absence), matching the
    ``build_industry_advance`` convention.
    """
    path = os.path.join(base, "market_breadth", "highs_lows.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype="float64")
    hl = pd.read_parquet(path)
    check_asset_read(path, HIGH_LOWS_ASSET, hl, require_valid_manifest=True)
    if "date" not in hl.columns or "high20" not in hl.columns or "low20" not in hl.columns:
        return pd.Series(dtype="float64")
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    hl = hl.set_index("date").sort_index()
    return (hl["high20"] / (hl["high20"] + hl["low20"]).replace(0, np.nan)).rename(
        "high_low_ratio")


def _file_sha256(path: str) -> str:
    # Deliberately differs from code_tree_hash._file_digest (None-vs-marker
    # convention): this builder records an explicit marker string so the manifest
    # distinguishes a healthy-but-absent optional input (<missing>) from a
    # present-but-unreadable file (<unreadable>) — see _upstream_roots.
    if not os.path.isfile(path):
        return "<missing>"
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "<unreadable>"


def _upstream_roots(data_dir: str) -> dict:
    """The exact input assets this market_env build derives from (§v18-7).

    ``daily`` is the dataset fingerprint's MANIFEST-ROOT Merkle digest over the
    per-stock ``daily/*.manifest.json`` sidecars (§v18-8) — the manifest, bound
    at write time, is the daily data's identity; the three single-file upstreams
    are streamed SHA-256 digests of the exact file the builder reads.  A
    present-but-unreadable input
    records ``"<unreadable>"`` and a healthy-but-absent optional input records
    ``"<missing>"``, so a downstream freshness check sees the root change
    rather than silently treating a broken OR absent input as fresh — without
    conflating the two.
    """
    base = os.path.join(data_dir, "a_shares")
    return {
        "daily": dataset_fingerprint(data_dir, ["daily"]),
        "highs_lows": _file_sha256(
            os.path.join(base, "market_breadth", "highs_lows.parquet")),
        "industry_ranking": _file_sha256(
            os.path.join(base, "industry_ranking.parquet")),
        "account_stats": _file_sha256(
            os.path.join(base, "market_breadth", "account_stats.parquet")),
    }


def _transform_code_hash() -> str:
    """Content hash of THIS builder's source — the transform identity (§v18-7).

    Any edit to this file flips the digest, so a downstream formal gate can tell
    whether an on-disk market_env was built by the code that is on disk RIGHT
    NOW.  Hashing this module's own bytes (not the whole tree) keeps the lineage
    scoped to the exact transform that produced the asset.
    """
    return _file_sha256(os.path.abspath(__file__))


def _transform_config_hash(parts: dict) -> str:
    """Config identity of a market_env build (§v18-7): output columns + parts."""
    return hash_json({
        "output_columns": sorted(OUTPUT_COLUMNS),
        "parts": parts,
    })


def compute_lineage(data_dir: str, parts: dict) -> dict:
    """The §v19 P0#2 derivation lineage, recomputable at read time.

    ``write_market_env`` records this at write time; the formal gate
    (``train_panel_panel._enforce_formal_manifests``) recomputes it from the
    CURRENT on-disk upstreams / this builder's current source / the recorded
    ``parts`` and compares via :func:`asset_contract.validate_derived_asset`.
    """
    return {
        "upstream_roots": _upstream_roots(data_dir),
        "transform_code_hash": _transform_code_hash(),
        "transform_config_hash": _transform_config_hash(parts),
    }


def write_market_env(data_dir: str, df: pd.DataFrame, parts: dict) -> str:
    """Atomically write the parquet + MARKET_ENV_ASSET manifest, then self-check.

    The manifest carries the channel-level vintage (``vintage_pit="proxy"`` —
    the STRICTER of the two parts, drawn from channel_vintage's ``market_env``
    entry), the per-part ``parts`` declaration, AND the §v18-7 DERIVATION
    LINEAGE: ``upstream_roots`` (content hash of every input asset the build
    read), ``transform_code_hash`` (this builder's own source), and
    ``transform_config_hash`` (the build's config inputs).  A formal read
    (``require_valid_manifest=True``) of the re-read file must pass, proving the
    schema_hash survives the parquet round-trip.
    """
    br = os.path.join(data_dir, "a_shares", "market_breadth")
    os.makedirs(br, exist_ok=True)
    out_path = os.path.join(br, "market_env_daily.parquet")
    with AtomicCommit(out_path) as ac:
        df.to_parquet(ac.tmp_path, compression="lz4")
    write_asset_manifest(
        out_path, MARKET_ENV_ASSET, df, parts=parts,
        **compute_lineage(data_dir, parts),
    )
    reread = pd.read_parquet(out_path)
    check_asset_read(out_path, MARKET_ENV_ASSET, reread, require_valid_manifest=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None,
                    help="project data root (default: resolved from config.yaml)")
    args = ap.parse_args()

    if args.data_dir is None:
        cfg = load_config()
        data_dir = cfg.project.data_dir
    else:
        data_dir = os.path.abspath(args.data_dir)

    df, parts = build_market_env(data_dir)
    if df.empty or not OUTPUT_COLUMNS:
        # A formal builder must fail loudly rather than write an empty
        # market_env_daily.parquet + manifest (a silent rows:0 asset would pass
        # the manifest gate yet carry no breadth signal — the §八 failure mode).
        raise SystemExit(
            "build_market_env: produced an EMPTY panel — no usable input data "
            f"under {data_dir}/a_shares (account_stats / highs_lows / "
            "industry_ranking / daily). Refusing to write an empty "
            "market_env_daily.parquet.")
    out_path = write_market_env(data_dir, df, parts)
    print(f"market_env_daily: {len(df)} dates "
          f"({df.index.min().date()} ~ {df.index.max().date()}), "
          f"{len(df.columns)} cols "
          f"(price={parts['price']['pit_alignment']}, "
          f"account={parts['account']['pit_alignment']})")
    print(f"manifest: {out_path}.manifest.json")


if __name__ == "__main__":
    main()
