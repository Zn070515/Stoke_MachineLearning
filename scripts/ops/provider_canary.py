"""Provider online canary (§十二-2) — periodic, NON-CI upstream drift detection.

The A-share test suite uses hand-recorded/constructed response fixtures, which
only prove "the parser is correct for THIS fixture".  A provider can change its
real field names or units overnight and every fixture still passes.  This canary
fetches 1–2 FIXED stocks from each K-line provider, fingerprints the returned
schema, and compares it against the frozen ``daily_equity``
(``research_qfq_daily``) contract.

It is meant to run on a SCHEDULE outside CI (cron-friendly: clear exit code, no
interactive prompts), not in the PR pipeline.

Design:
  * Module import + function definitions never touch the network.  The provider
    classes import their online deps lazily (T11 §十二), so importing and
    constructing them offline is safe.
  * ``fingerprint_frame`` — deterministic schema fingerprint: sorted column
    names with per-column canonical dtype tags, provenance attrs
    (source/adjustment_mode) when present, and the row count.
  * ``check_provider`` — runs the frozen contract's validators and returns
    ``(fingerprint, issues)``; drift == non-empty issues.
  * ``main`` — probes each provider, SKIPS unavailable ones (never fails on
    them, e.g. Tushare without a token), writes one JSON snapshot per probe to
    ``--state-dir``, prints a PASS/FAIL line per probe, and exits 0 iff every
    AVAILABLE provider's probes pass.  A daemon watchdog armed just before the
    probe loop force-exits with code 2 if a fetch hangs past ``--timeout``.

A pytest with a MOCKED provider validates the fingerprint/check logic with NO
network (see tests/scripts/test_provider_canary.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import pandas as pd

from stoke_ml.data.contract import (
    get_contract,
    validate_finite,
    validate_price_volume_amount_consistency,
    validate_schema,
    validate_source_metadata,
    validate_units,
)

# ── fixed probe configuration ────────────────────────────────────────────

# One SH + one SZ stock, so every exchange the providers route is probed.
PROBE_STOCKS = ("600519", "000001")
WINDOW_DAYS = 60  # calendar-day fetch window ending today
CONTRACT_NAME = "daily_equity"
# Snapshots live outside ``data/`` (no collision with tracked data) under the
# untracked ``reports/`` tree; override with ``--state-dir``.
DEFAULT_STATE_DIR = Path("reports") / "provider_canary"
# Hard kill-switch: a hung ``fetch_daily`` (WAF/network stall) must not leave a
# cron run stuck forever, so the probe loop runs under a daemon watchdog that
# force-exits the process (exit 2 = alert) after this many seconds.
DEFAULT_TIMEOUT = 300


# ── schema fingerprint ───────────────────────────────────────────────────

def _canonical_dtype(dtype) -> str:
    """Dtype tag stable across parquet's harmless dtype round-trips (mirrors
    ``stoke_ml.data.asset_contract._canonical_dtype``): every datetime64
    resolution → ``"datetime64"``, every string-ish (object / pandas str) →
    ``"string"``, numeric keeps its dtype.  A real column-type drift
    (string → number) still changes the tag.
    """
    if dtype.kind == "M":
        return "datetime64"
    if dtype.kind == "O" or str(dtype).startswith("string"):
        return "string"
    return str(dtype)


def fingerprint_frame(df: pd.DataFrame) -> dict:
    """Stable, deterministic schema fingerprint of one fetched frame.

    JSON-serializable: per-column canonical dtype tags (column names sorted),
    provenance ``attrs`` (``source`` / ``adjustment_mode``) when present, and
    the row count.  Deterministic — fixed key set, sorted columns.
    """
    columns = {
        name: _canonical_dtype(df[name].dtype)
        for name in sorted(df.columns)
    }
    fp: dict = {"columns": columns, "row_count": int(len(df))}
    if df.attrs.get("source") is not None:
        fp["source"] = str(df.attrs["source"])
    if df.attrs.get("adjustment_mode") is not None:
        fp["adjustment_mode"] = str(df.attrs["adjustment_mode"])
    return fp


# ── contract comparison ─────────────────────────────────────────────────

def check_provider(
    df: pd.DataFrame, contract
) -> tuple[dict, list[str]]:
    """Return ``(fingerprint, issues)`` for one provider frame vs the contract.

    Issues come from the frozen contract's validators: schema (missing required
    columns), units (sign/range), finite (non-finite OHLCV / too few rows),
    source metadata (empty ``source`` column / illegal ``adjustment_mode``).
    The price-volume-amount consistency diagnostic is also included because the
    canary's whole point is catching unit-MAGNITUDE drift (手 vs 股, 千元 vs 元)
    that the sign-only ``validate_units`` cannot see — its loose 100× band flags
    the classic volume/amount scale corruptions.  Drift == non-empty issues.
    """
    issues: list[str] = []
    issues += validate_schema(df, contract)
    issues += validate_units(df, contract)
    issues += validate_finite(df, contract)
    issues += validate_source_metadata(df, contract)
    issues += validate_price_volume_amount_consistency(df, contract)
    return fingerprint_frame(df), issues


# ── orchestration ───────────────────────────────────────────────────────

def _build_providers():
    """Instantiate the fixed provider chain.  Imports happen lazily so module
    import / function definition stays offline; a provider whose online dep is
    missing reports ``is_available()`` False and is skipped by ``main``."""
    from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
    from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
    from stoke_ml.data.sources.a_shares.tushare_source import TushareSource
    from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource

    return [
        EfinanceSource(),
        AKShareSource(),
        TushareSource(),
        BaostockSource(),
    ]


def _recent_window(window_days: int) -> tuple[str, str]:
    """ISO ``(start, end)`` covering the last ``window_days`` calendar days."""
    today = pd.Timestamp.now().normalize().date()
    start = today - pd.Timedelta(days=window_days)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _positive_int(value: str) -> int:
    """argparse type: a positive integer (rejects a vacuous 0-day window)."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    """Run the canary; return 0 iff every AVAILABLE provider's probes pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--state-dir", type=Path, default=DEFAULT_STATE_DIR,
        help=f"where per-probe JSON snapshots are written (default: %(default)s)",
    )
    parser.add_argument(
        "--window-days", type=_positive_int, default=WINDOW_DAYS,
        help="calendar-day fetch window ending today (>= 1)",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="hard kill-switch seconds for a hung fetch; force-exits with code "
             "2 (0 disables the watchdog)",
    )
    parser.add_argument(
        "--stocks", nargs="+", default=list(PROBE_STOCKS),
        help="fixed probe stock codes (default: 600519 000001)",
    )
    args = parser.parse_args(argv)

    contract = get_contract(CONTRACT_NAME)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    start, end = _recent_window(args.window_days)

    # Hard kill-switch: a hung fetch_daily (WAF/network stall) must not leave
    # the cron run stuck silently.  Mirrors the repo's established pattern
    # (CLAUDE.md Known Issues: threading.Timer + os._exit for WAF hang).  The
    # timer is CANCELLED on every normal return path (finally below), so a run
    # that finishes in time can never later fire os._exit(2) — a genuinely hung
    # probe still trips it before the finally runs.
    watchdog = (
        threading.Timer(args.timeout, lambda: os._exit(2))
        if args.timeout > 0
        else None
    )
    if watchdog is not None:
        watchdog.daemon = True
        watchdog.start()

    all_passed = True
    available = 0
    passed = 0
    try:
        for provider in _build_providers():
            name = provider.SOURCE_NAME
            try:
                if not provider.is_available():
                    print(f"[SKIP] {name}: not available (offline or missing token)")
                    continue
            except Exception as e:  # never let one provider's probe kill the run
                print(f"[SKIP] {name}: is_available() raised: {e!r}")
                continue
            available += 1
            for stock in args.stocks:
                try:
                    df = provider.fetch_daily(stock, start, end)
                except Exception as e:  # provider blew up — distinct triage label
                    print(f"[FAIL] {name} {stock}: exception during fetch: {e!r}")
                    all_passed = False
                    continue
                try:
                    fingerprint, issues = check_provider(df, contract)
                except Exception as e:  # canary's OWN validation bug — separate label
                    print(f"[FAIL] {name} {stock}: validation raised: {e!r}")
                    all_passed = False
                    continue
                snapshot = {
                    "contract": contract.dataset_name,
                    "source": name,
                    "stock": stock,
                    "start": start,
                    "end": end,
                    "fetched_at": pd.Timestamp.now().isoformat(),
                    "rows": fingerprint["row_count"],
                    "fingerprint": fingerprint,
                    "issues": issues,
                    "passed": not issues,
                }
                out = state_dir / f"{name}__{stock}.json"
                out.write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if issues:
                    all_passed = False
                    detail = issues[0]
                    if len(issues) > 1:
                        detail += f" (+{len(issues) - 1} more)"
                    print(f"[FAIL] {name} {stock}: {detail}")
                else:
                    passed += 1
                    print(
                        f"[PASS] {name} {stock}: {fingerprint['row_count']} rows, "
                        f"{len(fingerprint['columns'])} cols"
                    )

        print(
            f"[SUMMARY] providers_available={available} probes_passed={passed} "
            f"status={'OK' if all_passed else 'DRIFT'}"
        )
    finally:
        if watchdog is not None:
            watchdog.cancel()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
