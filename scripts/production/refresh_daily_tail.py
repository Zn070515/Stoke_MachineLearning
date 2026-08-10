"""Incrementally refresh the tails of the existing daily K-line store.

The v19 migration rewrote ``data/a_shares/daily/{code}.parquet`` in place
(normalized units + qfq provenance) but never fetched new market data, so most
files are stale (ends scattered across 2026-07-16 / 07-23 / 07-31) while the
market has since traded to 2026-08-07.  The formal data-quality gate refuses
data more than 4 trading days stale, so the panel-feature rebuild that consumes
this store is blocked until the tails are current.

``refresh_daily_tail`` brings every EXISTING daily file's tail current by
fetching ONLY the missing days per stock and merging them through the canonical
writer (``DataStorage.save_daily`` — non-destructive, last-write-wins, atomic
replace, manifest updated, §八-2 price-basis gate enforced).  It is driven by
the ACTUAL files on disk, NOT by ``download_data.py``'s index-member universe
(1579 stocks) and NOT by AKShare's current listing (which would create
truncated files for stocks not yet on disk).

Design notes
------------
- Universe = the ``*.parquet`` files already present; ``--codes`` only narrows
  it, never widens it.  No code is ever added that is not already on disk.
- Per stock: read the manifest ``end``; if ``end >= target_end`` → skip
  (already current).  Else fetch only ``[next_trading_day(end), target_end]``
  — never a full history, however far back ``end`` sits.
- ``target_end`` defaults to the most recent completed trading day (as of
  2026-08-10 that is 2026-08-07).  The current day never counts as complete.
- An empty fetch is a legitimate delisted/suspended outcome → counted as
  ``no_new_data``, NOT a failure.  A worker exception is a per-stock ``failed``
  and never kills the pool.
- Each worker process builds its OWN ``AShareDownloader`` and ``DataStorage``
  (both hold per-process circuit-breaker / connection / lock state and must not
  be shared across processes).

Known limitation: the qfq-anchor seam
-------------------------------------
Each tail fetch is a fresh 前复权 (qfq) batch anchored to the LATEST date, while
the stored history keeps its migrated anchor.  When a corporate action
(dividend / split) occurs INSIDE the stale window, the two anchors differ and a
price-level seam appears exactly at the merge boundary.  ``save_daily``'s §八-2
gate only compares the ``adjust`` LABEL ("qfq") — both sides carry it — so the
seam is not detected.  This is accepted for the unblock: August is low
corporate-action season, and a full re-download would re-anchor the history but
reintroduce the migration risk this incremental refresh exists to avoid.  The
panel feature build that consumes the refreshed daily inherits the same seam, so
the formal gate's return / ``feature_pct`` series stays internally self-
consistent even though cross-window price levels are not perfectly comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
from datetime import date

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.calendar import TradingCalendar, most_recent_completed_trading_day
from stoke_ml.data.download_cli import parse_stock_codes_arg
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.sources.a_shares.failover import AShareDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def manifest_end(data_dir: str, code: str) -> str | None:
    """Best-known last trading day on disk for ``code``.

    The per-stock contract manifest's ``end`` field is authoritative.  When it
    is missing, unparseable (bad JSON) or holds a non-date value, fall back to
    the flat parquet's max ``date`` (the manifest may be absent for a legacy
    file).  Returns ``None`` when neither yields a usable end — the caller then
    cannot compute a tail window and must not fetch a full history.

    The end VALUE is validated (parses as a real date) before it is trusted: a
    manifest that parses as JSON but carries ``"end": "garbage"`` must NOT crash
    the parent process when ``needs_refresh``/``tail_start_of`` parse it — it is
    routed to the parquet fallback and, failing that, to ``unknown``.
    """
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    manifest_path = os.path.join(daily_dir, f"{code}.manifest.json")
    end: str | None = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                end = json.load(f).get("end")
        except (OSError, ValueError):
            end = None
    if end:
        try:
            parsed = pd.Timestamp(str(end))
            if pd.isna(parsed):
                raise ValueError(f"{end!r} parses to NaT")
            return parsed.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001 - non-date manifest end → parquet fallback
            pass
    try:
        dates = pd.read_parquet(
            os.path.join(daily_dir, f"{code}.parquet"), columns=["date"]
        )
        if len(dates):
            return pd.to_datetime(dates["date"]).max().strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 - unreadable file → fall through to None
        pass
    return None


def needs_refresh(end: str | None, target_end: str) -> bool:
    """Whether a stock ending on ``end`` needs a tail refresh to ``target_end``.

    Boundary: ``end == target_end`` → already current → ``False`` (skip);
    ``end`` one trading day before ``target_end`` → ``True`` (fetch the missing
    day).  ``None`` (no known end) always refreshes.
    """
    if end is None:
        return True
    return pd.Timestamp(end).date() < pd.Timestamp(target_end).date()


def tail_start_of(end: str, cal: TradingCalendar) -> str:
    """First trading day strictly after ``end`` — the tail fetch window's start.

    Uses ``cal.next_trading_day`` so a weekend or holiday gap between the last
    stored bar and the first missing bar is skipped, not fetched as a stub.
    """
    return cal.next_trading_day(pd.Timestamp(end).date()).isoformat()


def build_plan(
    codes: list[str],
    data_dir: str,
    target_end: str,
    cal: TradingCalendar,
) -> tuple[list[str], list[dict], list[str]]:
    """Partition on-disk codes into current / to-refresh / unknown.

    Returns ``(current, plan, unknown)`` where each plan item is
    ``{"code": code, "tail_start": "YYYY-MM-DD"}``.  ``unknown`` holds codes
    whose end cannot be determined — they are reported (and later treated as
    failures) rather than risked with a full-history fetch.
    """
    current: list[str] = []
    plan: list[dict] = []
    unknown: list[str] = []
    for code in codes:
        end = manifest_end(data_dir, code)
        if end is None:
            unknown.append(code)
            continue
        if not needs_refresh(end, target_end):
            current.append(code)
            continue
        plan.append({"code": code, "tail_start": tail_start_of(end, cal)})
    return current, plan, unknown


def _process_one(task: tuple) -> dict:
    """Fetch the missing tail for one stock inside a dedicated worker process.

    The downloader and storage are constructed HERE, per process: the failover
    downloader carries per-process circuit-breaker/connection state and the
    storage holds a per-file writer lock, neither of which may be shared across
    processes.  Any exception is converted to a per-stock ``failed`` result so
    a crashing stock never kills the pool.
    """
    code, tail_start, end, data_dir = task
    try:
        downloader = AShareDownloader()
        storage = DataStorage(data_dir)
        df = downloader.fetch_daily(code, tail_start, end)
        if df.empty:
            return {"code": code, "status": "no_new_data", "rows": 0}
        # Canonical non-destructive merge: preserves migrated history, enforces
        # the §八-2 price-basis gate, and stamps the fetched df's attrs.
        storage.save_daily(df)
        dates = pd.to_datetime(df["date"])
        return {
            "code": code,
            "status": "refreshed",
            "rows": int(len(df)),
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
        }
    except Exception as exc:  # noqa: BLE001 - a worker crash is a per-stock failure
        return {"code": code, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bring stale daily K-line tails current (incremental, "
                    "existing files only)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Worker processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the refresh plan counts without fetching")
    parser.add_argument("--codes", type=str, default=None,
                        help="Comma-separated codes to restrict to "
                             "(default: all files on disk)")
    parser.add_argument("--end", type=str, default=None,
                        help="Target end YYYY-MM-DD (default: most recent "
                             "completed trading day)")
    args = parser.parse_args()

    if args.jobs < 1:
        logger.error("--jobs must be >= 1 (got %d)", args.jobs)
        sys.exit(2)

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.isdir(daily_dir):
        logger.error("no daily store at %s — refusing to run", daily_dir)
        sys.exit(2)

    cal = TradingCalendar("a_shares", calendar_dir=data_dir)
    target_end = (
        args.end
        or most_recent_completed_trading_day(cal, date.today()).strftime("%Y-%m-%d")
    )

    # Universe = existing files only.  --codes narrows it, never widens it.
    codes = sorted(
        f[: -len(".parquet")]
        for f in os.listdir(daily_dir) if f.endswith(".parquet")
    )
    if args.codes:
        requested = parse_stock_codes_arg(args.codes)
        codes = [c for c in codes if c in requested]
        for code in sorted(set(requested) - set(codes)):
            logger.warning(
                "--codes %s has no existing daily file — skipped "
                "(only existing files are touched)", code)
    if not codes:
        logger.error("no stock codes to refresh (empty daily store or "
                     "--codes matched nothing)")
        sys.exit(2)

    current, plan, unknown = build_plan(codes, data_dir, target_end, cal)
    logger.info(
        "target_end=%s codes=%d current=%d to_refresh=%d unknown_end=%d",
        target_end, len(codes), len(current), len(plan), len(unknown))
    for code in unknown:
        logger.warning(
            "%s: unknown end (no usable manifest/parquet date) — skipping, "
            "cannot compute a tail window", code)

    if args.dry_run:
        logger.info("DRY RUN: no fetch.  Refresh plan:")
        logger.info("  current (skip): %d", len(current))
        logger.info("  to refresh:     %d", len(plan))
        logger.info("  unknown-end:    %d", len(unknown))
        sys.exit(0)

    tasks = [
        (p["code"], p["tail_start"], target_end, data_dir) for p in plan
    ]
    refreshed: list[dict] = []
    no_new_data: list[str] = []
    failed: list[str] = []
    pool_error = False

    if tasks:
        logger.info("Refreshing %d stocks to %s (%d workers)",
                    len(tasks), target_end, args.jobs)
        with multiprocessing.Pool(args.jobs) as pool:
            try:
                for i, res in enumerate(pool.imap_unordered(_process_one, tasks), 1):
                    status = res["status"]
                    if status == "refreshed":
                        refreshed.append(res)
                        logger.info(
                            "[%d/%d] %s: %d rows [%s → %s]",
                            i, len(tasks), res["code"], res["rows"],
                            res["start"], res["end"])
                    elif status == "no_new_data":
                        no_new_data.append(res["code"])
                        logger.info(
                            "[%d/%d] %s: no new data "
                            "(delisted/suspended over window)",
                            i, len(tasks), res["code"])
                    else:
                        failed.append(res["code"])
                        logger.warning(
                            "[%d/%d] %s: FAILED %s",
                            i, len(tasks), res["code"], res.get("error"))
            except Exception as exc:  # noqa: BLE001 - a dead worker must not crash the run
                pool_error = True
                logger.error("worker pool crashed unexpectedly: %s", exc)

    logger.info(
        "Summary: refreshed=%d skipped_current=%d no_new_data=%d failed=%d",
        len(refreshed), len(current), len(no_new_data), len(failed))
    if failed:
        logger.error("FAILED %d stock(s): %s",
                     len(failed), ", ".join(failed[:20]))
    if unknown:
        logger.error("UNKNOWN-END %d stock(s) not refreshed: %s",
                     len(unknown), ", ".join(unknown[:20]))
    if pool_error or failed or unknown:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
