"""Download daily data for A-share stock universe via 4-source failover."""
import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta

import akshare as ak
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data import universe
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.codes import normalize_stock_code_series
from stoke_ml.data.download_manifest import default_path, write_manifest
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.utils.error_summary import classify_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_stock_codes(
    data_dir: str | None = None, indices: list[str] | None = None
) -> list[str]:
    """Stock codes for the given index symbols.

    The DEFAULT index universe is the HISTORICAL member union from
    ``membership.parquet`` (§七-2) — every stock that ever belonged to the index
    — NOT today's current constituents.  Downloading only current constituents
    turns the 2000-2026 backtest into a survivorship-adjacent "stocks that are
    in the index today" study (a stock that left CSI300 in 2019 still has
    legitimate 2015-2019 index-universe history).

    Indices with no historical membership data (e.g. 000852 CSI1000, which
    Baostock does not cover) fall back to AKShare's current constituent list.

    Args:
        data_dir: Project data dir; when provided, historical membership is read
            from ``{data_dir}/a_shares/index_constituents_hist/membership.parquet``.
        indices: List like ['000300', '000905']. Default: CSI 300 + CSI 500.
    """
    if indices is None:
        indices = list(universe.DEFAULT_INDICES)

    codes = set()
    if data_dir:
        mem = universe.load_index_membership(data_dir, indices)
        covered = set(mem["index_code"].astype(str))
        if covered:
            mem_codes = set(
                normalize_stock_code_series(mem["stock_code"]).dropna())
            codes.update(mem_codes)
            logger.info("Historical member union (%s): %d stocks",
                        ",".join(sorted(covered)), len(mem_codes))
            indices = [i for i in indices if str(i) not in covered]

    for symbol in indices:
        name = {"000300": "CSI 300", "000905": "CSI 500"}.get(symbol, symbol)
        try:
            df = ak.index_stock_cons_csindex(symbol=symbol)
            new_codes = set(df["成分券代码"].tolist())
            codes.update(new_codes)
            logger.info("Fetched %d stocks from %s (%s)", len(new_codes), name, symbol)
        except Exception as e:
            logger.error("Failed to fetch %s (category=%s): %s",
                         name, classify_error(e).value, e)

    return sorted(codes)


def get_all_a_share_codes(data_dir: str | None = None) -> list[str]:
    """Fetch ALL A-share stock codes via AKShare stock_info_a_code_name.

    With ``data_dir``, the currently-listed universe is UNIONED with the
    delisted records (§七-1): ``--all`` must download delisted stocks' history
    too, or the 2000-2026 "全 A" panel silently becomes the survivor set of
    stocks still visible today (survivorship bias).
    """
    logger.info("Fetching full A-share stock list (may take ~5s)...")
    df = ak.stock_info_a_code_name()
    codes = set(normalize_stock_code_series(df["code"]).dropna())
    if data_dir:
        from stoke_ml.data.universe import delisted_codes
        extra = delisted_codes(data_dir)
        if extra:
            codes.update(extra)
            logger.info("Including %d delisted stocks from universe records",
                        len(extra))
    return sorted(codes)


def _last_fully_closed_trading_day(calendar: TradingCalendar) -> date:
    """Most recent trading day strictly before today whose session has fully
    closed (and, we assume, been published by the data source) — the default
    completeness end.  Today's bar is never required (it has not closed), and
    the result is capped at the calendar's ``verified_until`` so forward
    estimates never count as published fact (§P0-3).
    """
    bound = min(date.today() - timedelta(days=1), calendar.verified_until)
    d = bound
    while not calendar.is_trading_day(d):
        d -= timedelta(days=1)
    return d


def _effective_range(
    code: str,
    req_start: pd.Timestamp | None,
    req_end: pd.Timestamp | None,
    status_by_code: dict[str, tuple],
    last_closed: date,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Per-stock lifecycle window (§P0-3).

    A completeness claim must be judged against the stock's OWN tradable life,
    not a global 2000→today ask: a 2018 IPO can never start in 2000, and a
    delisted stock can never have data past its last trading day.  So

        effective_start = max(requested_start, list_date)
        effective_end   = min(requested_end, delist_date, last_closed)

    ``req_end=None`` (caller did not bound the request) keeps no upper bound —
    the CLI default already resolves to ``last_closed``.
    """
    eff_start = req_start
    eff_end = req_end
    ld, dd = status_by_code.get(code, (None, None))
    if pd.notna(ld):
        ts = pd.Timestamp(ld)
        eff_start = max(eff_start, ts) if eff_start is not None else ts
    if pd.notna(dd):
        ts = pd.Timestamp(dd)
        eff_end = min(eff_end, ts) if eff_end is not None else ts
    if eff_end is not None:
        eff_end = min(eff_end, pd.Timestamp(last_closed))
    return eff_start, eff_end


def filter_existing(
    codes: list[str], data_dir: str,
    start_date: str | None = None, end_date: str | None = None,
) -> tuple[list[str], set[str]]:
    """Filter out stocks already complete on disk.

    File presence alone is NOT enough to skip (§五-3): a stock is skipped only
    when its per-stock manifest exists, validates against the parquet via
    ``DataStorage.validate_manifest`` (rows / start / end / schema-hash /
    provenance) AND covers the stock's *effective* date range — the requested
    window clipped to the stock's own list/delist dates and the last fully
    closed trading day (§P0-3).  A stale or invalid manifest, a schema drift, a
    partial file, or a file ending before the effective ``end_date`` is
    re-downloaded.

    Returns ``(to_download, complete_codes)``.
    """
    storage = DataStorage(data_dir)
    status = universe.load_universe_status(data_dir)
    status_by_code: dict[str, tuple] = {}
    if not status.empty:
        norm = normalize_stock_code_series(status["stock_code"]).fillna("")
        status_by_code = {
            c: (ld, dd)
            for c, ld, dd in zip(norm, status["list_date"], status["delist_date"])
            if c
        }
    calendar = TradingCalendar("a_shares", calendar_dir=data_dir)
    last_closed = _last_fully_closed_trading_day(calendar)
    req_start = pd.Timestamp(start_date) if start_date else None
    req_end = pd.Timestamp(end_date) if end_date else None
    complete: set[str] = set()
    for code in codes:
        report = storage.validate_manifest(code, "a_shares")
        if not report.get("ok"):
            continue
        actual = report.get("actual") or {}
        a = actual.get("start")
        b = actual.get("end")
        eff_start, eff_end = _effective_range(
            code, req_start, req_end, status_by_code, last_closed)
        if eff_start is not None and (a is None or pd.Timestamp(a) > eff_start):
            continue
        if eff_end is not None and (b is None or pd.Timestamp(b) < eff_end):
            continue
        complete.add(code)
    to_download = [c for c in codes if c not in complete]
    return to_download, complete


def main():
    parser = argparse.ArgumentParser(description="Download A-share daily data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: config)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: from config universe)")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds between stocks (default: 0)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated AKShare index symbols")
    parser.add_argument("--all", action="store_true", dest="all_stocks",
                        help="Download ALL A-shares (~5500 stocks)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stocks already in data/a_shares/daily/")
    parser.add_argument("--require-complete", action="store_true",
                        help="Exit non-zero if any requested stock is missing "
                             "after the run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    storage = DataStorage(cfg.project.data_dir)
    downloader = AShareDownloader()

    # Default end is the last fully closed trading day (§P0-3) — never today,
    # which has not closed and may not be published yet, nor a forward estimate
    # past verified_until.  An explicit --end is honored as-is.
    _cal = TradingCalendar("a_shares", calendar_dir=cfg.project.data_dir)
    start_date = args.start or cfg.markets.a_shares.start_date
    end_date = args.end or _last_fully_closed_trading_day(_cal).strftime("%Y-%m-%d")

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    elif args.all_stocks:
        codes = get_all_a_share_codes(cfg.project.data_dir)
    elif args.indices:
        codes = get_stock_codes(cfg.project.data_dir, args.indices.split(","))
    else:
        codes = get_stock_codes(cfg.project.data_dir)

    if not codes:
        logger.error("No stock codes to download.")
        sys.exit(1)

    # §P0-4: the run manifest must answer "is the ENTIRE requested universe
    # complete?", so the original request is captured before --skip-existing
    # narrows the working set, and skipped-complete stocks stay in `complete`.
    requested_codes = list(codes)
    n_skipped = 0
    complete_prior: set[str] = set()
    if args.skip_existing:
        codes, complete_prior = filter_existing(
            codes, cfg.project.data_dir, start_date, end_date
        )
        n_skipped = len(complete_prior)
        logger.info("Skipping %d complete stocks, %d to download",
                     n_skipped, len(codes))

    # §七-2: bounded-end semantics.  An explicit future --end is recorded
    # verbatim as `requested_end` while the fetch/completeness end is bounded to
    # the latest available trading day; the manifest then reports
    # BOUNDED_COMPLETE instead of claiming coverage of a date range no source
    # can serve.
    _cal = TradingCalendar("a_shares", calendar_dir=cfg.project.data_dir)
    latest_available = _last_fully_closed_trading_day(_cal).strftime("%Y-%m-%d")
    requested_end = end_date
    effective_end = (
        min(end_date, latest_available) if end_date else latest_available
    )
    if requested_end and requested_end > effective_end:
        logger.warning(
            "requested --end %s is past the latest available trading day %s — "
            "run is bounded to %s (BOUNDED_COMPLETE)",
            requested_end, latest_available, effective_end)
    fetch_end = effective_end

    manifest_path = default_path(cfg.project.data_dir)
    logger.info("Downloading %d stocks from %s to %s",
                len(codes), start_date, fetch_end)
    success, fail = 0, 0
    failed_codes: list[str] = []
    run_errors: dict[str, str] = {}
    checkpoint_step = max(len(codes) // 50, 1)  # ~2% of the universe

    def _write_manifest(*, final: bool) -> dict:
        # §七-1 / §P0-4: `complete` comes from filter_existing — validated
        # manifest AND full requested-date coverage — never from file presence,
        # so "every parquet on disk" does not imply `all_complete` (§五-4).
        # Skipped complete stocks are unioned back so the manifest reports the
        # WHOLE requested universe, not just what this run touched.  A final
        # write re-validates every stock; periodic checkpoints keep a lightweight
        # in_progress artifact so a hard kill still leaves a manifest.
        if final:
            _, newly_complete = filter_existing(
                codes, cfg.project.data_dir, start_date, fetch_end)
            complete_all = complete_prior | newly_complete
            status = None  # derive complete / bounded_complete from the data
        else:
            complete_all = set(complete_prior)
            status = "in_progress"
        return write_manifest(
            manifest_path,
            market="a_shares",
            start_date=start_date,
            end_date=fetch_end,
            requested_end=requested_end,
            effective_end=effective_end,
            latest_available_end=latest_available,
            status=status,
            requested=requested_codes,
            failed=failed_codes,
            complete=complete_all,
            success_count=success,
            skipped_existing_count=n_skipped,
        )

    # §七-1: per-code try/except/finally — one bad stock (provider crash, disk
    # full, contract/price-basis conflict, schema drift) must NOT abort the
    # whole universe; it is recorded as failed and the run continues.
    aborted: str | None = None
    try:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.sleep)
            try:
                df = downloader.fetch_daily(code, start_date, fetch_end)
                if df.empty:
                    logger.warning("[%d/%d] %s: EMPTY (all sources failed)",
                                   i + 1, len(codes), code)
                    fail += 1
                    failed_codes.append(code)
                    run_errors[code] = "EMPTY: all sources failed"
                    continue
                storage.save_daily(df)
                dates = pd.to_datetime(df["date"])
                logger.info("[%d/%d] %s: %d rows [%s → %s]",
                            i + 1, len(codes), code, len(df),
                            dates.min().strftime("%Y-%m-%d"),
                            dates.max().strftime("%Y-%m-%d"))
                success += 1
            except Exception as exc:
                cat = classify_error(exc).value
                fail += 1
                failed_codes.append(code)
                run_errors[code] = f"{cat}: {exc}"
                logger.warning("[%d/%d] %s: FAILED (category=%s): %s",
                               i + 1, len(codes), code, cat, exc)
            finally:
                # Periodic checkpoint so even a hard kill leaves a manifest.
                if (i + 1) % checkpoint_step == 0:
                    _write_manifest(final=False)
    except Exception as exc:
        aborted = f"{classify_error(exc).value}: {exc}"
        logger.error("download loop aborted unexpectedly: %s", exc)
    finally:
        # The final manifest ALWAYS runs — including when an unexpected
        # exception escapes the per-code loop.  The failure scenario is exactly
        # when the manifest is most needed (§七-1).
        try:
            manifest = _write_manifest(final=True)
        except Exception as exc:
            aborted = (aborted + "; " if aborted else "") + f"manifest-write: {exc}"
            logger.error("failed to write final manifest: %s", exc)
            manifest = None

    if manifest is None:
        logger.error("download_data: run incomplete and no final manifest — exit 3")
        sys.exit(3)
    logger.info("Download manifest: %s", manifest_path)
    logger.info(
        "Requested=%d success=%d failed=%d missing=%d all_complete=%s status=%s",
        manifest["requested_count"], manifest["success_count"],
        manifest["failed_count"], manifest["missing_count"],
        manifest["all_complete"], manifest["status"])
    if manifest["missing"]:
        logger.warning("MISSING %d stock(s) requested but not on disk: %s",
                       len(manifest["missing"]),
                       ", ".join(manifest["missing"][:20])
                       + (" ..." if len(manifest["missing"]) > 20 else ""))
    if run_errors:
        logger.error("FAILED %d stock(s): %s", len(run_errors),
                     ", ".join(list(run_errors)[:20])
                     + (" ..." if len(run_errors) > 20 else ""))
    if aborted:
        logger.error("download loop aborted: %s", aborted)
    # §七-1: any failure, missing stock, bounded request or aborted loop
    # → non-zero exit; a clean full run is the ONLY path to exit 0.  (The former
    # --require-complete gate is subsumed: the run now always fails loudly.)
    if aborted or fail > 0 or not manifest["all_complete"]:
        logger.error("download_data exit=2: aborted=%s failed=%d missing=%d",
                     bool(aborted), fail, manifest["missing_count"])
        sys.exit(2)


if __name__ == "__main__":
    main()
