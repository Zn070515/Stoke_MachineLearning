"""Download daily data for A-share stock universe via 4-source failover."""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import akshare as ak
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data import universe
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
            mem_codes = set(mem["stock_code"].astype(str).str.zfill(6))
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
    codes = set(df["code"].astype(str).str.zfill(6).tolist())
    if data_dir:
        from stoke_ml.data.universe import delisted_codes
        extra = delisted_codes(data_dir)
        if extra:
            codes.update(extra)
            logger.info("Including %d delisted stocks from universe records",
                        len(extra))
    return sorted(codes)


def filter_existing(
    codes: list[str], data_dir: str,
    start_date: str | None = None, end_date: str | None = None,
) -> tuple[list[str], set[str]]:
    """Filter out stocks already complete on disk.

    File presence alone is NOT enough to skip (§五-3): a stock is skipped only
    when its per-stock manifest exists, validates against the parquet via
    ``DataStorage.validate_manifest`` (rows / start / end / schema-hash /
    provenance) AND covers the requested date range.  A stale or invalid
    manifest, a schema drift, a partial file, or a file ending before the
    requested ``end_date`` is re-downloaded.

    Returns ``(to_download, complete_codes)``.
    """
    storage = DataStorage(data_dir)
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
        if req_start is not None and (a is None or pd.Timestamp(a) > req_start):
            continue
        if req_end is not None and (b is None or pd.Timestamp(b) < req_end):
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

    start_date = args.start or cfg.markets.a_shares.start_date
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

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

    n_skipped = 0
    if args.skip_existing:
        codes, existing = filter_existing(
            codes, cfg.project.data_dir, start_date, end_date
        )
        n_skipped = len(existing)
        logger.info("Skipping %d complete stocks, %d to download",
                     n_skipped, len(codes))

    logger.info("Downloading %d stocks from %s to %s", len(codes), start_date, end_date)
    success, fail = 0, 0
    failed_codes: list[str] = []

    for i, code in enumerate(codes):
        if i > 0:
            time.sleep(args.sleep)

        logger.info("[%d/%d] Fetching %s ...", i + 1, len(codes), code)
        df = downloader.fetch_daily(code, start_date, end_date)

        if df.empty:
            logger.warning("  %s: EMPTY (all sources failed)", code)
            fail += 1
            failed_codes.append(code)
            continue

        storage.save_daily(df)
        dates = pd.to_datetime(df["date"])
        logger.info("  %s: %d rows [%s → %s]", code, len(df),
                     dates.min().strftime("%Y-%m-%d"),
                     dates.max().strftime("%Y-%m-%d"))
        success += 1

    # Persist the download manifest so a PARTIAL run cannot pass for complete.
    # `complete` comes from filter_existing — validated manifest AND full
    # requested-date coverage — never from file presence, so "every parquet on
    # disk" does not imply `all_complete` (§五-4).
    _, complete = filter_existing(codes, cfg.project.data_dir, start_date, end_date)
    manifest = write_manifest(
        default_path(cfg.project.data_dir),
        market="a_shares",
        start_date=start_date,
        end_date=end_date,
        requested=codes,
        failed=failed_codes,
        complete=complete,
        success_count=success,
        skipped_existing_count=n_skipped,
    )
    logger.info("Download manifest: %s", default_path(cfg.project.data_dir))
    logger.info("Requested=%d success=%d failed=%d missing=%d all_complete=%s",
                manifest["requested_count"], manifest["success_count"],
                manifest["failed_count"], manifest["missing_count"],
                manifest["all_complete"])
    if manifest["missing"]:
        logger.warning("MISSING %d stock(s) requested but not on disk: %s",
                       len(manifest["missing"]),
                       ", ".join(manifest["missing"][:20])
                       + (" ..." if len(manifest["missing"]) > 20 else ""))
    if args.require_complete and not manifest["all_complete"]:
        logger.error("--require-complete set but %d stock(s) missing",
                     manifest["missing_count"])
        sys.exit(2)


if __name__ == "__main__":
    main()
