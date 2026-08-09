"""Preprocess new data types through the multi-shape preprocessing pipeline.

Reads raw data from MarketWideStorage (downloaded by download_datacenter.py),
runs the appropriate PreprocessingChain, saves preprocessed results.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/preprocess_new_data.py --type all
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/preprocess_new_data.py --type flow
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/preprocess_new_data.py --type block_trade
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/preprocess_new_data.py --type concept --stocks 600519,000001
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.date_normalize import as_date_us
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingQualityError,
)
from stoke_ml.utils.error_summary import ErrorSummary, log_summary

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Map script --type to (storage_key, pipeline_chain_name)
TYPE_MAP = {
    "flow": ("capital_flow", "flow"),
    "block_trade": ("block_trade", "event_block_trade"),
    "shareholder": ("shareholder", "event_shareholder"),
    "lockup": ("lockup", "event_lockup"),
    "dividend": ("dividend", "event_dividend"),
    "board": (None, "board"),  # needs multiple pool storages
    "sector": ("industry_ranking", "sector"),
    "concept": ("concept_blocks", "concept"),
}


def _build_provenance(cfg) -> dict:
    """Run-level provenance bound to every replace_range write.

    Each stock's write manifest carries run_id / git_commit / config_hash so a
    later reader can tell exactly which code + config produced the data.
    """
    git_commit = "unknown"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    config_hash = "unknown"
    try:
        from omegaconf import OmegaConf
        config_hash = hashlib.sha1(
            OmegaConf.to_yaml(cfg).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        pass
    return {
        "run_id": f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}",
        "git_commit": git_commit,
        "config_hash": config_hash,
    }


def get_stocks_from_disk(data_dir: str, storage_key: str) -> list[str]:
    """Discover available stock codes from partitioned storage."""
    base = os.path.join(data_dir, "a_shares", storage_key)
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def get_stocks_from_daily(data_dir: str) -> list[str]:
    """Fall back to daily K-line directory."""
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    codes = set()
    for root, _dirs, files in os.walk(daily_dir):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (exposed for tests)."""
    parser = argparse.ArgumentParser(
        description="Preprocess new data types through multi-shape pipeline",
    )
    parser.add_argument(
        "--type", type=str, default="all",
        choices=["all"] + list(TYPE_MAP.keys()),
        help="Data type to preprocess",
    )
    parser.add_argument("--event-type", type=str, default=None,
                        choices=["block_trade", "shareholder", "lockup", "dividend"],
                        help="Specific event type when --type=event")
    parser.add_argument("--start", type=str, default="2000-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes")
    parser.add_argument("--save-to", type=str, default=None,
                        help="Override output storage key (default: {storage_key}_processed)")
    parser.add_argument("--strict", action="store_true",
                        help="Block stocks whose output fails error-level quality "
                             "checks or whose daily K-line context cannot be loaded; "
                             "never persist degraded output")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="§十二: explicit opt-out from the formal-mode default "
                             "strictness — quality problems are logged and degraded "
                             "output is written instead of blocking.  Only meaningful "
                             "in formal mode; --strict still wins when both are given.")
    parser.add_argument("--no-formal", action="store_true",
                        help="§九-1: allow fold_train_only chains on the offline "
                             "full-history path (dev smoke only; production runs "
                             "must be formal so the validation chain is honest)")
    parser.add_argument("--degrade-threshold", type=float, default=0.2,
                        help="Max fraction of previously-present dates a "
                             "replace_range write may drop before it is rejected "
                             "(default 0.2)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the replace_range degradation guard: allow "
                             "an intentional destructive rewrite (schema "
                             "unification / full-range regeneration) even when "
                             "the new output drops columns or covers fewer "
                             "dates.  The coverage report is still written to "
                             "the sidecar manifest so the outcome stays "
                             "auditable (§v19 migration rebuilds).")
    parser.add_argument(
        "--sector-snapshot-asof", type=str, default=None,
        help="Date the current stock_sector_cache.csv snapshot is valid from "
             "(YYYY-MM-DD).  Rows before it are treated as unknown-sector "
             "(features zeroed) instead of backfilling today's classification "
             "onto history.  Defaults to the snapshot file's mtime.  Ignored "
             "when sector_membership.parquet (genuine PIT) exists (§十-4).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    # §十二: formal full-history preprocessing is STRICT by default — error-level
    # quality problems BLOCK and nothing is persisted (no degraded artifact).
    # --allow-degraded explicitly opts out so the caller may write degraded
    # output; --strict remains the explicit "always block" switch (also effective
    # in the --no-formal dev path).  The effective flag is passed down as the
    # daily-K-line-load block decision; pp.run() additionally receives
    # formal/allow_degraded so the quality gate is enforced at the pipeline layer.
    strict = args.strict or (not args.no_formal and not args.allow_degraded)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    # Resolve stock list
    if args.stocks:
        stock_list = [c.strip() for c in args.stocks.split(",")]
    else:
        stock_list = get_stocks_from_daily(data_dir)
        if not stock_list:
            logger.error("No stock codes found. Run download_data.py first.")
            sys.exit(1)

    # Build pipeline from config
    pp_cfg = cfg.get("preprocessing", {}) if hasattr(cfg, "get") else {}
    pp = PreprocessingPipeline.from_config(pp_cfg)
    provenance = _build_provenance(cfg)

    # Determine types to process
    if args.type == "all":
        to_process = list(TYPE_MAP.keys())
    elif args.type == "event" and args.event_type:
        to_process = [args.event_type]
    elif args.type == "event":
        logger.error("--type event requires --event-type")
        sys.exit(1)
    else:
        to_process = [args.type]

    for dtype in to_process:
        if dtype not in TYPE_MAP:
            logger.warning("Unknown type: %s", dtype)
            continue
        storage_key, chain_name = TYPE_MAP[dtype]
        chain = pp.get_chain(chain_name)
        if chain is None:
            logger.warning("Chain '%s' not configured, skipping %s", chain_name, dtype)
            continue

        if dtype == "board":
            _process_board(pp, chain_name, stock_list, data_dir, args, provenance,
                           strict=strict)
        elif dtype == "sector":
            _process_sector(pp, chain_name, stock_list, data_dir, args, provenance,
                            strict=strict)
        elif storage_key:
            _process_standard(
                dtype, storage_key, pp, chain_name, stock_list, data_dir,
                args, provenance, strict=strict,
            )


def _process_standard(dtype, storage_key, pp, chain_name, stock_list, data_dir,
                      args, provenance, strict):
    """Process standard per-stock data: load → transform → save.

    Passes K-line data as ``close_prices`` + ``trading_dates`` so that
    EventToDaily can forward-fill to daily and activate price-dependent
    features (dual-concentration, MCap normalization, etc.).
    ``daily_data`` is also passed for block_trade amount_ratio which
    accesses it separately via **kwargs.

    Under ``strict`` (explicit ``--strict``, or formal mode by default §十二) a
    stock whose daily K-line context fails to load, or whose transformed output
    trips error-level quality checks, is BLOCKED — nothing is persisted and the
    failure is counted (禁止 commit, 保留 staging 输出, 记录失败 manifest).
    ``--allow-degraded`` (formal mode only) relaxes both: the quality gate is
    lifted at the pipeline layer and the daily-load failure no longer blocks,
    so degraded output is written instead.

    Every replace_range write carries ``provenance`` and is guarded by the
    degradation check in MarketWideStorage.save().
    """
    logger.info("=== %s: %d stocks (%s to %s) ===",
                dtype, len(stock_list), args.start, args.end)
    t0 = time.time()
    source = MarketWideStorage(data_dir, storage_key)
    output_key = args.save_to or f"{storage_key}_processed"
    dest = MarketWideStorage(data_dir, output_key)
    prov = {
        **provenance,
        "source_snapshot": f"raw:{storage_key}:{len(stock_list)}stocks",
    }

    from stoke_ml.data.calendar import get_research_calendar
    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    calendar = get_research_calendar(strict=True, data_dir=data_dir)
    trading_dates = pd.DatetimeIndex(
        calendar.get_trading_days(args.start, args.end)
    )

    total = 0
    blocked = 0
    rejected = 0
    summary = ErrorSummary()
    for code in stock_list:
        try:
            raw = source.load(code, args.start, args.end)
            if raw.empty:
                continue
            # Inject stock_code when missing or any NaN (data quality fix)
            if "stock_code" not in raw.columns:
                raw["stock_code"] = code
            elif raw["stock_code"].isna().any():
                raw["stock_code"] = raw["stock_code"].fillna(code)
            # Load K-line daily data for market-cap context features
            daily_data = None
            try:
                daily_data = ds.load_daily(
                    code, args.start, args.end, require_valid_manifest=True)
                if not daily_data.empty and "stock_code" not in daily_data.columns:
                    daily_data["stock_code"] = code
            except Exception as exc:
                if strict:
                    logger.error(
                        "%s: daily K-line load failed for %s — blocking "
                        "(strict mode, §十二) (%s)", dtype, code, exc,
                    )
                    blocked += 1
                    continue
                logger.warning(
                    "%s: daily K-line load failed for %s — price-dependent "
                    "features degraded (%s)", dtype, code, exc,
                )
            processed = pp.run(
                chain_name, raw, strict=strict, formal=not args.no_formal,
                allow_degraded=args.allow_degraded,
                daily_data=daily_data,
                close_prices=daily_data,
                trading_dates=trading_dates,
            )
            if not processed.empty:
                rejected += dest.save(
                    processed, replace_range=True,
                    provenance=prov, degrade_threshold=args.degrade_threshold,
                    force=args.force,
                    replace_window=(args.start, args.end),
                )
                total += len(processed)
        except PreprocessingQualityError as exc:
            blocked += 1
            logger.error(
                "%s: quality gate blocked %s: %s "
                "(staging output retained, nothing persisted)",
                dtype, code, exc,
            )
        except Exception as exc:
            logger.warning("%s preprocessing failed for %s", dtype, code, exc_info=True)
            summary.record_exc(exc, f"preprocess:{dtype}")

    if blocked:
        logger.warning(
            "  %s: %d stocks blocked by quality gate / daily-load failure "
            "(not persisted)", dtype, blocked,
        )
    if rejected:
        logger.error(
            "  %s: %d stocks rejected by replace_range degradation guard "
            "(old files preserved)", dtype, rejected,
        )
    logger.info("  %s: %d rows saved (%.1fs)", dtype, total, time.time() - t0)
    if summary:
        log_summary(summary, logger, f"preprocess:{dtype}")


def _process_board(pp, chain_name, stock_list, data_dir, args, provenance,
                   strict):
    """Process board data: load limit_up pools → broadcast to stocks."""
    logger.info("=== board: %d stocks ===", len(stock_list))
    t0 = time.time()
    prov = {
        **provenance,
        "source_snapshot": f"raw:limit_up_pools:{len(stock_list)}stocks",
    }

    # Load all 4 limit-up pools
    pools = {}
    for pool_name in ["zt", "zb", "dt", "yzt"]:
        storage_key = f"limit_up_{pool_name}"
        pool_storage = MarketWideStorage(data_dir, storage_key)
        frames = []
        for code in stock_list:
            pdf = pool_storage.load(code, args.start, args.end)
            if not pdf.empty:
                if "stock_code" not in pdf.columns or pdf["stock_code"].isna().all():
                    pdf["stock_code"] = code
                frames.append(pdf)
        if frames:
            pools[pool_name] = pd.concat(frames, ignore_index=True)
        else:
            logger.warning("No %s pool data found for %s–%s", pool_name, args.start, args.end)

    # Load sentiment if available (single market-level time series, not per-stock)
    sentiment = None
    try:
        sent_path = os.path.join(data_dir, "a_shares", "limit_up_sentiment", "sentiment.parquet")
        if os.path.isfile(sent_path):
            sentiment = pd.read_parquet(sent_path)
            if "date" in sentiment.columns:
                sentiment["date"] = pd.to_datetime(sentiment["date"])
        if sentiment is not None and not sentiment.empty:
            logger.info("Loaded limit_up_sentiment: %d rows", len(sentiment))
    except Exception:
        logger.warning("Failed to load limit_up_sentiment", exc_info=True)

    # Build concept_map from concept block data for same-concept ZT stats
    concept_map = {}
    try:
        concept_storage = MarketWideStorage(data_dir, "concept_blocks")
        c_frames = []
        for code in stock_list:
            cdf = concept_storage.load(code, args.start, args.end)
            if not cdf.empty:
                c_frames.append(cdf)
        if c_frames:
            concept_all = pd.concat(c_frames, ignore_index=True)
            if "stock_code" in concept_all.columns and "board_name" in concept_all.columns:
                # Use most frequent concept per stock as primary
                concept_map = (
                    concept_all.groupby("stock_code")["board_name"]
                    .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
                    .dropna()
                    .to_dict()
                )
                # Normalize keys to str for consistent lookup against df["stock_code"]
                concept_map = {str(k): v for k, v in concept_map.items()}
            logger.info("  Built concept_map: %d stocks → concept", len(concept_map))
    except Exception:
        logger.debug("No concept data available for board ZT stats", exc_info=True)

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    dest = MarketWideStorage(data_dir, args.save_to or "board_processed")
    total = 0
    blocked = 0
    rejected = 0
    summary = ErrorSummary()
    for code in stock_list:
        try:
            base = ds.load_daily(code, args.start, args.end,
                                 require_valid_manifest=True)
            if base.empty:
                continue
            processed = pp.run(
                chain_name, base, strict=strict, formal=not args.no_formal,
                allow_degraded=args.allow_degraded,
                pools=pools, sentiment=sentiment,
                concept_map=concept_map if concept_map else None,
            )
            if not processed.empty:
                rejected += dest.save(
                    processed, replace_range=True,
                    provenance=prov, degrade_threshold=args.degrade_threshold,
                    force=args.force,
                    replace_window=(args.start, args.end),
                )
                total += len(processed)
        except PreprocessingQualityError as exc:
            blocked += 1
            logger.error(
                "board: quality gate blocked %s: %s "
                "(staging output retained, nothing persisted)", code, exc,
            )
        except Exception as exc:
            logger.warning("board preprocessing failed for %s", code, exc_info=True)
            summary.record_exc(exc, "preprocess:board")

    if blocked:
        logger.warning(
            "  board: %d stocks blocked by quality gate (not persisted)", blocked
        )
    if rejected:
        logger.error(
            "  board: %d stocks rejected by replace_range degradation guard "
            "(old files preserved)", rejected,
        )
    logger.info("  board: %d rows saved (%.1fs)", total, time.time() - t0)
    if summary:
        log_summary(summary, logger, "preprocess:board")


def _process_sector(pp, chain_name, stock_list, data_dir, args, provenance,
                    strict):
    """Process sector data: load industry ranking + sector map → broadcast to stocks."""
    logger.info("=== sector: %d stocks ===", len(stock_list))
    t0 = time.time()
    prov = {
        **provenance,
        "source_snapshot": f"raw:industry_ranking:{len(stock_list)}stocks",
    }

    # Load industry ranking from single market-wide parquet
    ir_path = os.path.join(data_dir, "a_shares", "industry_ranking.parquet")
    if not os.path.exists(ir_path):
        logger.warning(
            "No industry_ranking.parquet found — run download_industry_ranking.py first"
        )
        return
    industry_ranking = as_date_us(pd.read_parquet(ir_path))
    # Date filter
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    industry_ranking = industry_ranking[
        (industry_ranking["date"] >= start_ts)
        & (industry_ranking["date"] <= end_ts)
    ]
    if industry_ranking.empty:
        logger.warning("industry_ranking empty for %s–%s", args.start, args.end)
        return
    logger.info(
        "  Loaded industry_ranking: %d rows, %d sectors, %d dates",
        len(industry_ranking),
        industry_ranking["sector_code"].nunique() if "sector_code" in industry_ranking.columns else 0,
        industry_ranking["date"].nunique(),
    )

    # Build sector_map: stock_code → sector_code (from cache CSV + naming map)
    cache_path = os.path.join(data_dir, "a_shares", "stock_sector_cache.csv")
    if not os.path.exists(cache_path):
        logger.warning("No stock_sector_cache.csv found — sector preprocessing skipped")
        return
    sector_df = pd.read_csv(cache_path, dtype=str)
    # Build sector_name → sector_code mapping from industry_ranking
    if "sector_name" in industry_ranking.columns and "sector_code" in industry_ranking.columns:
        name_to_code = (
            industry_ranking.groupby("sector_name")["sector_code"]
            .first().to_dict()
        )
    else:
        name_to_code = {}
    sector_map = {}
    for _, row in sector_df.iterrows():
        code = name_to_code.get(row["sector"], row["sector"])
        sector_map[row["stock_code"]] = code

    # §十-4: stock→sector membership must be point-in-time.
    #   * sector_membership.parquet [date, stock_code, sector_code] is a genuine
    #     PIT source: each stock's sector is resolved by its row date.
    #   * stock_sector_cache.csv is a CURRENT-SNAPSHOT classification with no
    #     historical validity.  Applying it to the whole window backfills
    #     today's sector onto every historical row (present-backfill bias), so
    #     it is applied ONLY to rows >= sector_map_valid_from — the CLI override,
    #     or the snapshot file's mtime as the best available earliest-valid date.
    #     Older rows get NaN sector_code → sector features zeroed as unknown.
    membership = None
    membership_path = os.path.join(data_dir, "a_shares", "sector_membership.parquet")
    if os.path.exists(membership_path):
        membership = pd.read_parquet(membership_path)
        missing_cols = [c for c in ("date", "stock_code", "sector_code")
                        if c not in membership.columns]
        if missing_cols:
            logger.error(
                "sector_membership.parquet missing column(s) %s — falling back "
                "to the snapshot map", missing_cols,
            )
            membership = None
        else:
            membership = as_date_us(membership)
            membership["stock_code"] = membership["stock_code"].astype(str)
            logger.info(
                "  sector: using PIT sector_membership.parquet (%d records, "
                "%d stocks, %s–%s)",
                len(membership),
                membership["stock_code"].nunique(),
                membership["date"].min(), membership["date"].max(),
            )

    sector_map_valid_from = None
    if membership is None:
        sector_map_valid_from = args.sector_snapshot_asof
        if sector_map_valid_from is None:
            sector_map_valid_from = datetime.fromtimestamp(
                os.path.getmtime(cache_path)
            ).strftime("%Y-%m-%d")
        logger.info(
            "  sector: current-snapshot stock_sector_cache.csv applied only to "
            "rows >= %s (older rows: unknown sector, features zeroed). "
            "Pass --sector-snapshot-asof to override, or provide "
            "sector_membership.parquet for genuine PIT membership (§十-4).",
            sector_map_valid_from,
        )
    prov["sector_map_asof"] = "PIT" if membership is not None else sector_map_valid_from

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    dest = MarketWideStorage(data_dir, args.save_to or "industry_ranking_processed")

    # Precompute stock-independent sector features ONCE from the ranking,
    # then broadcast to every stock — avoids ~N recomputations of momentum /
    # RRG / breadth_z / relative_strength / alpha inside SectorBroadcaster.
    from stoke_ml.preprocessing.cross_sectional.sector import SectorBroadcaster
    chain = pp.get_chain(chain_name)
    sector_step = next(
        (s for s in chain.steps if isinstance(s, SectorBroadcaster)), None
    )
    if sector_step is None:
        sector_step = SectorBroadcaster()
    sector_features = sector_step.build_sector_features(industry_ranking)
    logger.info(
        "  Precomputed sector features: %d rows, %d sectors",
        len(sector_features),
        sector_features["sector_code"].nunique()
        if "sector_code" in sector_features.columns else 0,
    )

    total = 0
    blocked = 0
    rejected = 0
    unknown_sector_rows = 0
    summary = ErrorSummary()
    for i, code in enumerate(stock_list):
        try:
            base = ds.load_daily(code, args.start, args.end,
                                 require_valid_manifest=True)
            if base.empty:
                continue
            if membership is not None:
                # PIT: resolve each row's sector from the last membership
                # record on-or-before its date; earlier rows stay unknown.
                sub = membership[membership["stock_code"] == code]
                if sub.empty:
                    base["sector_code"] = np.nan
                else:
                    sub = (sub.sort_values("date")
                           .drop_duplicates(subset="date", keep="last"))
                    base = (base.sort_values("date")
                            .reset_index(drop=True))
                    base = pd.merge_asof(
                        base, sub[["date", "sector_code"]],
                        on="date", direction="backward",
                    )
            processed = pp.run(
                chain_name, base, strict=strict, formal=not args.no_formal,
                allow_degraded=args.allow_degraded,
                industry_ranking=industry_ranking,
                sector_map=None if membership is not None else sector_map,
                sector_features=sector_features,
                sector_map_valid_from=(
                    None if membership is not None else sector_map_valid_from
                ),
            )
            if not processed.empty:
                if membership is None and sector_map_valid_from is not None:
                    unknown_sector_rows += int(
                        (processed["date"] < pd.Timestamp(sector_map_valid_from)).sum()
                    )
                rejected += dest.save(
                    processed, replace_range=True,
                    provenance=prov, degrade_threshold=args.degrade_threshold,
                    force=args.force,
                    replace_window=(args.start, args.end),
                )
                total += len(processed)
            if (i + 1) % 500 == 0:
                logger.info("  sector progress: %d/%d stocks, %d rows",
                            i + 1, len(stock_list), total)
        except PreprocessingQualityError as exc:
            blocked += 1
            logger.error(
                "sector: quality gate blocked %s: %s "
                "(staging output retained, nothing persisted)", code, exc,
            )
        except Exception as exc:
            logger.warning("sector preprocessing failed for %s", code, exc_info=True)
            summary.record_exc(exc, "preprocess:sector")

    if blocked:
        logger.warning(
            "  sector: %d stocks blocked by quality gate (not persisted)", blocked
        )
    if rejected:
        logger.error(
            "  sector: %d stocks rejected by replace_range degradation guard "
            "(old files preserved)", rejected,
        )
    if membership is None and unknown_sector_rows:
        logger.info(
            "  sector: %d rows older than snapshot-asof %s treated as "
            "unknown-sector (features zeroed) — no present-backfill (§十-4)",
            unknown_sector_rows, sector_map_valid_from,
        )
    logger.info("  sector: %d rows saved (%.1fs)", total, time.time() - t0)
    if summary:
        log_summary(summary, logger, "preprocess:sector")


if __name__ == "__main__":
    main()
