"""Download A-share announcements with sentiment (CNINFO or EastMoney).

CNINFO (巨潮资讯网): CSRC official, PDF body extraction, full historical coverage.
EastMoney (东方财富): HTTP API, titles only, no PDF needed, faster but no bodies.

Sentiment computed via financial lexicon on title text (no GPU needed).

Resume: reads existing {code}.parquet, finds max date, only fetches newer data.

Usage:
  python scripts/production/download_cninfo_announcements.py                             # all stocks, EastMoney
  python scripts/production/download_cninfo_announcements.py --source cninfo             # CNINFO (PDF bodies)
  python scripts/production/download_cninfo_announcements.py --shard 0/4                 # 1/4 of stocks
  python scripts/production/download_cninfo_announcements.py --stocks 000001,600519      # specific stocks
  python scripts/production/download_cninfo_announcements.py --skip-sentiment            # raw only
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.cninfo_source import CninfoSource
from stoke_ml.data.sources.a_shares.announcement_source import AnnouncementSource
from stoke_ml.data.download_manifest import write_run_manifest
from stoke_ml.features.news_nlp import compute_raw_sentiment, NewsSentimentAnalyzer

sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

_SENTIMENT_COLS = [
    "sentiment_mean", "sentiment_std", "announce_count",
    "positive_ratio", "negative_ratio", "has_announce",
]


def available_stocks(data_dir: str) -> list[str]:
    base = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(base):
        return []
    codes = set()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".parquet"):
                codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def build_daily_sentiment(raw_dir: str, stock_code: str) -> pd.DataFrame:
    """Lexicon-based daily sentiment from announcement body text."""
    path = os.path.join(raw_dir, f"{stock_code}.parquet")
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["date"] + _SENTIMENT_COLS)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    sent_col = None
    for c in ["sentiment_body", "sentiment_title"]:
        if c in df.columns:
            sent_col = c
            break
    if sent_col is None:
        return pd.DataFrame(columns=["date"] + _SENTIMENT_COLS)

    df["sentiment"] = pd.to_numeric(df[sent_col], errors="coerce").fillna(0)

    daily = df.groupby("date").agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", lambda x: x.std() if len(x) > 1 else 0.0),
        announce_count=("sentiment", "count"),
        positive=("sentiment", lambda x: (x > 0.05).sum()),
        negative=("sentiment", lambda x: (x < -0.05).sum()),
    ).reset_index()

    daily["positive_ratio"] = daily["positive"] / daily["announce_count"]
    daily["negative_ratio"] = daily["negative"] / daily["announce_count"]
    daily["has_announce"] = daily["announce_count"] > 0
    daily = daily.drop(columns=["positive", "negative"])
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily = daily.sort_values("date").reset_index(drop=True)

    out_dir = os.path.join(raw_dir, "sentiment")
    os.makedirs(out_dir, exist_ok=True)
    daily.to_parquet(os.path.join(out_dir, f"{stock_code}.parquet"), index=False, compression='lz4')
    return daily


def _read_existing_stock(raw_dir: str, code: str) -> pd.DataFrame | None:
    """Read an existing parquet file, returning None if missing/corrupt."""
    path = os.path.join(raw_dir, f"{code}.parquet")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        logger.warning("Corrupt file for %s, re-downloading", code)
        os.remove(path)
        return None


def _resume_filter(raw_dir: str, codes: list[str], start_date: str,
                   end_date: str) -> tuple[list[str], dict[str, str], int]:
    """Determine which stocks need downloading and from what date.

    Returns (pending_codes, resume_map, n_skipped).
    resume_map: code → resume_start_date (may be later than the global start_date).
    """
    end_ts = pd.Timestamp(end_date)
    pending = []
    resume_map = {}
    skipped = 0

    for code in codes:
        existing = _read_existing_stock(raw_dir, code)
        if existing is not None:
            max_date = existing["date"].max()
            if max_date >= end_ts:
                skipped += 1
                continue
            # Resume from day after last covered date
            resume_start = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            pending.append(code)
            resume_map[code] = resume_start
        else:
            pending.append(code)
            resume_map[code] = start_date

    return pending, resume_map, skipped


def main():
    parser = argparse.ArgumentParser(description="Download A-share announcements")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--source", type=str, default="eastmoney",
                        choices=["cninfo", "eastmoney"],
                        help="Data source: cninfo (PDF bodies, may be rate-limited) "
                             "or eastmoney (titles only, faster) (default: eastmoney)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip lexicon sentiment + daily aggregation")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-download all stocks (ignore existing files)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="[CNINFO only] Skip PDF body download (titles only)")
    parser.add_argument("--pdf-workers", type=int, default=8,
                        help="[CNINFO only] Concurrent PDF download threads per stock (default: 8)")
    parser.add_argument("--sleep", type=float, default=0,
                        help="Seconds between stocks (default: 0)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec: k/N (e.g. 0/4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    raw_dir = os.path.join(data_dir, "a_shares", "announcements")
    os.makedirs(raw_dir, exist_ok=True)

    codes = (args.stocks.split(",") if args.stocks
             else available_stocks(data_dir))

    if not codes:
        logger.error("No stocks found")
        sys.exit(1)

    if args.shard:
        k, n = args.shard.split("/")
        k, n = int(k), int(n)
        codes = [c for i, c in enumerate(codes) if i % n == k]
        logger.info("Shard %s/%s: %d stocks", k, n, len(codes))

    end_date = args.end or time.strftime("%Y-%m-%d")

    if args.no_resume:
        resume_map = {c: args.start for c in codes}
        n_skipped = 0
    else:
        codes, resume_map, n_skipped = _resume_filter(
            raw_dir, codes, args.start, end_date,
        )
        if n_skipped:
            logger.info("Skipping %d already-complete stocks, %d remaining",
                       n_skipped, len(codes))

    if not codes:
        logger.info("All stocks already downloaded. Nothing to do.")
        sys.exit(0)

    if args.source == "cninfo":
        source = CninfoSource(pdf_bodies=not args.no_pdf, pdf_workers=args.pdf_workers)
        logger.info("Downloading via CNINFO for %d stocks (%s to %s), PDF bodies: %s",
                    len(codes), args.start, end_date, not args.no_pdf)
    else:
        source = AnnouncementSource()
        if args.sleep > 0:
            logger.info("Downloading via EastMoney for %d stocks (%s to %s), sleep=%.1fs",
                        len(codes), args.start, end_date, args.sleep)

    analyzer = None if args.skip_sentiment else NewsSentimentAnalyzer(force_lexicon=True)

    success = 0
    empty_count = 0
    done_codes: set[str] = set()
    failed_codes: list[str] = []
    for i, code in enumerate(codes):
        try:
            if i > 0 and args.sleep > 0:
                time.sleep(args.sleep)

            resume_start = resume_map.get(code, args.start)

            new_df = source.fetch_announcements(code, resume_start, end_date)
            if new_df.empty:
                empty_count += 1
                if (i + 1) % 10 == 0:
                    logger.info("[%d/%d] %s: 0 new (empties so far: %d)",
                                i + 1, len(codes), code, empty_count)
                continue

            new_df["stock_code"] = code

            # Merge with existing data if resuming
            if resume_start != args.start:
                existing = _read_existing_stock(raw_dir, code)
                if existing is not None:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined["date"] = pd.to_datetime(combined["date"])
                    combined = combined.drop_duplicates(subset=["title", "date"])
                    combined = combined.sort_values("date").reset_index(drop=True)
                    df = combined
                else:
                    df = new_df
            else:
                df = new_df

            if analyzer is not None and len(df) > 0:
                df = compute_raw_sentiment(df, analyzer)

            out_path = os.path.join(raw_dir, f"{code}.parquet")
            df.to_parquet(out_path, index=False, compression='lz4')

            if not args.skip_sentiment:
                build_daily_sentiment(raw_dir, code)

            success += 1
            done_codes.add(code)
            if success % 10 == 0:
                logger.info("[%d/%d] %s: %d announcements (total: %d with data, %d empty)",
                            i + 1, len(codes), code, len(df), success, empty_count)

        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", i + 1, len(codes), code, e)
            failed_codes.append(code)

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/announcements",
            start_date=args.start, end_date=end_date,
            requested=codes, failed=failed_codes, complete=done_codes,
            success_count=success,
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)

    logger.info("Done: %d/%d stocks with announcements", success, len(codes))


if __name__ == "__main__":
    main()
