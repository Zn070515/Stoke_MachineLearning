# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Backfill K-line (2000-2014 gap) + turnover via Baostock in a single pass.

One Baostock query per stock covering 2000-01-01→today, then:
  - Pre-existing dates: use Baostock OHLCV + turnover (backfill)
  - Post-existing dates: merge Baostock turnover into existing efinance/AKShare data

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_kline_and_turnover.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_kline_and_turnover.py --stocks 000001,600519
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_kline_and_turnover.py --dry-run
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
TARGET_START = "2000-01-01"
TARGET_END = "2026-07-26"


def _bs_code(code: str) -> str:
    """Convert A-share code to Baostock symbol."""
    if code.startswith("6") or code.startswith("9"):
        return f"sh.{code}"
    elif code.startswith("8") or code.startswith("4"):
        return f"bj.{code}"
    else:
        return f"sz.{code}"


def main():
    parser = argparse.ArgumentParser(
        description="Backfill K-line 2000-2014 + turnover via Baostock"
    )
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all)")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Delay between stocks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan only, don't download")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.storage import DataStorage

    cfg = load_config()
    daily_dir = Path(cfg.project.data_dir) / "a_shares" / "daily"
    storage = DataStorage(cfg.project.data_dir)

    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([f.stem for f in daily_dir.glob("*.parquet")])

    # ── Scan: what does each stock need? ──
    need_backfill = []   # (code, min_date, max_date) — gap before min_date
    need_turnover = []   # (code, min_date, max_date) — turnover missing
    already_ok = 0

    for code in codes:
        path = daily_dir / f"{code}.parquet"
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            min_d = df["date"].min()
            max_d = df["date"].max()

            has_gap = min_d > pd.Timestamp(TARGET_START)
            has_turnover = "turnover" in df.columns and df["turnover"].notna().sum() > 0

            if has_gap:
                need_backfill.append((code, min_d.strftime("%Y-%m-%d"), max_d.strftime("%Y-%m-%d")))
            if not has_turnover:
                need_turnover.append((code, min_d.strftime("%Y-%m-%d"), max_d.strftime("%Y-%m-%d")))

            if not has_gap and has_turnover:
                already_ok += 1
        except Exception as e:
            logger.warning("%s: scan failed: %s", code, e)

    logger.info("Scan complete: %d stocks total", len(codes))
    logger.info("  Need K-line backfill (2000-2014): %d", len(need_backfill))
    logger.info("  Need turnover: %d", len(need_turnover))
    logger.info("  Already complete: %d", already_ok)

    if args.dry_run:
        if need_backfill:
            logger.info("Backfill samples: %s", need_backfill[:5])
        if need_turnover:
            logger.info("Turnover samples: %s", need_turnover[:5])
        return 0

    # Build unified todo list: each stock queried once
    todo = set()
    for code, mn, mx in need_backfill:
        todo.add((code, mn, mx))
    for code, mn, mx in need_turnover:
        todo.add((code, mn, mx))
    todo = sorted(todo)

    if not todo:
        logger.info("Nothing to do. All stocks complete.")
        return 0

    logger.info("%d stocks to process (one Baostock query each)", len(todo))

    import baostock as bs

    success, fail = 0, 0
    t_start = time.time()
    total_backfill_rows = 0
    total_turnover_merged = 0

    for i, (code, min_existing, max_existing) in enumerate(todo):
        if i > 0:
            time.sleep(args.sleep)

        path = daily_dir / f"{code}.parquet"

        try:
            # ── Login ──
            lg = bs.login()
            if lg is None or lg.error_code != "0":
                logger.error("[%d/%d] %s: Baostock login failed", i + 1, len(todo), code)
                fail += 1
                continue

            # ── One query: full date range ──
            rs = bs.query_history_k_data_plus(
                _bs_code(code),
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=TARGET_START,
                end_date=TARGET_END,
                frequency="d",
                adjustflag="2",  # forward-adjusted
            )

            if rs is None or rs.error_code != "0":
                err = "None" if rs is None else rs.error_msg
                logger.warning("[%d/%d] %s: query failed: %s", i + 1, len(todo), code, err)
                bs.logout()
                fail += 1
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            bs.logout()

            if not rows:
                logger.info("[%d/%d] %s: Baostock returned empty", i + 1, len(todo), code)
                fail += 1
                continue

            bs_df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                                  "volume", "amount", "turnover", "pct_change"])
            bs_df["date"] = pd.to_datetime(bs_df["date"])
            for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]:
                bs_df[col] = pd.to_numeric(bs_df[col], errors="coerce")
            bs_df = bs_df.dropna(subset=["open", "high", "low", "close"])
            bs_df["pct_change"] = bs_df["pct_change"].fillna(0.0)

            if bs_df.empty:
                fail += 1
                continue

            # ── Read existing data ──
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"])
            existing["stock_code"] = existing["stock_code"].fillna(code)

            # Capture pre-backfill provenance BEFORE any write: save_daily
            # overwrites the manifest's flat source field, so reading it after
            # the first write would mis-attribute the existing rows (§八-1).
            m = storage.manifest(code, market="a_shares")
            existing_source = (m or {}).get("source", "unknown")
            existing_adjust = (m or {}).get("adjust", "unknown")

            # ── Split Baostock result ──
            bs_backfill = bs_df[bs_df["date"] < existing["date"].min()].copy()
            bs_overlap = bs_df[bs_df["date"] >= existing["date"].min()].copy()

            backfill_rows = len(bs_backfill)
            total_backfill_rows += backfill_rows

            # ── Strategy: prefer existing OHLCV, add Baostock turnover ──
            # 1. Add turnover to existing data from overlap period
            if not bs_overlap.empty and "turnover" in bs_overlap.columns:
                turnover_map = bs_overlap[["date", "turnover"]].dropna(subset=["turnover"])
                if not turnover_map.empty:
                    existing = existing.merge(turnover_map, on="date", how="left",
                                              suffixes=("", "_bs"))
                    if "turnover_bs" in existing.columns:
                        if "turnover" in existing.columns:
                            existing["turnover"] = existing["turnover"].fillna(existing["turnover_bs"])
                        else:
                            existing["turnover"] = existing["turnover_bs"]
                        existing = existing.drop(columns=["turnover_bs"])
                    total_turnover_merged += turnover_map["date"].nunique()

            # 2. Backfill: Baostock OHLCV + turnover for pre-existing dates
            bs_backfill["stock_code"] = code
            bs_backfill = bs_backfill[["date", "open", "high", "low", "close",
                                         "volume", "amount", "turnover", "pct_change", "stock_code"]]

            # Fill turnover gaps with 0 (established canonical convention).
            if "turnover" in existing.columns:
                existing["turnover"] = existing["turnover"].fillna(0.0)

            # Persist via the storage API so the manifest + source segments
            # stay in sync (§八-1). Existing rows keep the pre-backfill provider
            # attribution; the backfill dates are new Baostock rows (per-date
            # segments are derived inside _build_source_segments).
            existing.attrs["source"] = existing_source
            existing.attrs["adjustment_mode"] = existing_adjust
            storage.save_daily(existing, market="a_shares")
            if not bs_backfill.empty:
                bs_backfill.attrs["source"] = "baostock"
                bs_backfill.attrs["adjustment_mode"] = "qfq"
                storage.save_daily(bs_backfill, market="a_shares")

            all_dates = (
                pd.concat([bs_backfill["date"], existing["date"]], ignore_index=True)
                if not bs_backfill.empty else existing["date"]
            )
            min_d = all_dates.min().strftime("%Y-%m-%d")
            max_d = all_dates.max().strftime("%Y-%m-%d")
            has_t = "turnover" in existing.columns and existing["turnover"].notna().sum() > 0
            total_rows = len(existing) + (len(bs_backfill) if not bs_backfill.empty else 0)
            logger.info("[%d/%d] %s: %d rows [%s → %s] +%d backfill, turnover=%s",
                        i + 1, len(todo), code, total_rows, min_d, max_d,
                        backfill_rows, "Y" if has_t else "N")

            success += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                logger.info("  ... %d/%d done, %.1f stk/min",
                            i + 1, len(todo), (i + 1) / elapsed * 60)

        except Exception as e:
            logger.error("[%d/%d] %s: %s", i + 1, len(todo), code, e)
            try:
                bs.logout()
            except Exception:
                pass
            fail += 1

    elapsed = time.time() - t_start
    logger.info("Done: %d ok, %d fail — %d backfill rows, %d turnover merged (%.1f min)",
                success, fail, total_backfill_rows, total_turnover_merged, elapsed / 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
