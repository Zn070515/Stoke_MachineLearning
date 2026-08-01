"""Download valuation ratios (PE/PB/PS/PCF) from EastMoney K-line API.

EastMoney push2his K-line fields:
  f115 = PE TTM, f116 = PB MRQ, f120 = PS TTM, f164 = PCF TTM

Alternative to Baostock-based download_valuation.py. Uses curl-cffi TLS
spoofing (same as efinance_source) so it survives EastMoney WAF.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_valuation_em.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_valuation_em.py --stocks 600519,000001
  PYTHONPATH=. ./.venv/Scripts/python scripts/_download_valuation_em.py --start 2015-01-01

Output: data/a_shares/valuation/{code}.parquet (same format as Baostock version)
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from curl_cffi import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent

EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

VAL_FIELD_MAP = {
    "f51": "date",
    "f115": "pe_ttm",
    "f116": "pb_mrq",
    "f120": "ps_ttm",
    "f164": "pcf_ttm",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def _to_secid(code: str) -> str:
    prefix = "1" if code.startswith("6") else "0"
    return f"{prefix}.{code}"


def fetch_valuation(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": ",".join(VAL_FIELD_MAP.keys()),
        "beg": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "rtntype": "6",
        "secid": _to_secid(code),
        "klt": "101",
        "fqt": "1",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                EM_URL, params=params, headers=EM_HEADERS,
                impersonate="chrome120", timeout=30,
            )
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
            continue

        if resp.status_code != 200:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
            continue

        try:
            data = resp.json()
        except ValueError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
            continue

        rc = data.get("rc")
        if rc is not None and rc != 0:
            return pd.DataFrame()

        klines = data.get("data", {})
        if not isinstance(klines, dict):
            return pd.DataFrame()
        klines = klines.get("klines")
        if not klines:
            return pd.DataFrame()

        field_codes = params["fields2"].split(",")
        rows = [k.split(",") for k in klines]
        df = pd.DataFrame(rows, columns=field_codes)
        df.rename(columns=VAL_FIELD_MAP, inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        for col in ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"], how="all")
        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Download daily valuation ratios from EastMoney K-line API"
    )
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all missing)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between stocks (default: 0.3)")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    cfg = load_config()
    data_dir = Path(cfg.project.data_dir) / "a_shares"

    daily_dir = data_dir / "daily"
    all_stocks = {f.stem for f in daily_dir.glob("*.parquet")}

    val_dir = data_dir / "valuation"
    val_dir.mkdir(exist_ok=True)
    existing = {f.stem for f in val_dir.glob("*.parquet")}

    if args.stocks:
        stocks = [c.strip() for c in args.stocks.split(",")]
    else:
        stocks = sorted(all_stocks - existing)

    if not stocks:
        logger.info("Nothing to download — all %d stocks have valuation data.", len(existing))
        return 0

    logger.info("Valuation: %d/%d stocks cached, %d to fetch (start=%s, end=%s)",
                len(existing), len(all_stocks), len(stocks), args.start, args.end)

    t0 = time.time()
    done = fail = 0
    n = len(stocks)

    for i, code in enumerate(stocks):
        try:
            df = fetch_valuation(code, args.start, args.end)
            if not df.empty:
                df["stock_code"] = code
                out_path = val_dir / f"{code}.parquet"
                df.to_parquet(out_path, index=False, compression="lz4")
                done += 1
            else:
                fail += 1
        except Exception:
            fail += 1

        if (i + 1) % 100 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            logger.info("  [%d/%d] done=%d fail=%d (%.1f/s, ETA %.0fs)",
                        i + 1, n, done, fail, rate, eta)

        if i < n - 1:
            time.sleep(args.sleep)

    elapsed = time.time() - t0
    logger.info("Done: %d ok, %d fail, %.0fs", done, fail, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
