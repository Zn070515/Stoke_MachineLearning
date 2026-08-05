#!/usr/bin/env python3

# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Monitor EastMoney connectivity and launch guba shards when ban lifts.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_wait_and_launch_guba.py
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parent.parent
CHECK_URL = "https://guba.eastmoney.com/list,600519,f.html"
CHECK_INTERVAL = 90  # seconds between checks
MAX_WAIT = 7200  # 2 hours max

SHARD_COUNT = 6
LOG_DIR = PROJECT

# Stock files prepped earlier
SPLIT_DIR = Path.home() / "AppData" / "Local" / "Temp" / "guba_splits"


def check():
    try:
        resp = requests.get(
            CHECK_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def launch_shards():
    procs = []
    for s in range(SHARD_COUNT):
        stock_file = SPLIT_DIR / f"v2_{s}.txt"
        if not stock_file.exists():
            print(f"  Missing {stock_file}, skipping shard {s}")
            continue
        stocks = stock_file.read_text().strip()
        log_file = LOG_DIR / f"download_guba_v2_{s}.log"
        cmd = (
            f'PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_guba.py '
            f'--stocks "{stocks}" --max-pages 500 --concurrent --workers 2 '
            f'--page-delay 2.0 --sleep 0.5 '
            f'> {log_file} 2>&1'
        )
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(PROJECT),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        procs.append(proc)
        print(f"  Shard {s}: PID {proc.pid}, {len(stocks.split(','))} stocks")
    return procs


def main():
    print(f"Waiting for EastMoney ban to lift (checking every {CHECK_INTERVAL}s)...")
    started = time.time()

    while time.time() - started < MAX_WAIT:
        if check():
            elapsed = time.time() - started
            print(f"Connection restored after {elapsed:.0f}s! Launching {SHARD_COUNT} shards...")
            procs = launch_shards()
            print(f"Launched {len(procs)} shards. Monitoring progress...")
            # Keep running to show progress every 5 min
            while True:
                time.sleep(300)
                guba_count = len(list((PROJECT / "data/a_shares/guba_raw").glob("*.parquet")))
                daily_count = len(list((PROJECT / "data/a_shares/daily").glob("*.parquet")))
                print(f"  Progress: {guba_count}/{daily_count} ({100*guba_count/daily_count:.1f}%)")
                if guba_count >= daily_count:
                    print("DONE! All stocks have guba data.")
                    return 0
        else:
            elapsed = time.time() - started
            print(f"  [{elapsed:.0f}s] Still blocked, retrying in {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)

    print(f"Timed out after {MAX_WAIT}s. Please retry manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
