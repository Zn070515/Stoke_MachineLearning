# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Generic shard launcher — splits stock list across N subprocesses.

Launcher args go BEFORE the script name:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_launch_shards.py --shards 4 download_fundamentals.py --sleep 0 --start 2015-01-01
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_launch_shards.py --shards 4 download_comment.py --history --sleep 0
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_launch_shards.py --shards 4 download_news.py --max-pages 20 --sleep 0 --skip-sentiment
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_launch_shards.py --shards 4 download_guba.py --sort comment --page-delay 0.1 --max-pages 500 --sleep 0 --skip-sentiment --no-bodies
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")
N_SHARDS = 4


def get_stocks_from_disk(data_dir: str) -> list[str]:
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    if not os.path.exists(daily_dir):
        return []
    codes = set()
    for f in os.listdir(daily_dir):
        if f.endswith(".parquet"):
            codes.add(f.replace(".parquet", ""))
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description="Launch N shards of a stock-download script")
    parser.add_argument("--shards", type=int, default=N_SHARDS,
                        help=f"Number of parallel shards (default: {N_SHARDS})")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Override stock list (comma-separated)")
    parser.add_argument("script", help="Script name (e.g. download_fundamentals.py)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to the child script")
    args = parser.parse_args()

    script_path = str(PROJECT / "scripts" / args.script)
    if not os.path.exists(script_path):
        print(f"ERROR: script not found: {script_path}")
        return 1

    # Get stock list
    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        sys.path.insert(0, str(PROJECT))
        from stoke_ml.config import load_config
        cfg = load_config()
        codes = get_stocks_from_disk(cfg.project.data_dir)

    if not codes:
        print("ERROR: no stock codes found")
        return 1

    n_shards = min(args.shards, len(codes))
    shard_size = len(codes) // n_shards
    print(f"Launching {args.script} with {len(codes)} stocks across {n_shards} shards (~{shard_size}/shard)")

    procs = []
    for k in range(n_shards):
        start = k * shard_size
        end = start + shard_size if k < n_shards - 1 else len(codes)
        chunk = codes[start:end]
        chunk_str = ",".join(chunk)

        log_dir = PROJECT / "logs"
        log_dir.mkdir(exist_ok=True)
        # Include --type in log name to prevent multiple launchers from
        # overwriting each other's output (e.g. block_trade vs lockup).
        type_suffix = ""
        for i, arg in enumerate(args.script_args):
            if arg == "--type" and i + 1 < len(args.script_args):
                type_suffix = "_" + args.script_args[i + 1]
                break
        log_name = args.script.replace(".py", "") + type_suffix + f"_{k}.log"
        log = open(log_dir / log_name, "w")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(PROJECT)

        cmd = [
            PYTHON, script_path,
            "--stocks", chunk_str,
            *args.script_args,
        ]
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((k, p, log))
        print(f"  Shard {k}/{n_shards}: PID={p.pid}, {len(chunk)} stocks [{chunk[0]}...{chunk[-1]}]")

    print(f"\nAll {n_shards} shards running.")
    log_dir = PROJECT / "logs"
    print(f"  tail -f {log_dir}/{args.script.replace('.py', '')}_0.log")

    try:
        ok, fail = 0, 0
        while procs:
            for k, p, log in list(procs):
                ret = p.poll()
                if ret is not None:
                    log.close()
                    if ret == 0:
                        status = "OK"
                        ok += 1
                    else:
                        status = f"FAIL({ret})"
                        fail += 1
                    print(f"  Shard {k}: {status}")
                    procs.remove((k, p, log))
            if procs:
                time.sleep(5)
        print(f"\nDone: {ok} ok, {fail} fail")
    except KeyboardInterrupt:
        print("\nStopping...")
        for k, p, log in procs:
            p.terminate()
            log.close()


if __name__ == "__main__":
    sys.exit(main())
