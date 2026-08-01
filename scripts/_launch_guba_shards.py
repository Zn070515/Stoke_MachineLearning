"""Launch Guba download shards as subprocesses.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/_launch_guba_shards.py
"""
import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")
N_SHARDS = 8

procs = []
for k in range(N_SHARDS):
    log = open(PROJECT / f"download_guba_{k}.log", "w")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT)
    cmd = [
        PYTHON,
        str(PROJECT / "scripts" / "download_guba.py"),
        "--shard", f"{k}/{N_SHARDS}",
        "--sort", "comment",
        "--page-delay", "0.1",
        "--max-pages", "500",
        "--sleep", "0",
        "--start", "2025-01-01",
        "--no-resume",
        "--skip-sentiment",
        "--no-bodies",
    ]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    procs.append((k, p, log))
    print(f"Shard {k}/{N_SHARDS} PID={p.pid}")

print(f"\nAll {N_SHARDS} shards running. Monitor with:")
print(f"  tail -f {PROJECT}/download_guba_0.log")

try:
    while procs:
        for k, p, log in list(procs):
            ret = p.poll()
            if ret is not None:
                log.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"Shard {k}: {status}")
                procs.remove((k, p, log))
        if procs:
            time.sleep(5)
    print("\nAll shards complete.")
except KeyboardInterrupt:
    print("\nStopping...")
    for k, p, log in procs:
        p.terminate()
        log.close()
