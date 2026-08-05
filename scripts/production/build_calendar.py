#!/usr/bin/env python
"""Materialize + validate the external trading-calendar artifact.

The A-share calendar is the EXCHANGE-published calendar, not "workdays minus
holidays".  This script persists it as a self-describing
`data/exchange_calendar/a_shares.parquet` (date / is_open / exchange / source /
version) — the single artifact all calendar consumers read.
It then cross-checks the artifact against the verified in-code generator and
exits 1 on any drift so the mismatch surfaces loudly.

Run:
    PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_calendar.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from stoke_ml.config import load_config
from stoke_ml.data.calendar import VERIFIED_UNTIL, save_calendar, validate_calendar


def main() -> int:
    cfg = load_config()
    data_dir = cfg.project.data_dir
    path = save_calendar(data_dir, "a_shares")
    report = validate_calendar(data_dir, "a_shares")
    print(f"calendar artifact: {path}")
    print(f"trading days:      {report['trading_days']}")
    # Dates past verified_until are forward estimates, not verified
    # exchange fact — strict formal flows fail beyond this point.
    print(f"verified_until:    {VERIFIED_UNTIL['a_shares']}")
    if report["ok"]:
        print("validate: OK — artifact matches the verified generator")
        return 0
    print(f"validate: FAIL — {report['reason']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
