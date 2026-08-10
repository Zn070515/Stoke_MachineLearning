"""Unit tests for ``scripts/production/refresh_daily_tail.py``.

Pure decision logic only — no network, no real downloads, no ``AShareDownloader``:

- ``needs_refresh``: skip-vs-refresh boundary (``end == target_end`` → skip;
  ``end == target_end - 1 trading day`` → refresh).
- ``tail_start_of``: next trading day strictly after ``end``, across a weekend
  and across the 2026 春节 / 国庆 closures, using the real ``TradingCalendar``
  backed by the shipped ``data/exchange_calendar/a_shares.parquet`` artifact.
- ``manifest_end``: manifest ``end`` preferred; missing/unparseable manifest
  falls back to the parquet's max ``date``.
- ``build_plan``: the dry-run plan (current / to-refresh / unknown counts) for
  a temp dir with fake parquet + manifest files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.production.refresh_daily_tail import (
    build_plan,
    manifest_end,
    needs_refresh,
    tail_start_of,
)
from stoke_ml.data.calendar import TradingCalendar

# Real, read-only repo calendar artifact — the authoritative trading-day set.
_REPO_DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
CAL = TradingCalendar("a_shares", calendar_dir=_REPO_DATA_DIR)


def _write_fake_stock(
    data_dir: str,
    code: str,
    parquet_end: str,
    write_manifest: bool = True,
    manifest_end: str | None = None,
) -> None:
    """Create a fake daily parquet (+ optional manifest) for a code under a
    temp ``data_dir``.  ``write_manifest=False`` exercises the parquet fallback.
    """
    daily = os.path.join(data_dir, "a_shares", "daily")
    os.makedirs(daily, exist_ok=True)
    pd.DataFrame({"date": pd.to_datetime([parquet_end])}).to_parquet(
        os.path.join(daily, f"{code}.parquet"), index=False
    )
    if write_manifest:
        end = manifest_end or parquet_end
        with open(os.path.join(daily, f"{code}.manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"stock": code, "end": end}, f)


def _write_unreadable_parquet(data_dir: str, code: str) -> None:
    """A parquet with no ``date`` column → the fallback cannot derive an end."""
    daily = os.path.join(data_dir, "a_shares", "daily")
    os.makedirs(daily, exist_ok=True)
    pd.DataFrame({"other": [1.0]}).to_parquet(
        os.path.join(daily, f"{code}.parquet"), index=False
    )


# ── needs_refresh: skip-vs-refresh boundary ────────────────────────────────

def test_needs_refresh_end_equals_target_skips():
    assert needs_refresh("2026-08-07", "2026-08-07") is False


def test_needs_refresh_one_trading_day_behind_refreshes():
    # 2026-08-06 is the trading day immediately before the target.
    assert needs_refresh("2026-08-06", "2026-08-07") is True


def test_needs_refresh_far_behind_refreshes():
    # The stale-tail case the script exists for: 2026-07-23 vs target 2026-08-07.
    assert needs_refresh("2026-07-23", "2026-08-07") is True


def test_needs_refresh_none_refreshes():
    # No known end (no manifest / no parquet date) must refresh, never skip.
    assert needs_refresh(None, "2026-08-07") is True


def test_needs_refresh_end_past_target_skips():
    # A file already beyond the target is current — never fetched backward.
    assert needs_refresh("2026-08-10", "2026-08-07") is False


# ── tail_start_of: next trading day strictly after end ─────────────────────

def test_tail_start_across_weekend():
    # 2026-08-07 is a Friday; the next trading day is Monday 2026-08-10.
    assert tail_start_of("2026-08-07", CAL) == "2026-08-10"


def test_tail_start_across_spring_festival_holiday():
    # 2026-02-13 (Fri) is followed by the 春节 closure (2/16-2/23);
    # trading resumes Tuesday 2026-02-24.
    assert tail_start_of("2026-02-13", CAL) == "2026-02-24"


def test_tail_start_across_national_day_holiday():
    # 2026-10-07 (Wed) is a 国庆 closure; the next trading day is 2026-10-08.
    assert tail_start_of("2026-10-07", CAL) == "2026-10-08"


# ── manifest_end: manifest preferred, parquet fallback ─────────────────────

def test_manifest_end_reads_manifest(tmp_path):
    _write_fake_stock(str(tmp_path), "000001", "2026-07-23",
                      manifest_end="2026-08-07")
    # Manifest end wins even though the fake parquet's date is older.
    assert manifest_end(str(tmp_path), "000001") == "2026-08-07"


def test_manifest_end_falls_back_to_parquet_date(tmp_path):
    _write_fake_stock(str(tmp_path), "000002", "2026-07-23",
                      write_manifest=False)
    assert manifest_end(str(tmp_path), "000002") == "2026-07-23"


def test_manifest_end_none_when_unreadable(tmp_path):
    _write_unreadable_parquet(str(tmp_path), "000003")
    assert manifest_end(str(tmp_path), "000003") is None


# ── build_plan: dry-run counts ─────────────────────────────────────────────

def test_build_plan_counts_and_tail_starts(tmp_path):
    target = "2026-08-07"
    _write_fake_stock(str(tmp_path), "000001", target)              # current
    _write_fake_stock(str(tmp_path), "000002", "2026-08-06")        # to refresh
    _write_fake_stock(str(tmp_path), "000003", "2026-07-23",
                      write_manifest=False)                          # fallback → refresh

    current, plan, unknown = build_plan(
        ["000001", "000002", "000003"], str(tmp_path), target, CAL)

    assert current == ["000001"]
    assert unknown == []
    assert {p["code"] for p in plan} == {"000002", "000003"}
    by_code = {p["code"]: p for p in plan}
    assert by_code["000002"]["tail_start"] == "2026-08-07"  # day after 08-06
    assert by_code["000003"]["tail_start"] == "2026-07-24"  # day after 07-23


def test_build_plan_unknown_end(tmp_path):
    target = "2026-08-07"
    _write_fake_stock(str(tmp_path), "000001", target)      # current
    _write_unreadable_parquet(str(tmp_path), "000002")      # unknown end

    current, plan, unknown = build_plan(
        ["000001", "000002"], str(tmp_path), target, CAL)

    assert current == ["000001"]
    assert plan == []
    assert unknown == ["000002"]
