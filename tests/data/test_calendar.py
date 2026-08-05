"""TradingCalendar correctness tests.

A-shares NEVER trade on weekends — not even on 调休 makeup workdays.  The
calendar is weekdays MINUS official SSE/SZSE holiday closures, verified to
match the SSE official trading calendar exactly for 2001-2026.
"""
import datetime as dt
import hashlib

import pandas as pd
import pytest
from omegaconf import OmegaConf

from stoke_ml.data.calendar import (
    VERIFIED_UNTIL,
    TradingCalendar,
    build_calendar_frame,
    calendar_artifact_hash,
    get_research_calendar,
    load_calendar,
    most_recent_completed_trading_day,
    save_calendar,
    validate_calendar,
)


@pytest.fixture
def cal():
    return TradingCalendar("a_shares")


class TestNoWeekendTrading:
    """Weekends are never trading days, even 调休 makeup workdays."""

    @pytest.mark.parametrize("d", [
        dt.date(2026, 2, 14),   # 调休 makeup Sat (春节 2026)
        dt.date(2026, 2, 22),   # 调休 makeup Sun
        dt.date(2026, 5, 9),    # 调休 makeup Sat (劳动节 2026)
        dt.date(2026, 10, 10),  # 调休 makeup Sat (国庆 2026)
        dt.date(2015, 10, 10),  # 调休 makeup Sat (国庆 2015)
        dt.date(2024, 2, 18),   # 调休 makeup Sun (春节 2024)
    ])
    def test_makeup_weekend_not_trading(self, cal, d):
        assert d.weekday() >= 5
        assert not cal.is_trading_day(d)

    def test_plain_weekend_not_trading(self, cal):
        assert not cal.is_trading_day(dt.date(2026, 8, 2))   # Sunday
        assert not cal.is_trading_day(dt.date(2026, 8, 1))   # Saturday


class Test2026Holidays:
    """2026 closures per 上证公告〔2025〕45号."""

    @pytest.mark.parametrize("d", [
        dt.date(2026, 1, 1),    # 元旦
        dt.date(2026, 1, 2),    # 元旦
        dt.date(2026, 2, 16),   # 春节
        dt.date(2026, 2, 17), dt.date(2026, 2, 18), dt.date(2026, 2, 19),
        dt.date(2026, 2, 20), dt.date(2026, 2, 23),
        dt.date(2026, 4, 6),    # 清明
        dt.date(2026, 5, 1), dt.date(2026, 5, 4), dt.date(2026, 5, 5),  # 劳动节
        dt.date(2026, 6, 19),   # 端午 (was 6/22 wrongly, 6/19 missing)
        dt.date(2026, 9, 25),   # 中秋 (was 9/28 wrongly)
        dt.date(2026, 10, 1), dt.date(2026, 10, 2), dt.date(2026, 10, 5),
        dt.date(2026, 10, 6), dt.date(2026, 10, 7),  # 国庆
    ])
    def test_closure(self, cal, d):
        assert not cal.is_trading_day(d)

    @pytest.mark.parametrize("d", [
        dt.date(2026, 1, 5),    # 元旦 resume
        dt.date(2026, 2, 24),   # 春节 resume
        dt.date(2026, 6, 22),   # 端午 resume (real trading day, per data)
        dt.date(2026, 9, 28),   # 中秋 resume
        dt.date(2026, 10, 8),   # 国庆 resume
        dt.date(2026, 7, 15),   # normal Wednesday
    ])
    def test_trading(self, cal, d):
        assert cal.is_trading_day(d)


class TestHistoricalClosures:
    """Pre-2015 closures the calendar used to fabricate as trading days."""

    @pytest.mark.parametrize("d", [
        dt.date(2001, 10, 1),   # 国庆 2001
        dt.date(2002, 2, 20),   # 春节 2002
        dt.date(2004, 5, 3),    # 五一 2004
        dt.date(2008, 6, 9),    # 端午 2008 (first 端午 holiday)
        dt.date(2008, 4, 4),    # 清明 2008 (first 清明 holiday)
        dt.date(2009, 10, 8),   # 国庆 2009
        dt.date(2010, 2, 16),   # 春节 2010
        dt.date(2013, 9, 19),   # 中秋 2013
        dt.date(2014, 10, 1),   # 国庆 2014
        dt.date(2018, 12, 31),  # 2019 元旦 arrangement (was missing)
    ])
    def test_closure(self, cal, d):
        assert not cal.is_trading_day(d)

    @pytest.mark.parametrize("d", [
        dt.date(2001, 7, 6),    # normal Friday 2001
        dt.date(2008, 6, 18),   # normal Wednesday 2008
        dt.date(2014, 9, 11),   # normal Thursday 2014
    ])
    def test_normal_trading(self, cal, d):
        assert cal.is_trading_day(d)


class TestSequences:
    def test_get_trading_days_excludes_holidays_and_weekends(self, cal):
        days = cal.get_trading_days("2010-02-01", "2010-02-28")
        assert days == sorted(set(days))
        # 春节 2010 closures (2/15-2/19) excluded; all returned are weekdays
        # outside those dates
        for d in days:
            assert d.weekday() < 5
        assert dt.date(2010, 2, 15) not in days
        assert dt.date(2010, 2, 16) not in days
        assert dt.date(2010, 2, 24) in days  # normal Wed after holiday

    def test_get_trading_days_2001_golden_week(self, cal):
        # 2001 国庆 7-day break: 10/1-10/7 closed, resumes 10/8
        days = cal.get_trading_days("2001-09-28", "2001-10-10")
        assert days == [
            dt.date(2001, 9, 28),
            dt.date(2001, 10, 8), dt.date(2001, 10, 9), dt.date(2001, 10, 10),
        ]

    def test_next_trading_day_across_spring_festival(self, cal):
        # 2026-02-13 (Fri) is the last day before 春节; next is 2/24 (Tue)
        assert cal.next_trading_day(dt.date(2026, 2, 13)) == dt.date(2026, 2, 24)

    def test_next_trading_day_after_weekend(self, cal):
        assert cal.next_trading_day(dt.date(2026, 7, 31)) == dt.date(2026, 8, 3)


class TestSeededCounts:
    """A few year-end snapshots to guard against wholesale regression."""

    @pytest.mark.parametrize("year,expected_days", [
        (2002, 237), (2008, 246), (2013, 238), (2019, 244),
    ])
    def test_year_trading_day_count(self, cal, year, expected_days):
        days = cal.get_trading_days(f"{year}-01-01", f"{year}-12-31")
        assert len(days) == expected_days, (
            f"{year}: {len(days)} != {expected_days}")


class TestExternalCalendar:
    """The calendar is externalized as a self-describing
    exchange_calendar.parquet (date / is_open / exchange / source / version)
    so consumers read a data artifact instead of parsing holiday rules.  The
    artifact is authoritative when present and must match the verified
    generator exactly."""

    def test_build_calendar_frame_schema(self):
        frame = build_calendar_frame("a_shares")
        assert {"date", "is_open", "exchange", "source", "version",
                "verified_until", "generated_at",
                "status_after_verified_until"} <= set(frame.columns)
        assert frame["date"].is_monotonic_increasing
        assert not frame["date"].duplicated().any()
        # Every date is a weekday; is_open never true on a weekend.
        assert (frame["date"].dt.weekday < 5).all()
        assert not frame.loc[frame["date"].dt.weekday >= 5, "is_open"].any()
        # Version + exchange stamped on every row.
        assert (frame["version"] == TradingCalendar.CALENDAR_VERSION).all()
        assert (frame["exchange"] == "SSE/SZSE/BSE").all()
        # Sanity: a known holiday is closed, a normal mid-week day is open.
        assert not frame.loc[frame["date"] == pd.Timestamp("2018-12-31"), "is_open"].iloc[0]
        assert frame.loc[frame["date"] == pd.Timestamp("2010-02-24"), "is_open"].iloc[0]
        # The artifact records the verified window as metadata.
        assert (frame["verified_until"] == pd.Timestamp("2026-12-31")).all()
        assert (frame["status_after_verified_until"] == "UNKNOWN").all()
        assert frame["generated_at"].notna().all()
        # A row past verified_until (a 2027 weekday) is only a forward estimate.
        assert pd.Timestamp(frame.loc[frame["date"] == pd.Timestamp("2027-01-04"),
                                      "date"].iloc[0]).date() > VERIFIED_UNTIL["a_shares"]

    def test_save_load_roundtrip_is_authoritative(self, tmp_path):
        path = save_calendar(tmp_path, "a_shares")
        assert path.exists()
        # A calendar loaded from the artifact answers identically to the
        # code-only calendar across closures, resume days and ranges.
        code = TradingCalendar("a_shares")
        ext = TradingCalendar("a_shares", calendar_dir=tmp_path)
        assert ext._external is not None
        for d in [dt.date(2018, 12, 31), dt.date(2010, 2, 16),
                  dt.date(2026, 6, 19), dt.date(2010, 2, 24),
                  dt.date(2026, 8, 3), dt.date(2026, 2, 14)]:
            assert ext.is_trading_day(d) == code.is_trading_day(d), d
        assert (ext.get_trading_days("2010-02-01", "2010-02-28")
                == code.get_trading_days("2010-02-01", "2010-02-28"))
        assert (ext.get_trading_days("2001-09-28", "2001-10-10")
                == code.get_trading_days("2001-09-28", "2001-10-10"))
        # next_trading_day flows through the artifact.
        assert (ext.next_trading_day(dt.date(2026, 2, 13))
                == code.next_trading_day(dt.date(2026, 2, 13)))

    def test_validate_calendar_clean_after_save(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        report = validate_calendar(tmp_path, "a_shares")
        assert report["ok"], report
        assert report["exists"] and report["mismatches"] == 0
        assert report["trading_days"] > 0
        # No artifact → validate reports not-ok with a reason, does not raise.
        assert not validate_calendar(tmp_path / "empty", "a_shares")["ok"]

    def test_validate_calendar_flags_drift(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        # Flip one real trading day to closed: the drift guard must catch it.
        frame.loc[frame["date"] == pd.Timestamp("2010-02-24"), "is_open"] = False
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"] and report["mismatches"] == 1, report

    def test_malformed_artifact_raises(self, tmp_path):
        cal_dir = tmp_path / "exchange_calendar"
        cal_dir.mkdir()
        # Non-empty but missing required columns → load must raise loudly.
        pd.DataFrame({"date": [pd.Timestamp("2026-01-01")]}).to_parquet(
            cal_dir / "a_shares.parquet")
        with pytest.raises(ValueError, match="missing columns"):
            TradingCalendar("a_shares", calendar_dir=tmp_path)

    def test_empty_artifact_raises(self, tmp_path):
        cal_dir = tmp_path / "exchange_calendar"
        cal_dir.mkdir()
        pd.DataFrame().to_parquet(cal_dir / "a_shares.parquet")
        with pytest.raises(ValueError, match="empty"):
            load_calendar(tmp_path, "a_shares")

    def test_absent_artifact_falls_back_to_code(self, tmp_path):
        ext = TradingCalendar("a_shares", calendar_dir=tmp_path)
        assert ext._external is None
        # Identical behaviour to the code-only calendar (no artifact present).
        code = TradingCalendar("a_shares")
        assert ext.is_trading_day(dt.date(2026, 6, 19)) == code.is_trading_day(dt.date(2026, 6, 19))
        assert (ext.get_trading_days("2010-02-01", "2010-02-28")
                == code.get_trading_days("2010-02-01", "2010-02-28"))

    # ── verified_until + strict mode ────────────────────────────────

    def test_artifact_carries_verified_metadata(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        assert pd.Timestamp(frame["verified_until"].iloc[0]).date() == VERIFIED_UNTIL["a_shares"]
        assert frame["status_after_verified_until"].iloc[0] == "UNKNOWN"
        assert pd.notna(frame["generated_at"].iloc[0])
        # A strict calendar inherits verified_until from the artifact.
        ext = TradingCalendar("a_shares", calendar_dir=tmp_path, strict=True)
        assert ext.verified_until == VERIFIED_UNTIL["a_shares"]

    def test_strict_raises_beyond_verified_until(self, cal):
        strict = TradingCalendar("a_shares", strict=True)
        assert strict.verified_until == VERIFIED_UNTIL["a_shares"]
        # 2027 is a forward estimate — a formal flow must fail, not guess.
        with pytest.raises(ValueError, match="verified_until"):
            strict.is_trading_day(dt.date(2027, 1, 4))
        with pytest.raises(ValueError, match="verified_until"):
            strict.get_trading_days("2027-01-01", "2027-01-31")
        # Even a query that only ENDS past verified_until fails.
        with pytest.raises(ValueError, match="verified_until"):
            strict.get_trading_days("2026-12-30", "2027-01-02")

    def test_strict_within_verified_ok(self, cal):
        strict = TradingCalendar("a_shares", strict=True)
        assert strict.is_trading_day(dt.date(2026, 7, 15))
        assert strict.is_trading_day(dt.date(2026, 6, 19)) is False  # 端午 2026
        assert strict.get_trading_days("2026-02-09", "2026-02-27")[0] == dt.date(2026, 2, 9)
        assert strict.next_trading_day(dt.date(2026, 2, 13)) == dt.date(2026, 2, 24)

    def test_non_strict_still_answers_forward_estimates(self, cal):
        # Non-strict (downloader/scheduling) keeps answering 2027 estimates.
        assert cal.is_trading_day(dt.date(2027, 1, 4))
        assert dt.date(2027, 1, 4) in cal.get_trading_days("2027-01-01", "2027-01-31")

    def test_artifact_gap_raises_at_load(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        # Drop a weekday INSIDE the artifact's window: a torn/partial artifact.
        frame = frame[frame["date"] != pd.Timestamp("2010-02-24")]
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        with pytest.raises(ValueError, match="incomplete"):
            load_calendar(tmp_path, "a_shares")
        with pytest.raises(ValueError, match="incomplete"):
            TradingCalendar("a_shares", calendar_dir=tmp_path)

    # ── full outer join validation (no vanishing dates) ───────────

    def test_validate_calendar_detects_missing_date(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        # Truncate the window to 2000-2020: the artifact is complete INSIDE its
        # own window (load succeeds) but missing every 2020+ weekday that the
        # generator covers — only a full outer join surfaces that.
        frame = frame[frame["date"] <= pd.Timestamp("2020-12-31")]
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"]
        assert report["problems"]["missing_dates"] > 0, report

    def test_validate_calendar_reports_interior_gap(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        # Dropping ONE interior weekday makes load_calendar raise "incomplete";
        # validate_calendar must report that loudly, not silently pass.
        frame = frame[frame["date"] != pd.Timestamp("2010-02-24")]
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"]
        assert "incomplete" in report["reason"], report

    def test_validate_calendar_detects_extra_date(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        # A spurious Saturday row that the generator would never produce.
        extra = pd.DataFrame([{
            "date": pd.Timestamp("2026-08-01"), "is_open": True,
            "exchange": "SSE/SZSE/BSE", "source": "tampered",
            "version": TradingCalendar.CALENDAR_VERSION,
            "verified_until": pd.Timestamp("2026-12-31"),
            "generated_at": pd.Timestamp.now(tz="UTC"),
            "status_after_verified_until": "UNKNOWN",
        }])
        pd.concat([frame, extra], ignore_index=True).to_parquet(
            tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"]
        assert report["problems"]["extra_dates"] == 1, report

    def test_validate_calendar_detects_version_and_source_mismatch(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        frame["version"] = "stale-version"
        frame["source"] = "tampered"
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"]
        assert report["problems"]["version_mismatch"] > 0, report
        assert report["problems"]["source_mismatch"] > 0, report

    def test_validate_calendar_detects_verified_until_mismatch(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares")
        frame["verified_until"] = pd.Timestamp("2025-12-31")
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        report = validate_calendar(tmp_path, "a_shares")
        assert not report["ok"]
        assert report["problems"]["verified_until_mismatch"] is True, report


class TestResearchCalendarFactory:
    """get_research_calendar unifies every formal consumer on the frozen
    exchange_calendar artifact — no module silently builds a hardcoded default
    calendar.  The factory attaches the artifact when present, inherits strict
    mode, and resolves data_dir lazily from the project config when omitted."""

    def test_factory_with_artifact_uses_external_calendar(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        cal = get_research_calendar(data_dir=tmp_path)
        assert cal._external is not None
        assert cal.verified_until == VERIFIED_UNTIL["a_shares"]
        # Queries flow through the artifact, not the code fallback.
        assert cal.is_trading_day(dt.date(2026, 6, 19)) is False  # 端午 2026
        assert cal.is_trading_day(dt.date(2026, 7, 15)) is True

    def test_factory_strict_fails_beyond_verified_until(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        cal = get_research_calendar(strict=True, data_dir=tmp_path)
        assert cal.verified_until == VERIFIED_UNTIL["a_shares"]
        with pytest.raises(ValueError, match="verified_until"):
            cal.is_trading_day(dt.date(2027, 1, 4))
        with pytest.raises(ValueError, match="verified_until"):
            cal.get_trading_days("2026-12-30", "2027-01-02")
        # Non-strict (aux PIT mapping) keeps answering forward estimates.
        loose = get_research_calendar(data_dir=tmp_path)
        assert loose.is_trading_day(dt.date(2027, 1, 4)) is True

    def test_factory_absent_artifact_falls_back_to_code(self, tmp_path):
        cal = get_research_calendar(data_dir=tmp_path)
        assert cal._external is None
        # Semantics identical to the artifact-free calendar (fallback).
        assert cal.is_trading_day(dt.date(2026, 6, 19)) is False

    def test_factory_default_data_dir_from_config(self, tmp_path, monkeypatch):
        # data_dir omitted → resolved lazily from project config; the calendar
        # still reads the artifact that lives under that data root.
        save_calendar(tmp_path, "a_shares")
        cfg = OmegaConf.create({"project": {"data_dir": str(tmp_path)}})
        monkeypatch.setattr("stoke_ml.config.load_config", lambda **k: cfg)
        cal = get_research_calendar()
        assert cal._external is not None
        assert cal.verified_until == VERIFIED_UNTIL["a_shares"]


class TestCalendarArtifactHash:
    """§九: the gate report must bind the artifact's CONTENT hash, not a version
    string.  The hash is text-canonical (drops generated_at / parquet metadata)
    so a fresh save round-trips identically, and it matches the digest
    ``train_panel`` freezes as experiment identity for the same data root."""

    def test_hash_is_deterministic_after_roundtrip(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        h = calendar_artifact_hash(tmp_path)
        assert h == calendar_artifact_hash(tmp_path)
        assert len(h) == 16
        assert all(ch in "0123456789abcdef" for ch in h)

    def test_hash_is_content_sensitive(self, tmp_path):
        save_calendar(tmp_path, "a_shares")
        h = calendar_artifact_hash(tmp_path)
        frame = load_calendar(tmp_path, "a_shares")
        # Flip one real trading day to closed: content changed → hash flips.
        frame.loc[frame["date"] == pd.Timestamp("2010-02-24"), "is_open"] = False
        frame.to_parquet(tmp_path / "exchange_calendar" / "a_shares.parquet")
        assert calendar_artifact_hash(tmp_path) != h

    def test_missing_artifact_hashes_code_fallback(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        h = calendar_artifact_hash(empty)
        assert h == calendar_artifact_hash(empty)
        assert len(h) == 16

    def test_hash_matches_train_panel_canonical_digest(self, tmp_path):
        """The digest equals the canonical text encoding train_panel records as
        experiment identity — a gate report can be cross-checked against it."""
        save_calendar(tmp_path, "a_shares")
        frame = load_calendar(tmp_path, "a_shares").drop(columns=["generated_at"])
        cols = sorted(frame.columns)
        frame = frame.sort_values("date").reset_index(drop=True)
        digest = "\n".join(
            "|".join(str(row[c]) for c in cols) for _, row in frame.iterrows()
        ).encode("utf-8")
        assert calendar_artifact_hash(tmp_path) == hashlib.sha1(digest).hexdigest()[:16]


class TestMostRecentCompletedTradingDay:
    """§九: the freshness reference is the most recent COMPLETED session — a
    dataset current through it stays fresh across 春节/国庆 7-8 day closures."""

    def test_ref_on_closure_day_walks_back_over_holiday(self, cal):
        # 2026-02-23 is a 春节 closure Monday; the last real session is 2/13.
        assert most_recent_completed_trading_day(
            cal, dt.date(2026, 2, 23)) == dt.date(2026, 2, 13)

    def test_ref_on_trading_day_is_previous_session(self, cal):
        # Today's own session is not yet complete — the most recent completed is
        # the previous trading day.
        assert most_recent_completed_trading_day(
            cal, dt.date(2026, 8, 5)) == dt.date(2026, 8, 4)
        assert most_recent_completed_trading_day(
            cal, dt.date(2026, 2, 13)) == dt.date(2026, 2, 12)

    def test_ref_on_weekend_is_previous_friday(self, cal):
        assert most_recent_completed_trading_day(
            cal, dt.date(2026, 8, 1)) == dt.date(2026, 7, 31)

    def test_ref_before_first_session_clamps_to_first_day(self, cal):
        """A ref_date before the market's earliest session must clamp to the
        first trading day instead of fabricating a pre-market weekday (which an
        unbounded backward walk would return)."""
        assert cal.first_trading_day() == dt.date(2000, 1, 3)
        assert most_recent_completed_trading_day(
            cal, dt.date(1990, 1, 1)) == cal.first_trading_day()

    def test_ref_before_first_session_clamps_external(self, tmp_path):
        """The same clamp holds for an artifact-backed calendar (earliest
        is_open row of the frozen frame)."""
        save_calendar(tmp_path, "a_shares")
        ext = TradingCalendar("a_shares", calendar_dir=tmp_path)
        assert ext.first_trading_day() == dt.date(2000, 1, 3)
        assert most_recent_completed_trading_day(
            ext, dt.date(1990, 1, 1)) == dt.date(2000, 1, 3)
