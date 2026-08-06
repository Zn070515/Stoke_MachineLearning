"""Data sources honor the frozen exchange_calendar artifact (§九).

Each source that internally enumerates trading days accepts a ``calendar_dir``
(data root) at construction.  When the data root carries a frozen
``exchange_calendar/a_shares.parquet`` artifact, that artifact is authoritative
for which dates the source fetches — not the code-builtin calendar.  With no
``calendar_dir`` the code-builtin calendar is used unchanged (backward compat).

These tests write an artifact that DIFFERS from the code-builtin calendar (a
normal weekday marked closed) so they can prove which calendar the source
actually read, instead of silently passing because both calendars agree.
"""
import datetime as dt
import sys
import types

import pandas as pd
import pytest

from stoke_ml.data.calendar import TradingCalendar, build_calendar_frame
from stoke_ml.data.sources.a_shares.dragon_tiger_source import DragonTigerSource
from stoke_ml.data.sources.a_shares.limit_up_source import (
    LimitUpSource,
    SENTIMENT_COLS,
)
from stoke_ml.data.sources.a_shares.margin_source import MarginTradingSource
from stoke_ml.data.sources.a_shares.sector_source import (
    INDUSTRY_RANKING_COLS,
    IndustryRankingSource,
)

# A normal weekday that the code-builtin calendar treats as OPEN; the fixture
# artifact marks it CLOSED so the two calendars provably differ.
CLOSED_DAY = dt.date(2026, 8, 11)  # Tuesday — open in code, closed in artifact
START, END = "2026-08-10", "2026-08-14"


@pytest.fixture
def frozen_dir(tmp_path):
    """Data root whose frozen artifact marks CLOSED_DAY closed."""
    frame = build_calendar_frame("a_shares")
    frame.loc[frame["date"].dt.date == CLOSED_DAY, "is_open"] = False
    out = tmp_path / "exchange_calendar"
    out.mkdir()
    frame.to_parquet(out / "a_shares.parquet", index=False)
    return str(tmp_path)


@pytest.fixture
def fake_akshare(monkeypatch):
    """Stub akshare to record the trading dates each source queries."""
    recorded: list[str] = []

    class _FakeAKShare:
        def stock_margin_detail_sse(self, date):
            recorded.append(date)  # YYYYMMDD
            return pd.DataFrame()

        def stock_margin_detail_szse(self, date):
            return pd.DataFrame()

        def stock_lhb_stock_detail_em(self, symbol, date):
            recorded.append(date)  # YYYYMMDD
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "akshare", _FakeAKShare())
    return recorded


def _artifact_days(frozen_dir: str) -> list[str]:
    cal = TradingCalendar("a_shares", calendar_dir=frozen_dir)
    return [d.strftime("%Y-%m-%d") for d in cal.get_trading_days(START, END)]


def _artifact_days8(frozen_dir: str) -> list[str]:
    return [d.replace("-", "") for d in _artifact_days(frozen_dir)]


# ── IndustryRankingSource ──────────────────────────────────────────────

class TestIndustryRankingSource:
    def test_artifact_is_authoritative(self, frozen_dir, monkeypatch):
        captured: list[str] = []
        src = IndustryRankingSource(calendar_dir=frozen_dir)

        def fake_fetch(self, date=None):
            captured.append(date)
            return pd.DataFrame(columns=INDUSTRY_RANKING_COLS)

        monkeypatch.setattr(IndustryRankingSource, "fetch", fake_fetch)
        src.fetch_batch(START, END)

        assert captured == _artifact_days(frozen_dir)
        # The artifact really excludes CLOSED_DAY, so the source did read it.
        assert "2026-08-11" not in captured

    def test_no_calendar_dir_keeps_code_builtin(self, monkeypatch):
        captured: list[str] = []
        src = IndustryRankingSource()

        def fake_fetch(self, date=None):
            captured.append(date)
            return pd.DataFrame(columns=INDUSTRY_RANKING_COLS)

        monkeypatch.setattr(IndustryRankingSource, "fetch", fake_fetch)
        src.fetch_batch(START, END)

        # Backward compat: code-builtin calendar still treats CLOSED_DAY as open.
        assert "2026-08-11" in captured


# ── LimitUpSource ──────────────────────────────────────────────────────

class TestLimitUpSource:
    def test_fetch_batch_uses_artifact(self, frozen_dir, monkeypatch):
        captured: list[str] = []
        src = LimitUpSource(calendar_dir=frozen_dir)

        def make_fake(pool):
            def fake_pool(self, date):
                captured.append((pool, date))
                return pd.DataFrame()

            return fake_pool

        for pool in ("zt", "zb", "dt", "yzt"):
            monkeypatch.setattr(LimitUpSource, f"fetch_{pool}_pool", make_fake(pool))
        src.fetch_batch(START, END)

        dates = {d for _, d in captured}
        assert sorted(dates) == _artifact_days(frozen_dir)
        assert "2026-08-11" not in dates

    def test_fetch_sentiment_batch_uses_artifact(self, frozen_dir, monkeypatch):
        captured: list[str] = []
        src = LimitUpSource(calendar_dir=frozen_dir)

        def fake_fetch_sentiment(self, date):
            captured.append(date)
            return {"date": date, **{c: 0 for c in SENTIMENT_COLS if c != "date"}}

        monkeypatch.setattr(LimitUpSource, "fetch_sentiment", fake_fetch_sentiment)
        src.fetch_sentiment_batch(START, END)

        assert sorted(captured) == _artifact_days(frozen_dir)
        assert "2026-08-11" not in captured

    def test_no_calendar_dir_keeps_code_builtin(self, monkeypatch):
        captured: list[str] = []
        src = LimitUpSource()

        def fake_fetch_sentiment(self, date):
            captured.append(date)
            return {"date": date, **{c: 0 for c in SENTIMENT_COLS if c != "date"}}

        monkeypatch.setattr(LimitUpSource, "fetch_sentiment", fake_fetch_sentiment)
        src.fetch_sentiment_batch(START, END)

        assert "2026-08-11" in captured


# ── MarginTradingSource ────────────────────────────────────────────────

class TestMarginTradingSource:
    def test_fetch_daily_uses_artifact(self, frozen_dir, fake_akshare):
        src = MarginTradingSource(calendar_dir=frozen_dir)
        src.fetch_daily(START, END, sleep=0)
        assert sorted(fake_akshare) == _artifact_days8(frozen_dir)
        assert "20260811" not in fake_akshare

    def test_no_calendar_dir_keeps_code_builtin(self, fake_akshare):
        src = MarginTradingSource()
        src.fetch_daily(START, END, sleep=0)
        assert "20260811" in fake_akshare


# ── DragonTigerSource ──────────────────────────────────────────────────

class TestDragonTigerSource:
    def test_fetch_by_stock_uses_artifact(self, frozen_dir, fake_akshare):
        src = DragonTigerSource(calendar_dir=frozen_dir)
        src.fetch_by_stock("600519", START, END, sleep=0)
        assert sorted(fake_akshare) == _artifact_days8(frozen_dir)
        assert "20260811" not in fake_akshare

    def test_no_calendar_dir_keeps_code_builtin(self, fake_akshare):
        src = DragonTigerSource()
        src.fetch_by_stock("600519", START, END, sleep=0)
        assert "20260811" in fake_akshare
