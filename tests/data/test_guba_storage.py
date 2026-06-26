"""Tests for GubaStorage — 3-layer medallion storage for Guba forum posts."""
import tempfile

import pandas as pd
import pytest

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.guba_storage import GubaStorage, GUBA_COLS


class TestGubaStorage:

    @staticmethod
    def _sample_posts():
        return pd.DataFrame({
            "date": ["2026-06-25", "2026-06-25", "2026-06-23"],
            "time": ["10:30:00", "16:00:00", "14:00:00"],
            "title": ["大涨了", "明天要跌", "稳住了"],
            "body": ["利好！", "利空消息", "没什么"],
            "post_id": ["123", "456", "789"],
            "url": ["http://x.com/1", "http://x.com/2", "http://x.com/3"],
        })

    def test_save_and_load_raw(self):
        """Bronze: save + load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            df = self._sample_posts()
            storage.save_raw("600519", df)

            loaded = storage.load_raw("600519")
            assert len(loaded) == 3
            assert set(loaded["post_id"]) == {"123", "456", "789"}

    def test_bronze_to_silver_pit_alignment(self):
        """16:00 post moves to next trading day, 10:30 post stays put."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            df = self._sample_posts()
            storage.save_raw("600519", df)

            silver = storage.bronze_to_silver("600519")

            # Post 123 at 10:30 on Thursday — same day
            post_123 = silver[silver["post_id"] == "123"].iloc[0]
            assert pd.to_datetime(post_123["aligned_date"]).date() == pd.Timestamp(
                "2026-06-25"
            ).date()

            # Post 456 at 16:00 on Thursday → aligned to Friday
            post_456 = silver[silver["post_id"] == "456"].iloc[0]
            assert pd.to_datetime(post_456["aligned_date"]).date() == pd.Timestamp(
                "2026-06-26"
            ).date()

            # Post 789 at 14:00 on 2026-06-23 — same day
            post_789 = silver[silver["post_id"] == "789"].iloc[0]
            assert pd.to_datetime(post_789["aligned_date"]).date() == pd.Timestamp(
                "2026-06-23"
            ).date()

    def test_silver_to_gold_daily_aggregation(self):
        """Gold: correct columns and aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)
            # Use trading-day aligned_dates so ZI fill does not drop rows.
            # 2026-06-23 (Tue) and 2026-06-24 (Wed) are both trading days.
            df = pd.DataFrame({
                "date": ["2026-06-22", "2026-06-22", "2026-06-23"],
                "time": ["10:30:00", "16:00:00", "14:00:00"],
                "title": ["大涨了", "明天要跌", "稳住了"],
                "body": ["利好！", "利空消息", "没什么"],
                "post_id": ["123", "456", "789"],
                "url": ["http://x.com/1", "http://x.com/2", "http://x.com/3"],
                "sentiment_title": [0.8, -0.5, 0.0],
                "aligned_date": pd.to_datetime(
                    ["2026-06-23", "2026-06-24", "2026-06-24"]
                ),
            })
            storage.save_silver("600519", df)

            gold = storage.silver_to_gold("600519")

            for col in GUBA_COLS:
                assert col in gold.columns, f"Missing column: {col}"

            # Day 23: 1 post (sentiment 0.8)
            day_23 = gold[gold["date"] == pd.Timestamp("2026-06-23").date()]
            assert len(day_23) == 1
            row = day_23.iloc[0]
            assert bool(row["has_guba_post"]) is True
            assert row["guba_post_count"] == 1
            assert abs(row["guba_sentiment_mean"] - 0.8) < 0.01
            assert row["guba_positive_ratio"] == 1.0  # 0.8 > 0.2
            assert row["guba_negative_ratio"] == 0.0  # 0.8 is not < -0.2

            # Day 24: 2 posts (sentiment -0.5, 0.0)
            day_24 = gold[gold["date"] == pd.Timestamp("2026-06-24").date()]
            assert len(day_24) == 1
            row = day_24.iloc[0]
            assert bool(row["has_guba_post"]) is True
            assert row["guba_post_count"] == 2
            assert abs(row["guba_sentiment_mean"] - (-0.25)) < 0.01
            assert row["guba_positive_ratio"] == 0.0  # neither > 0.2
            assert row["guba_negative_ratio"] == 0.5  # -0.5 < -0.2, 0.0 is not

    def test_load_daily_sentiment_date_range(self):
        """Date range filtering works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = GubaStorage(tmpdir)

            # Create gold data with multiple dates (all trading days)
            gold_df = pd.DataFrame({
                "date": pd.to_datetime(["2026-06-23", "2026-06-24", "2026-06-25"]),
                "stock_code": ["600519"] * 3,
                "guba_sentiment_mean": [0.5, -0.3, 0.1],
                "guba_sentiment_std": [0.2, 0.1, 0.0],
                "guba_post_count": [5, 3, 1],
                "guba_positive_ratio": [0.6, 0.0, 0.0],
                "guba_negative_ratio": [0.0, 0.33, 0.0],
                "has_guba_post": [True, True, True],
            })
            storage.save_daily_sentiment(gold_df)

            # Load sub-range
            result = storage.load_daily_sentiment("600519", "2026-06-23", "2026-06-24")
            assert len(result) == 2
            dates = set(pd.to_datetime(result["date"]).dt.date)
            assert pd.Timestamp("2026-06-23").date() in dates
            assert pd.Timestamp("2026-06-24").date() in dates
            assert pd.Timestamp("2026-06-25").date() not in dates
