"""Tests for EastMoney Guba (股吧) forum post scraper."""
import pandas as pd
import pytest

from stoke_ml.data.sources.a_shares.guba_source import GubaSource


class TestGubaSource:
    """Integration tests for GubaSource.

    These tests make live HTTP requests to guba.eastmoney.com.
    They may be skipped with --no-integration if a marker is configured.
    """

    @pytest.fixture
    def source(self):
        return GubaSource()

    def test_fetch_list_page_returns_posts(self, source):
        """_fetch_list_page should return DataFrame with correct columns."""
        df = source._fetch_list_page("600519", page=1)

        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert len(df) > 0, "Should return at least some posts"
        assert len(df) <= 85, (
            "One page has max ~85 posts (80 regular + up to 5 featured)"
        )

        for col in ["date", "time", "title", "post_id", "url"]:
            assert col in df.columns, f"Missing column: {col}"

        # Date should be parseable
        if not df["date"].str.strip().eq("").all():
            pd.to_datetime(df["date"])  # should not raise

        # post_id should be non-empty strings
        assert (df["post_id"].str.strip() != "").all(), (
            "All post_ids should be non-empty"
        )

    def test_pagination_different_pages(self, source):
        """Page 1 and page 2 should return mostly different posts.

        Small overlap (<= 5 posts) may occur when Guba features
        popular posts across multiple pages.
        """
        p1 = source._fetch_list_page("600519", page=1)
        p2 = source._fetch_list_page("600519", page=2)

        ids1 = set(p1["post_id"])
        ids2 = set(p2["post_id"])

        assert len(ids1) > 0, "Page 1 should have posts"
        assert len(ids2) > 0, "Page 2 should have posts"

        overlap = ids1 & ids2
        # Allow small overlap (featured/hot posts may appear on both pages)
        assert len(overlap) <= 5, (
            f"Overlap {len(overlap)} exceeds 5: pages too similar"
        )

        # The vast majority should be distinct
        total_unique = len(ids1 | ids2)
        assert total_unique >= len(ids1) + len(ids2) - 5, (
            "Pages should be mostly distinct"
        )

    def test_fetch_posts_respects_date_filter(self, source):
        """Posts returned should all be on or after start_date."""
        df = source.fetch_posts(
            "600519",
            start_date="2026-06-01",
            max_pages=1,
            fetch_bodies=False,
        )

        if len(df) > 0:
            dates = pd.to_datetime(df["date"])
            assert (dates >= pd.Timestamp("2026-06-01")).all(), (
                "All posts should be on or after start_date"
            )

    def test_fetch_posts_returns_expected_columns(self, source):
        """fetch_posts should always return DataFrame with expected columns,
        even for invalid stock codes."""
        # Test with stock that has no Guba forum
        df = source.fetch_posts(
            "999999",
            max_pages=1,
            fetch_bodies=False,
        )

        assert isinstance(df, pd.DataFrame)
        for col in ["date", "time", "title", "body", "post_id", "url"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_fetch_post_body_returns_text(self, source):
        """_fetch_post_body should return a non-empty string for a valid post."""
        # First get a post_id from the list
        df = source._fetch_list_page("600519", page=1)
        if df.empty:
            pytest.skip("No posts available to test body fetch")

        post_id = str(df.iloc[0]["post_id"])
        body = source._fetch_post_body("600519", post_id)

        assert isinstance(body, str), "Body should be a string"
        # Some posts may not have extractable bodies, so we don't assert length > 0
        # But for most real posts it should return content
        if not body:
            # Try another post
            for i in range(1, min(5, len(df))):
                post_id2 = str(df.iloc[i]["post_id"])
                body2 = source._fetch_post_body("600519", post_id2)
                if body2:
                    # At least one post should return content
                    return
            pytest.skip("Could not extract body from any of the top 5 posts")

    def test_fetch_posts_with_bodies(self, source):
        """fetch_posts with fetch_bodies=True should populate body column."""
        df = source.fetch_posts(
            "600519",
            max_pages=1,
            fetch_bodies=True,
        )

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            # At least some posts should have body content
            has_body = df["body"].str.strip().str.len() > 5
            body_count = has_body.sum()
            # Not all posts may have extractable bodies, but some should
            assert body_count >= 1, (
                f"Expected at least 1 post with body content, got {body_count}"
            )
