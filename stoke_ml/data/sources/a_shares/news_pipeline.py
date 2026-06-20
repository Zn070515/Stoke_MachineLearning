"""Multi-source news pipeline aggregating Sina, Xueqiu, and THS news."""
import logging

import pandas as pd

from stoke_ml.data.sources.a_shares.news_source import SinaNewsSource
from stoke_ml.data.sources.a_shares.xueqiu_source import XueqiuNewsSource
from stoke_ml.data.sources.a_shares.ths_source import THSNewsSource

logger = logging.getLogger(__name__)

SOURCE_MAP = {
    "sina": SinaNewsSource,
    "xueqiu": XueqiuNewsSource,
    "ths": THSNewsSource,
}


class NewsPipeline:
    """Aggregate news from multiple sources with deduplication."""

    def __init__(self, active_sources: list[str] | None = None):
        """
        Args:
            active_sources: List of source names to use (default: all available).
        """
        sources_to_use = active_sources or list(SOURCE_MAP.keys())
        self._sources = {
            name: SOURCE_MAP[name]()
            for name in sources_to_use
            if name in SOURCE_MAP
        }
        if not self._sources:
            self._sources = {"sina": SinaNewsSource()}

    def fetch_all_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Fetch news from all active sources, deduplicate by (title, date).

        Returns DataFrame with columns: date, title, url, source.
        """
        all_frames = []

        for source_name, source in self._sources.items():
            try:
                df = source.fetch_news(
                    stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    max_pages=max_pages,
                )
                if not df.empty:
                    df["source"] = source_name
                    all_frames.append(df)
                    logger.debug(
                        "  %s: %d articles from %s", stock_code, len(df), source_name,
                    )
            except Exception as e:
                logger.debug("  %s: source %s failed: %s", stock_code, source_name, e)

        if not all_frames:
            return pd.DataFrame(columns=["date", "title", "body", "url", "source"])

        combined = pd.concat(all_frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])

        # Deduplicate across sources: prefer rows with body text
        if "body" in combined.columns:
            combined["_body_len"] = combined["body"].str.len().fillna(0)
            combined = combined.sort_values("_body_len", ascending=False)
            combined = combined.drop_duplicates(subset=["title", "date"])
            combined = combined.drop(columns=["_body_len"])
        else:
            combined = combined.drop_duplicates(subset=["title", "date"])

        combined = combined.sort_values("date", ascending=False)

        return combined.reset_index(drop=True)
