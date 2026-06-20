"""THS / 10jqka (同花顺) news source via AKShare EastMoney backend."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

THS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


class THSNewsSource:
    """Fetch stock news via AKShare EastMoney news (same underlying data as THS)."""

    def fetch_news(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Fetch news from EastMoney news via AKShare.

        AKShare's stock_news_em uses EastMoney's API which covers
        the same news pool that 同花顺 draws from.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for THS news")
            return pd.DataFrame(columns=["date", "title", "url"])

        try:
            df = ak.stock_news_em(stock=stock_code)
        except Exception as e:
            logger.debug("THS/AKShare news failed for %s: %s", stock_code, e)
            return pd.DataFrame(columns=["date", "title", "url"])

        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "title", "url"])

        # AKShare returns columns: 关键词, 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
        # Map to standard format
        col_map = {}
        for col in df.columns:
            if "时间" in col or "发布时间" in col:
                col_map[col] = "date"
            elif "标题" in col:
                col_map[col] = "title"
            elif "链接" in col:
                col_map[col] = "url"

        if col_map:
            df = df.rename(columns=col_map)
            # Keep only standard columns
            keep_cols = [c for c in ["date", "title", "url"] if c in df.columns]
            df = df[keep_cols]
        else:
            # Try positional: first col = title, last col = url, date from content
            cols = list(df.columns)
            result = pd.DataFrame()
            if len(cols) >= 1:
                result["title"] = df[cols[0]].astype(str)
            result["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            result["url"] = ""
            df = result

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        else:
            df["date"] = pd.Timestamp.now()

        if "title" not in df.columns:
            return pd.DataFrame(columns=["date", "title", "url"])

        # Truncate long titles
        df["title"] = df["title"].astype(str).str[:300]
        if "url" not in df.columns:
            df["url"] = ""

        df = df.drop_duplicates(subset=["title", "date"])
        df = df.sort_values("date", ascending=False)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df[["date", "title", "url"]].reset_index(drop=True)
