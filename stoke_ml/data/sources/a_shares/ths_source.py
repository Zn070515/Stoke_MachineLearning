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
            # Work around pandas pyarrow backend incompatibility in AKShare
            old_backend = pd.options.mode.string_storage
            try:
                pd.options.mode.string_storage = "python"
                df = ak.stock_news_em(symbol=stock_code)
            finally:
                pd.options.mode.string_storage = old_backend
        except Exception as e:
            logger.debug("THS/AKShare news failed for %s: %s", stock_code, e)
            return pd.DataFrame(columns=["date", "title", "url"])

        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "title", "url"])

        # AKShare returns: 关键词, 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
        # Map to standard format — keep body text when available
        col_map = {}
        body_col = None
        for col in df.columns:
            if "时间" in col or "发布时间" in col:
                col_map[col] = "date"
            elif "标题" in col:
                col_map[col] = "title"
            elif "内容" in col:
                body_col = col
            elif "链接" in col:
                col_map[col] = "url"

        if col_map:
            df = df.rename(columns=col_map)
            keep_cols = [c for c in ["date", "title", "url"] if c in df.columns]
            if body_col and body_col not in col_map:
                df["body"] = df[body_col].astype(str)
            elif body_col and body_col in col_map:
                pass  # already renamed
            else:
                df["body"] = ""
            df = df[keep_cols + ["body"]]
        else:
            cols = list(df.columns)
            if len(cols) >= 1:
                df["title"] = df[cols[0]].astype(str)
            df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            df["url"] = ""
            df["body"] = ""

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        else:
            df["date"] = pd.Timestamp.now()

        if "title" not in df.columns:
            return pd.DataFrame(columns=["date", "title", "url", "body"])

        df["title"] = df["title"].astype(str).str[:300]
        df["body"] = df.get("body", "")
        if df["body"].dtype != object:
            df["body"] = df["body"].astype(str)
        df["body"] = df["body"].str[:2000]
        if "url" not in df.columns:
            df["url"] = ""

        df = df.drop_duplicates(subset=["title", "date"])
        df = df.sort_values("date", ascending=False)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        if max_pages and len(df) > max_pages * 20:
            df = df.head(max_pages * 20)

        return df[["date", "title", "url", "body"]].reset_index(drop=True)
