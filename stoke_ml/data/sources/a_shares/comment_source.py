"""AKShare market comment sentiment — comprehensive scores for all A-share stocks.

Provides two layers:
1. stock_comment_em() — 5184 stocks × 14 metrics (综合得分, 关注指数, 机构参与度)
2. stock_comment_detail_zhpj_lspf_em() — per-stock 30-day daily rating history
"""
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


class CommentSource:
    """Fetch market comment sentiment from AKShare (EastMoney data)."""

    @staticmethod
    def fetch_all_snapshot() -> pd.DataFrame:
        """Fetch comprehensive comment scores for ALL A-share stocks.

        Returns one row per stock with: code, score, attention, institution,
        trend, rank, and metadata columns.
        """
        import akshare as ak

        df = ak.stock_comment_em()
        df = df.rename(columns={
            "代码": "stock_code",
            "综合得分": "comment_score",
            "关注指数": "comment_attention",
            "机构参与度": "comment_institution",
            "上升": "comment_trend",
            "目前排名": "comment_rank",
            "交易日": "date",
            "最新价": "latest_price",
            "涨跌幅": "pct_change",
            "换手率": "turnover",
            "市盈率": "pe_ratio",
            "主力成本": "main_cost",
            "名称": "stock_name",
        })
        df["date"] = pd.to_datetime(df["date"])
        # Keep only the stock codes we track (6-digit)
        df = df[df["stock_code"].str.match(r"^\d{6}$")]
        # Core sentiment columns as float32
        for col in ["comment_score", "comment_attention", "comment_institution",
                     "comment_trend", "comment_rank"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df.reset_index(drop=True)

    @staticmethod
    def fetch_stock_history(stock_code: str) -> pd.DataFrame:
        """Fetch 30-day daily rating history for a single stock.

        Returns DataFrame with columns: date, comment_score.
        """
        import akshare as ak

        try:
            df = ak.stock_comment_detail_zhpj_lspf_em(stock_code)
            df = df.rename(columns={
                "交易日": "date",
                "评分": "comment_score",
            })
            df["stock_code"] = stock_code
            df["date"] = pd.to_datetime(df["date"])
            df["comment_score"] = pd.to_numeric(
                df["comment_score"], errors="coerce"
            ).astype("float32")
            return df[["date", "stock_code", "comment_score"]].reset_index(drop=True)
        except Exception as e:
            logger.debug("comment_detail %s failed: %s", stock_code, e)
            return pd.DataFrame(columns=["date", "stock_code", "comment_score"])

    @staticmethod
    def fetch_history_batch(
        stock_codes: list[str], sleep: float = 0.3
    ) -> pd.DataFrame:
        """Download 30-day history for multiple stocks sequentially."""
        frames = []
        for i, code in enumerate(stock_codes):
            if i > 0:
                time.sleep(sleep)
            df = CommentSource.fetch_stock_history(code)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["date", "stock_code", "comment_score"])
        return pd.concat(frames, ignore_index=True)
