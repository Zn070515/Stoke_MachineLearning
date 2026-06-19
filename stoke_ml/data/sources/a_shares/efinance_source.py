"""Efinance (东方财富) data source for A-shares."""
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase


class EfinanceSource(AShareSourceBase):
    """Fast, preferred A-share data source via East Money API."""

    SOURCE_NAME = "efinance"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            import efinance as ef
            df = ef.stock.get_quote_history(
                stock_code, beg=start_date, end=end_date
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return self._normalize(df, stock_code)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Efinance fetch failed for %s: %s", stock_code, e
            )
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        col_map = {
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def is_available(self) -> bool:
        try:
            import efinance
            return True
        except ImportError:
            return False
