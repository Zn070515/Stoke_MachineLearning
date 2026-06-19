"""AKShare data source using Sina finance API (not EastMoney)."""
import logging
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase

logger = logging.getLogger(__name__)


class AKShareSource(AShareSourceBase):
    """A-share data via AKShare's Sina finance wrapper."""

    SOURCE_NAME = "akshare"

    @staticmethod
    def _to_sina_symbol(stock_code: str) -> str:
        prefix = "sh" if stock_code.startswith("6") else "sz"
        return f"{prefix}{stock_code}"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            import akshare as ak
            symbol = self._to_sina_symbol(stock_code)
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return self._normalize(df, stock_code)
        except Exception as e:
            logger.warning("AKShare fetch failed for %s: %s", stock_code, e)
            return pd.DataFrame()

    CN_COL_MAP = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "pct_change",
    }

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        # Try Chinese column names first (older AKShare), then English
        if any(c in df.columns for c in self.CN_COL_MAP):
            df = df.rename(columns={
                k: v for k, v in self.CN_COL_MAP.items() if k in df.columns
            })
        cols = ["date", "open", "high", "low", "close", "volume", "amount"]
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pct_change"] = 0.0
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def is_available(self) -> bool:
        try:
            import akshare  # noqa: F401
            return True
        except ImportError:
            return False
