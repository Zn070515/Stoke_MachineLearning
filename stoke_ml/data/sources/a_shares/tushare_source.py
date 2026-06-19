"""Tushare data source for A-shares (optional, requires token)."""
import os
import logging
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase

logger = logging.getLogger(__name__)


class TushareSource(AShareSourceBase):
    """Tushare A-share data source. Requires a Tushare token."""

    SOURCE_NAME = "tushare"

    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("TUSHARE_TOKEN", "")
        self._pro: object | None = None

    def _get_pro(self):
        if self._pro is not None:
            return self._pro
        if not self._token:
            return None
        try:
            import tushare as ts
            ts.set_token(self._token)
            self._pro = ts.pro_api()
            return self._pro
        except Exception:
            return None

    def is_available(self) -> bool:
        if not self._token:
            return False
        try:
            import tushare
            return True
        except ImportError:
            return False

    @staticmethod
    def _to_ts_code(stock_code: str) -> str:
        if stock_code.startswith("6"):
            return f"{stock_code}.SH"
        elif stock_code.startswith("8") or stock_code.startswith("4"):
            return f"{stock_code}.BJ"
        else:
            return f"{stock_code}.SZ"

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            pro = self._get_pro()
            if pro is None:
                return pd.DataFrame()
            ts_code = self._to_ts_code(stock_code)
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return self._normalize(df, stock_code)
        except Exception as e:
            logger.warning("Tushare fetch failed for %s: %s", stock_code, e)
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        col_map = {
            "trade_date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "vol": "volume",
            "amount": "amount", "pct_chg": "pct_change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
        return df
