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
            if self._get_pro() is None:
                return pd.DataFrame()
            import tushare as ts
            ts_code = self._to_ts_code(stock_code)
            # pro_bar(adj="qfq") returns 前复权 prices, matching efinance /
            # akshare / baostock.  Plain pro.daily() returns UNadjusted bars
            # which would inject a fake 涨跌 seam on failover (v6 §六).
            df = ts.pro_bar(
                ts_code=ts_code,
                adj="qfq",
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
            "turnover_rate": "turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        cols = ["date", "open", "high", "low", "close", "volume", "amount",
                "pct_change", "turnover"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        # Tushare units differ from the stored convention (v6 §六):
        #   vol    手 → ×100 股
        #   amount 千元 → ×1000 元
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000.0
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
        df.attrs["source"] = self.SOURCE_NAME
        df.attrs["adjustment_mode"] = "qfq"
        return df
