"""AKShare data source using Sina finance API (not EastMoney)."""
import logging
import pandas as pd
from stoke_ml.data.codes import (
    UnsupportedMarketError,
    market_of_code,
    normalize_stock_code,
)
from stoke_ml.data.sources.a_shares.base import AShareSourceBase

logger = logging.getLogger(__name__)


class AKShareSource(AShareSourceBase):
    """A-share data via AKShare's Sina finance wrapper."""

    SOURCE_NAME = "akshare"

    @staticmethod
    def _to_sina_symbol(stock_code: str) -> str:
        """Sina symbol: ``sh600519`` / ``sz000001`` / ``bj920001``.

        Sina accepts the ``bj`` prefix for BSE codes (verified against
        CN_MarketData.getKLineData — ``bj920001`` returns bars while
        ``sz920001`` returns null).
        """
        code = normalize_stock_code(stock_code)
        if code is None:
            raise UnsupportedMarketError(
                f"Unusable stock code: {stock_code!r}"
            )
        market = market_of_code(code)
        if market is None:
            raise UnsupportedMarketError(
                f"Sina cannot route non-A-share code {code}"
            )
        return f"{market.lower()}{code}"

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
        "振幅": "amplitude", "换手率": "turnover",
    }

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        # Try Chinese column names first (older AKShare), then English
        if any(c in df.columns for c in self.CN_COL_MAP):
            df = df.rename(columns={
                k: v for k, v in self.CN_COL_MAP.items() if k in df.columns
            })
        cols = ["date", "open", "high", "low", "close", "volume", "amount",
                "turnover", "amplitude", "pct_change"]
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
        for col in ["open", "high", "low", "close", "volume", "amount",
                     "turnover", "amplitude", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "pct_change" not in df.columns or df["pct_change"].fillna(0).abs().sum() == 0:
            # stock_zh_a_daily (Sina qfq) omits 涨跌幅; derive from close instead
            # of persisting zeros. qfq preserves daily returns within a factor
            # epoch, so close.pct_change() matches the exchange 涨跌幅.
            df["pct_change"] = df["close"].pct_change() * 100.0
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.attrs["source"] = self.SOURCE_NAME
        df.attrs["adjustment_mode"] = "qfq"
        return df

    def is_available(self) -> bool:
        try:
            import akshare  # noqa: F401
            return True
        except ImportError:
            return False
