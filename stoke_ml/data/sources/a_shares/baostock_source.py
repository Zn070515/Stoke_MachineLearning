"""Baostock data source for A-shares (last resort, free)."""
import logging
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase

logger = logging.getLogger(__name__)


class BaostockSource(AShareSourceBase):
    """Baostock A-share data source. Free, no authentication needed."""

    SOURCE_NAME = "baostock"

    def is_available(self) -> bool:
        try:
            import baostock as bs
            return True
        except ImportError:
            return False

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import baostock as bs
        lg = None
        try:
            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("Baostock login failed: %s", lg.error_msg)
                return pd.DataFrame()

            if stock_code.startswith("6") or stock_code.startswith("9"):
                bs_code = f"sh.{stock_code}"
            elif stock_code.startswith("8") or stock_code.startswith("4"):
                bs_code = f"bj.{stock_code}"
            else:
                bs_code = f"sz.{stock_code}"

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,pctChg",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                logger.warning("Baostock query failed: %s", rs.error_msg)
                bs.logout()
                return pd.DataFrame()

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                bs.logout()
                return pd.DataFrame()
            df = pd.DataFrame(
                rows,
                columns=["date", "open", "high", "low", "close",
                          "volume", "amount", "pct_change"],
            )
            result = self._normalize(df, stock_code)
            bs.logout()
            return result
        except Exception as e:
            logger.warning("Baostock fetch failed for %s: %s", stock_code, e)
            if lg is not None and lg.error_code == "0":
                try:
                    bs.logout()
                except Exception:
                    pass
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
