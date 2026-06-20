"""Dragon-Tiger board (龙虎榜) data via AKShare."""
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

LHB_COLS = [
    "date", "stock_code", "stock_name", "lhb_reason",
    "buy_amount", "sell_amount", "net_amount",
    "buy_inst_amount", "sell_inst_amount",
    "buy_broker_count", "sell_broker_count",
]


class DragonTigerSource:
    """Fetch daily dragon-tiger board (龙虎榜) data."""

    SOURCE_NAME = "akshare_lhb"

    def fetch_daily(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch LHB daily detail for a date range (all stocks)."""
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for LHB data")
            return pd.DataFrame()

        try:
            df = ak.stock_lhb_detail_em(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is None or df.empty:
                return pd.DataFrame()
            return self._normalize(df)
        except Exception as e:
            logger.error("LHB daily fetch failed: %s", e)
            return pd.DataFrame()

    def fetch_by_stock(
        self, stock_code: str, start_date: str, end_date: str,
        sleep: float = 0.3,
    ) -> pd.DataFrame:
        """Fetch LHB history for a single stock over a date range.

        AKShare stock_lhb_stock_detail_em takes a single date, so we loop.
        """
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        frames = []
        for d in dates:
            date_str = d.strftime("%Y%m%d")
            try:
                df = ak.stock_lhb_stock_detail_em(
                    symbol=stock_code, date=date_str, flag="买入",
                )
                if df is not None and not df.empty:
                    df = df.copy()
                    df["date"] = d.strftime("%Y-%m-%d")
                    frames.append(df)
            except Exception:
                pass
            time.sleep(sleep)

        if not frames:
            return pd.DataFrame()
        return self._normalize(pd.concat(frames, ignore_index=True))

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize AKShare LHB output to standard columns."""
        df = df.copy()

        # Map Chinese column names to standard names (handles both daily and per-stock APIs)
        col_map = {}
        for col in df.columns:
            if col in ("上榜日", "日期"):
                col_map[col] = "date"
            elif col in ("代码",):
                col_map[col] = "stock_code"
            elif col in ("名称",):
                col_map[col] = "stock_name"
            elif col in ("上榜原因", "类型"):
                col_map[col] = "lhb_reason"
            elif col in ("龙虎榜买入额", "买入金额"):
                col_map[col] = "buy_amount"
            elif col in ("龙虎榜卖出额", "卖出金额"):
                col_map[col] = "sell_amount"
            elif col in ("龙虎榜净买额", "净额"):
                col_map[col] = "net_amount"

        if col_map:
            df = df.rename(columns=col_map)

        # Drop duplicate columns that may result from mapping ambiguity
        df = df.loc[:, ~df.columns.duplicated()]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.replace(".0", "").str.zfill(6)

        # Ensure numeric columns (only those present)
        for col in ["buy_amount", "sell_amount", "net_amount"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute net if missing
        if "net_amount" not in df.columns:
            if "buy_amount" in df.columns and "sell_amount" in df.columns:
                df["net_amount"] = df["buy_amount"] - df["sell_amount"]

        keep = [c for c in LHB_COLS if c in df.columns]
        return df[keep].reset_index(drop=True)
