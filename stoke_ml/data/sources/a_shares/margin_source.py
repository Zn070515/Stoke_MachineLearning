"""Margin trading & short selling data (融资融券) via AKShare."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

MARGIN_COLS = [
    "date", "stock_code", "margin_balance", "margin_buy",
    "margin_repay", "short_balance", "short_sell_vol",
    "short_repay_vol", "margin_net",
]


class MarginTradingSource:
    """Fetch daily margin trading and short selling data."""

    SOURCE_NAME = "akshare_margin"

    def fetch_daily(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch margin trading details for all stocks.

        Returns DataFrame with columns: date, stock_code, margin_balance,
        margin_buy, margin_repay, short_balance, short_sell_vol, etc.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for margin data")
            return pd.DataFrame()

        frames = []

        # SSE margin data
        try:
            sse = ak.stock_margin_detail_sse(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if sse is not None and not sse.empty:
                sse = self._normalize(sse, "sse")
                frames.append(sse)
        except Exception as e:
            logger.debug("SSE margin fetch failed: %s", e)

        # SZSE margin data
        try:
            szse = ak.stock_margin_detail_szse(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if szse is not None and not szse.empty:
                szse = self._normalize(szse, "szse")
                frames.append(szse)
        except Exception as e:
            logger.debug("SZSE margin fetch failed: %s", e)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["date", "stock_code"])
        return result.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _normalize(self, df: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """Normalize AKShare output to standard columns."""
        df = df.copy()

        col_map = {}
        for col in df.columns:
            if "日期" in col or "date" in col.lower():
                col_map[col] = "date"
            elif "股票代码" in col or "code" in col.lower():
                col_map[col] = "stock_code"
            elif "融资余额" in col:
                col_map[col] = "margin_balance"
            elif "融资买入额" in col:
                col_map[col] = "margin_buy"
            elif "融资偿还额" in col:
                col_map[col] = "margin_repay"
            elif "融券余量" in col or "融券余额" in col:
                col_map[col] = "short_balance"
            elif "融券卖出量" in col:
                col_map[col] = "short_sell_vol"
            elif "融券偿还量" in col:
                col_map[col] = "short_repay_vol"

        if col_map:
            df = df.rename(columns=col_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Ensure stock_code is string
        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

        # Compute net margin flow
        if "margin_buy" in df.columns and "margin_repay" in df.columns:
            df["margin_buy"] = pd.to_numeric(df["margin_buy"], errors="coerce")
            df["margin_repay"] = pd.to_numeric(df["margin_repay"], errors="coerce")
            df["margin_net"] = df["margin_buy"] - df["margin_repay"]

        # Keep only standard columns that exist
        keep = [c for c in MARGIN_COLS if c in df.columns]
        return df[keep]
