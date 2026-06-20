"""Margin trading & short selling data (融资融券) via AKShare."""
import logging
import time

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
        self, start_date: str, end_date: str, sleep: float = 0.3
    ) -> pd.DataFrame:
        """Fetch margin trading details for all stocks over a date range.

        AKShare margin APIs accept a single date at a time, so we loop.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for margin data")
            return pd.DataFrame()

        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        all_frames = []

        for d in dates:
            date_str = d.strftime("%Y%m%d")
            day_frames = []

            # SSE margin data
            try:
                sse = ak.stock_margin_detail_sse(date=date_str)
                if sse is not None and not sse.empty:
                    sse = self._normalize(sse)
                    day_frames.append(sse)
            except Exception:
                pass

            # SZSE margin data
            try:
                szse = ak.stock_margin_detail_szse(date=date_str)
                if szse is not None and not szse.empty:
                    szse = self._normalize(szse)
                    day_frames.append(szse)
            except Exception:
                pass

            if day_frames:
                day_df = pd.concat(day_frames, ignore_index=True)
                day_df["date"] = d.strftime("%Y-%m-%d")
                all_frames.append(day_df)

            time.sleep(sleep)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = result.drop_duplicates(subset=["date", "stock_code"])
        return result.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize AKShare output to standard columns.

        Handles both SSE (9 columns with dates) and SZSE (8 columns, no date).
        """
        df = df.copy()

        # Use exact column name matching to avoid ambiguity
        sse_map = {
            "信用交易日期": "date",
            "标的证券代码": "stock_code",
            "融资余额": "margin_balance",
            "融资买入额": "margin_buy",
            "融资偿还额": "margin_repay",
            "融券余量": "short_balance",
            "融券卖出量": "short_sell_vol",
            "融券偿还量": "short_repay_vol",
        }
        szse_map = {
            "证券代码": "stock_code",
            "融资余额": "margin_balance",
            "融资买入额": "margin_buy",
            "融券余量": "short_balance",
            "融券卖出量": "short_sell_vol",
        }

        # Determine which mapping to use based on available columns
        if "信用交易日期" in df.columns:
            col_map = sse_map
        else:
            col_map = szse_map

        # Only map columns that exist
        safe_map = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=safe_map)

        # Drop duplicates from ambiguous mappings
        df = df.loc[:, ~df.columns.duplicated()]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.replace(".0", "").str.zfill(6)

        # Ensure numeric columns
        for col in ["margin_buy", "margin_repay", "short_sell_vol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute net margin flow
        if "margin_buy" in df.columns and "margin_repay" in df.columns:
            df["margin_net"] = df["margin_buy"] - df["margin_repay"]

        keep = [c for c in MARGIN_COLS if c in df.columns]
        return df[keep]
