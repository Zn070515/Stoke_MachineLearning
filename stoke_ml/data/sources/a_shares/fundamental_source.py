"""Quarterly fundamental data source (ROE, PE, PB, revenue growth etc.) via AKShare."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FUNDAMENTAL_COLS = [
    "report_date", "disclose_date", "stock_code",
    "roe", "roa", "pe", "pb", "eps",
    "revenue_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "gross_margin", "net_margin",
    "total_revenue", "net_profit",
]


class FundamentalSource:
    """Fetch quarterly financial indicators for A-share stocks."""

    SOURCE_NAME = "akshare_fundamental"

    def fetch_indicators(
        self, stock_code: str
    ) -> pd.DataFrame:
        """Fetch fundamental indicators for a single stock.

        Returns DataFrame with quarterly data including report_date
        (quarter-end) and disclose_date (announcement date).
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for fundamental data")
            return pd.DataFrame()

        try:
            df = ak.stock_financial_analysis_indicator(symbol=stock_code)
        except Exception as e:
            logger.debug("Fundamental fetch for %s failed: %s", stock_code, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        return self._normalize(df, stock_code)

    def _normalize(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Map AKShare Chinese columns to standard English names."""
        df = df.copy()

        col_map = {}
        for col in df.columns:
            if "日期" in col and "披露" not in col:
                col_map[col] = "report_date"
            elif "披露" in col:
                col_map[col] = "disclose_date"
            elif "净资产收益率" in col or col == "ROE":
                col_map[col] = "roe"
            elif "总资产收益率" in col or col == "ROA":
                col_map[col] = "roa"
            elif "市盈率" in col or col == "PE":
                col_map[col] = "pe"
            elif "市净率" in col or col == "PB":
                col_map[col] = "pb"
            elif "每股收益" in col or col == "EPS":
                col_map[col] = "eps"
            elif "营业收入同比增长" in col:
                col_map[col] = "revenue_yoy"
            elif "净利润同比增长" in col:
                col_map[col] = "profit_yoy"
            elif "资产负债率" in col:
                col_map[col] = "debt_ratio"
            elif "流动比率" in col:
                col_map[col] = "current_ratio"
            elif "毛利率" in col:
                col_map[col] = "gross_margin"
            elif "净利率" in col:
                col_map[col] = "net_margin"
            elif "营业总收入" in col:
                col_map[col] = "total_revenue"
            elif "净利润" in col:
                col_map[col] = "net_profit"

        if col_map:
            df = df.rename(columns=col_map)

        # Add stock_code
        df["stock_code"] = str(stock_code).zfill(6)

        # Date handling
        for date_col in ["report_date", "disclose_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # If disclose_date is missing, use report_date
        if "report_date" in df.columns and "disclose_date" not in df.columns:
            df["disclose_date"] = df["report_date"]

        # Convert percentage columns from "XX%" or decimal
        for pct_col in ["roe", "roa", "revenue_yoy", "profit_yoy",
                         "debt_ratio", "gross_margin", "net_margin"]:
            if pct_col in df.columns:
                df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False)
                df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")

        for num_col in ["pe", "pb", "eps", "total_revenue", "net_profit",
                         "current_ratio"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        keep = [c for c in FUNDAMENTAL_COLS if c in df.columns]
        result = df[keep].dropna(subset=["report_date"])
        return result.sort_values("report_date").reset_index(drop=True)
