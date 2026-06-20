"""Quarterly fundamental data source (ROE, PE, PB, revenue growth etc.) via AKShare."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

FUNDAMENTAL_COLS = [
    "report_date", "disclose_date", "stock_code",
    "roe", "roa", "pe", "pb", "eps",
    "revenue_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "gross_margin", "net_margin",
    "total_revenue", "net_profit",
]

# Map Chinese indicator names → English column names
# Using exact matches on (选项, 指标) tuples to avoid ambiguity
INDICATOR_MAP = {
    "净资产收益率(ROE)": "roe",
    "总资产报酬率(ROA)": "roa",
    "基本每股收益": "eps",
    "营业总收入增长率": "revenue_yoy",
    "归属母公司净利润增长率": "profit_yoy",
    "资产负债率": "debt_ratio",
    "流动比率": "current_ratio",
    "毛利率": "gross_margin",
    "销售净利率": "net_margin",
    "营业总收入": "total_revenue",
    "净利润": "net_profit",
    "归母净利润": "net_profit",  # preference for 净利润 but use either
}


class FundamentalSource:
    """Fetch quarterly financial indicators for A-share stocks."""

    SOURCE_NAME = "akshare_fundamental"

    def fetch_indicators(
        self, stock_code: str
    ) -> pd.DataFrame:
        """Fetch fundamental indicators for a single stock.

        Returns DataFrame with quarterly data.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for fundamental data")
            return pd.DataFrame()

        try:
            df = ak.stock_financial_abstract(symbol=stock_code)
        except Exception as e:
            logger.debug("Fundamental fetch for %s failed: %s", stock_code, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        return self._normalize(df, stock_code)

    def _normalize(self, raw: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Convert AKShare wide-format financial abstract to long-format DataFrame."""
        df = raw.copy()

        # Date columns are YYYYMMDD format, indicator rows have 选项 + 指标
        date_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        if not date_cols:
            return pd.DataFrame()

        # Only keep rows where 指标 matches our mapping
        df = df[df["指标"].isin(INDICATOR_MAP)]

        # For duplicate indicators (e.g., 归母净利润 appears in both 常用指标 and 成长能力),
        # prefer the first occurrence
        df = df.drop_duplicates(subset=["指标"], keep="first")

        # Melt date columns into rows
        id_vars = ["指标"]
        df = df.melt(id_vars=id_vars, value_vars=date_cols,
                     var_name="report_date", value_name="value")

        # Convert to numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d")

        # Map indicator names to English
        df["indicator"] = df["指标"].map(INDICATOR_MAP)
        df = df.dropna(subset=["indicator"])

        # Pivot to wide format: one row per date, one column per indicator
        result = df.pivot_table(
            index="report_date", columns="indicator", values="value", aggfunc="first"
        ).reset_index()

        result["stock_code"] = str(stock_code).zfill(6)

        # No disclose_date from this API — use report_date as proxy
        result["disclose_date"] = result["report_date"]

        # PE and PB not available from financial_abstract — leave as NaN
        # They will be None/missing in the output; can be added from daily data later

        keep = [c for c in FUNDAMENTAL_COLS if c in result.columns]
        result = result[keep].dropna(subset=["report_date"])
        return result.sort_values("report_date").reset_index(drop=True)
