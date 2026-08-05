"""Quarterly fundamental data source (ROE, PE, PB, revenue growth etc.) via AKShare."""
import logging

import pandas as pd

from stoke_ml.data.codes import normalize_stock_code

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


def _statutory_disclosure_date(report_date):
    """Map a report-period end to its statutory disclosure deadline.

    A-share quarterly/annual financial reports carry legal disclosure
    deadlines (法定披露截止日):
        Q1   (period end 03-31) → 04-30 of the SAME year
        H1   (period end 06-30) → 08-31 of the SAME year
        Q3   (period end 09-30) → 10-31 of the SAME year
        Annual (period end 12-31) → 04-30 of the FOLLOWING year

    Conservative alignment: the statutory deadline is the LATEST legal date a
    company may disclose, so anchoring on it never reveals information before
    that date passes — no lookahead into an unpublished filing.  Most
    companies actually disclose well before the deadline (e.g. an annual
    report often lands in March), so this alignment is deliberately LATE
    relative to the true announcement: safe, but not the real publish time.
    It is used as a leakage-free proxy because the upstream API
    (``ak.stock_financial_abstract``) does not return an announcement-date
    field.

    ``report_date`` may be ``pd.Timestamp``, ``datetime.date`` or ``str``.
    """
    report_date = pd.Timestamp(report_date)
    quarter = (report_date.month - 1) // 3  # 0=Q1, 1=Q2, 2=Q3, 3=Q4
    year = report_date.year + (1 if quarter == 3 else 0)
    y, m, d = {
        0: (year, 4, 30),
        1: (year, 8, 31),
        2: (year, 10, 31),
        3: (year, 4, 30),
    }[quarter]
    return pd.Timestamp(y, m, d)


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

        result["stock_code"] = normalize_stock_code(stock_code)

        # No disclose_date from this API.  Anchoring on the statutory
        # disclosure deadline (法定披露截止日) instead of report_date: the
        # report-period end (e.g. 2025-03-31) is months before the filing is
        # actually published (often late April), so using it as the disclosure
        # day would leak the figures forward into the model.  The statutory
        # deadline is the LATEST legal publish date — conservative, never
        # early, hence leakage-free (see _statutory_disclosure_date).
        result["disclose_date"] = result["report_date"].map(_statutory_disclosure_date)

        # PE and PB not available from financial_abstract — leave as NaN
        # They will be None/missing in the output; can be added from daily data later

        keep = [c for c in FUNDAMENTAL_COLS if c in result.columns]
        result = result[keep].dropna(subset=["report_date"])
        return result.sort_values("report_date").reset_index(drop=True)
