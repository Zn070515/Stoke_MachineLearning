"""Earnings forecast & express report data source (业绩预告 / 业绩快报).

Free sources via AKShare:
  - stock_yjyg_em()  — 业绩预告 (earnings forecast): net-profit growth band,
                       announced ahead of the quarter-end report.
  - stock_yjbb_em()  — 业绩快报 (earnings express): early full/quarter figures.

These are the highest-timeliness fundamental signals — a forecast/express
lands weeks before the actual quarterly report, so it is the first market
reaction to a beat/miss. Quarterly fundamentals are backwards-looking and
lag by 1-4 months; this closes that gap.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns emitted by fetch_forecasts() regardless of akshare schema version.
YJYG_COLS = [
    "stock_code", "stock_name", "report_type", "forecast_metric",
    "announce_date", "change_reason", "net_profit_yoy", "net_profit",
]


class EarningsSource:
    """Fetch A-share earnings forecasts (业绩预告) and express reports (业绩快报)."""

    def fetch_forecasts(self, date: str | None = None) -> pd.DataFrame:
        """Fetch 业绩预告 (earnings forecasts) market-wide.

        Args:
            date: optional report-period filter (e.g. '20260331'). When None,
                  the API returns forecasts across recent report periods.

        Returns:
            DataFrame with YJYG_COLS. net_profit in 万元, yoy in %.
        """
        import akshare as ak
        logger.info("Fetching 业绩预告 (date=%s)...", date)
        try:
            df = ak.stock_yjyg_em(date=date) if date else ak.stock_yjyg_em()
        except Exception as e:
            logger.warning("业绩预告 failed: %s", e)
            return pd.DataFrame()
        if df.empty:
            logger.warning("业绩预告 empty")
            return pd.DataFrame()
        return self._rename_yjyg(df)

    def fetch_express(self, date: str | None = None) -> pd.DataFrame:
        """Fetch 业绩快报 (earnings express reports) market-wide.

        Args:
            date: optional report-period filter (e.g. '20260331'). When None,
                  the API returns the latest batch.

        Returns:
            DataFrame with a stock_code/report_date/announce_date plus the
            express financials (revenue, net profit, yoy rates).
        """
        import akshare as ak
        logger.info("Fetching 业绩快报 (date=%s)...", date)
        try:
            df = ak.stock_yjbb_em(date=date) if date else ak.stock_yjbb_em()
        except Exception as e:
            logger.warning("业绩快报 failed: %s", e)
            return pd.DataFrame()
        if df.empty:
            logger.warning("业绩快报 empty")
            return pd.DataFrame()
        df = df.rename(columns={
            "股票代码": "stock_code",
            "股票简称": "stock_name",
            "每股收益": "eps",
            "营业收入-营业收入": "revenue",
            "营业收入-同比增长": "revenue_yoy",
            "营业收入-季度环比增长": "revenue_qoq",
            "净利润-净利润": "net_profit",
            "净利润-同比增长": "net_profit_yoy",
            "净利润-季度环比增长": "net_profit_qoq",
            "每股净资产": "bvps",
            "净资产收益率": "roe",
            "每股经营现金流量": "ocf_per_share",
            "销售毛利率": "gross_margin",
            "所处行业": "industry",
            "最新公告日期": "announce_date",
        })
        if "report_date" not in df.columns:
            # YJBB has no report-date column; announce_date is the signal date.
            df["report_date"] = df.get("announce_date")
        for col in ["eps", "revenue", "revenue_yoy", "revenue_qoq",
                    "net_profit", "net_profit_yoy", "net_profit_qoq",
                    "bvps", "roe", "ocf_per_share", "gross_margin"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        logger.info("业绩快报: %d rows", len(df))
        return df

    @staticmethod
    def _rename_yjyg(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize 业绩预告 across two AKShare schemas.

        Old schema (pre-2024): 预告净利润下限/上限 (万元), 预告净利润变动幅度下限/
        上限, 报告日期. New schema (current): 预告类型, 预测指标, 业绩变动幅度
        (single yoy %), 预测数值 (net profit in 元), no report-date column. Both
        are mapped onto YJYG_COLS.
        """
        mapping = {
            "股票代码": "stock_code",
            "股票简称": "stock_name",
            "报告日期": "report_date",
            "业绩变动类型": "report_type",
            "业绩变动原因": "change_reason",
            "公告日期": "announce_date",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

        # New schema columns.
        if "预告类型" in df.columns:
            df["report_type"] = df["预告类型"]
        if "预测指标" in df.columns:
            df["forecast_metric"] = df["预测指标"]
        if "业绩变动幅度" in df.columns:
            df["net_profit_yoy"] = df["业绩变动幅度"]
        if "预测数值" in df.columns:
            df["net_profit"] = df["预测数值"]

        # Old schema columns (override the singles when present).
        if "预告净利润变动幅度下限" in df.columns:
            df["net_profit_yoy"] = df["预告净利润变动幅度下限"]
            df["net_profit_yoy_high"] = df["预告净利润变动幅度上限"]
        if "预告净利润下限" in df.columns:
            df["net_profit"] = df["预告净利润下限"]
            df["net_profit_high"] = df["预告净利润上限"]

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        for col in ["net_profit_yoy", "net_profit_yoy_high",
                    "net_profit", "net_profit_high"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # If no schema matched, net_profit_yoy/net_profit simply won't exist;
        # the storage layer treats missing columns as all-NaN.
        logger.info("业绩预告: %d rows", len(df))
        return df
