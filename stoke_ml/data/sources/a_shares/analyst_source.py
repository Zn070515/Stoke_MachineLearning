"""Analyst rating and profit forecast data source.

Free sources via AKShare:
  - stock_profit_forecast_em() — consensus EPS forecast (market-wide, single call)
  - stock_analyst_rank_em() — analyst ranking/accuracy (market-wide)
"""
import logging

import pandas as pd

from stoke_ml.data.codes import normalize_stock_code_series

logger = logging.getLogger(__name__)


class AnalystSource:
    """Fetch analyst consensus forecasts and rating histories.

    Profit forecasts are one of the strongest alpha signals in A-shares
    because they are forward-looking (unlike fundamental data which is
    backward-looking quarterly accounting data).
    """

    def fetch_profit_forecast(self) -> pd.DataFrame:
        """Fetch consensus profit forecast for all A-shares (single call).

        Returns columns: stock_code, stock_name, report_count,
        rating_buy, rating_overweight, rating_neutral,
        rating_underweight, rating_sell, eps_2025~2028.
        """
        import akshare as ak
        logger.info("Fetching profit forecast (market-wide)...")
        try:
            df = ak.stock_profit_forecast_em()
        except Exception as e:
            logger.warning("Profit forecast failed: %s", e)
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "序号": "rank",
            "代码": "stock_code",
            "名称": "stock_name",
            "研报数": "report_count",
            "机构投资评级(近六个月)-买入": "rating_buy",
            "机构投资评级(近六个月)-增持": "rating_overweight",
            "机构投资评级(近六个月)-中性": "rating_neutral",
            "机构投资评级(近六个月)-减持": "rating_underweight",
            "机构投资评级(近六个月)-卖出": "rating_sell",
            "2025预测每股收益": "eps_2025",
            "2026预测每股收益": "eps_2026",
            "2027预测每股收益": "eps_2027",
            "2028预测每股收益": "eps_2028",
        })
        df["stock_code"] = normalize_stock_code_series(df["stock_code"])
        for col in ["report_count", "rating_buy", "rating_overweight",
                     "rating_neutral", "rating_underweight", "rating_sell"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["eps_2025", "eps_2026", "eps_2027", "eps_2028"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info("Profit forecast: %d stocks", len(df))
        return df

    def fetch_analyst_ranking(self, year: str = "2024") -> pd.DataFrame:
        """Fetch analyst ranking/accuracy table (market-wide)."""
        import akshare as ak
        logger.info("Fetching analyst ranking (year=%s)...", year)
        try:
            df = ak.stock_analyst_rank_em(year=year)
            logger.info("Analyst ranking: %d analysts", len(df))
            return df
        except Exception as e:
            logger.warning("Analyst ranking failed: %s", e)
            return pd.DataFrame()
