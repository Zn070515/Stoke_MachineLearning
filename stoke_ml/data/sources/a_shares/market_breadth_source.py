"""Market breadth indicators — daily advance/decline, highs/lows, volume stats.

Free sources via AKShare:
  - stock_a_high_low_statistics() — new highs / new lows by sector
  - stock_zh_a_stop_em() — limit-up/down stocks daily
  - stock_zt_pool_strong_em() — strong limit-up pool (连板)
  - stock_account_statistics_em() — new A-share accounts (sentiment proxy)
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MarketBreadthSource:
    """Fetch daily market-wide breadth and sentiment indicators.

    These capture the overall market environment — regime detection
    for training and live trading. Individual stock returns are heavily
    influenced by the market regime (bull vs bear, high vs low volatility).
    """

    def fetch_new_highs_lows(self) -> pd.DataFrame:
        """Fetch daily new high / new low counts.

        Returns DataFrame with date, new_high_count, new_low_count per sector.
        """
        import akshare as ak
        logger.info("Fetching new highs/lows statistics...")
        try:
            df = ak.stock_a_high_low_statistics()
            if "date" not in df.columns and "日期" in df.columns:
                df["date"] = pd.to_datetime(df["日期"])
            elif "item" in df.columns:
                # Pivot: item column contains dates, value columns are sectors
                df = df.rename(columns={"item": "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            logger.info("New highs/lows: %d days", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to fetch new highs/lows: %s", e)
            return pd.DataFrame()

    def fetch_daily_limit_stats(self, date: str | None = None) -> pd.DataFrame:
        """Fetch daily limit-up/down stocks.

        Args:
            date: Trading date YYYYMMDD. Default: latest trading day.

        Returns DataFrame with limit-up/down counts and stock lists.
        """
        import akshare as ak
        try:
            df = ak.stock_zh_a_stop_em(date=date) if date else ak.stock_zh_a_stop_em()
            if "日期" in df.columns:
                df["date"] = pd.to_datetime(df["日期"])
            logger.info("Daily limit stats: %d rows", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to fetch daily limit stats: %s", e)
            return pd.DataFrame()

    def fetch_strong_limit_up_pool(self, date: str | None = None) -> pd.DataFrame:
        """Fetch strong limit-up pool (连板 stocks with >2 consecutive limits).

        These are momentum leaders — useful for sentiment and board effect.
        """
        import akshare as ak
        try:
            df = ak.stock_zt_pool_strong_em(date=date) if date else ak.stock_zt_pool_strong_em()
            logger.info("Strong limit-up pool: %d stocks", len(df) if not df.empty else 0)
            return df
        except Exception as e:
            logger.warning("Failed to fetch strong ZT pool: %s", e)
            return pd.DataFrame()

    def fetch_account_statistics(self) -> pd.DataFrame:
        """Fetch monthly new A-share investor accounts (sentiment proxy).

        Historically, new account openings peak near market tops and
        trough near market bottoms — a contrarian sentiment indicator.
        """
        import akshare as ak
        logger.info("Fetching account statistics...")
        try:
            df = ak.stock_account_statistics_em()
            if "日期" in df.columns:
                df["date"] = pd.to_datetime(df["日期"])
            logger.info("Account stats: %d months", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to fetch account stats: %s", e)
            return pd.DataFrame()

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch all breadth indicators.

        Returns dict with keys: highs_lows, account_stats.
        """
        return {
            "highs_lows": self.fetch_new_highs_lows(),
            "account_stats": self.fetch_account_statistics(),
        }
