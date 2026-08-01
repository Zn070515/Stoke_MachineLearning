"""Share pledge data source — major risk factor in A-shares.

In A-shares, controlling shareholders frequently pledge their shares
as collateral for loans. High pledge ratios are a leading indicator of:
  - Forced liquidation cascades (margin calls → stock dumps)
  - Corporate governance risk (tunneling, capital occupation)
  - Credit stress at the controlling shareholder level

Free sources via AKShare:
  - stock_gpzy_pledge_ratio_em() — aggregate pledge ratio per stock
  - stock_gpzy_pledge_ratio_detail_em() — per-pledgor detail
"""
import logging
import socket
import time

import pandas as pd

logger = logging.getLogger(__name__)

# AKShare internally calls requests.get() without a timeout parameter,
# which causes TCP connections to hang indefinitely when EastMoney
# rate-limits. Setting a global socket timeout prevents this.
socket.setdefaulttimeout(15)


class PledgeSource:
    """Fetch share pledge ratios and details for A-share stocks."""

    @staticmethod
    def _zfill(code: str) -> str:
        return str(code).zfill(6)

    def fetch_pledge_ratio(self, code: str) -> pd.DataFrame:
        """Fetch individual pledge ratio detail for one stock.

        Returns columns: stock_code, pledge_ratio, pledge_count, etc.
        """
        import akshare as ak
        df = None
        for attempt in range(2):
            try:
                df = ak.stock_gpzy_individual_pledge_ratio_detail_em(symbol=self._zfill(code))
                break
            except (TypeError, ValueError, KeyError):
                break  # AKShare internal bug — retry won't help
            except Exception:
                if attempt < 1:
                    time.sleep(1)
        if df is None or df.empty:
            return pd.DataFrame()
        df["stock_code"] = self._zfill(code)
        return df

    def fetch_all_pledge_ratios(
        self, codes: list[str], sleep: float = 0.3
    ) -> pd.DataFrame:
        """Fetch pledge ratios for multiple stocks.

        Returns consolidated DataFrame.
        """
        all_data = []
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(sleep)
            if (i + 1) % 100 == 0:
                logger.info("  pledge %d/%d", i + 1, len(codes))
            df = self.fetch_pledge_ratio(code)
            if not df.empty:
                all_data.append(df)
        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        logger.info("Pledge ratios: %d rows for %d stocks",
                      len(result), result["stock_code"].nunique())
        return result

    def fetch_market_pledge_stats(self) -> pd.DataFrame:
        """Fetch market-wide pledge statistics (single call).

        Returns aggregate pledge stats for the entire A-share market.
        """
        import akshare as ak
        logger.info("Fetching market-wide pledge statistics...")
        try:
            df = ak.stock_gpzy_pledge_ratio_em()
        except Exception as e:
            logger.warning("Market pledge stats failed: %s", e)
            return pd.DataFrame()
        logger.info("Market pledge: %d rows", len(df))
        return df
