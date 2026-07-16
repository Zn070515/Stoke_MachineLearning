"""Industry index data source via THS (同花顺) through AKShare.

Uses AKShare → THS API routing, which is independent of the EastMoney IP block.

Provides:
  - Daily industry index returns (90 THS industry boards)
  - Stock-to-industry mapping (via EastMoney constituent API, fallback to THS)
"""
import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndustrySource:
    """Fetch THS industry board indices and stock-to-industry mappings."""

    def fetch_industry_boards(self) -> pd.DataFrame:
        """Get list of all THS industry boards (name + code)."""
        import akshare as ak

        df = ak.stock_board_industry_name_ths()
        # Columns: name, code (e.g. 半导体, 881121)
        df = df.rename(columns={"name": "name", "code": "code"})
        logger.info("Got %d THS industry boards", len(df))
        return df

    def fetch_industry_index(
        self, name: str, start_date: str = "20150101", end_date: str = "20260716",
    ) -> pd.DataFrame:
        """Fetch daily index data for a single THS industry by name.

        Returns DataFrame with date index and columns:
          open, close, high, low, volume, amount.
        """
        import akshare as ak

        df = ak.stock_board_industry_index_ths(
            symbol=name, start_date=start_date, end_date=end_date,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # Standardize columns: 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 成交额
        col_map = {}
        for c in df.columns:
            c_str = str(c)
            if "日期" in c_str:
                col_map[c] = "date"
            elif "开盘" in c_str:
                col_map[c] = "open"
            elif "最高" in c_str:
                col_map[c] = "high"
            elif "最低" in c_str:
                col_map[c] = "low"
            elif "收盘" in c_str:
                col_map[c] = "close"
            elif "成交" in c_str and "量" in c_str:
                col_map[c] = "volume"
            elif "成交" in c_str and "额" in c_str:
                col_map[c] = "amount"

        df = df.rename(columns=col_map)
        if "date" not in df.columns and len(df.columns) > 0:
            df["date"] = pd.to_datetime(df.iloc[:, 0])
        else:
            df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        for col in ["open", "close", "high", "low", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        return df

    def fetch_all_returns(
        self, start_date: str = "20150101", end_date: str = "20260716",
        sleep: float = 0.3,
    ) -> pd.DataFrame:
        """Fetch daily pct_chg for all THS industry boards.

        Returns DataFrame with date index, industry names as columns.
        """
        boards = self.fetch_industry_boards()
        all_returns = {}

        for i, row in boards.iterrows():
            name = row["name"]
            if i > 0:
                time.sleep(sleep)
            try:
                df = self.fetch_industry_index(name, start_date, end_date)
                if not df.empty and "close" in df.columns:
                    # Compute daily pct change
                    ret = df["close"].pct_change() * 100.0
                    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(ret) > 100:  # require at least 100 trading days
                        all_returns[name] = ret.astype("float32")
            except Exception:
                logger.debug("Industry %s fetch failed", name, exc_info=True)

        if not all_returns:
            return pd.DataFrame()

        result = pd.DataFrame(all_returns).sort_index()
        result.columns.name = "industry"
        logger.info(
            "Industry returns: %d days × %d industries, %s to %s",
            len(result), len(result.columns),
            result.index.min().date(), result.index.max().date(),
        )
        return result

    def fetch_stock_industry_map(self) -> pd.DataFrame:
        """Build stock-to-industry mapping.

        Tries EastMoney constituent API first, falls back to THS-based approach.
        Returns DataFrame with columns: stock_code, industry.
        """
        mappings = self._try_em_mapping()
        if mappings is not None and len(mappings) > 100:
            return mappings
        return self._ths_mapping()

    def _try_em_mapping(self) -> pd.DataFrame | None:
        """Try EastMoney constituent API for stock-industry mapping."""
        import akshare as ak

        try:
            boards = ak.stock_board_industry_name_em()
            if boards is None or boards.empty:
                return None

            mappings = []
            for _, row in boards.head(90).iterrows():
                code = row.get("板块代码", row.iloc[1] if len(row) > 1 else None)
                name = row.get("板块名称", row.iloc[0] if len(row) > 0 else None)
                if code is None:
                    continue
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=name)
                    if cons is not None and not cons.empty:
                        stock_col = None
                        for c in cons.columns:
                            if "代码" in str(c):
                                stock_col = c
                                break
                        if stock_col is None:
                            stock_col = cons.columns[0]
                        for _, srow in cons.iterrows():
                            mappings.append({
                                "stock_code": str(srow[stock_col]).strip(),
                                "industry": name,
                            })
                except Exception:
                    pass
                time.sleep(0.2)

            if mappings:
                df = pd.DataFrame(mappings)
                df = df.drop_duplicates(subset=["stock_code"], keep="first")
                return df
        except Exception:
            pass
        return None

    def _ths_mapping(self) -> pd.DataFrame:
        """Build stock-industry mapping from THS industry board data.

        Uses the fact that each THS industry board has constituent stocks
        accessible through AKShare.
        """
        import akshare as ak

        try:
            industries = ak.stock_board_industry_name_ths()
        except Exception:
            return pd.DataFrame(columns=["stock_code", "industry"])

        mappings = []
        for _, row in industries.iterrows():
            name = row["name"]
            try:
                # THS industry spot data includes constituent info
                spot = ak.stock_board_industry_spot_em(symbol=name)
                if spot is not None and not spot.empty:
                    stock_col = None
                    for c in spot.columns:
                        c_str = str(c)
                        if "代码" in c_str:
                            stock_col = c
                            break
                    if stock_col:
                        for _, srow in spot.iterrows():
                            mappings.append({
                                "stock_code": str(srow[stock_col]).strip(),
                                "industry": name,
                            })
            except Exception:
                pass
            time.sleep(0.1)

        if mappings:
            df = pd.DataFrame(mappings)
            df = df.drop_duplicates(subset=["stock_code"], keep="first")
            return df
        return pd.DataFrame(columns=["stock_code", "industry"])
