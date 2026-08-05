"""IPO calendar and ST/delisting records for A-share stock universe construction.

Free sources via AKShare:
  - stock_ipo_info() — upcoming + recent IPO data
  - stock_zh_a_st_em() — current ST (special treatment) stocks
  - stock_info_sh_delist() / stock_info_sz_delist() — delisted stocks
"""
import logging

import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code_series

logger = logging.getLogger(__name__)


class IPOStSource:
    """Fetch IPO calendar, ST list, and delisting records.

    These are used for universe filtering — new IPOs (< 1 year) should be
    excluded from training because they have no history, ST stocks have
    unusual trading rules (5% limit, no margin), and delisted stocks
    help with survivorship bias correction.
    """

    def fetch_ipo_calendar(self) -> pd.DataFrame:
        """Fetch IPO calendar for A-shares.

        Returns DataFrame with columns: stock_code, stock_name, ipo_date,
        list_date, issue_price, total_shares, pe_ratio.
        """
        import akshare as ak
        logger.info("Fetching IPO calendar...")
        df = ak.stock_new_ipo_cninfo()
        df = df.rename(columns={
            "证劵代码": "stock_code",
            "证券简称": "stock_name",
            "申购日期": "ipo_date",
            "上市日期": "list_date",
            "发行价": "issue_price",
            "总发行数量": "total_shares",
            "发行市盈率": "pe_ratio",
        })
        df["stock_code"] = normalize_stock_code_series(df["stock_code"])
        df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        df["issue_price"] = pd.to_numeric(df["issue_price"], errors="coerce")
        df["total_shares"] = pd.to_numeric(df["total_shares"], errors="coerce")
        df["pe_ratio"] = pd.to_numeric(df["pe_ratio"], errors="coerce")
        keep = ["stock_code", "stock_name", "ipo_date", "list_date",
                "issue_price", "total_shares", "pe_ratio"]
        logger.info("IPO calendar: %d stocks", len(df))
        return df[[c for c in keep if c in df.columns]]

    def fetch_st_list(self) -> pd.DataFrame:
        """Fetch current ST (special treatment) stocks.

        Returns DataFrame with columns: stock_code, stock_name, st_type.
        """
        import akshare as ak
        logger.info("Fetching current ST list...")
        df = ak.stock_zh_a_st_em()
        df = df.rename(columns={
            "代码": "stock_code",
            "名称": "stock_name",
            "相关性": "st_type",
        })
        df["stock_code"] = normalize_stock_code_series(df["stock_code"])
        logger.info("ST list: %d stocks", len(df))
        return df

    def fetch_delisted(self) -> pd.DataFrame:
        """Fetch delisted stocks from SSE + SZSE.

        Returns DataFrame with explicit semantic columns: stock_code,
        stock_name, list_date, suspension_date, delist_effective_date,
        delist_reason, market.

        The two exchanges report the exit in DIFFERENT fields — SSE's
        ``stock_info_sh_delist`` gives 暂停上市日期 (suspension = the day trading
        stopped), SZSE's ``stock_info_sz_delist`` gives 终止上市日期 (formal
        removal).  §八-2: both are kept as SEPARATE columns rather than collapsed
        into one Chinese field, so the universe layer can resolve a conservative
        exit date per market without guessing which single field to trust.
        """
        import akshare as ak
        logger.info("Fetching delisted stocks...")
        frames = []
        for market, func in [("SSE", ak.stock_info_sh_delist),
                              ("SZSE", ak.stock_info_sz_delist)]:
            try:
                df = func()
                df["market"] = market
                frames.append(df)
            except Exception as e:
                logger.warning("Failed to fetch %s delisted: %s", market, e)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)

        # Both markets put the code in a different column — SSE rows carry only
        # 公司代码, SZSE rows only 证券代码/股票代码.
        for col in ("公司代码", "证券代码", "股票代码"):
            if col not in result.columns:
                result[col] = np.nan
        code = result["公司代码"].where(result["公司代码"].notna(), result["证券代码"])
        code = code.where(code.notna(), result["股票代码"])
        result["stock_code"] = normalize_stock_code_series(code)

        for col in ("公司简称", "证券简称", "股票简称"):
            if col not in result.columns:
                result[col] = np.nan
        name = result["公司简称"].where(result["公司简称"].notna(), result["证券简称"])
        name = name.where(name.notna(), result["股票简称"])
        result["stock_name"] = name

        for out, src in (
            ("list_date", "上市日期"),
            ("suspension_date", "暂停上市日期"),
            ("delist_effective_date", "终止上市日期"),
        ):
            result[out] = pd.to_datetime(
                result[src] if src in result.columns
                else pd.Series(pd.NaT, index=result.index),
                errors="coerce",
            )
        result["delist_reason"] = (
            result["终止上市原因"] if "终止上市原因" in result.columns
            else pd.Series(np.nan, index=result.index)
        )

        keep = ["stock_code", "stock_name", "list_date", "suspension_date",
                "delist_effective_date", "delist_reason", "market"]
        result = result[keep].dropna(subset=["stock_code"])
        logger.info("Delisted: %d stocks", len(result))
        return result

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch all IPO/ST/delisted data.

        Returns dict with keys: ipo, st_list, delisted.
        """
        return {
            "ipo": self.fetch_ipo_calendar(),
            "st_list": self.fetch_st_list(),
            "delisted": self.fetch_delisted(),
        }
