"""Dragon-Tiger board (龙虎榜) data via AKShare."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

LHB_COLS = [
    "date", "stock_code", "stock_name", "lhb_reason",
    "buy_amount", "sell_amount", "net_amount",
    "buy_inst_amount", "sell_inst_amount",
    "buy_broker_count", "sell_broker_count",
]


class DragonTigerSource:
    """Fetch daily dragon-tiger board (龙虎榜) data."""

    SOURCE_NAME = "akshare_lhb"

    def fetch_daily(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch LHB daily detail for a date range.

        Returns per-stock LHB appearance data.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for LHB data")
            return pd.DataFrame()

        all_frames = []

        # Generate date range and fetch each day
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
        except Exception:
            logger.error("Invalid date range: %s to %s", start_date, end_date)
            return pd.DataFrame()

        for d in dates:
            date_str = d.strftime("%Y%m%d")
            try:
                daily = ak.stock_lhb_detail_daily(
                    trade_date=date_str, adjust=""
                )
                if daily is not None and not daily.empty:
                    daily = daily.copy()
                    daily["date"] = d.strftime("%Y-%m-%d")
                    all_frames.append(daily)
            except Exception as e:
                logger.debug("LHB daily %s failed: %s", date_str, e)
                continue

        if not all_frames:
            return pd.DataFrame()

        df = pd.concat(all_frames, ignore_index=True)
        return self._normalize(df)

    def fetch_by_stock(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch LHB history for a single stock."""
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        try:
            df = ak.stock_lhb_stock_detail_em(
                symbol=stock_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            if "date" not in df.columns and "日期" in df.columns:
                # Infer from trade_date parameter
                df["date"] = pd.date_range(start_date, end_date, periods=len(df))
            return self._normalize(df)
        except Exception as e:
            logger.debug("LHB stock %s fetch failed: %s", stock_code, e)
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize to standard LHB columns."""
        df = df.copy()

        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "日期" in col or col == "date":
                col_map[col] = "date"
            elif "代码" in col or "code" in cl:
                col_map[col] = "stock_code"
            elif "名称" in col or "name" in cl:
                col_map[col] = "stock_name"
            elif "解读" in col or "reason" in cl:
                col_map[col] = "lhb_reason"
            elif "买入" in col and "金额" in col:
                col_map[col] = "buy_amount"
            elif "卖出" in col and "金额" in col:
                col_map[col] = "sell_amount"
            elif "净买" in col or "净买额" in col:
                col_map[col] = "net_amount"
            elif "机构" in col and "买入" in col:
                col_map[col] = "buy_inst_amount"
            elif "机构" in col and "卖出" in col:
                col_map[col] = "sell_inst_amount"

        if col_map:
            df = df.rename(columns=col_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        elif "stock_code" not in df.columns:
            # Try to extract from other code columns
            for col in df.columns:
                if "代码" in str(col):
                    df["stock_code"] = df[col].astype(str).str.zfill(6)
                    break

        # Ensure numeric columns
        for col in ["buy_amount", "sell_amount", "net_amount",
                     "buy_inst_amount", "sell_inst_amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute net if missing
        if "net_amount" not in df.columns:
            if "buy_amount" in df.columns and "sell_amount" in df.columns:
                df["net_amount"] = df["buy_amount"] - df["sell_amount"]

        keep = [c for c in LHB_COLS if c in df.columns]
        return df[keep].reset_index(drop=True)
