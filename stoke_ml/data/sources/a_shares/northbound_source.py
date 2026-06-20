"""North-bound capital / Stock Connect (北向资金) data via AKShare."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

NB_COLS = [
    "date", "stock_code", "north_hold_shares", "north_hold_value",
    "north_hold_pct", "north_net_buy", "north_buy_amount",
    "north_sell_amount",
]


class NorthboundSource:
    """Fetch north-bound capital flow (沪深港通) data."""

    SOURCE_NAME = "akshare_hsgt"

    def fetch_individual(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch individual stock northbound holdings and flows.

        Returns per-stock daily data: holdings, value, net buy.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for northbound data")
            return pd.DataFrame()

        frames = []

        # Try SSE (沪股通)
        for market, func in [
            ("north", ak.stock_hsgt_individual_em),
        ]:
            try:
                # AKShare stock_hsgt_individual_em returns latest data
                # For historical, use stock_hsgt_hist_em
                df = func(
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
                if df is not None and not df.empty:
                    df = self._normalize(df)
                    frames.append(df)
            except Exception as e:
                logger.debug("Northbound individual fetch failed: %s", e)
                break

        # Also try historical net flow
        try:
            import akshare as ak
            hist = ak.stock_hsgt_hist_em(symbol="北向资金")
            if hist is not None and not hist.empty:
                hist["stock_code"] = "999999"  # market aggregate
                hist = self._normalize(hist)
                frames.append(hist)
        except Exception:
            pass

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["date", "stock_code"])
        return result.sort_values(["date", "stock_code"]).reset_index(drop=True)

    def fetch_aggregate(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch aggregate north-south capital flow."""
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        try:
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df is None or df.empty:
                return pd.DataFrame()
            return self._normalize(df)
        except Exception as e:
            logger.debug("Northbound aggregate fetch failed: %s", e)
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize AKShare output to standard columns."""
        df = df.copy()

        col_map = {}
        for col in df.columns:
            if "日期" in col or "date" in col.lower():
                col_map[col] = "date"
            elif "代码" in col or "code" in col.lower():
                col_map[col] = "stock_code"
            elif "持股数" in col or "hold_shares" in col.lower():
                col_map[col] = "north_hold_shares"
            elif "持股市值" in col or "hold_value" in col.lower():
                col_map[col] = "north_hold_value"
            elif "持股比例" in col or "hold_pct" in col.lower():
                col_map[col] = "north_hold_pct"
            elif "净买入" in col or "net_buy" in col.lower():
                col_map[col] = "north_net_buy"
            elif "买入金额" in col or "buy_amount" in col.lower():
                col_map[col] = "north_buy_amount"
            elif "卖出金额" in col or "sell_amount" in col.lower():
                col_map[col] = "north_sell_amount"

        if col_map:
            df = df.rename(columns=col_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

        for col in ["north_hold_pct", "north_net_buy"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        keep = [c for c in NB_COLS if c in df.columns]
        return df[keep].reset_index(drop=True)
