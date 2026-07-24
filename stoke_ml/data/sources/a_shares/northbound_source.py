"""North-bound capital / Stock Connect (北向资金) data via AKShare."""
import logging
import socket
import time

import pandas as pd

logger = logging.getLogger(__name__)

# AKShare internally calls requests.get() without a timeout parameter,
# which causes TCP connections to hang indefinitely when EastMoney
# rate-limits. Setting a global socket timeout prevents this.
socket.setdefaulttimeout(15)

NB_COLS = [
    "date", "stock_code", "north_hold_shares", "north_hold_value",
    "north_hold_pct", "north_net_buy", "north_buy_amount",
    "north_sell_amount",
]


class NorthboundSource:
    """Fetch north-bound capital flow (沪深港通) data."""

    SOURCE_NAME = "akshare_hsgt"

    def fetch_individual(
        self,
        start_date: str,
        end_date: str,
        stock_codes: list[str] | None = None,
        sleep: float = 0.5,
    ) -> pd.DataFrame:
        """Fetch individual stock northbound holdings and flows.

        Uses stock_hsgt_individual_em per-stock (returns full history, date-filtered after).
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for northbound data")
            return pd.DataFrame()

        if not stock_codes:
            logger.warning("No northbound stock codes specified")
            return pd.DataFrame()

        frames = []
        for i, code in enumerate(stock_codes):
            df = None
            for attempt in range(2):
                try:
                    df = ak.stock_hsgt_individual_em(symbol=code)
                    break
                except (TypeError, ValueError, KeyError):
                    break  # AKShare internal bug — retry won't help
                except Exception:
                    if attempt < 1:
                        time.sleep(1)
            if df is not None and not df.empty:
                df = self._normalize(df)
                df["stock_code"] = str(code).replace(".0", "").zfill(6)
                if "date" in df.columns:
                    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                if not df.empty:
                    frames.append(df)
            if len(stock_codes) > 25 and (i + 1) % 25 == 0:
                logger.info("  northbound %d/%d done, %d with data",
                            i + 1, len(stock_codes), len(frames))
            time.sleep(sleep)

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
            if col in ("持股日期",) or "date" in col.lower():
                col_map[col] = "date"
            elif "代码" in col or "code" in col.lower():
                col_map[col] = "stock_code"
            elif col in ("持股数量",) or "hold_shares" in col.lower():
                col_map[col] = "north_hold_shares"
            elif col in ("持股市值",) or "hold_value" in col.lower():
                col_map[col] = "north_hold_value"
            elif "持股比例" in col or "百分比" in col or "hold_pct" in col.lower():
                col_map[col] = "north_hold_pct"
            elif col in ("今日增持资金",) or "net_buy" in col.lower():
                col_map[col] = "north_net_buy"

        if col_map:
            df = df.rename(columns=col_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.replace(".0", "").str.zfill(6)

        for col in ["north_hold_pct", "north_net_buy"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        keep = [c for c in NB_COLS if c in df.columns]
        return df[keep].reset_index(drop=True)
