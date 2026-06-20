"""Sector ETF fund flow data via AKShare (Sina source)."""
import logging
import os

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "config", "sector_etf_mapping.yaml",
)


class SectorETFFlowSource:
    """Fetch sector-level ETF fund flows and attribute to stocks.

    Uses AKShare fund_etf_hist_sina (Sina-based, accessible globally)
    for daily OHLCV data, then computes flow from volume × price changes.
    """

    def __init__(self, mapping_path: str | None = None):
        self._mapping_path = mapping_path or DEFAULT_MAPPING_PATH
        self._sector_map = self._load_mapping()

    def _load_mapping(self) -> dict:
        try:
            with open(self._mapping_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("sectors", {})
        except FileNotFoundError:
            logger.warning("Sector ETF mapping not found: %s", self._mapping_path)
            return {}
        except Exception as e:
            logger.warning("Failed to load sector ETF mapping: %s", e)
            return {}

    def get_etf_codes(self) -> list[str]:
        codes = []
        for sector_info in self._sector_map.values():
            codes.extend(sector_info.get("etf_codes", []))
        return list(set(codes))

    @staticmethod
    def _to_sina_symbol(etf_code: str) -> str:
        """Convert 6-digit code to Sina symbol (shXXXXXX or szXXXXXX)."""
        code = str(etf_code).zfill(6)
        if code.startswith(("5", "6", "9")):
            return f"sh{code}"
        return f"sz{code}"

    def fetch_etf_daily(
        self, etf_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily K-line for a single ETF via AKShare (Sina source)."""
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare not available for ETF flow data")
            return pd.DataFrame()

        symbol = self._to_sina_symbol(etf_code)
        try:
            raw = ak.fund_etf_hist_sina(symbol=symbol)
        except Exception as e:
            logger.debug("ETF %s (%s) fetch failed: %s", etf_code, symbol, e)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # Filter date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        if df.empty:
            return pd.DataFrame()

        df["etf_code"] = etf_code
        df["pct_change"] = df["close"].pct_change()
        df["etf_flow"] = df["amount"] * np.sign(df["pct_change"])
        return df[["date", "etf_code", "volume", "amount", "pct_change", "etf_flow"]]

    def fetch_sector_flow(
        self, sector_name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch aggregated flow for a sector (all representative ETFs)."""
        sector_info = self._sector_map.get(sector_name)
        if not sector_info:
            logger.warning("Unknown sector: %s", sector_name)
            return pd.DataFrame()

        etf_codes = sector_info.get("etf_codes", [])
        frames = []
        for code in etf_codes:
            df = self.fetch_etf_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        # Aggregate by date
        agg = (
            combined.groupby("date")
            .agg(
                etf_flow_sum=("etf_flow", "sum"),
                etf_amount_sum=("amount", "sum"),
                etf_count=("etf_code", "nunique"),
            )
            .reset_index()
        )
        agg["sector_name"] = sector_name
        return agg

    def fetch_all_sectors(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch flow for all configured sectors."""
        all_frames = []
        for sector_name in self._sector_map:
            df = self.fetch_sector_flow(sector_name, start_date, end_date)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames, ignore_index=True)
