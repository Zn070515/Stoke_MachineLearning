"""Sector ETF fund flow data via EastMoney API."""
import logging
import os

import numpy as np
import pandas as pd
import yaml

from curl_cffi import requests

logger = logging.getLogger(__name__)

ETF_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

ETF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# Mapping file relative to project root
DEFAULT_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "config", "sector_etf_mapping.yaml",
)


class SectorETFFlowSource:
    """Fetch sector-level ETF fund flows and attribute to stocks."""

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
        """Get all ETF codes from the mapping."""
        codes = []
        for sector_info in self._sector_map.values():
            codes.extend(sector_info.get("etf_codes", []))
        return list(set(codes))

    def fetch_etf_daily(
        self, etf_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily K-line (includes volume/amount) for a single ETF.

        Uses volume and amount as proxies for fund flow.
        """
        secid = self._to_eastmoney_secid(etf_code)
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "101",
            "fqt": "1",
            "secid": secid,
            "beg": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "rtntype": "6",
        }

        try:
            resp = requests.get(
                ETF_KLINE_URL, params=params, headers=ETF_HEADERS,
                impersonate="chrome120", timeout=15,
            )
            if resp.status_code != 200:
                logger.debug("ETF %s fetch failed: HTTP %d", etf_code, resp.status_code)
                return pd.DataFrame()

            data = resp.json()
            if data.get("data") is None or data["data"].get("klines") is None:
                return pd.DataFrame()

            klines = data["data"]["klines"]
            if not klines:
                return pd.DataFrame()

            rows = [line.split(",") for line in klines]
            df = pd.DataFrame(rows, columns=[
                "date", "open", "close", "high", "low",
                "volume", "amount",
            ])
            df["date"] = pd.to_datetime(df["date"])
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["pct_change"] = pd.to_numeric(df["close"], errors="coerce").pct_change()
            df["etf_code"] = etf_code
            df["etf_flow"] = df["amount"] * np.sign(df["pct_change"])
            return df[["date", "etf_code", "volume", "amount", "pct_change", "etf_flow"]]

        except Exception as e:
            logger.debug("ETF %s fetch error: %s", etf_code, e)
            return pd.DataFrame()

    @staticmethod
    def _to_eastmoney_secid(etf_code: str) -> str:
        if etf_code.startswith(("5", "6", "9")):
            return f"1.{etf_code}"
        return f"0.{etf_code}"

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
