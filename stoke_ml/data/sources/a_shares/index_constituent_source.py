"""CSI index constituent history source.

Tracks which stocks entered/left major A-share indices over time.
Critical for survivorship-bias-free backtesting — you can't trade
a stock as part of the CSI 300 universe before it was added.

Sources:
  - index_stock_cons_csindex() — current constituents of CSI indices
  - index_stock_cons_weight_csindex() — current constituents with weights
"""
import logging
import time
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Major CSI indices for universe construction
_DEFAULT_INDICES = {
    "000300": "CSI 300",
    "000905": "CSI 500",
    "000852": "CSI 1000",
    "000016": "SSE 50",
    "399006": "ChiNext",
    "000688": "STAR 50",
}


class IndexConstituentSource:
    """Fetch current index constituents with weights.

    Note: AKShare does not provide historical constituent snapshots,
    only current membership. For historical tracking, we snapshot
    periodically and detect changes by comparing snapshots over time.

    The current snapshot is still useful for:
    - Universe filtering (only trade stocks in target indices)
    - Index-weight features (large-cap bias, passive flow)
    """

    def fetch_current_constituents(
        self, symbol: str
    ) -> pd.DataFrame:
        """Fetch current constituents for one CSI index.

        Returns DataFrame with stock_code, stock_name, and optionally weight.
        """
        import akshare as ak
        logger.info("Fetching constituents for %s (%s)...",
                      symbol, _DEFAULT_INDICES.get(symbol, symbol))
        try:
            df = ak.index_stock_cons_weight_csindex(symbol=symbol)
        except Exception:
            try:
                df = ak.index_stock_cons_csindex(symbol=symbol)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()

        df["index_code"] = symbol
        df["index_name"] = _DEFAULT_INDICES.get(symbol, symbol)
        # Normalize column names (varies by AKShare version)
        col_map = {
            "成分券代码": "stock_code", "股票代码": "stock_code",
            "成分券名称": "stock_name", "股票名称": "stock_name",
            "权重": "weight", "成分券权重": "weight",
        }
        df = df.rename(columns={k: v for k, v in col_map.items()
                                 if k in df.columns})
        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        if "weight" not in df.columns:
            df["weight"] = 0.0
        now = datetime.now().strftime("%Y-%m-%d")
        df["snapshot_date"] = now
        return df

    def fetch_all_indices(
        self, indices: list[str] | None = None, sleep: float = 0.3
    ) -> pd.DataFrame:
        """Fetch constituents for multiple indices.

        Returns combined DataFrame with all index memberships.
        """
        if indices is None:
            indices = list(_DEFAULT_INDICES.keys())
        frames = []
        for i, sym in enumerate(indices):
            if i > 0:
                time.sleep(sleep)
            df = self.fetch_current_constituents(sym)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        logger.info("Index constituents: %d rows across %d indices",
                      len(result), result["index_code"].nunique())
        return result
