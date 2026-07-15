"""EastMoney datacenter unified sources — thin wrappers around EastMoneyClient.

Each source is a separate class but they all use EastMoneyClient.datacenter()
internally. The datacenter API is EastMoney's unified data platform covering:

- Block trades (大宗交易) → RPT_DATA_BLOCKTRADE
- Shareholder count (股东户数) → RPT_HOLDERNUMLATEST
- Lockup expiry (限售解禁) → RPT_LIFT_STAGE
- Dividend history (分红送转) → RPT_SHAREBONUS_DET

All sources are per-stock with date filtering. Amounts in CNY (元).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from stoke_ml.crawler.eastmoney import EastMoneyClient

logger = logging.getLogger(__name__)

# ── Block trade (大宗交易) ───────────────────────────────────────────────

BLOCK_TRADE_COLS = [
    "date", "stock_code",
    "deal_price", "close_price", "premium_pct",
    "volume", "amount", "buyer", "seller",
]


class BlockTradeSource:
    """Fetch block trade records (大宗交易) per stock.

    Block trades are large off-exchange transactions that signal
    institutional activity. Premium/discount to market close is a
    key sentiment indicator.
    """

    SOURCE_NAME = "eastmoney_block_trade"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch(self, code: str, page_size: int = 50) -> pd.DataFrame:
        """Fetch recent block trade records for a stock.

        Returns DataFrame with: date, deal_price, close_price,
        premium_pct (溢价率%), volume, amount, buyer, seller.
        """
        raw = self._client.datacenter(
            "RPT_DATA_BLOCKTRADE",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="TRADE_DATE",
            sort_types="-1",
        )
        if not raw:
            return pd.DataFrame(columns=BLOCK_TRADE_COLS)

        rows = []
        for r in raw:
            close = float(r.get("CLOSE_PRICE") or 0)
            deal_price = float(r.get("DEAL_PRICE") or 0)
            premium = ((deal_price / close - 1) * 100) if close else 0
            rows.append({
                "date": str(r.get("TRADE_DATE", ""))[:10],
                "stock_code": code,
                "deal_price": float(deal_price),
                "close_price": float(close),
                "premium_pct": round(premium, 2),
                "volume": float(r.get("DEAL_VOLUME") or 0),
                "amount": float(r.get("DEAL_AMT") or 0),
                "buyer": str(r.get("BUYER_NAME") or ""),
                "seller": str(r.get("SELLER_NAME") or ""),
            })

        df = pd.DataFrame(rows, columns=BLOCK_TRADE_COLS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, codes: list[str], start_date: Optional[str] = None,
        end_date: Optional[str] = None, page_size: int = 50,
    ) -> pd.DataFrame:
        """Fetch block trades for multiple stocks."""
        frames = []
        for code in codes:
            df = self.fetch(code, page_size=page_size)
            if df.empty:
                continue
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=BLOCK_TRADE_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._client.close()


# ── Shareholder count (股东户数变化) ────────────────────────────────────

SHAREHOLDER_COLS = [
    "date", "stock_code",
    "holder_num", "change_num", "change_ratio", "avg_shares",
]


class ShareholderSource:
    """Fetch shareholder count changes (股东户数变化) per stock.

    Declining shareholder count = ownership concentration = bullish signal
    (retail exits, institutions accumulate). Quarterly frequency.
    """

    SOURCE_NAME = "eastmoney_shareholder"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch(self, code: str, page_size: int = 20) -> pd.DataFrame:
        """Fetch recent shareholder count records.

        Returns DataFrame with: date, holder_num, change_num(环比变化量),
        change_ratio(环比变化率%), avg_shares(户均持股).
        """
        raw = self._client.datacenter(
            "RPT_HOLDERNUMLATEST",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="END_DATE",
            sort_types="-1",
        )
        if not raw:
            return pd.DataFrame(columns=SHAREHOLDER_COLS)

        rows = []
        for r in raw:
            rows.append({
                "date": str(r.get("END_DATE", ""))[:10],
                "stock_code": code,
                "holder_num": int(r.get("HOLDER_NUM") or 0),
                "change_num": int(r.get("HOLDER_NUM_CHANGE") or 0),
                "change_ratio": float(r.get("HOLDER_NUM_RATIO") or 0),
                "avg_shares": float(r.get("AVG_FREE_SHARES") or 0),
            })

        df = pd.DataFrame(rows, columns=SHAREHOLDER_COLS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, codes: list[str], start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch shareholder data for multiple stocks."""
        frames = []
        for code in codes:
            df = self.fetch(code)
            if df.empty:
                continue
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=SHAREHOLDER_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._client.close()


# ── Lockup expiry (限售解禁) ────────────────────────────────────────────

LOCKUP_COLS = [
    "date", "stock_code",
    "free_type", "free_shares", "able_shares", "free_ratio",
]


class LockupExpirySource:
    """Fetch lockup expiry calendar (限售解禁) per stock.

    Lockup expiry releases previously restricted shares into the
    market — a major supply-side event. Tracks both historical
    unlocks and upcoming (future 90 days) unlocks.
    """

    SOURCE_NAME = "eastmoney_lockup"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch_history(self, code: str, page_size: int = 15) -> pd.DataFrame:
        """Fetch historical lockup expiry records.

        Returns DataFrame with: date, free_type(解禁类型),
        free_shares(本次解禁,万股), able_shares(实际可流通,万股),
        free_ratio(占总股本比,小数×100=百分比).
        """
        raw = self._client.datacenter(
            "RPT_LIFT_STAGE",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="FREE_DATE",
            sort_types="-1",
        )
        return self._parse_lockup(raw, code)

    def fetch_upcoming(
        self, code: str, trade_date: Optional[str] = None,
        forward_days: int = 90, page_size: int = 20,
    ) -> pd.DataFrame:
        """Fetch upcoming lockup expiry within forward_days from trade_date.

        Args:
            code: Stock code.
            trade_date: Reference date (YYYY-MM-DD). Defaults to today.
            forward_days: Days to look ahead (default 90).
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (
            datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=forward_days)
        ).strftime("%Y-%m-%d")

        raw = self._client.datacenter(
            "RPT_LIFT_STAGE",
            filter_str=(
                f'(SECURITY_CODE="{code}")'
                f"(FREE_DATE>='{trade_date}')"
                f"(FREE_DATE<='{end_date}')"
            ),
            page_size=page_size,
            sort_columns="FREE_DATE",
            sort_types="1",
        )
        return self._parse_lockup(raw, code)

    def _parse_lockup(self, raw: list[dict], code: str) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=LOCKUP_COLS)

        rows = []
        for r in raw:
            rows.append({
                "date": str(r.get("FREE_DATE", ""))[:10],
                "stock_code": code,
                "free_type": str(r.get("FREE_SHARES_TYPE") or ""),
                "free_shares": float(r.get("FREE_SHARES") or 0),
                "able_shares": float(r.get("ABLE_FREE_SHARES") or 0),
                "free_ratio": float(r.get("FREE_RATIO") or 0),
            })

        df = pd.DataFrame(rows, columns=LOCKUP_COLS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_all(
        self, code: str, trade_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch both history and upcoming lockup data.

        Returns: {"history": DataFrame, "upcoming": DataFrame}
        """
        return {
            "history": self.fetch_history(code),
            "upcoming": self.fetch_upcoming(code, trade_date=trade_date),
        }

    def close(self):
        self._client.close()


# ── Dividend history (分红送转) ──────────────────────────────────────────

DIVIDEND_COLS = [
    "date", "stock_code",
    "bonus_rmb", "transfer_ratio", "bonus_ratio", "plan",
]


class DividendSource:
    """Fetch dividend & split history (分红送转) per stock.

    Tracks per-share dividend (每股派息), transfer ratio (转增比例),
    and bonus share ratio (送股比例).
    """

    SOURCE_NAME = "eastmoney_dividend"

    def __init__(self, min_interval: float = 1.2):
        self._client = EastMoneyClient(min_interval=min_interval)

    def fetch(self, code: str, page_size: int = 20) -> pd.DataFrame:
        """Fetch dividend history for a stock.

        Returns DataFrame with: date(ex-dividend date),
        bonus_rmb(每股派息税前), transfer_ratio(每10股转增),
        bonus_ratio(每10股送股), plan(进度).
        """
        raw = self._client.datacenter(
            "RPT_SHAREBONUS_DET",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=page_size,
            sort_columns="EX_DIVIDEND_DATE",
            sort_types="-1",
        )
        if not raw:
            return pd.DataFrame(columns=DIVIDEND_COLS)

        rows = []
        for r in raw:
            rows.append({
                "date": str(r.get("EX_DIVIDEND_DATE", ""))[:10],
                "stock_code": code,
                "bonus_rmb": float(r.get("PRETAX_BONUS_RMB") or 0),
                "transfer_ratio": float(r.get("TRANSFER_RATIO") or 0),
                "bonus_ratio": float(r.get("BONUS_RATIO") or 0),
                "plan": str(r.get("ASSIGN_PROGRESS") or ""),
            })

        df = pd.DataFrame(rows, columns=DIVIDEND_COLS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_batch(
        self, codes: list[str], start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch dividend data for multiple stocks."""
        frames = []
        for code in codes:
            df = self.fetch(code)
            if df.empty:
                continue
            if start_date:
                df = df[df["date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["date"] <= pd.Timestamp(end_date)]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=DIVIDEND_COLS)
        return pd.concat(frames, ignore_index=True)

    def close(self):
        self._client.close()
