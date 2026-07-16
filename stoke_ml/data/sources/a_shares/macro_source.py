"""Macro-economic data source for A-share market.

Free, no-registration sources via AKShare:
  - Shibor (daily, all tenors): O/N, 1W, 2W, 1M, 3M, 6M, 9M, 1Y
  - Exchange rates (daily): USD/CNY, EUR/CNY, JPY/CNY, HKD/CNY, GBP/CNY
  - China-US bond yields (daily): 2Y/5Y/10Y/30Y for both countries + spreads
  - PMI (monthly): manufacturing + non-manufacturing
  - Money supply (monthly): M0/M1/M2
  - Social financing (monthly): total + breakdown
  - CPI (monthly): YoY change
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroSource:
    """Fetch macro-economic indicators and align to daily trading calendar."""

    # Monthly indicators → column names in the daily-aligned output
    _MONTHLY_SPECS = {
        "pmi": {
            "func": "macro_china_pmi",
            "rename": {
                "制造业-指标": "pmi_manufacturing",
                "非制造业-指标": "pmi_non_manufacturing",
            },
        },
        "money_supply": {
            "func": "macro_china_money_supply",
            "rename": {
                "货币和准货币(M2)-金额(亿元)": "m2_amount",
                "货币和准货币(M2)-同比增长": "m2_yoy",
                "货币(M1)-金额(亿元)": "m1_amount",
                "货币(M1)-同比增长": "m1_yoy",
                "流通中现金(M0)-金额(亿元)": "m0_amount",
                "流通中现金(M0)-同比增长": "m0_yoy",
            },
        },
        "social_financing": {
            "func": "macro_china_shrzgm",
            "rename": {
                "社会融资规模增量": "sf_total",
                "新增-人民币贷款": "sf_rmb_loans",
                "新增-企业债券": "sf_corp_bonds",
                "新增-非金融企业境内股票融资": "sf_equity",
            },
        },
        "cpi": {
            "func": "macro_china_cpi_yearly",
            "rename": {
                "今值": "cpi_yoy",
            },
        },
    }

    @staticmethod
    def _parse_ym_date(date_val: str) -> pd.Timestamp | None:
        """Parse Chinese month-format dates like '2026年07月份' or '202607'.

        Returns the last day of the month as a Timestamp, or None on failure.
        """
        import re

        s = str(date_val).strip()
        # "2026年07月份" style
        m = re.match(r"(\d{4})年(\d{2})月", s)
        if m:
            yr, mo = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=yr, month=mo, day=1) + pd.offsets.MonthEnd(0)
        # "202607" style
        m = re.match(r"(\d{4})(\d{2})$", s)
        if m:
            yr, mo = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=yr, month=mo, day=1) + pd.offsets.MonthEnd(0)
        # "2008年03月份" with full-width
        m = re.match(r"(\d{4})\D+(\d{2})", s)
        if m:
            yr, mo = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=yr, month=mo, day=1) + pd.offsets.MonthEnd(0)
        return None

    def fetch_all(self, calendar=None) -> pd.DataFrame:
        """Fetch all macro indicators and align to daily trading calendar.

        Args:
            calendar: Optional TradingCalendar instance. If provided, dates are
                aligned to trading days only.

        Returns:
            DataFrame with date index and all macro columns at daily frequency.
            Monthly indicators are forward-filled.
        """
        import akshare as ak

        daily_dfs = []

        # ---- daily indicators ----
        daily_dfs.append(self._fetch_shibor(ak))
        daily_dfs.append(self._fetch_exchange_rates(ak))
        daily_dfs.append(self._fetch_bond_yields(ak))

        # ---- monthly indicators ----
        monthly_dfs = []
        for key, spec in self._MONTHLY_SPECS.items():
            try:
                df = self._fetch_monthly(ak, spec["func"], spec["rename"], key)
                if df is not None:
                    monthly_dfs.append(df)
            except Exception:
                logger.debug("Macro monthly %s fetch failed", key, exc_info=True)

        # Merge all daily
        daily = daily_dfs[0]
        for df in daily_dfs[1:]:
            daily = daily.join(df, how="outer")

        daily = daily.sort_index()

        # Forward-fill daily indicators (some may have gaps on weekends)
        daily = daily.ffill()

        # Merge monthly indicators on the daily index
        if monthly_dfs:
            for mdf in monthly_dfs:
                for col in mdf.columns:
                    # Create a daily series: only valid on month-end dates
                    daily_series = pd.Series(np.nan, index=daily.index, dtype="float32")
                    for ts, val in mdf[col].items():
                        if pd.notna(val):
                            daily_series.loc[ts] = val
                    daily[col] = daily_series.ffill()

        # Restrict to trading calendar if provided
        if calendar is not None:
            trading_days = calendar.get_trading_days(
                start=daily.index.min().strftime("%Y-%m-%d"),
                end=daily.index.max().strftime("%Y-%m-%d"),
            )
            tday_set = set(pd.Timestamp(d).date() for d in trading_days)
            daily = daily[daily.index.to_series().dt.date.isin(tday_set)]

        # Drop rows where all Shibor columns are NaN (before data starts)
        shibor_cols = [c for c in daily.columns if c.startswith("shibor_")]
        if shibor_cols:
            daily = daily.dropna(subset=shibor_cols, how="all")

        # Fill any remaining NaN with 0 (safety)
        daily = daily.fillna(0.0)

        return daily

    def _fetch_shibor(self, ak) -> pd.DataFrame:
        """Fetch daily Shibor rates. Returns DataFrame with date index."""
        df = ak.macro_china_shibor_all()
        # Columns: 日期, O/N-定价, O/N-涨跌幅, 1W-定价, 1W-涨跌幅, ...
        # AKShare may change suffix between 指标/定价 — detect dynamically.
        df = df.rename(columns={"日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        out = pd.DataFrame(index=df.index)
        tenors = ["O/N", "1W", "2W", "1M", "3M", "6M", "9M", "1Y"]
        for t in tenors:
            # Match the column that starts with the tenor code and contains the rate value
            for col in df.columns:
                if col.startswith(t) and "涨跌" not in col:
                    key = f"shibor_{t.replace('/', '_')}"
                    out[key] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                    break
        return out

    def _fetch_exchange_rates(self, ak) -> pd.DataFrame:
        """Fetch daily BOC exchange rates. Returns DataFrame with date index."""
        df = ak.currency_boc_safe()
        # Columns: 日期, 美元, 欧元, 日元, 港元, 英镑, ...
        rename = {
            "日期": "date",
            "美元": "fx_usd_cny",
            "欧元": "fx_eur_cny",
            "日元": "fx_jpy_cny",
            "港元": "fx_hkd_cny",
            "英镑": "fx_gbp_cny",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in df.columns:
            if col in rename.values():
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        keep = [v for v in rename.values() if v in df.columns]
        return df[keep]

    def _fetch_bond_yields(self, ak) -> pd.DataFrame:
        """Fetch daily China-US bond yields. Returns DataFrame with date index."""
        df = ak.bond_zh_us_rate()
        rename = {
            "日期": "date",
            "中国国债收益率2年": "bond_cn_2y",
            "中国国债收益率5年": "bond_cn_5y",
            "中国国债收益率10年": "bond_cn_10y",
            "中国国债收益率30年": "bond_cn_30y",
            "中国国债收益率10年-2年": "bond_cn_10y2y_spread",
            "美国国债收益率2年": "bond_us_2y",
            "美国国债收益率5年": "bond_us_5y",
            "美国国债收益率10年": "bond_us_10y",
            "美国国债收益率30年": "bond_us_30y",
            "美国国债收益率10年-2年": "bond_us_10y2y_spread",
            "中国GDP年增率": "gdp_cn_yoy",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in df.columns:
            if col in rename.values():
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        keep = [v for v in rename.values() if v in df.columns]
        return df[keep]

    def _fetch_monthly(self, ak, func_name: str, rename: dict, _key: str) -> pd.DataFrame | None:
        """Fetch a monthly indicator and return a DataFrame with date index."""
        func = getattr(ak, func_name)
        df = func()
        if df is None or df.empty:
            return None

        # Identify the date column (first column with 年/月/日期 in name)
        date_col = None
        for c in df.columns:
            if any(kw in str(c) for kw in ["月", "日期", "时间"]):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "raw_date"})
        df = df.rename(
            columns={k: v for k, v in rename.items() if k in df.columns}
        )

        dates = df["raw_date"].apply(self._parse_ym_date)
        df["date"] = dates
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

        keep = [v for v in rename.values() if v in df.columns]
        if not keep:
            return None
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df[keep]
