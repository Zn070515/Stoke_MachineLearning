# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""Build a global daily market-environment panel (limit-up temperature DEFERRED).

Merges: account_stats (monthly investor/mkt-cap), highs_lows (daily breadth),
industry advance ratio, and a market-turnover z-score from daily K-line flats.
No limit-up ecology columns (deferred family).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config


def _z(s: pd.Series, win: int = 20) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0)


def build_turnover_daily(base: str) -> pd.Series:
    """Sum 'amount' across all daily flat files per date -> z-scored turnover."""
    amounts = []
    for f in glob.glob(os.path.join(base, "daily", "[0-9][0-9][0-9][0-9][0-9][0-9].parquet")):
        try:
            d = pd.read_parquet(f, columns=["date", "amount"])
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            amounts.append(d.groupby("date")["amount"].sum())
        except Exception:
            continue
    if not amounts:
        return pd.Series(dtype="float64")
    tot = pd.concat(amounts).groupby(level=0).sum()
    return _z(tot)


def build_industry_advance(base: str) -> pd.Series:
    """Fraction of industries with positive ind_return per date (long format)."""
    path = os.path.join(base, "industry", "industry_ranking_computed.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype="float64")
    raw = pd.read_parquet(path)
    if "date" not in raw.columns or "ind_return" not in raw.columns:
        return pd.Series(dtype="float64")
    d = raw[["date", "ind_return"]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.dropna(subset=["ind_return"])
    if d.empty:
        return pd.Series(dtype="float64")
    adv = d.groupby("date")["ind_return"].apply(lambda x: float((x > 0).mean()))
    return adv.rename("market_adv_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-turnover", action="store_true",
                    help="skip the slow daily-file turnover scan")
    args = ap.parse_args()

    cfg = load_config()
    base = os.path.join(cfg.project.data_dir, "a_shares")
    br = os.path.join(base, "market_breadth")

    # account_stats: 数据日期 is month-level ("2015-04") -> append -28 (month-end)
    # so the monthly figure takes effect near month-end before daily resample/ffill.
    acc = pd.read_parquet(os.path.join(br, "account_stats.parquet"))
    acc["date"] = pd.to_datetime(acc["数据日期"].astype(str) + "-28", errors="coerce")
    acc = acc.dropna(subset=["date"])
    acc = acc.sort_values("date").set_index("date")
    acc = acc.rename(columns={
        "新增投资者-数量": "investor_new_num",
        "沪深总市值": "mkt_cap_total",
        "沪深户均市值": "avg_account_cap",
    })
    acc_raw = acc[["investor_new_num", "mkt_cap_total", "avg_account_cap"]].resample("D").ffill()
    acc_z_monthly = acc[["investor_new_num", "mkt_cap_total", "avg_account_cap"]].apply(_z)
    acc_z_daily = acc_z_monthly.resample("D").ffill()

    hl = pd.read_parquet(os.path.join(br, "highs_lows.parquet"))
    hl["date"] = pd.to_datetime(hl["date"]).dt.normalize()
    hl = hl.set_index("date").sort_index()
    hl["high_low_ratio"] = hl["high20"] / (hl["high20"] + hl["low20"]).replace(0, np.nan)

    series = {
        "high_low_ratio": hl["high_low_ratio"],
        "mkt_cap_total_z": acc_z_daily["mkt_cap_total"],
        "avg_account_cap_z": acc_z_daily["avg_account_cap"],
        "investor_new_num": acc_raw["investor_new_num"],
        "investor_new_z": acc_z_daily["investor_new_num"],
    }
    adv = build_industry_advance(base)
    if not adv.empty:
        series["market_adv_ratio"] = adv
    if not args.skip_turnover:
        series["market_turnover_z"] = build_turnover_daily(base)

    out = pd.DataFrame(series).sort_index()
    out = out.fillna(0.0)
    out.index.name = "date"
    out.to_parquet(os.path.join(br, "market_env_daily.parquet"), compression="lz4")
    print(f"market_env_daily: {len(out)} dates "
          f"({out.index.min().date()} ~ {out.index.max().date()}), "
          f"{len(out.columns)} cols")


if __name__ == "__main__":
    main()
