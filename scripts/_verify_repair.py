"""Verify stock_code repair: no dup dates, no NaN stock_code."""
import pandas as pd

R = r"D:\Projects\Stoke_MachineLearning\data\a_shares"

for t in ["capital_flow", "margin", "dragon_tiger", "northbound"]:
    for c in ["000001", "600519", "000725"]:
        p = f"{R}/{t}/{c}.parquet"
        try:
            df = pd.read_parquet(p)
        except FileNotFoundError:
            continue
        d = pd.to_datetime(df["date"])
        nan = df["stock_code"].isna().sum()
        print(
            f"{t}/{c}: rows={len(df)} dup_dates={df['date'].duplicated().sum()} "
            f"stock_code_NaN={nan} range={d.min().date()}~{d.max().date()}"
        )
