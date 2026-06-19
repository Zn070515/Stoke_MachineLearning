"""Data storage — Parquet partitioned by year/month."""
import os
import pandas as pd


class DataStorage:
    """Save and load market data as partitioned Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_daily(self, df: pd.DataFrame, market: str = "a_shares"):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(
                self._root, market, "daily", str(year), f"{month:02d}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])
            save_df.to_parquet(out_path, index=False)

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str,
        market: str = "a_shares"
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = os.path.join(self._root, market, "daily")
        if not os.path.exists(base):
            return pd.DataFrame()

        all_data = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    path = os.path.join(root, f)
                    df = pd.read_parquet(path)
                    df["date"] = pd.to_datetime(df["date"])
                    mask = (df["date"] >= start) & (df["date"] <= end)
                    all_data.append(df[mask])

        if not all_data:
            return pd.DataFrame()
        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values("date").reset_index(drop=True)
