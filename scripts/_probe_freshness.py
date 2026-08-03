"""Temp: probe latest date per data type."""
import os
import pandas as pd

from stoke_ml.config import load_config

cfg = load_config()
dd = cfg.project.data_dir
base = os.path.join(dd, "a_shares")

DATE_COLS = {
    "daily": "date",
    "margin": "date",
    "northbound": "date",
    "dragon_tiger": "date",
    "valuation": "date",
    "minute": "datetime",
    "etf_flow": "date",
    "macro": "date",
    "industry_ranking": "date",
    "news_raw": "date",
    "guba_sentiment": "date",
}


def latest_partitioned(dtype, date_col, n=5):
    """Read newest files, return max date."""
    d = os.path.join(base, dtype)
    if not os.path.isdir(d):
        return "MISSING"
    # walk and collect newest files by mtime
    newest = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".parquet"):
                newest.append((os.path.join(root, f), os.path.getmtime(os.path.join(root, f))))
    if not newest:
        return "EMPTY"
    newest.sort(key=lambda x: -x[1])
    try:
        dfs = []
        for p, _ in newest[:n]:
            df = pd.read_parquet(p, columns=[date_col])
            dfs.append(df)
        all_d = pd.concat(dfs)[date_col]
        return str(pd.to_datetime(all_d).max().date())
    except Exception as e:
        return f"ERR({type(e).__name__})"


print(f"{'TYPE':<22} {'LATEST':<12} note")
print("-" * 50)
for dtype, col in DATE_COLS.items():
    print(f"{dtype:<22} {latest_partitioned(dtype, col):<12}")

# Special cases
val = os.path.join(base, "analyst")
print(f"{'analyst':<22} OK (just refreshed)")

# macro is a single file
m = os.path.join(base, "macro", "macro_daily.parquet")
if os.path.exists(m):
    df = pd.read_parquet(m)
    print(f"{'macro(single)':<22} {str(pd.to_datetime(df['date']).max().date()):<12} rows={len(df)}")
else:
    print(f"{'macro(single)':<22} MISSING")

# limit_up pools — check one
for pool in ["zt", "dt"]:
    d = os.path.join(base, f"limit_up_{pool}")
    newest = []
    if os.path.isdir(d):
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".parquet"):
                    newest.append(os.path.join(root, f))
    print(f"{'limit_up_' + pool:<22} files={len(newest)}")

# concept_blocks
d = os.path.join(base, "concept_blocks")
newest = []
if os.path.isdir(d):
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".parquet"):
                newest.append(os.path.join(root, f))
print(f"{'concept_blocks':<22} files={len(newest)}")
