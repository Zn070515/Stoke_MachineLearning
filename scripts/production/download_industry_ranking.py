"""Build industry ranking data from daily K-line + sector cache.

Computes sector-level daily features from existing stock data:
- change_pct: sector return (equal-weighted mean of constituent returns)
- rank: cross-sectional rank by change_pct per date
- up_count / down_count: constituent stock advance/decline counts
- leader: best-performing stock in sector each day

Output: single market-wide ``industry_ranking.parquet`` for SectorBroadcaster.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_industry_ranking.py
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.download_manifest import write_run_manifest_or_exit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    data_dir = cfg.project.data_dir
    daily_dir = os.path.join(data_dir, "a_shares", "daily")
    cache_path = os.path.join(data_dir, "a_shares", "stock_sector_cache.csv")
    output_path = os.path.join(data_dir, "a_shares", "industry_ranking.parquet")

    # 1. Load sector map
    if not os.path.exists(cache_path):
        logger.error("No stock_sector_cache.csv — run download_data.py first")
        sys.exit(1)
    sector_df = pd.read_csv(cache_path, dtype=str)
    sector_map = dict(zip(sector_df["stock_code"], sector_df["sector"]))
    unique_sectors = sorted(set(sector_map.values()))
    logger.info("Sector map: %d stocks → %d sectors", len(sector_map), len(unique_sectors))

    # 2. Assign short sector codes for the output
    sector_code_map = {
        name: f"SEC{i:04d}" for i, name in enumerate(unique_sectors)
    }

    # 3. Load all stock daily returns into a single massive DataFrame
    logger.info("Loading daily returns for all stocks...")
    t0 = time.time()
    daily_files = sorted(
        f for f in os.listdir(daily_dir) if f.endswith(".parquet")
    )
    frames = []
    processed = 0
    for fname in daily_files:
        code = fname.replace(".parquet", "")
        sector = sector_map.get(code)
        if sector is None:
            continue
        try:
            df = pd.read_parquet(
                os.path.join(daily_dir, fname),
                columns=["date", "pct_change"],
            )
        except Exception:
            continue
        if df.empty:
            continue
        df["stock_code"] = code
        df["sector_name"] = sector
        df["sector_code"] = sector_code_map[sector]
        df.rename(columns={"pct_change": "pct_chg"}, inplace=True)
        frames.append(df)
        processed += 1
        if processed % 1000 == 0:
            logger.info("  Loaded %d/%d stocks...", processed, len(daily_files))
    logger.info("Loaded %d stocks in %.1fs", processed, time.time() - t0)

    if not frames:
        logger.error("No data loaded")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    all_data["date"] = pd.to_datetime(all_data["date"], errors="coerce")
    all_data["pct_chg"] = pd.to_numeric(all_data["pct_chg"], errors="coerce")
    all_data = all_data.dropna(subset=["date", "pct_chg"])

    # 4. Compute sector-level aggregates per date
    logger.info("Computing sector aggregates per date...")
    t0 = time.time()

    # Sector return: equal-weighted mean of constituent pct_chg
    sector_ret = (
        all_data.groupby(["date", "sector_code", "sector_name"])["pct_chg"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sector_ret.columns = ["date", "sector_code", "sector_name",
                          "change_pct", "ret_std", "n_stocks"]

    # Up/down counts
    all_data["is_up"] = (all_data["pct_chg"] > 0).astype(int)
    all_data["is_down"] = (all_data["pct_chg"] < 0).astype(int)
    updown = (
        all_data.groupby(["date", "sector_code"])[["is_up", "is_down"]]
        .sum()
        .reset_index()
    )
    updown.columns = ["date", "sector_code", "up_count", "down_count"]

    # 5. Find leader (highest pct_chg per sector per date)
    idx = all_data.groupby(["date", "sector_code"])["pct_chg"].idxmax()
    leaders = all_data.loc[idx, ["date", "sector_code", "stock_code", "pct_chg"]].copy()
    leaders.columns = ["date", "sector_code", "leader", "leader_change"]

    # 6. Compute rank per date (1 = best sector)
    sector_ret["rank"] = (
        sector_ret.groupby("date")["change_pct"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 7. Merge all together
    result = sector_ret.merge(updown, on=["date", "sector_code"], how="left")
    result = result.merge(leaders, on=["date", "sector_code"], how="left")

    # Fill missing up/down counts
    result["up_count"] = result["up_count"].fillna(0).astype(int)
    result["down_count"] = result["down_count"].fillna(0).astype(int)
    result["leader"] = result["leader"].fillna("")
    result["leader_change"] = result["leader_change"].fillna(0.0)

    # Cast types
    result["change_pct"] = result["change_pct"].astype(np.float32)
    result["n_stocks"] = result["n_stocks"].astype(np.int16)
    result["rank"] = result["rank"].astype(np.int16)
    result["up_count"] = result["up_count"].astype(np.int16)
    result["down_count"] = result["down_count"].astype(np.int16)
    result["leader_change"] = result["leader_change"].astype(np.float32)

    # Drop rows without date (shouldn't happen)
    result = result.dropna(subset=["date"])

    # 8. Save
    result.to_parquet(output_path, index=False, compression='lz4')
    logger.info("Saved %d rows (%d dates, %d sectors) to %s (%.1fs)",
                len(result),
                result["date"].nunique(),
                result["sector_code"].nunique(),
                output_path,
                time.time() - t0)

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    # A run that cannot record its own coverage fails loudly (§v18-10).
    write_run_manifest_or_exit(
        data_dir, "a_shares/industry_ranking",
        requested=["industry_ranking"], failed=[],
        complete={"industry_ranking"}, success_count=1,
    )

    # Print sector listing for reference
    sector_listing = result.groupby("sector_code")["sector_name"].first()
    logger.info("Sectors:")
    for code, name in sorted(sector_listing.items()):
        n_dates = result[result["sector_code"] == code]["date"].nunique()
        logger.info("  %s → %s (%d days)", code, name, n_dates)


if __name__ == "__main__":
    main()
