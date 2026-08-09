"""Build industry ranking data from daily K-line + sector membership.

Computes sector-level daily features from existing stock data:
- change_pct: sector return (equal-weighted mean of constituent returns)
- rank: cross-sectional rank by change_pct per date
- up_count / down_count: constituent stock advance/decline counts
- leader: best-performing stock in sector each day

Output: single market-wide ``industry_ranking.parquet`` for SectorBroadcaster.

§v19 P0#1: sectors are derived from the genuine-PIT ``sector_membership.parquet``
(per-date ``[date, stock_code, sector_code, sector_name]``) when present — every
daily row is joined on ``(date, stock_code)``, so a stock with no asserted CSRC
gate that date is EXCLUDED (honest unclassified, never present-backfilled).  The
legacy ``stock_sector_cache.csv`` snapshot (SEC#### short codes) is only the
fallback for when the PIT artifact is absent.  §v19 P0#4: every daily file is
read through the canonical validated store (``require_valid_manifest=True``) and
ANY invalid file aborts the whole build — no silent skip.

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
from stoke_ml.data.codes import normalize_stock_code_series
from stoke_ml.data.download_manifest import write_run_manifest_or_exit
from stoke_ml.data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_industry_ranking(base: str) -> pd.DataFrame:
    """Derive the per-date sector ranking from daily K-line + sector membership.

    ``base`` is the ``a_shares`` dir (``data_dir/a_shares``).

    §v19 P0#4 (fail-closed): every daily file is read through
    ``DataStorage.load_daily(..., require_valid_manifest=True)``; ANY file that
    fails canonical validation aborts the build (a missing stock would silently
    change the sector returns).

    §v19 P0#1 (genuine-PIT sectors): when ``sector_membership.parquet`` is
    present, each daily row is joined to it on ``(date, stock_code)`` via an
    INNER join — rows without an asserted gate that date (pre-gate history,
    no-gate stocks) are honestly excluded from the sector aggregates, never
    present-backfilled.  Without it, the legacy snapshot cache is used with its
    SEC#### short codes (pre-P0#1 behavior preserved).
    """
    storage = DataStorage(os.path.dirname(base))
    codes = storage.list_stocks("a_shares")

    # 1. Load daily returns fail-closed (§v19 P0#4): a present-but-invalid daily
    #    file must abort, not silently drop a stock from the sector aggregates.
    problems: list[str] = []
    frames: list[pd.DataFrame] = []
    for code in codes:
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
        except (ValueError, OSError) as exc:
            problems.append(f"{code}: {exc}")
            continue
        if d is None or d.empty:
            continue
        d = d.copy()
        d["stock_code"] = code
        d = d.rename(columns={"pct_change": "pct_chg"})
        frames.append(d[["date", "stock_code", "pct_chg"]])
    if problems:
        raise SystemExit(
            f"download_industry_ranking: {len(problems)} daily files FAILED "
            "canonical validation — refusing to build industry ranking over "
            "incomplete inputs (§v19 P0#4):\n  " + "\n  ".join(problems[:20]))
    if not frames:
        raise SystemExit("download_industry_ranking: no data loaded")

    all_data = pd.concat(frames, ignore_index=True)
    all_data["date"] = pd.to_datetime(all_data["date"], errors="coerce")
    all_data["pct_chg"] = pd.to_numeric(all_data["pct_chg"], errors="coerce")
    all_data = all_data.dropna(subset=["date", "pct_chg"])

    # 2. Resolve sector membership.  PIT artifact first; snapshot fallback.
    membership_path = os.path.join(base, "sector_membership.parquet")
    if os.path.isfile(membership_path):
        mem = pd.read_parquet(membership_path)
        mem["date"] = pd.to_datetime(mem["date"], errors="coerce")
        mem["stock_code"] = normalize_stock_code_series(mem["stock_code"])
        mem = mem.dropna(subset=["date", "stock_code", "sector_code",
                                 "sector_name"])
        all_data["stock_code"] = normalize_stock_code_series(all_data["stock_code"])
        all_data = all_data.dropna(subset=["stock_code"])
        # INNER join: only days where CNINFO asserts a gate contribute.  A stock
        # with no gate that date is honestly unclassified — excluded, never
        # present-backfilled with today's classification (§v19 P0#1).
        all_data = all_data.merge(
            mem[["date", "stock_code", "sector_code", "sector_name"]],
            on=["date", "stock_code"], how="inner",
        )
        if all_data.empty:
            raise SystemExit(
                "download_industry_ranking: sector_membership.parquet asserts a "
                "gate on no daily row — refusing to write an empty ranking")
        logger.info("Sector membership: %d stocks in the PIT file",
                    mem["stock_code"].nunique())
    else:
        cache_path = os.path.join(base, "stock_sector_cache.csv")
        if not os.path.isfile(cache_path):
            raise SystemExit(
                "download_industry_ranking: neither sector_membership.parquet "
                "nor stock_sector_cache.csv — run download_sector_membership.py "
                "(or download_data.py) first")
        sector_df = pd.read_csv(cache_path, dtype=str)
        sector_map = dict(zip(normalize_stock_code_series(sector_df["stock_code"]),
                              sector_df["sector"]))
        sector_map = {k: v for k, v in sector_map.items() if k is not None}
        unique_sectors = sorted(set(sector_map.values()))
        sector_code_map = {
            name: f"SEC{i:04d}" for i, name in enumerate(unique_sectors)
        }
        logger.info("Sector map: %d stocks → %d sectors (snapshot fallback)",
                    len(sector_map), len(unique_sectors))
        all_data["stock_code"] = normalize_stock_code_series(all_data["stock_code"])
        all_data = all_data.dropna(subset=["stock_code"])
        all_data["sector_name"] = all_data["stock_code"].map(sector_map)
        all_data["sector_code"] = all_data["sector_name"].map(sector_code_map)
        all_data = all_data.dropna(subset=["sector_code", "sector_name"])
        if all_data.empty:
            raise SystemExit(
                "download_industry_ranking: snapshot cache matches no loaded "
                "stock — refusing to write an empty ranking")

    # 3. Compute sector-level aggregates per date
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

    # 4. Find leader (highest pct_chg per sector per date)
    idx = all_data.groupby(["date", "sector_code"])["pct_chg"].idxmax()
    leaders = all_data.loc[idx, ["date", "sector_code", "stock_code", "pct_chg"]].copy()
    leaders.columns = ["date", "sector_code", "leader", "leader_change"]

    # 5. Compute rank per date (1 = best sector)
    sector_ret["rank"] = (
        sector_ret.groupby("date")["change_pct"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 6. Merge all together
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

    logger.info("Built %d sector-day rows (%d sectors) in %.1fs",
                len(result), result["sector_code"].nunique(), time.time() - t0)
    return result


def main():
    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares")
    output_path = os.path.join(base, "industry_ranking.parquet")

    result = build_industry_ranking(base)

    # Save
    result.to_parquet(output_path, index=False, compression='lz4')
    logger.info("Saved %d rows (%d dates, %d sectors) to %s",
                len(result),
                result["date"].nunique(),
                result["sector_code"].nunique(),
                output_path)

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
