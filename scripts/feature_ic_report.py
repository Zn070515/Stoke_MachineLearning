"""L4 gate: per-feature IC report over pre-built feature panels.

Computes, per feature column:
  - ic_cross   : mean cross-sectional Spearman RankIC vs forward return (per date)
  - icir_cross : ic_cross / std(IC across dates)
  - ic_pos_ratio: fraction of dates with positive IC
  - coverage   : fraction of dates with >= min_stocks observations
  - ic_ts      : mean per-stock time-series Spearman IC (handles global/regime features)
Dual window: full history + 2021+ primary window.

Global features (identical across stocks per date) have no cross-sectional IC
and are reported with ic_cross=NaN and their ic_ts intact.

Output: reports/feature_ic_report.csv
"""
import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from stoke_ml.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HORIZON = 1
MIN_STOCKS = 20
PRIMARY = pd.Timestamp("2021-01-01")

# Columns to skip (metadata / target / non-features).
SKIP = {"date", "stock_code", "sector", "sector_code", "size_proxy",
        "open", "high", "low", "close", "volume", "amount"}


def forward_return(df: pd.DataFrame, h: int = HORIZON) -> pd.Series:
    close = df["close"]
    return close.shift(-h) / close - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--max-stocks", type=int, default=None,
                    help="cap stock count (dev runs); None = all")
    ap.add_argument("--min-stocks", type=int, default=MIN_STOCKS,
                    help="min stocks per date for cross-sectional IC (dev runs can lower)")
    ap.add_argument("--feature-subset", default=None,
                    help="comma-separated features; None = all (minus SKIP)")
    args = ap.parse_args()
    min_stocks = args.min_stocks

    cfg = load_config()
    feat_dir = args.features_dir or os.path.join(cfg.project.data_dir, "features")
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if args.max_stocks:
        files = files[: args.max_stocks]
    if not files:
        log.error("no feature files under %s", feat_dir)
        return

    # Discover feature columns as a union across a sample of files.
    # Some columns are sparse (present for some stocks, absent for others);
    # reading only the first file would silently drop them from the report.
    # Read schema metadata only (no data) via ParquetFile for cheap discovery.
    union = set()
    n_sample = min(50, len(files))
    for f in files[:n_sample]:
        union.update(pq.ParquetFile(f).schema.names)
    cols = [c for c in union
            if c not in SKIP and not c.startswith("fwd_")]
    if args.feature_subset:
        cols = [c for c in cols if c in set(args.feature_subset.split(","))]
    log.info("%d stocks, %d candidate features (sampled %d files)",
             len(files), len(cols), n_sample)

    # Per-window: collect (date, stock, feature, fwd_ret) long frames.
    windows = {"full": [], "primary": []}
    n_loaded = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date", "stock_code", "close"] + cols)
        except Exception:
            # Some discovered column is absent in this file; re-read the full
            # file and intersect against what is actually available so a missing
            # sparse feature never drops the whole stock.
            try:
                df = pd.read_parquet(f)
            except Exception as e:
                log.warning("skip %s: %s", os.path.basename(f), e)
                continue
        if len(df) < 30 or "close" not in df:
            continue
        want = [c for c in cols if c in df.columns]
        df["fwd_ret"] = forward_return(df)
        df["date"] = pd.to_datetime(df["date"])
        n_loaded += 1
        for win, mask in (("full", df["date"] >= pd.Timestamp("2010-01-01")),
                          ("primary", df["date"] >= PRIMARY)):
            sub = df.loc[mask, ["date", "stock_code", "fwd_ret"] + want]
            sub = sub.dropna(subset=["fwd_ret"])
            windows[win].append(sub)
    log.info("loaded %d stocks", n_loaded)

    results = []
    for win, parts in windows.items():
        if not parts:
            continue
        panel = pd.concat(parts, ignore_index=True)
        dates = panel["date"].unique()
        for col in cols:
            # ---- cross-sectional IC (per date across stocks) ----
            ics = []
            for d in dates:
                sub = panel[panel["date"] == d][[col, "fwd_ret"]].dropna()
                if len(sub) < min_stocks or sub[col].nunique() < 2:
                    continue
                rho, _ = spearmanr(sub[col], sub["fwd_ret"])
                if np.isfinite(rho):
                    ics.append(rho)
            if ics:
                ics = np.asarray(ics)
                ic_cross = float(ics.mean())
                icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
                pos = float((ics > 0).mean())
                coverage = float(len(ics) / len(dates))
            else:
                # No date had >= min_stocks distinct values (e.g. global feature).
                ic_cross = np.nan
                icir = np.nan
                pos = np.nan
                coverage = 0.0
            # ---- time-series IC (per stock over time) ----
            ts_ics = []
            for code, g in panel.groupby("stock_code"):
                gg = g[[col, "fwd_ret"]].dropna()
                if len(gg) >= 30 and gg[col].nunique() >= 2:
                    rho, _ = spearmanr(gg[col], gg["fwd_ret"])
                    if np.isfinite(rho):
                        ts_ics.append(rho)
            ic_ts = float(np.mean(ts_ics)) if ts_ics else np.nan
            results.append({
                "window": win, "feature": col,
                "ic_cross": ic_cross, "icir_cross": icir,
                "ic_pos_ratio": pos, "coverage": coverage, "ic_ts": ic_ts,
            })
            log.info("[%s] %s ic=%.4f icir=%.2f ts=%.4f", win, col, ic_cross, icir, ic_ts)

    rep = pd.DataFrame(results)
    if rep.empty:
        log.error("no results produced")
        return

    def _grade(r):
        ic_c = abs(r["ic_cross"]) if np.isfinite(r["ic_cross"]) else 0.0
        ic_t = abs(r["ic_ts"]) if np.isfinite(r["ic_ts"]) else 0.0
        ic = max(ic_c, ic_t)
        return "high" if ic >= 0.02 else ("medium" if ic >= 0.01 else "low")

    rep["grade"] = rep.apply(_grade, axis=1)
    rep = rep.sort_values(["window", "grade", "ic_cross"],
                          ascending=[True, True, False],
                          na_position="last")
    os.makedirs("reports", exist_ok=True)
    rep.to_csv("reports/feature_ic_report.csv", index=False)
    log.info("wrote reports/feature_ic_report.csv (%d rows)", len(rep))


if __name__ == "__main__":
    main()
