"""Per-feature sparsity report over pre-built feature panels.

The hot-board / concept-heat feature families (has_hot_board, board_momentum,
avg_concept_heat, cb_*, ...) are constant zero for non-participating stocks
over long windows, so their all-history mean IC is dominated by the recent
active window and their time-series IC on most stocks is undefined. This report
quantifies, per feature column, how sparse it is ACROSS the panel:

  - mean_nonzero_ratio : fraction of stock-days with value != 0
  - constant_stock_ratio: fraction of stocks where the column never varies
    (unique value count == 1 over that stock's full history)
  - sparse (tier)     : "event" if mean_nonzero_ratio < 0.01
                        (signal exists only for a small subset of the panel)

Event-sparse features are informational: they carry cross-sectional signal on
the few dates where they activate (that is exactly what the IC report shows for
has_hot_board), but their time-series IC on the broad panel is meaningless.
The Panel model handles them via their has_* mask columns (already emitted).

Output: reports/feature_sparsity_report.csv
"""
import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd

from stoke_ml.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Non-metadata, non-price columns only.
SKIP = {"date", "stock_code", "sector", "sector_code", "size_proxy",
        "open", "high", "low", "close", "volume", "amount", "pct_change"}

EVENT_SPARSE_RATIO = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--max-stocks", type=int, default=None,
                    help="cap stock count (dev runs); None = all")
    ap.add_argument("--event-ratio", type=float, default=EVENT_SPARSE_RATIO,
                    help="mean_nonzero_ratio below this => event-sparse")
    args = ap.parse_args()

    cfg = load_config()
    feat_dir = args.features_dir or os.path.join(cfg.project.data_dir, "features")
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if args.max_stocks:
        files = files[: args.max_stocks]
    if not files:
        log.error("no feature files under %s", feat_dir)
        return
    log.info("%d stocks", len(files))

    # Per-stock counts per column: (n_nonzero, n_rows, unique==1 flag).
    agg: dict[str, dict] = {}
    n_loaded = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            log.warning("skip %s: %s", os.path.basename(f), e)
            continue
        if df.empty:
            continue
        for c in df.columns:
            if c in SKIP or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x = df[c].to_numpy()
            if x.size == 0:
                continue
            a = agg.setdefault(c, {"nz": 0.0, "n": 0.0, "const": 0.0})
            a["nz"] += float((x != 0).sum())
            a["n"] += float(x.size)
            if np.unique(x).size <= 1:
                a["const"] += 1.0
        n_loaded += 1
    log.info("loaded %d stocks, %d numeric features", n_loaded, len(agg))
    if not agg:
        log.error("no features measured")
        return

    rows = []
    for c, a in agg.items():
        nz_ratio = a["nz"] / a["n"] if a["n"] else 0.0
        const_ratio = a["const"] / n_loaded if n_loaded else 0.0
        tier = "event" if nz_ratio < args.event_ratio else "dense"
        rows.append({
            "feature": c,
            "mean_nonzero_ratio": round(nz_ratio, 6),
            "constant_stock_ratio": round(const_ratio, 6),
            "tier": tier,
        })
    rep = pd.DataFrame(rows)
    rep = rep.sort_values(["tier", "mean_nonzero_ratio", "feature"],
                          ascending=[False, True, True])

    os.makedirs("reports", exist_ok=True)
    out = "reports/feature_sparsity_report.csv"
    rep.to_csv(out, index=False)
    log.info("wrote %s (%d features)", out, len(rep))

    n_event = int((rep["tier"] == "event").sum())
    n_const = int((rep["constant_stock_ratio"] >= 0.9).sum())
    log.info("event-sparse (nz_ratio<%.2f): %d features", args.event_ratio, n_event)
    log.info("constant-on->=90%%-of-stocks: %d features", n_const)
    log.info("top-20 sparsest:\n%s", rep[rep["tier"] == "event"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
