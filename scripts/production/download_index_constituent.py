"""Download current index constituent lists for major CSI indices.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_index_constituent.py
"""
import argparse
import logging
import os

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.index_constituent_source import IndexConstituentSource
from stoke_ml.data.download_manifest import write_run_manifest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download index constituent data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: data/a_shares/index_constituents/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output or os.path.join(
        cfg.project.data_dir, "a_shares", "index_constituents"
    )
    os.makedirs(out_dir, exist_ok=True)

    src = IndexConstituentSource()
    df = src.fetch_all_indices()

    if df.empty:
        logger.error("No constituent data fetched.")
        # Unified run manifest (§五-5): an empty fetch is a failed run, not a
        # complete one — recorded so the gap is never mistaken for coverage.
        try:
            write_run_manifest(
                data_dir, "a_shares/index_constituents",
                requested=["constituents"], failed=["constituents"],
                complete=set(), success_count=0,
            )
        except Exception as exc:
            logger.error("run manifest write failed: %s", exc)
        return

    path = os.path.join(out_dir, "constituents.parquet")
    df.to_parquet(path, index=False, compression='lz4')
    logger.info("Saved %d rows across %d indices → %s",
                  len(df), df["index_code"].nunique(), path)

    # Also save per-index summary
    for idx_code, group in df.groupby("index_code"):
        idx_name = group["index_name"].iloc[0]
        logger.info("  %s (%s): %d stocks, weight range [%.4f, %.4f]",
                      idx_code, idx_name, len(group),
                      group["weight"].min(), group["weight"].max())

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/index_constituents",
            requested=["constituents"], failed=[], complete={"constituents"},
            success_count=1,
        )
    except Exception as exc:
        logger.error("run manifest write failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
