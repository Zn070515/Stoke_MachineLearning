"""Download IPO calendar, ST list, and delisting records.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_ipo_st.py
"""
import argparse
import logging
import os

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.sources.a_shares.ipo_st_source import IPOStSource
from stoke_ml.data.download_manifest import write_run_manifest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download IPO/ST/delisting data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: data/a_shares/universe/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output or os.path.join(cfg.project.data_dir, "a_shares", "universe")
    os.makedirs(out_dir, exist_ok=True)

    src = IPOStSource()
    data = src.fetch_all()

    done: set[str] = set()
    failed: list[str] = []
    for name, df in data.items():
        try:
            if df.empty:
                failed.append(name)
                logger.warning("Skipping %s (empty)", name)
                continue
            path = os.path.join(out_dir, f"{name}.parquet")
            df.to_parquet(path, index=False, compression='lz4')
            logger.info("Saved %s: %d rows → %s", name, len(df), path)
            done.add(name)
        except Exception as e:
            failed.append(name)
            logger.error("%s: %s", name, str(e)[:120])

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/universe",
            requested=list(data.keys()), failed=failed, complete=done,
            success_count=len(done),
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
