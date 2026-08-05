"""Download macro-economic indicators and save as daily features.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_macro.py
"""
import argparse
import logging
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.download_manifest import write_run_manifest
from stoke_ml.data.generation_store import write_generation
from stoke_ml.data.sources.a_shares.macro_source import MacroSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download macro-economic indicators")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    logger.info("Fetching macro indicators...")
    try:
        ms = MacroSource()
        df = ms.fetch_all()

        gen_name = write_generation(
            data_dir, "a_shares/macro/macro_daily", df,
            manifest={
                "dataset": "macro_daily",
                "rows": len(df),
                "columns": list(df.columns),
                "date_min": str(df.index.min().date()),
                "date_max": str(df.index.max().date()),
                "source": "akshare",
                "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        logger.info("Wrote macro generation %s", gen_name)

        done = {"macro_daily"}
        failed: list[str] = []
        logger.info("Done. Columns: %s", list(df.columns))
        logger.info("Date range: %s to %s", df.index.min().date(), df.index.max().date())
    except Exception as e:
        done: set[str] = set()
        failed = ["macro_daily"]
        logger.error("macro_daily: %s", str(e)[:120])

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    try:
        write_run_manifest(
            data_dir, "a_shares/macro",
            requested=["macro_daily"], failed=failed, complete=done,
            success_count=len(done),
        )
    except Exception as exc:
        logger.warning("run manifest write failed: %s", exc)


if __name__ == "__main__":
    main()
