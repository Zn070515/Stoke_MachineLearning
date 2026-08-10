"""Download market breadth indicators — new highs/lows, account statistics.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_market_breadth.py
"""
import argparse
import logging
import os

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.asset_contract import (
    AtomicCommit,
    DataAssetContract,
    contract_for_channel,
    write_asset_manifest,
)
from stoke_ml.data.sources.a_shares.market_breadth_source import MarketBreadthSource
from stoke_ml.data.download_manifest import write_run_manifest_or_exit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

#: File-level asset contract for the single market-wide ``highs_lows.parquet``
#: (the market_env PRICE upstream feeding ``build_market_env._build_high_low_ratio``).
#: Mirrors ``INDUSTRY_RANKING_ASSET``: one file, ``date`` extent (values recorded
#: FOR a trading day → ``record_date``).  Same-day trade breadth → verified-by-
#: nature PIT (immutable_snapshot / raw / verified per channel_vintage).
HIGH_LOWS_ASSET: DataAssetContract = contract_for_channel(
    "highs_lows",
    data_type="highs_lows",
    partition="single_file",
    extent_column="date",
    effective_date_policy="record_date",
)


def main():
    parser = argparse.ArgumentParser(description="Download market breadth data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: data/a_shares/market_breadth/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    out_dir = args.output or os.path.join(
        data_dir, "a_shares", "market_breadth"
    )
    os.makedirs(out_dir, exist_ok=True)

    src = MarketBreadthSource()
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
            if name == "highs_lows":
                # §v19: the market_env PRICE upstream is a GOVERNED asset — atomic
                # write (temp + os.replace) + the HIGH_LOWS_ASSET per-file content
                # manifest, so build_market_env._build_high_low_ratio's formal read
                # (require_valid_manifest=True) can verify the file.  The manifest
                # records schema/source/extent/vintage (same-day trade breadth →
                # verified PIT) and flags a tampered/altered file at read time.
                df.attrs["source"] = ("AKShare stock_a_high_low_statistics "
                                      "(same-day market-wide breadth)")
                with AtomicCommit(path) as ac:
                    df.to_parquet(ac.tmp_path, index=False, compression='lz4')
                write_asset_manifest(path, HIGH_LOWS_ASSET, df)
            else:
                df.to_parquet(path, index=False, compression='lz4')
            logger.info("Saved %s: %d rows → %s", name, len(df), path)
            done.add(name)
        except Exception as e:
            failed.append(name)
            logger.error("%s: %s", name, str(e)[:120])

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    # A manifest-write failure is FATAL — a run that cannot record its own
    # coverage must fail loudly, never exit 0 (§十一).
    write_run_manifest_or_exit(
        data_dir, "a_shares/market_breadth",
        requested=list(data.keys()), failed=failed, complete=done,
        success_count=len(done),
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
