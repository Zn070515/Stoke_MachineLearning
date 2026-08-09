"""Broadcast single-file asset contracts (§十七): industry / market_env.

Unlike the per-stock channels, these two are MARKET-WIDE single parquet files
whose trading dates live in the file's DatetimeIndex, not a column.  The
``extent_column="date"`` contract is therefore INDEX-backed: ``_extent`` falls
back to the DatetimeIndex when the named column is absent, so the manifest's
``start``/``end`` bound the trading-date span of the file.  Both channels are
immutable_snapshot / formula_versioned / proxy — the manifest carries the same
labels the training admission is judged against.
"""
import json
import os

import pandas as pd
import pytest

from stoke_ml.data.asset_contract import (
    check_asset_read,
    validate_asset_manifest,
    write_asset_manifest,
)
from stoke_ml.data.broadcast_assets import INDUSTRY_ASSET, MARKET_ENV_ASSET


def _manifest_of(parquet_path: str) -> dict:
    with open(parquet_path + ".manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _broadcast_frame(asset=None):
    """An index-dated frame shaped like the broadcast assets' output.

    ``MARKET_ENV_ASSET`` declares ``column_contract="market_env_daily"``, so its
    frame must carry the 3 required PRICE columns (the ACCOUNT part is optional
    and its absence is schema-valid, §v19-11); ``INDUSTRY_ASSET`` carries the
    sector-return columns download_industry.py produces.
    """
    if asset is MARKET_ENV_ASSET:
        return pd.DataFrame(
            {
                "high_low_ratio": [0.5, 0.4, 0.6],
                "market_adv_ratio": [0.6, 0.55, 0.7],
                "market_turnover_z": [1.0, -0.5, 0.2],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                name="date",
            ),
        )
    return pd.DataFrame(
        {
            "银行": [0.5, -0.1, 0.2],
            "白酒": [1.0, 0.3, -0.5],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            name="date",
        ),
    )


@pytest.mark.parametrize("asset", [INDUSTRY_ASSET, MARKET_ENV_ASSET])
def test_broadcast_index_extent_round_trip(tmp_path, asset):
    df = _broadcast_frame(asset)
    path = os.path.join(str(tmp_path), f"{asset.data_type}.parquet")
    df.to_parquet(path)
    write_asset_manifest(path, asset, df)

    manifest = _manifest_of(path)
    # index-backed extent bounds the trading-date span
    assert manifest["start"] == "2024-01-02"
    assert manifest["end"] == "2024-01-04"
    assert manifest["effective_date_policy"] == "index_date"
    assert manifest["vintage_source"] == "immutable_snapshot"
    assert manifest["vintage_transform"] == "formula_versioned"
    assert manifest["vintage_pit"] == "proxy"

    # validate re-reads the file with its index intact and still agrees
    report = validate_asset_manifest(path, asset)
    assert report["ok"], report


@pytest.mark.parametrize("asset", [INDUSTRY_ASSET, MARKET_ENV_ASSET])
def test_broadcast_tamper_detected(tmp_path, asset):
    df = _broadcast_frame(asset)
    path = os.path.join(str(tmp_path), f"{asset.data_type}.parquet")
    df.to_parquet(path)
    write_asset_manifest(path, asset, df)

    manifest = _manifest_of(path)
    manifest["schema_hash"] = "deadbeef"
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    report = validate_asset_manifest(path, asset)
    assert not report["ok"]
    assert any("schema_hash" in s for s in report["mismatches"])


@pytest.mark.parametrize("asset", [INDUSTRY_ASSET, MARKET_ENV_ASSET])
def test_broadcast_legacy_manifestless_read_is_lenient(tmp_path, asset):
    """A manifest-less broadcast file (predates the contract) reads fine."""
    df = _broadcast_frame(asset)
    path = os.path.join(str(tmp_path), f"{asset.data_type}.parquet")
    df.to_parquet(path)

    raw = pd.read_parquet(path)
    check_asset_read(path, asset, raw)  # lenient: no raise, no manifest

    with pytest.raises(ValueError, match="manifest missing"):
        check_asset_read(path, asset, raw, require_valid_manifest=True)
