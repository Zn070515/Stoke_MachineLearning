"""Asset-contract tests for auxiliary data stores (§十三).

The refactor adds file-level governance to the auxiliary stores: an atomically
written manifest sidecar per parquet, a schema-hash / rows / extent / data-type
cross-check that flags a tampered file, and an atomic-commit guarantee that a
failed write leaves the prior file + manifest untouched while a successful write
leaves no ``.tmp`` residue.

Backward-compat contract: default reads stay lenient — a manifest-less file is
read (legacy), only a PRESENT-but-mismatched manifest is flagged (warning log).
Formal reads (``require_valid_manifest=True``) raise on either.
"""
import json
import logging
import os

import pandas as pd
import pytest

from stoke_ml.data.announcement_storage import (
    ANNOUNCEMENT_ASSET,
    ANNOUNCEMENT_SENTIMENT_ASSET,
    AnnouncementStorage,
)
from stoke_ml.data.asset_contract import validate_asset_manifest
from stoke_ml.data.etf_storage import ETF_FLOW_ASSET, ETFStorage
from stoke_ml.data.fundamental_storage import FUNDAMENTAL_ASSET, FundamentalStorage


def _manifest_of(parquet_path: str) -> dict:
    with open(parquet_path + ".manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _tmp_files(root: str) -> list[str]:
    return [
        os.path.join(d, f)
        for d, _dirs, files in os.walk(root)
        for f in files if ".tmp" in f
    ]


def _rewrite_manifest(path: str, mutator) -> None:
    manifest = _manifest_of(path)
    mutator(manifest)
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


# ── round-trip ─────────────────────────────────────────────────────────────

def test_fundamental_round_trip_writes_valid_manifest(tmp_path):
    storage = FundamentalStorage(str(tmp_path))
    df = pd.DataFrame({
        "stock_code": ["000001"] * 3,
        "report_date": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"]),
        "disclose_date": pd.to_datetime(["2024-04-30", "2024-08-31", "2024-10-31"]),
        "roe": [10.0, 11.0, 12.0],
    })
    storage.save(df)

    path = os.path.join(
        str(tmp_path), "a_shares", "fundamentals", "2024", "Q2", "000001.parquet",
    )
    assert os.path.isfile(path + ".manifest.json")
    report = validate_asset_manifest(path, FUNDAMENTAL_ASSET)
    assert report["ok"], report

    loaded = storage.load("000001", "2024-01-01", "2024-12-31")
    assert len(loaded) == 3
    assert list(loaded["report_date"]) == [
        pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30"),
        pd.Timestamp("2024-09-30"),
    ]
    assert _tmp_files(str(tmp_path)) == []


def test_etf_round_trip_writes_valid_manifest(tmp_path):
    storage = ETFStorage(str(tmp_path))
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
        "sector_name": ["半导体"] * 3,
        "net_inflow": [1.0, 2.0, 3.0],
    })
    storage.save(df)

    path = os.path.join(
        str(tmp_path), "a_shares", "etf_flow", "2024", "01", "sector_半导体.parquet",
    )
    assert os.path.isfile(path + ".manifest.json")
    assert validate_asset_manifest(path, ETF_FLOW_ASSET)["ok"]

    loaded = storage.load_sector_flow("半导体", "2024-01-01", "2024-12-31")
    assert len(loaded) == 3
    assert list(loaded["net_inflow"]) == [1.0, 2.0, 3.0]
    assert _tmp_files(str(tmp_path)) == []


def test_announcement_raw_and_sentiment_round_trip(tmp_path):
    storage = AnnouncementStorage(str(tmp_path))
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "stock_code": ["000001"] * 3,
        "title": ["A", "B", "C"],
        "sentiment_title": [0.1, -0.2, 0.3],
    })
    storage.save_raw("000001", raw)

    raw_path = os.path.join(str(tmp_path), "a_shares", "announcements",
                            "000001.parquet")
    assert validate_asset_manifest(raw_path, ANNOUNCEMENT_ASSET)["ok"]
    assert len(storage.load_raw("000001")) == 3

    daily = storage.build_daily_sentiment("000001", save=True)
    sent_path = os.path.join(str(tmp_path), "a_shares", "announcements",
                             "sentiment", "000001.parquet")
    assert os.path.isfile(sent_path + ".manifest.json")
    assert validate_asset_manifest(sent_path, ANNOUNCEMENT_SENTIMENT_ASSET)["ok"]

    loaded = storage.load_daily_sentiment("000001")
    assert list(loaded["date"]) == list(daily["date"])
    assert _tmp_files(str(tmp_path)) == []


# ── tamper detection ───────────────────────────────────────────────────────

def test_tampered_schema_hash_detected(tmp_path):
    storage = FundamentalStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "stock_code": ["000001"] * 2,
        "report_date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "disclose_date": pd.to_datetime(["2024-04-30", "2024-08-31"]),
        "roe": [10.0, 11.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "fundamentals", "2024",
                        "Q2", "000001.parquet")
    _rewrite_manifest(path, lambda m: m.update(schema_hash="deadbeef"))

    report = validate_asset_manifest(path, FUNDAMENTAL_ASSET)
    assert not report["ok"]
    assert any("schema_hash" in s for s in report["mismatches"])


def test_tampered_rows_detected(tmp_path):
    storage = ETFStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "sector_name": ["半导体"] * 2,
        "net_inflow": [1.0, 2.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "etf_flow", "2024", "01",
                        "sector_半导体.parquet")
    _rewrite_manifest(path, lambda m: m.update(rows=999))

    report = validate_asset_manifest(path, ETF_FLOW_ASSET)
    assert not report["ok"]
    assert any("rows" in s for s in report["mismatches"])


def test_tampered_data_type_detected(tmp_path):
    storage = AnnouncementStorage(str(tmp_path))
    storage.save_raw("000001", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "stock_code": ["000001"],
        "title": ["A"],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "announcements",
                        "000001.parquet")
    _rewrite_manifest(path, lambda m: m.update(data_type="northbound"))

    report = validate_asset_manifest(path, ANNOUNCEMENT_ASSET)
    assert not report["ok"]
    assert any("data_type" in s for s in report["mismatches"])


def test_truncated_parquet_detected(tmp_path):
    storage = FundamentalStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "stock_code": ["000001"],
        "report_date": pd.to_datetime(["2024-06-30"]),
        "disclose_date": pd.to_datetime(["2024-08-31"]),
        "roe": [11.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "fundamentals", "2024",
                        "Q2", "000001.parquet")
    with open(path, "wb") as f:
        f.write(b"not a parquet")

    report = validate_asset_manifest(path, FUNDAMENTAL_ASSET)
    assert not report["ok"]
    assert report["reason"] and "unreadable" in report["reason"]


def test_tampered_manifest_read_logs_warning_but_returns(tmp_path, caplog):
    """Default (lenient) read flags a mismatch via warning but still returns."""
    storage = FundamentalStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "stock_code": ["000001"],
        "report_date": pd.to_datetime(["2024-06-30"]),
        "disclose_date": pd.to_datetime(["2024-08-31"]),
        "roe": [11.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "fundamentals", "2024",
                        "Q2", "000001.parquet")
    _rewrite_manifest(path, lambda m: m.update(schema_hash="deadbeef"))

    with caplog.at_level(logging.WARNING, logger="stoke_ml.data.asset_contract"):
        loaded = storage.load("000001", "2024-01-01", "2024-12-31")

    assert len(loaded) == 1
    assert any("mismatch" in r.getMessage() for r in caplog.records)


# ── atomic commit ──────────────────────────────────────────────────────────

def test_failed_write_preserves_prior_file_and_manifest(tmp_path, monkeypatch):
    storage = FundamentalStorage(str(tmp_path))
    df = pd.DataFrame({
        "stock_code": ["000001"],
        "report_date": pd.to_datetime(["2024-06-30"]),
        "disclose_date": pd.to_datetime(["2024-08-31"]),
        "roe": [11.0],
    })
    storage.save(df)  # first write lands file + manifest

    path = os.path.join(str(tmp_path), "a_shares", "fundamentals", "2024",
                        "Q2", "000001.parquet")
    prior_bytes = open(path, "rb").read()
    prior_manifest = _manifest_of(path)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(RuntimeError):
        storage.save(df)  # same partition -> would overwrite

    assert open(path, "rb").read() == prior_bytes
    assert _manifest_of(path) == prior_manifest
    assert _tmp_files(str(tmp_path)) == []


def test_require_valid_manifest_raises_on_missing_manifest(tmp_path):
    storage = ETFStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "sector_name": ["半导体"],
        "net_inflow": [1.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "etf_flow", "2024", "01",
                        "sector_半导体.parquet")
    os.remove(path + ".manifest.json")

    # lenient read still serves the legacy (manifest-less) file
    assert len(storage.load_sector_flow("半导体", "2024-01-01", "2024-12-31")) == 1
    # formal read refuses it
    with pytest.raises(ValueError, match="manifest missing"):
        storage.load_sector_flow(
            "半导体", "2024-01-01", "2024-12-31", require_valid_manifest=True,
        )


def test_require_valid_manifest_raises_on_mismatch(tmp_path):
    storage = AnnouncementStorage(str(tmp_path))
    storage.save_raw("000001", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "stock_code": ["000001"],
        "title": ["A"],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "announcements",
                        "000001.parquet")
    _rewrite_manifest(path, lambda m: m.update(rows=7))

    with pytest.raises(ValueError, match="require_valid_manifest=True"):
        storage.load_raw("000001", require_valid_manifest=True)
