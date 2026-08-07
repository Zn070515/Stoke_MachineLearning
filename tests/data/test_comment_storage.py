"""CommentStorage tests: per-stock daily round-trip + the ``comment`` channel's
file-level asset contract (§十七).

The daily per-stock files (``save_daily`` / ``load_daily``) carry a
``COMMENT_SENTIMENT_ASSET`` manifest written atomically beside each parquet.
A tampered file is flagged; a manifest-less legacy file still reads (lenient
default), and ``require_valid_manifest=True`` refuses it.
"""
import json
import os

import pandas as pd
import pytest

from stoke_ml.data.asset_contract import validate_asset_manifest
from stoke_ml.data.comment_storage import COMMENT_SENTIMENT_ASSET, CommentStorage


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


def _daily_df(dates=("2024-01-02", "2024-01-03"), code="000001"):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "stock_code": [code] * n,
        "comment_score": [0.5, -0.3][:n],
        "comment_attention": [1.0, 2.0][:n],
        "comment_institution": [0.1, 0.2][:n],
        "comment_trend": [0.0, -0.1][:n],
    })


def _daily_path(tmp_path, code="000001"):
    return os.path.join(str(tmp_path), "a_shares", "comment_sentiment",
                        f"{code}.parquet")


# ── round-trip ─────────────────────────────────────────────────────────────

def test_comment_round_trip_writes_valid_manifest(tmp_path):
    storage = CommentStorage(str(tmp_path))
    storage.save_daily(_daily_df())

    path = _daily_path(tmp_path)
    assert os.path.isfile(path + ".manifest.json")
    report = validate_asset_manifest(path, COMMENT_SENTIMENT_ASSET)
    assert report["ok"], report

    manifest = _manifest_of(path)
    assert manifest["data_type"] == "comment_sentiment"
    assert manifest["effective_date_policy"] == "record_date"
    assert manifest["vintage_source"] == "immutable_snapshot"
    assert manifest["vintage_transform"] == "raw"
    assert manifest["vintage_pit"] == "verified"
    assert manifest["start"] == "2024-01-02"
    assert manifest["end"] == "2024-01-03"

    loaded = storage.load_daily("000001", "2024-01-01", "2024-01-31")
    assert len(loaded) == 2
    assert list(loaded["comment_score"]) == [0.5, -0.3]
    assert _tmp_files(str(tmp_path)) == []


def test_comment_save_daily_merges_with_existing(tmp_path):
    storage = CommentStorage(str(tmp_path))
    storage.save_daily(_daily_df(("2024-01-02",)))
    storage.save_daily(_daily_df(("2024-01-03", "2024-01-04")))

    loaded = storage.load_daily("000001", "2024-01-01", "2024-01-31")
    assert len(loaded) == 3  # merged, deduped by date (keep=last)


# ── tamper detection ───────────────────────────────────────────────────────

def test_comment_tampered_rows_detected(tmp_path):
    storage = CommentStorage(str(tmp_path))
    storage.save_daily(_daily_df())
    path = _daily_path(tmp_path)
    _rewrite_manifest(path, lambda m: m.update(rows=999))

    report = validate_asset_manifest(path, COMMENT_SENTIMENT_ASSET)
    assert not report["ok"]
    assert any("rows" in s for s in report["mismatches"])


def test_comment_tampered_data_type_detected(tmp_path):
    storage = CommentStorage(str(tmp_path))
    storage.save_daily(_daily_df())
    path = _daily_path(tmp_path)
    _rewrite_manifest(path, lambda m: m.update(data_type="northbound"))

    report = validate_asset_manifest(path, COMMENT_SENTIMENT_ASSET)
    assert not report["ok"]
    assert any("data_type" in s for s in report["mismatches"])


def test_comment_require_valid_manifest_raises(tmp_path):
    storage = CommentStorage(str(tmp_path))
    storage.save_daily(_daily_df())
    path = _daily_path(tmp_path)
    os.remove(path + ".manifest.json")

    # lenient read still serves the legacy (manifest-less) file
    assert len(storage.load_daily("000001", "2024-01-01", "2024-01-31")) == 2
    # formal read refuses it
    with pytest.raises(ValueError, match="manifest missing"):
        storage.load_daily(
            "000001", "2024-01-01", "2024-01-31", require_valid_manifest=True,
        )
