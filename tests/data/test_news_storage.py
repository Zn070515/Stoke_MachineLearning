"""NewsStorage tests: 3-layer medallion + the gold ``sentiment`` channel's
file-level asset contract (§十七).

The gold layer (``save_daily_sentiment`` / ``load_daily_sentiment``) carries a
``SENTIMENT_ASSET`` manifest (rows / schema hash / date extent / source /
effective-date policy / vintage labels) written atomically beside each
partitioned parquet.  A tampered file is flagged; a manifest-less legacy file
still reads (lenient default), and ``require_valid_manifest=True`` refuses it.
"""
import json
import os

import pandas as pd
import pytest

from stoke_ml.data.asset_contract import (
    parse_era_coverage,
    validate_asset_manifest,
)
from stoke_ml.data.download_resume import write_stock_manifest
from stoke_ml.data.news_storage import SENTIMENT_ASSET, NewsStorage


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


def _gold_df(dates=("2024-01-02", "2024-01-03"), code="000001"):
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "stock_code": [code] * len(dates),
        "sentiment_mean": [0.5, -0.3],
        "sentiment_std": [0.1, 0.2],
        "news_count": [3, 1],
        "positive_ratio": [0.6, 0.0],
        "negative_ratio": [0.0, 0.5],
        "has_news": [True, True],
    })


def _gold_path(tmp_path, year="2024", month="01", code="000001"):
    return os.path.join(str(tmp_path), "a_shares", "sentiment", year, month,
                        f"{code}.parquet")


# ── bronze / silver round-trip (baseline storage tests) ────────────────────

def test_raw_news_round_trip(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_raw_news("000001", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "title": ["A"],
        "body": ["b"],
        "source": ["eastmoney"],
    }))
    loaded = storage.load_raw_news("000001")
    assert len(loaded) == 1
    assert loaded["title"].iloc[0] == "A"


# ── gold asset contract: round-trip ────────────────────────────────────────

def test_sentiment_round_trip_writes_valid_manifest(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())

    path = _gold_path(tmp_path)
    assert os.path.isfile(path + ".manifest.json")
    report = validate_asset_manifest(path, SENTIMENT_ASSET)
    assert report["ok"], report

    manifest = _manifest_of(path)
    # The four §十七 aspects land in the manifest.
    assert manifest["data_type"] == "sentiment"
    assert manifest["effective_date_policy"] == "post_close_next_trading_day"
    assert manifest["vintage_source"] == "immutable_snapshot"
    assert manifest["vintage_transform"] == "model_versioned"
    assert manifest["vintage_pit"] == "verified"
    assert manifest["start"] == "2024-01-02"
    assert manifest["end"] == "2024-01-03"

    loaded = storage.load_daily_sentiment("000001", "2024-01-01", "2024-01-31")
    assert len(loaded) == 2
    assert _tmp_files(str(tmp_path)) == []


# ── §T8 write-end: provider-era fields from the downloader manifest ─────────

def _news_downloader_manifest(tmp_path, code="000001", **kw):
    """Write a news per-stock downloader manifest and return its path."""
    raw_dir = os.path.join(str(tmp_path), "a_shares", "news_raw")
    defaults = dict(
        dataset="news_raw",
        requested_start="2024-01-01", requested_end="2024-01-31",
        effective_start="2024-01-01", effective_end="2024-01-31",
        actual_start="2024-01-01", actual_end="2024-01-31",
        status="COMPLETE", provider_exhausted=True,
    )
    defaults.update(kw)
    return write_stock_manifest(raw_dir, code, **defaults)


def test_sentiment_write_end_records_provider_era_fields(tmp_path):
    """§T8 write-end round-trip: a gold sentiment manifest records the three
    provider-era fields derived from the stock's downloader manifest — a window
    fully retrieved over a provider era is era-covered even when the gold rows
    are sparse (no_event is covered, not a gap)."""
    _news_downloader_manifest(tmp_path, missing_intervals=[["2024-01-15", "2024-01-20"]])
    storage = NewsStorage(str(tmp_path))
    # sparse: events only on 2 days inside the fully-retrieved era window
    storage.save_daily_sentiment(_gold_df())

    path = _gold_path(tmp_path)
    manifest = _manifest_of(path)
    assert manifest["provider_available_start"] == "2024-01-01"
    assert manifest["provider_available_end"] == "2024-01-31"
    # actual [01-01, 01-31] minus the gap -> two disjoint retrieved ranges
    assert manifest["retrieved_ranges"] == [
        ["2024-01-01", "2024-01-14"], ["2024-01-21", "2024-01-31"]]
    assert manifest["known_gaps"] == [["2024-01-15", "2024-01-20"]]

    # the 6-day gap genuinely reduces what was retrieved: 25 of 31 era days
    report = parse_era_coverage(manifest)
    assert report["not_observed"] is False
    assert report["era_covered"] == pytest.approx(25 / 31)


def test_sentiment_write_end_without_downloader_manifest_not_observed(tmp_path):
    """§T8: a stock with NO downloader manifest writes a gold manifest WITHOUT
    the provider-era fields — not_observed (no crash, fields absent)."""
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())
    path = _gold_path(tmp_path)
    manifest = _manifest_of(path)
    assert "provider_available_start" not in manifest
    assert "provider_available_end" not in manifest
    assert "retrieved_ranges" not in manifest
    assert "known_gaps" not in manifest
    assert parse_era_coverage(manifest)["not_observed"] is True


# ── gold asset contract: tamper detection ──────────────────────────────────

def test_sentiment_tampered_schema_hash_detected(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())
    path = _gold_path(tmp_path)
    _rewrite_manifest(path, lambda m: m.update(schema_hash="deadbeef"))

    report = validate_asset_manifest(path, SENTIMENT_ASSET)
    assert not report["ok"]
    assert any("schema_hash" in s for s in report["mismatches"])


def test_sentiment_tampered_vintage_detected(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())
    path = _gold_path(tmp_path)
    _rewrite_manifest(path, lambda m: m.update(vintage_source="latest_revised"))

    report = validate_asset_manifest(path, SENTIMENT_ASSET)
    assert not report["ok"]
    assert any("vintage_source" in s for s in report["mismatches"])


def test_sentiment_require_valid_manifest_raises(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())
    path = _gold_path(tmp_path)
    os.remove(path + ".manifest.json")

    # lenient read still serves the legacy (manifest-less) file
    assert len(storage.load_daily_sentiment("000001", "2024-01-01", "2024-01-31")) == 2
    # formal read refuses it
    with pytest.raises(ValueError, match="manifest missing"):
        storage.load_daily_sentiment(
            "000001", "2024-01-01", "2024-01-31", require_valid_manifest=True,
        )


def test_sentiment_tampered_read_raises_when_required(tmp_path):
    storage = NewsStorage(str(tmp_path))
    storage.save_daily_sentiment(_gold_df())
    path = _gold_path(tmp_path)
    _rewrite_manifest(path, lambda m: m.update(rows=999))

    with pytest.raises(ValueError, match="require_valid_manifest=True"):
        storage.load_daily_sentiment(
            "000001", "2024-01-01", "2024-01-31", require_valid_manifest=True,
        )
