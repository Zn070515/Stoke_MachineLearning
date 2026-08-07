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
from stoke_ml.data.asset_contract import (
    DataAssetContract,
    contract_for_channel,
    downloader_era_fields,
    parse_era_coverage,
    provider_era_fields,
    validate_asset_manifest,
    write_asset_manifest,
)
from stoke_ml.data.download_resume import write_stock_manifest
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


def test_announcement_write_end_records_provider_era_fields(tmp_path):
    """§T8 write-end round-trip (announcement): BOTH the raw and the sentiment
    asset manifests record the three provider-era fields from the downloader
    manifest (which lives in the same a_shares/announcements dir)."""
    base = os.path.join(str(tmp_path), "a_shares", "announcements")
    write_stock_manifest(
        base, "000001", dataset="announcements",
        requested_start="2024-01-01", requested_end="2024-01-31",
        effective_start="2024-01-01", effective_end="2024-01-31",
        actual_start="2024-01-02", actual_end="2024-01-03",
        status="COMPLETE", provider_exhausted=True,
    )
    storage = AnnouncementStorage(str(tmp_path))
    storage.save_raw("000001", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "title": ["A"],
        "sentiment_title": [0.1],
    }))
    raw_path = os.path.join(base, "000001.parquet")
    raw_manifest = _manifest_of(raw_path)
    assert raw_manifest["provider_available_start"] == "2024-01-01"
    assert raw_manifest["retrieved_ranges"] == [["2024-01-02", "2024-01-03"]]
    assert raw_manifest["known_gaps"] == []

    storage.build_daily_sentiment("000001", save=True)
    sent_path = os.path.join(base, "sentiment", "000001.parquet")
    sent_manifest = _manifest_of(sent_path)
    assert sent_manifest["provider_available_start"] == "2024-01-01"
    assert sent_manifest["retrieved_ranges"] == [["2024-01-02", "2024-01-03"]]
    assert parse_era_coverage(sent_manifest)["era_covered"] == 2 / 31


def test_announcement_write_end_without_downloader_manifest_not_observed(tmp_path):
    """§T8: no downloader manifest -> both asset manifests stay era-less."""
    storage = AnnouncementStorage(str(tmp_path))
    storage.save_raw("000001", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "title": ["A"],
    }))
    base = os.path.join(str(tmp_path), "a_shares", "announcements")
    raw_manifest = _manifest_of(os.path.join(base, "000001.parquet"))
    assert "provider_available_start" not in raw_manifest
    assert "retrieved_ranges" not in raw_manifest
    assert parse_era_coverage(raw_manifest)["not_observed"] is True


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


# ── §十七 interface extension: effective-date policy + vintage status ──────

def test_contract_for_channel_fills_declared_vintage():
    asset = contract_for_channel(
        "margin",
        data_type="margin",
        partition="year/month/stock_code",
        extent_column="date",
        column_contract="margin",
        effective_date_policy="record_date",
    )
    assert asset.vintage_source == "immutable_snapshot"
    assert asset.vintage_transform == "raw"
    assert asset.vintage_pit == "verified"


def test_contract_for_channel_undeclared_falls_back_to_unknown():
    asset = contract_for_channel(
        "no_such_channel", data_type="x", partition="y",
    )
    assert asset.vintage_source == "unknown"
    assert asset.vintage_transform == "unknown"
    assert asset.vintage_pit == "unknown"


def test_new_fields_land_in_manifest_and_validate(tmp_path):
    asset = contract_for_channel(
        "capital_flow",
        data_type="capital_flow",
        partition="year/month/stock_code",
        extent_column="date",
        effective_date_policy="record_date",
    )
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "stock_code": ["000001"] * 2,
        "net_inflow": [1.0, 2.0],
    })
    path = os.path.join(str(tmp_path), "000001.parquet")
    df.to_parquet(path, index=False)
    write_asset_manifest(path, asset, df, entity="000001")

    m = _manifest_of(path)
    assert m["effective_date_policy"] == "record_date"
    assert m["vintage_source"] == "immutable_snapshot"
    assert m["vintage_transform"] == "formula_versioned"
    assert m["vintage_pit"] == "verified"

    assert validate_asset_manifest(path, asset)["ok"]


def test_validate_cross_checks_vintage_tamper(tmp_path):
    asset = contract_for_channel(
        "margin", data_type="margin", partition="year/month/stock_code",
        extent_column="date", effective_date_policy="record_date",
    )
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "margin_balance": [1.0],
    })
    path = os.path.join(str(tmp_path), "000001.parquet")
    df.to_parquet(path, index=False)
    write_asset_manifest(path, asset, df, entity="000001")

    _rewrite_manifest(path, lambda m: m.update(
        vintage_source="latest_revised", effective_date_policy="event_date"))

    report = validate_asset_manifest(path, asset)
    assert not report["ok"]
    assert any("vintage_source" in s for s in report["mismatches"])
    assert any("effective_date_policy" in s for s in report["mismatches"])


def test_vintage_extension_leaves_original_contracts_untouched(tmp_path):
    """The v15 T9 assets (None vintage) keep OLD manifest shape + validate."""
    storage = FundamentalStorage(str(tmp_path))
    storage.save(pd.DataFrame({
        "stock_code": ["000001"],
        "report_date": pd.to_datetime(["2024-06-30"]),
        "disclose_date": pd.to_datetime(["2024-08-31"]),
        "roe": [11.0],
    }))
    path = os.path.join(str(tmp_path), "a_shares", "fundamentals", "2024",
                        "Q2", "000001.parquet")
    m = _manifest_of(path)
    assert "vintage_source" not in m
    assert "effective_date_policy" not in m
    assert FUNDAMENTAL_ASSET.vintage_source is None
    assert validate_asset_manifest(path, FUNDAMENTAL_ASSET)["ok"]


def test_validate_skips_new_aspects_when_asset_does_not_declare_them(tmp_path):
    """A legacy-format manifest + a None-vintage asset still validates OK —
    the new aspects are only cross-checked when the asset declares them."""
    asset = DataAssetContract(data_type="sentiment", partition="stock_code")
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "score": [0.5],
    })
    path = os.path.join(str(tmp_path), "000001.parquet")
    df.to_parquet(path, index=False)
    write_asset_manifest(path, asset, df, entity="000001")

    m = _manifest_of(path)
    assert "vintage_source" not in m
    assert validate_asset_manifest(path, asset)["ok"]


# ── §T8 provider-era fields (no_event vs not_observed) ─────────────────────

def _downloader_manifest(tmp_path, code="000001", **kw):
    """Write a per-stock downloader manifest and return its payload."""
    raw_dir = os.path.join(str(tmp_path), "a_shares", "news_raw")
    defaults = dict(
        dataset="news_raw",
        requested_start="2023-01-01", requested_end="2023-01-31",
        effective_start="2023-01-01", effective_end="2023-01-31",
        actual_start="2023-01-01", actual_end="2023-01-31",
        status="COMPLETE", provider_exhausted=True,
    )
    defaults.update(kw)
    path = write_stock_manifest(raw_dir, code, **defaults)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_provider_era_fields_maps_downloader_manifest(tmp_path):
    """§T8 write-end mapping: the 3 era fields derive from the downloader
    manifest's effective window + actual fetch + missing_intervals."""
    dm = _downloader_manifest(
        tmp_path,
        requested_start="2023-01-01", requested_end="2023-12-31",
        effective_start="2023-01-01", effective_end="2023-12-31",
        actual_start="2023-01-01", actual_end="2023-12-31",
        missing_intervals=[["2023-06-01", "2023-06-30"]],
    )
    assert provider_era_fields(dm) == {
        "provider_available_start": "2023-01-01",
        "provider_available_end": "2023-12-31",
        "retrieved_ranges": [["2023-01-01", "2023-05-31"],
                             ["2023-07-01", "2023-12-31"]],
        "known_gaps": [["2023-06-01", "2023-06-30"]],
    }


def test_provider_era_fields_restricts_retrieval_to_era_window(tmp_path):
    """A fetch that starts after the era window records ONLY the overlap as
    retrieved — the un-fetched head of the era is not claimed as observed."""
    dm = _downloader_manifest(
        tmp_path,
        requested_start="2023-01-01", requested_end="2023-12-31",
        effective_start="2023-01-01", effective_end="2023-12-31",
        actual_start="2023-03-01", actual_end="2023-08-31",
    )
    fields = provider_era_fields(dm)
    assert fields["retrieved_ranges"] == [["2023-03-01", "2023-08-31"]]


def test_provider_era_fields_empty_without_manifest_or_window(tmp_path):
    """No downloader manifest → {}; a manifest with no window → {} — in both
    cases the stock is not_observed (no provider era recorded)."""
    assert provider_era_fields(None) == {}
    assert provider_era_fields({}) == {}
    dm = _downloader_manifest(
        tmp_path, requested_start=None, requested_end=None,
        effective_start=None, effective_end=None)
    assert provider_era_fields(dm) == {}


def test_parse_era_coverage_full_retrieval_ignores_zero_rows():
    """§T8 acceptance (1): a provider window FULLY covered by retrieved_ranges
    → era_covered == 1.0 even when the file's rows are sparse/absent — a day
    inside a retrieved range is OBSERVED whether or not an event happened
    (no_event is legitimate, not a data gap)."""
    report = parse_era_coverage({
        "provider_available_start": "2023-01-01",
        "provider_available_end": "2023-01-10",
        "retrieved_ranges": [["2023-01-01", "2023-01-10"]],
        "known_gaps": [],
    })
    assert report["not_observed"] is False
    assert report["era_covered"] == 1.0


def test_parse_era_coverage_not_observed_without_window():
    """§T8 acceptance (2): a manifest with NO provider window / retrieved_ranges
    → not_observed, era_covered None — excluded from the era numerator, never
    reported as zero coverage."""
    for manifest in ({}, {"retrieved_ranges": [["2023-01-01", "2023-01-10"]]}):
        report = parse_era_coverage(manifest)
        assert report["not_observed"] is True
        assert report["era_covered"] is None


def test_parse_era_coverage_partial_retrieval_is_fraction():
    """§T8 acceptance (3): a window 60% covered → era_covered ≈ 0.6."""
    report = parse_era_coverage({
        "provider_available_start": "2023-01-01",
        "provider_available_end": "2023-01-10",
        "retrieved_ranges": [["2023-01-01", "2023-01-06"]],
    })
    assert report["not_observed"] is False
    assert report["era_covered"] == 0.6


def test_parse_era_coverage_gap_reduces_covered_days():
    """A known gap inside the window reduces the covered fraction — the gap is
    retrieved_ranges already split, so era_covered reflects the retrieval."""
    report = parse_era_coverage({
        "provider_available_start": "2023-01-01",
        "provider_available_end": "2023-01-10",
        "retrieved_ranges": [["2023-01-01", "2023-01-03"],
                             ["2023-01-06", "2023-01-10"]],
        "known_gaps": [["2023-01-04", "2023-01-05"]],
    })
    assert report["era_covered"] == 0.8


def test_parse_era_coverage_clamps_retrieval_outside_window():
    """A retrieved range extending past the era window is clamped to it."""
    report = parse_era_coverage({
        "provider_available_start": "2023-01-04",
        "provider_available_end": "2023-01-07",
        "retrieved_ranges": [["2023-01-01", "2023-01-10"]],
    })
    assert report["era_covered"] == 1.0  # 4 of 4 era days observed


def test_downloader_era_fields_reads_manifest_and_round_trips(tmp_path):
    """§T8 write-end unit: downloader_era_fields(raw_dir, code) maps the on-disk
    downloader manifest, and the fields land in a write_asset_manifest call."""
    _downloader_manifest(tmp_path, code="000001")
    fields = downloader_era_fields(
        os.path.join(str(tmp_path), "a_shares", "news_raw"), "000001")
    assert fields["provider_available_start"] == "2023-01-01"
    assert fields["retrieved_ranges"] == [["2023-01-01", "2023-01-31"]]

    asset = DataAssetContract(data_type="sentiment", partition="stock_code")
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-02"]),
                       "stock_code": ["000001"], "has_news": [False]})
    path = os.path.join(str(tmp_path), "000001.parquet")
    df.to_parquet(path, index=False)
    write_asset_manifest(path, asset, df, entity="000001", **fields)
    m = _manifest_of(path)
    assert m["provider_available_start"] == "2023-01-01"
    assert m["provider_available_end"] == "2023-01-31"
    assert m["retrieved_ranges"] == [["2023-01-01", "2023-01-31"]]
    assert m["known_gaps"] == []
    # the full chain: the written gold manifest parses back to full era coverage
    assert parse_era_coverage(m)["era_covered"] == 1.0


def test_downloader_era_fields_missing_manifest_is_empty(tmp_path):
    """A stock with no downloader manifest maps to {} — not_observed, no crash."""
    assert downloader_era_fields(
        os.path.join(str(tmp_path), "a_shares", "news_raw"), "999999") == {}
