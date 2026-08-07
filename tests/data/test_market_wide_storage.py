"""MarketWideStorage.replace_range degradation-guard tests.

A destructive window replacement must never delete previously-present dates
(or columns) and silently write a partial output back.  The guard rejects such
writes by default; ``force=True`` opts into an intentional rewrite.  Provenance
manifests record every accepted/rejected write for later auditing.
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stoke_ml.data.asset_contract import validate_asset_manifest
from stoke_ml.data.market_wide_storage import MARKET_WIDE_ASSETS, MarketWideStorage


def _frame(dates, code="000001", extra=None):
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "stock_code": code,
        "value": [float(i) for i in range(len(dates))],
    })
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def _storage(tmp_path, data_type="margin"):
    return MarketWideStorage(str(tmp_path), data_type)


PROV = {
    "run_id": "run-1",
    "source_snapshot": "raw:margin:3stocks",
    "config_hash": "abc123",
    "git_commit": "deadbeef",
}


def _read_manifest(tmp_path, code="000001", data_type="margin"):
    path = os.path.join(
        str(tmp_path), "a_shares", data_type, ".manifests", f"{code}.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _asset_manifest_of(parquet_path: str) -> dict:
    with open(parquet_path + ".manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _rewrite_asset_manifest(path: str, mutator) -> None:
    manifest = _asset_manifest_of(path)
    mutator(manifest)
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


ADOPTED_HEADLINE_CHANNELS = sorted(MARKET_WIDE_ASSETS)


class TestAssetContract:
    """§十七: the 7 headline channels persisted through MarketWideStorage carry
    a file-level asset manifest alongside the provenance sidecar."""

    @pytest.mark.parametrize("data_type", ADOPTED_HEADLINE_CHANNELS)
    def test_adopted_channel_round_trip_writes_valid_manifest(self, tmp_path,
                                                              data_type):
        s = _storage(tmp_path, data_type)
        s.save(_frame(["2024-01-02", "2024-01-03"], extra={"v2": 1.0}))

        path = os.path.join(str(tmp_path), "a_shares", data_type,
                            "000001.parquet")
        assert os.path.isfile(path + ".manifest.json")
        report = validate_asset_manifest(path, MARKET_WIDE_ASSETS[data_type])
        assert report["ok"], report

        manifest = _asset_manifest_of(path)
        assert manifest["data_type"] == data_type
        assert manifest["vintage_source"] == "immutable_snapshot"
        assert manifest["start"] == "2024-01-02"
        assert manifest["end"] == "2024-01-03"

        out = s.load("000001", "2024-01-01", "2024-01-31")
        assert len(out) == 2

    @pytest.mark.parametrize("data_type", ADOPTED_HEADLINE_CHANNELS)
    def test_adopted_channel_tamper_raises_when_required(self, tmp_path,
                                                         data_type):
        s = _storage(tmp_path, data_type)
        s.save(_frame(["2024-01-02", "2024-01-03"]))
        path = os.path.join(str(tmp_path), "a_shares", data_type,
                            "000001.parquet")
        _rewrite_asset_manifest(path, lambda m: m.update(rows=999))

        # lenient read still returns the data
        assert len(s.load("000001", "2024-01-01", "2024-01-31")) == 2
        # formal read refuses it
        with pytest.raises(ValueError, match="require_valid_manifest=True"):
            s.load("000001", "2024-01-01", "2024-01-31",
                   require_valid_manifest=True)

    def test_unadopted_data_type_gets_no_manifest(self, tmp_path):
        """A MARKET_DATA_TYPES value outside the headline set stays
        manifest-less (non-breaking) — e.g. ``shareholder``."""
        s = _storage(tmp_path, "shareholder")
        s.save(_frame(["2024-01-02"]))
        path = os.path.join(str(tmp_path), "a_shares", "shareholder",
                            "000001.parquet")
        assert os.path.isfile(path)  # parquet written
        assert not os.path.isfile(path + ".manifest.json")  # no asset manifest
        assert MARKET_WIDE_ASSETS.get("shareholder") is None

    def test_vintage_tamper_detected(self, tmp_path):
        s = _storage(tmp_path, "margin")
        s.save(_frame(["2024-01-02"]))
        path = os.path.join(str(tmp_path), "a_shares", "margin",
                            "000001.parquet")
        _rewrite_asset_manifest(path, lambda m: m.update(vintage_source="unknown"))

        report = validate_asset_manifest(path, MARKET_WIDE_ASSETS["margin"])
        assert not report["ok"]
        assert any("vintage_source" in s_ for s_ in report["mismatches"])


class TestReplaceRangeGuard:
    def test_no_degradation_accepted(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03"]))
        # Same coverage: replace accepted, old rows inside window replaced.
        s.save(
            _frame(["2024-01-01", "2024-01-02", "2024-01-03"], extra={"v2": 1.0}),
            replace_range=True,
        )
        out = s.load("000001", "2024-01-01", "2024-01-03")
        assert len(out) == 3
        assert "v2" in out.columns

    def test_partial_new_data_rejected_preserves_old(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03"]))
        # Only the last date is regenerated -> measured against the intended
        # window [01-01, 01-03] that is 2/3 dates lost (67% > 20%).
        rejected = s.save(
            _frame(["2024-01-03"]), replace_range=True,
            provenance=PROV,
            replace_window=("2024-01-01", "2024-01-03"),
        )
        assert rejected == 1
        # Old file must be untouched.
        out = s.load("000001", "2024-01-01", "2024-01-03")
        assert len(out) == 3

    def test_partial_default_window_preserves_outside_dates(self, tmp_path):
        """Without an explicit replace_window the window defaults to the new
        extent, so rows outside it are preserved (partial-run support)."""
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03"]))
        rejected = s.save(_frame(["2024-01-03"]), replace_range=True)
        assert rejected == 0
        out = s.load("000001", "2024-01-01", "2024-01-03")
        assert len(out) == 3  # 01-01/01-02 preserved, 01-03 refreshed

    def test_mild_degradation_within_threshold_accepted(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
        # Intended window [01-01, 01-05]: new output drops one of five dates ->
        # exactly 20% degradation, allowed at threshold 0.2.
        rejected = s.save(
            _frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            replace_range=True,
            replace_window=("2024-01-01", "2024-01-05"),
        )
        assert rejected == 0
        out = s.load("000001", "2024-01-01", "2024-01-05")
        assert len(out) == 4  # 2024-01-05 dropped (intended rewrite)

    def test_force_bypasses_guard(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03"]))
        rejected = s.save(
            _frame(["2024-01-03"]), replace_range=True, force=True,
            replace_window=("2024-01-01", "2024-01-03"),
        )
        assert rejected == 0
        out = s.load("000001", "2024-01-01", "2024-01-03")
        assert len(out) == 1  # intentional rewrite to one date

    def test_schema_drop_rejected(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02"], extra={"aux": 0.0}))
        # New output drops the 'aux' column -> schema degradation.
        rejected = s.save(
            _frame(["2024-01-01", "2024-01-02"]), replace_range=True,
        )
        assert rejected == 1
        out = s.load("000001", "2024-01-01", "2024-01-02")
        assert "aux" in out.columns  # old file preserved

    def test_return_counts_rejected_across_stocks(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02"], code="000001"))
        s.save(_frame(["2024-01-01", "2024-01-02"], code="600519"))
        # Both stocks would drop 50% of dates within the intended window.
        rejected = s.save(
            pd.concat([
                _frame(["2024-01-02"], code="000001"),
                _frame(["2024-01-02"], code="600519"),
            ], ignore_index=True),
            replace_range=True,
            replace_window=("2024-01-01", "2024-01-02"),
        )
        assert rejected == 2

    def test_first_write_no_existing_no_guard(self, tmp_path):
        s = _storage(tmp_path)
        rejected = s.save(_frame(["2024-01-01"]), replace_range=True)
        assert rejected == 0
        out = s.load("000001", "2024-01-01", "2024-01-01")
        assert len(out) == 1


class TestReplaceRangeManifest:
    def test_accepted_write_records_provenance(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01"]))
        s.save(_frame(["2024-01-01", "2024-01-02"]), replace_range=True, provenance=PROV)
        m = _read_manifest(tmp_path)
        assert m["decision"] == "accepted"
        assert m["run_id"] == "run-1"
        assert m["git_commit"] == "deadbeef"
        assert m["config_hash"] == "abc123"
        assert m["source_snapshot"] == "raw:margin:3stocks"
        assert m["coverage"]["new_dates_in_window"] == 2

    def test_rejected_write_records_decision_and_coverage(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01", "2024-01-02", "2024-01-03"]))
        s.save(_frame(["2024-01-03"]), replace_range=True, provenance=PROV,
               replace_window=("2024-01-01", "2024-01-03"))
        m = _read_manifest(tmp_path)
        assert m["decision"] == "rejected"
        assert m["coverage"]["missing_dates"] == 2
        assert m["coverage"]["degradation_ratio"] > 0.2

    def test_no_provenance_no_manifest(self, tmp_path):
        s = _storage(tmp_path)
        s.save(_frame(["2024-01-01"]))
        s.save(_frame(["2024-01-01", "2024-01-02"]), replace_range=True)
        assert not os.path.exists(os.path.join(
            str(tmp_path), "a_shares", "margin", ".manifests", "000001.json"
        ))


class TestLoadDateErrorReporting:
    """§T18: a corrupt partition must be dropped but ALSO aggregated/reported,
    not silently swallowed by the load_date loop."""

    def _corrupt_partition(self, tmp_path):
        part = os.path.join(str(tmp_path), "a_shares", "margin", "2024", "01")
        os.makedirs(part, exist_ok=True)
        (Path(part) / "corrupt.parquet").write_bytes(b"not a parquet")
        return part

    def test_load_date_reports_unreadable_partitions(self, tmp_path, caplog):
        s = _storage(tmp_path)
        part = self._corrupt_partition(tmp_path)
        pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "stock_code": ["000001", "000001"],
        }).to_parquet(os.path.join(part, "000001.parquet"), index=False)

        with caplog.at_level("WARNING", logger="stoke_ml.data.market_wide_storage"):
            out = s.load_date("2024-01-02")
        # Drop semantics preserved: the valid partition still loads.
        assert len(out) == 1
        assert out["stock_code"].iloc[0] == "000001"
        # §T18: the corrupt partition is aggregated + reported, not silently dropped.
        assert any("Error summary" in r.getMessage() for r in caplog.records)
        assert any("load_date:margin" in r.getMessage() for r in caplog.records)
