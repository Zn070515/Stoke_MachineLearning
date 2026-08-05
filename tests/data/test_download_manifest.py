"""Download-run manifest tests.

A partially-successful download ("4990 of 5000, 10 failed, program said
success") must never be mistaken for complete.  The manifest records requested
vs success vs failed vs missing so training / QA can read the true coverage.
``complete`` is the caller-supplied set of codes that validated AND covered the
requested range — never a mere "file exists" scan (§五-4).
"""
import json
import os

from stoke_ml.data.download_manifest import (
    default_path, load_manifest, run_manifest_path, write_manifest, write_run_manifest,
)


def test_write_manifest_full_report(tmp_path):
    path = default_path(str(tmp_path))
    manifest = write_manifest(
        path, market="a_shares", start_date="2024-01-01", end_date="2024-06-30",
        requested=[f"{i:06d}" for i in range(1, 501)],
        failed=["000002", "000003"],
        complete={f"{i:06d}" for i in range(1, 501)} - {"000002", "000003"},
        success_count=498,
    )
    assert manifest["requested_count"] == 500
    assert manifest["success_count"] == 498
    assert manifest["failed_count"] == 2
    assert manifest["complete_count"] == 498
    assert manifest["missing_count"] == 2
    assert manifest["missing"] == ["000002", "000003"]
    assert manifest["all_complete"] is False
    assert manifest["status"] == "complete"
    assert manifest["requested_end"] is None
    assert manifest["effective_end"] is None
    assert manifest["latest_available_end"] is None
    # §P0-4: the full request and the validated-complete set are persisted so
    # "is the ENTIRE requested universe complete" is auditable after the fact.
    assert manifest["requested"] == [f"{i:06d}" for i in range(1, 501)]
    assert manifest["complete"] == [
        f"{i:06d}" for i in range(1, 501) if i not in (2, 3)
    ]
    # Round-trips through disk.
    assert load_manifest(path) == manifest
    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f)["schema_version"] == "1.3"


def test_write_manifest_all_complete(tmp_path):
    codes = [f"{i:06d}" for i in range(10)]
    manifest = write_manifest(
        str(tmp_path / "m.json"), market="a_shares",
        start_date="2024-01-01", end_date="2024-06-30",
        requested=codes, failed=[], complete=set(codes), success_count=10,
    )
    assert manifest["all_complete"] is True
    assert manifest["missing"] == []


def test_missing_detects_save_failures_too(tmp_path):
    """A fetch that 'succeeded' but never validated as complete must count as
    missing — the caller's validated set, not the in-loop counter, is truth."""
    codes = [f"{i:06d}" for i in range(5)]
    manifest = write_manifest(
        str(tmp_path / "m.json"), market="a_shares",
        start_date="2024-01-01", end_date="2024-06-30",
        requested=codes, failed=["000003"], complete={"000000", "000001", "000002"},
        success_count=4,
    )
    assert manifest["missing"] == ["000003", "000004"]
    assert manifest["missing_count"] == 2
    assert manifest["all_complete"] is False


def test_load_manifest_none_when_absent(tmp_path):
    assert load_manifest(str(tmp_path / "nope.json")) is None


def test_bounded_future_end_never_claims_full_coverage(tmp_path):
    """§七-2: an explicit future end whose request got bounded to the latest
    available trading day is recorded as ``bounded_complete`` and must NOT
    report ``all_complete`` — even when every requested stock is on disk —
    because the run did not cover the original request."""
    codes = [f"{i:06d}" for i in range(3)]
    manifest = write_manifest(
        str(tmp_path / "b.json"), market="a_shares",
        start_date="2020-01-01",
        requested_end="2026-12-31",
        effective_end="2026-08-04",
        latest_available_end="2026-08-04",
        requested=codes, failed=[], complete=set(codes), success_count=3,
    )
    assert manifest["status"] == "bounded_complete"
    assert manifest["requested_end"] == "2026-12-31"
    assert manifest["effective_end"] == "2026-08-04"
    assert manifest["latest_available_end"] == "2026-08-04"
    assert manifest["missing"] == []
    assert manifest["all_complete"] is False  # must not claim the future range


def test_bounded_end_equal_to_available_is_plain_complete(tmp_path):
    """A request whose end equals the latest available day is NOT bounded."""
    codes = [f"{i:06d}" for i in range(3)]
    manifest = write_manifest(
        str(tmp_path / "c.json"), market="a_shares",
        requested_end="2026-08-04", effective_end="2026-08-04",
        requested=codes, failed=[], complete=set(codes), success_count=3,
    )
    assert manifest["status"] == "complete"
    assert manifest["all_complete"] is True


def test_write_run_manifest_dataset_scoped(tmp_path):
    """Dataset-scoped wrapper writes to {data_dir}/{dataset}/download_manifest.json."""
    manifest = write_run_manifest(
        str(tmp_path), "a_shares/earnings",
        start_date="2026-03-01", end_date="2026-03-31",
        requested=["forecasts", "express"], failed=["express"],
        complete={"forecasts"},
    )
    assert run_manifest_path(str(tmp_path), "a_shares/earnings") == \
        os.path.join(str(tmp_path), "a_shares/earnings", "download_manifest.json")
    assert manifest["market"] == "a_shares/earnings"
    assert manifest["missing"] == ["express"]
    assert manifest["all_complete"] is False
    # success_count defaults to len(complete) when not passed.
    assert manifest["success_count"] == 1


def test_write_run_manifest_success_count_and_skipped(tmp_path):
    manifest = write_run_manifest(
        str(tmp_path), "a_shares/index_constituents_hist",
        start_date="2026-01-01", end_date="2026-06-01",
        requested=["000300/2026-01-01", "000905/2026-01-01", "000905/2026-02-01"],
        failed=[], complete={"000300/2026-01-01", "000905/2026-01-01"},
        success_count=2, skipped_existing_count=1,
    )
    assert manifest["missing"] == ["000905/2026-02-01"]
    assert manifest["all_complete"] is False
    assert manifest["skipped_existing_count"] == 1
