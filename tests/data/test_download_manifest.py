"""Download-run manifest tests (review v8 §二-2).

A partially-successful download ("4990 of 5000, 10 failed, program said
success") must never be mistaken for complete.  The manifest records requested
vs success vs failed vs missing so training / QA can read the true coverage.
"""
import json

from stoke_ml.data.download_manifest import default_path, load_manifest, write_manifest


def test_write_manifest_full_report(tmp_path):
    path = default_path(str(tmp_path))
    manifest = write_manifest(
        path, market="a_shares", start_date="2024-01-01", end_date="2024-06-30",
        requested=[f"{i:06d}" for i in range(1, 501)],
        failed=["000002", "000003"],
        on_disk={f"{i:06d}" for i in range(1, 501)} - {"000002", "000003"},
        success_count=498,
    )
    assert manifest["requested_count"] == 500
    assert manifest["success_count"] == 498
    assert manifest["failed_count"] == 2
    assert manifest["missing_count"] == 2
    assert manifest["missing"] == ["000002", "000003"]
    assert manifest["all_complete"] is False
    # Round-trips through disk.
    assert load_manifest(path) == manifest
    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f)["schema_version"] == "1.0"


def test_write_manifest_all_complete(tmp_path):
    codes = [f"{i:06d}" for i in range(10)]
    manifest = write_manifest(
        str(tmp_path / "m.json"), market="a_shares",
        start_date="2024-01-01", end_date="2024-06-30",
        requested=codes, failed=[], on_disk=set(codes), success_count=10,
    )
    assert manifest["all_complete"] is True
    assert manifest["missing"] == []


def test_missing_detects_save_failures_too(tmp_path):
    """A fetch that 'succeeded' but never landed on disk must count as missing —
    the on-disk scan, not the in-loop counter, is the coverage truth."""
    codes = [f"{i:06d}" for i in range(5)]
    manifest = write_manifest(
        str(tmp_path / "m.json"), market="a_shares",
        start_date="2024-01-01", end_date="2024-06-30",
        requested=codes, failed=["000003"], on_disk={"000000", "000001", "000002"},
        success_count=4,
    )
    assert manifest["missing"] == ["000003", "000004"]
    assert manifest["missing_count"] == 2
    assert manifest["all_complete"] is False


def test_load_manifest_none_when_absent(tmp_path):
    assert load_manifest(str(tmp_path / "nope.json")) is None
