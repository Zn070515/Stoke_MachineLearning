"""Unit tests for download_index_hist.py membership reconstruction (§T2).

Covers:
- rebuild_membership: out_date is the FIRST ABSENT snapshot month (so the
  half-open interval ``in_date <= d < out_date`` every consumer reads holds),
  never the last-present snapshot month.
- _finalize: the run-manifest write is never swallowed; partial runs exit
  non-zero and skip the membership rebuild; manifest-write failures exit
  non-zero via write_run_manifest_or_exit.

None of these touch the network or baostock.
"""

import os

import pandas as pd
import pytest

from scripts.production.download_index_hist import _finalize, rebuild_membership


def _snapshot(snap_dir, index, date, rows):
    """rows: list of (stock_code, update_date_str).  A stock ABSENT from a
    snapshot simply has no row (that is how Baostock absence manifests)."""
    d = snap_dir / index
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "code": [r[0] for r in rows],
        "updateDate": [pd.Timestamp(r[1]) for r in rows],
        "query_date": pd.Timestamp(date),
    })
    df.to_parquet(d / f"{date}.parquet", index=False)


def _mem(snap):
    return rebuild_membership(str(snap)).set_index("stock_code")


# ── rebuild_membership: out_date = first absent snapshot ──────────────

def test_out_date_is_first_absent_snapshot(tmp_path):
    """The audit's exact bug: B present Jan+Feb, absent Mar → out_date must be
    2026-03-01 (the FIRST ABSENT snapshot), NOT 2026-02-01 (last present)."""
    snap = tmp_path / "snapshots"
    _snapshot(snap, "000300", "2026-01-01",
              [("sh.600000", "2025-12-31"), ("sh.600004", "2025-12-31")])
    _snapshot(snap, "000300", "2026-02-01",
              [("sh.600000", "2026-01-30"), ("sh.600004", "2026-01-30")])
    _snapshot(snap, "000300", "2026-03-01",
              [("sh.600000", "2026-02-27")])

    mem = _mem(snap)
    assert mem.loc["600004", "index_code"] == "000300"
    assert not pd.isna(mem.loc["600004", "out_date"])
    assert mem.loc["600004", "out_date"] == pd.Timestamp("2026-03-01")
    # A confirmed member day in Feb must satisfy the half-open interval.
    assert mem.loc["600004", "in_date"] <= pd.Timestamp("2026-02-15") \
        < mem.loc["600004", "out_date"]
    # A is still present at the final grid (Mar) → open-ended.
    assert pd.isna(mem.loc["600000", "out_date"])


def test_open_ended_at_final_grid(tmp_path):
    """A present through the last queried snapshot → out_date NaT."""
    snap = tmp_path / "snapshots"
    _snapshot(snap, "000300", "2026-01-01", [("sh.600000", "2025-12-31")])
    _snapshot(snap, "000300", "2026-02-01", [("sh.600000", "2026-01-30")])
    mem = _mem(snap)
    assert pd.isna(mem.loc["600000", "out_date"])


def test_reentry_produces_two_spells(tmp_path):
    """A absent in Feb (no Feb row) → two spells: spell1 closes at the first
    absent month (2026-02-01), spell2 is open-ended at the last grid."""
    snap = tmp_path / "snapshots"
    _snapshot(snap, "000300", "2026-01-01", [("sh.600000", "2025-12-31")])
    _snapshot(snap, "000300", "2026-03-01", [("sh.600000", "2026-02-27")])
    mem = _mem(snap)
    spells = mem.loc["600000"]
    assert len(spells) == 2  # long-form: one row per spell
    assert sorted(pd.to_datetime(spells["in_date"])) == [
        pd.Timestamp("2025-12-31"), pd.Timestamp("2026-02-27")]
    outs = set(pd.Timestamp(v) if not pd.isna(v) else None
               for v in spells["out_date"])
    assert pd.Timestamp("2026-02-01") in outs   # first absent month (Feb)
    assert None in outs                          # open-ended second spell


def test_still_active_boundary_last_present_before_final_grid(tmp_path):
    """The final grid month was queried successfully but the stock was absent
    there → out_date is the concrete first-absent month, not NaT."""
    snap = tmp_path / "snapshots"
    _snapshot(snap, "000300", "2026-01-01", [("sh.600000", "2025-12-31")])
    _snapshot(snap, "000300", "2026-02-01", [("sh.600000", "2026-01-30")])
    _snapshot(snap, "000300", "2026-03-01", [("sh.600004", "2026-02-27")])
    mem = _mem(snap)
    assert not pd.isna(mem.loc["600000", "out_date"])
    assert mem.loc["600000", "out_date"] == pd.Timestamp("2026-03-01")


# ── _finalize: manifest write + non-zero exit on partial/failure ──────

def _finalize_args(tmp_path):
    data_dir = tmp_path / "data"
    base = data_dir / "a_shares" / "index_constituents_hist"
    snap = base / "snapshots"
    _snapshot(snap, "000300", "2026-01-01",
              [("sh.600000", "2025-12-31"), ("sh.600004", "2025-12-31")])
    _snapshot(snap, "000300", "2026-02-01",
              [("sh.600000", "2026-01-30"), ("sh.600004", "2026-01-30")])
    _snapshot(snap, "000300", "2026-03-01",
              [("sh.600000", "2026-02-27")])
    requested = ["000300/2026-01-01", "000300/2026-02-01", "000300/2026-03-01"]
    return data_dir, snap, base, requested


def _call_finalize(data_dir, base, snap, requested, failed, complete, **kw):
    return _finalize(str(data_dir), "2026-01-01", "2026-03-01", requested,
                     failed=failed, complete=complete, skipped_existing=0,
                     snap_dir=str(snap), base=str(base), **kw)


def test_finalize_success_writes_manifest_and_membership(tmp_path):
    data_dir, snap, base, requested = _finalize_args(tmp_path)
    rc = _call_finalize(data_dir, base, snap, requested,
                        failed=[], complete=set(requested))
    assert rc == 0
    assert os.path.isfile(data_dir / "a_shares" / "index_constituents_hist"
                          / "download_manifest.json")
    mem_path = base / "membership.parquet"
    assert os.path.isfile(mem_path)
    mem = pd.read_parquet(mem_path)
    assert len(mem) == 2  # 600000 (open-ended NaT) + 600004 (first-absent Mar)


def test_finalize_partial_run_exits_nonzero_and_skips_membership(tmp_path):
    data_dir, snap, base, requested = _finalize_args(tmp_path)
    failed = ["000300/2026-01-01"]
    rc = _call_finalize(data_dir, base, snap, requested,
                        failed=failed, complete=set(requested) - set(failed))
    assert rc == 1
    # The manifest is STILL written (partial status recorded honestly)...
    assert os.path.isfile(data_dir / "a_shares" / "index_constituents_hist"
                          / "download_manifest.json")
    # ...but a partial run never rebuilds membership.
    assert not os.path.isfile(base / "membership.parquet")


def test_finalize_manifest_write_failure_exits_nonzero(tmp_path, monkeypatch):
    from stoke_ml.data import download_manifest as dm

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    # write_run_manifest_or_exit internally calls the shared write_run_manifest;
    # patching the helper makes it raise, which or_exit converts to SystemExit(1).
    monkeypatch.setattr(dm, "write_run_manifest", _boom)
    data_dir, snap, base, requested = _finalize_args(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _call_finalize(data_dir, base, snap, requested,
                       failed=[], complete=set(requested))
    assert excinfo.value.code == 1
